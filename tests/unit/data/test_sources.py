"""InkML sources: an extracted directory tree and a tar archive read as a stream.

Every archive in this module is **built here** from the synthetic InkML fixtures
(``synth_inkml_dataset``, seed 1234). Nothing is downloaded: the real
``mathwriting-2024.tgz`` sits behind the ``DATASET_DOWNLOAD`` gate and these tests only prove
that the code path which will read it is correct on an archive of the same shape.
"""

from __future__ import annotations

from pathlib import Path
import tarfile

import pytest

from aicls.core.errors import InputValidationError, NotFoundError
from aicls.data import synth_inkml_dataset
from aicls.data.sources import (
    DirectoryInkmlSource,
    InkmlRecord,
    TarInkmlSource,
    looks_like_tar,
    open_inkml_source,
)

#: The five official MathWriting split directories, in the layout the released archive uses.
OFFICIAL_LAYOUT: dict[str, int] = {
    "train": 5,
    "valid": 2,
    "test": 2,
    "symbols": 2,
    "synthetic": 3,
}


def _official_tree(root: Path, *, seed: int = 1234) -> Path:
    """Write ``mathwriting-2024/<split>/*.inkml`` under ``root`` and return the corpus root."""
    corpus = root / "mathwriting-2024"
    for index, (split, count) in enumerate(OFFICIAL_LAYOUT.items()):
        synth_inkml_dataset(corpus / split, count, seed + 100 * index, prefix=split)
    return corpus


def _tar(archive: Path, tree: Path, *, arcname: str | None = None, mode: str = "w:gz") -> Path:
    """Pack ``tree`` into ``archive`` (gzip by default, like the released ``.tgz``)."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode) as bundle:
        bundle.add(tree, arcname=arcname if arcname is not None else tree.name)
    return archive


@pytest.fixture
def official_archive(tmp_path: Path) -> Path:
    """A ``.tgz`` with the official ``<prefix>/<split>/*.inkml`` layout."""
    return _tar(tmp_path / "mathwriting-2024.tgz", _official_tree(tmp_path / "src"))


# ---------------------------------------------------------------------------------- directory
def test_directory_source_reads_split_subdirectories_in_sorted_order(tmp_path: Path) -> None:
    corpus = _official_tree(tmp_path)
    source = DirectoryInkmlSource(corpus)

    assert source.kind == "directory"
    assert source.split_dir("train") == corpus / "train"
    assert source.source_label("valid") == str(corpus / "valid")

    records = list(source.iter_records(["train", "valid"]))
    assert [split for split, _ in records] == ["train"] * 5 + ["valid"] * 2
    ids = [record.sample_id for split, record in records if split == "train"]
    assert ids == sorted(ids)
    assert all(isinstance(record.payload, Path) for _, record in records)


def test_directory_source_falls_back_to_the_root_for_a_flat_corpus(tmp_path: Path) -> None:
    """A flat directory has no split sub-directories; the root answers whatever is asked."""
    flat = synth_inkml_dataset(tmp_path / "flat", 4, 1234)
    source = DirectoryInkmlSource(flat)

    assert source.split_dir("train") == flat
    assert len(list(source.iter_records(["train"]))) == 4


def test_directory_source_counts_and_caps_without_reading_payloads(tmp_path: Path) -> None:
    corpus = _official_tree(tmp_path)
    counts = DirectoryInkmlSource(corpus).count_splits(["train", "valid", "nope"], cap=3)

    assert counts["train"].n_files == 3 and counts["train"].capped is True
    assert counts["valid"].n_files == 2 and counts["valid"].capped is False
    assert counts["valid"].available is True
    # A split without its own directory falls back to the corpus root, so it sees *everything*
    # below it. That is the historical behaviour of prepare_shards and it is a trap: ask a flat
    # or mislabelled corpus for one split at a time.
    assert counts["nope"].source == str(corpus)
    assert counts["nope"].n_files == 3 and counts["nope"].capped is True


def test_directory_source_rejects_a_missing_root(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError):
        DirectoryInkmlSource(tmp_path / "absent")


# ---------------------------------------------------------------------------------------- tar
def test_tar_source_reads_the_official_split_layout_without_extracting(
    official_archive: Path, tmp_path: Path
) -> None:
    """train/valid/test/symbols/synthetic are found under the archive's top-level directory."""
    source = TarInkmlSource(official_archive)

    assert source.kind == "tar"
    counts = source.count_splits(list(OFFICIAL_LAYOUT))
    assert {name: count.n_files for name, count in counts.items()} == OFFICIAL_LAYOUT
    assert all(count.available for count in counts.values())

    records = list(source.iter_records(list(OFFICIAL_LAYOUT)))
    assert len(records) == sum(OFFICIAL_LAYOUT.values())
    assert {split for split, _ in records} == set(OFFICIAL_LAYOUT)
    assert all(isinstance(record.payload, bytes) for _, record in records)
    assert all(record.payload.startswith(b"<?xml") for _, record in records)
    # nothing was written next to the archive: no extracted tree exists
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mathwriting-2024.tgz", "src"]


def test_tar_source_sample_ids_and_labels_match_the_directory_source(
    tmp_path: Path, official_archive: Path
) -> None:
    """The same corpus through both sources yields the same ids and the same document bytes."""
    from aicls.ocr import read_inkml

    corpus = tmp_path / "src" / "mathwriting-2024"
    directory = DirectoryInkmlSource(corpus)
    archive = TarInkmlSource(official_archive)

    def labelled(records: list[tuple[str, InkmlRecord]]) -> dict[str, str]:
        return {r.sample_id: (read_inkml(r.payload).label or "") for _, r in records}

    assert labelled(list(directory.iter_records(["train"]))) == labelled(
        list(archive.iter_records(["train"]))
    )


def test_tar_source_requests_only_the_splits_it_was_asked_for(official_archive: Path) -> None:
    records = list(TarInkmlSource(official_archive).iter_records(["valid"]))

    assert {split for split, _ in records} == {"valid"}
    assert len(records) == OFFICIAL_LAYOUT["valid"]
    assert list(TarInkmlSource(official_archive).iter_records([])) == []


def test_tar_source_limit_is_per_split_and_stops_the_pass_early(official_archive: Path) -> None:
    source = TarInkmlSource(official_archive)

    records = list(source.iter_records(["train", "synthetic"], limit=1))
    assert sorted(split for split, _ in records) == ["synthetic", "train"]

    # a split that fills up while another is still open is skipped, not re-read
    mixed = list(source.iter_records(["train", "valid"], limit=2))
    assert sorted(split for split, _ in mixed) == ["train", "train", "valid", "valid"]

    # limit=0 reads nothing at all and stops on the first member it recognises
    assert list(source.iter_records(["train"], limit=0)) == []


def test_tar_source_counting_can_be_capped_per_split(official_archive: Path) -> None:
    """A preview must not walk 630 000 members: every split stops at the cap."""
    counts = TarInkmlSource(official_archive).count_splits(list(OFFICIAL_LAYOUT), cap=2)

    assert counts["train"].n_files == 2 and counts["train"].capped is True
    assert counts["valid"].n_files == 2 and counts["valid"].capped is True
    assert counts["synthetic"].n_files == 2 and counts["synthetic"].capped is True


def test_sources_expose_where_they_read_from(official_archive: Path, tmp_path: Path) -> None:
    corpus = tmp_path / "src" / "mathwriting-2024"

    assert DirectoryInkmlSource(corpus).root == corpus
    assert TarInkmlSource(official_archive).archive == official_archive


def test_tar_source_ignores_members_outside_a_split_unless_a_default_is_given(
    tmp_path: Path,
) -> None:
    """A flat archive has no split directories, so the caller must name the split explicitly."""
    flat = synth_inkml_dataset(tmp_path / "flat", 3, 1234)
    (flat / "README.txt").write_text("not inkml\n", encoding="utf-8", newline="\n")
    archive = _tar(tmp_path / "flat.tgz", flat, arcname="")

    assert list(TarInkmlSource(archive).iter_records(["train"])) == []
    records = list(TarInkmlSource(archive, default_split="train").iter_records(["train"]))
    assert len(records) == 3
    assert all(record.member.endswith(".inkml") for _, record in records)


def test_tar_source_matches_the_nearest_split_directory(tmp_path: Path) -> None:
    """``.../test/train/x.inkml`` belongs to ``train``: the directory closest to the file wins."""
    nested = tmp_path / "corpus" / "test" / "train"
    synth_inkml_dataset(nested, 2, 1234)
    archive = _tar(tmp_path / "nested.tgz", tmp_path / "corpus", arcname="corpus")

    source = TarInkmlSource(archive)
    assert source.count_splits(["train"])["train"].n_files == 2
    assert source.count_splits(["test"])["test"].n_files == 0


def test_tar_source_caps_the_bytes_it_reads_from_one_member(tmp_path: Path) -> None:
    """A huge member cannot make the source allocate an unbounded buffer."""
    payload = tmp_path / "big.inkml"
    payload.write_bytes(b"<ink>" + b"x" * 4096 + b"</ink>")
    archive = _tar(tmp_path / "big.tgz", payload, arcname="train/big.inkml")

    records = list(TarInkmlSource(archive, max_member_bytes=64).iter_records(["train"]))

    assert len(records) == 1
    assert len(records[0][1].payload) == 65  # the cap plus the byte that proves it was exceeded


def test_tar_source_reports_a_missing_or_unreadable_archive(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError):
        TarInkmlSource(tmp_path / "absent.tgz")

    not_a_tar = tmp_path / "junk.tgz"
    not_a_tar.write_bytes(b"this is not a tar archive at all")
    with pytest.raises(InputValidationError):
        list(TarInkmlSource(not_a_tar).iter_records(["train"]))

    with pytest.raises(InputValidationError):
        TarInkmlSource(_tar(tmp_path / "ok.tgz", tmp_path / "junk.tgz"), max_member_bytes=0)


def test_tar_source_labels_the_split_with_the_archive_it_came_from(official_archive: Path) -> None:
    label = TarInkmlSource(official_archive).source_label("test")

    assert label.startswith("tar:")
    assert label.endswith("!test")
    assert official_archive.name in label


# ------------------------------------------------------------------------------------ factory
def test_open_inkml_source_picks_the_implementation_from_the_path(
    tmp_path: Path, official_archive: Path
) -> None:
    corpus = tmp_path / "src" / "mathwriting-2024"

    assert isinstance(open_inkml_source(corpus), DirectoryInkmlSource)
    assert isinstance(open_inkml_source(official_archive), TarInkmlSource)
    with pytest.raises(NotFoundError):
        open_inkml_source(tmp_path / "neither")


def test_looks_like_tar_recognises_the_released_suffixes() -> None:
    assert looks_like_tar(Path("mathwriting-2024.tgz"))
    assert looks_like_tar(Path("D:/data/MATHWRITING-2024.TAR.GZ"))
    assert not looks_like_tar(Path("mathwriting-2024.zip"))
    assert not looks_like_tar(Path("raw"))
