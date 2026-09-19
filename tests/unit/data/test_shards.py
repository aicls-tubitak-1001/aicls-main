"""InkML -> memory-mapped npz shard conversion (docs/ml/egitim-altyapisi.md §2).

Two input paths are exercised: an extracted directory tree and a tar archive streamed member by
member. The archives are built here from the synthetic InkML fixtures; the real
``mathwriting-2024.tgz`` is behind the ``DATASET_DOWNLOAD`` gate and is never fetched.
"""

from __future__ import annotations

import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from aicls.core.errors import InputValidationError, NotFoundError
from aicls.data import prepare_shards, synth_inkml_dataset
from aicls.data.shards import (
    SHARD_MANIFEST_FILENAME,
    ShardManifest,
    ShardReader,
    build_vocab_from_shards,
    discover_shard_splits,
    load_shard_manifest,
    normalize_label,
    prepare_shards_from_source,
    read_shard_manifest,
    shard_fingerprints,
    shards_root_fingerprint,
)
from aicls.data.sources import DirectoryInkmlSource, TarInkmlSource
from aicls.ocr import read_inkml
from aicls.ocr.types import OcrInputError

#: Split layout of the fixture archives, mirroring the released corpus.
ARCHIVE_LAYOUT: dict[str, int] = {"train": 6, "valid": 2, "test": 2, "symbols": 2, "synthetic": 3}


def _official_archive(tmp_path: Path, *, mode: str = "w:gz") -> Path:
    """Build ``mathwriting-2024.tgz`` with the official ``<prefix>/<split>/*.inkml`` layout."""
    corpus = tmp_path / "src" / "mathwriting-2024"
    for index, (split, count) in enumerate(ARCHIVE_LAYOUT.items()):
        synth_inkml_dataset(corpus / split, count, 1234 + 100 * index, prefix=split)
    archive = tmp_path / "mathwriting-2024.tgz"
    with tarfile.open(archive, mode) as bundle:
        bundle.add(corpus, arcname=corpus.name)
    return archive


def test_prepare_mathwriting_shard_roundtrip_on_synthetic_inkml(
    inkml_root: Path, tmp_path: Path
) -> None:
    out = tmp_path / "shards"
    manifest = prepare_shards(inkml_root, out, split="train", shard_size=5)

    assert isinstance(manifest, ShardManifest)
    assert manifest.split == "train"
    assert manifest.n_samples == 12
    assert manifest.n_shards == 3  # 5 + 5 + 2
    assert [entry.n_samples for entry in manifest.shards] == [5, 5, 2]
    assert all(len(entry.sha256) == 64 for entry in manifest.shards)
    assert manifest.fingerprint.startswith("sha256:")

    # the source inks, keyed by sample id, must survive the round-trip point for point
    sources = {p.stem: read_inkml(p) for p in sorted(inkml_root.glob("*.inkml"))}
    assert len(sources) == 12

    seen: set[str] = set()
    for entry in manifest.shards:
        with ShardReader(out / "train" / entry.name) as reader:
            assert len(reader) == entry.n_samples
            for index in range(len(reader)):
                sample = reader.sample(index)
                original = sources[sample.sample_id]
                seen.add(sample.sample_id)
                assert len(sample.strokes) == original.n_strokes
                for produced, expected in zip(sample.strokes, original.strokes, strict=True):
                    assert produced.dtype == np.float32
                    np.testing.assert_allclose(produced, expected.points)
                assert sample.label == normalize_label(original.label or "")
                assert sample.writer_id == original.writer_id
                assert sample.is_synthetic is False
                assert sample.image is None
    assert seen == set(sources)


def test_prepare_shards_writes_a_manifest_that_can_be_reloaded(
    inkml_root: Path, tmp_path: Path
) -> None:
    out = tmp_path / "shards"
    manifest = prepare_shards(inkml_root, out, split="train", shard_size=5)

    path = out / "train" / SHARD_MANIFEST_FILENAME
    assert path.is_file()
    reloaded = load_shard_manifest(out, "train")
    assert reloaded.fingerprint == manifest.fingerprint
    assert reloaded.shards == manifest.shards
    assert json.loads(path.read_text(encoding="utf-8"))["split"] == "train"

    # the recorded sha256 really is the shard file's digest
    from aicls.core.hashing import sha256_file

    for entry in reloaded.shards:
        assert sha256_file(out / "train" / entry.name) == entry.sha256


def test_shard_arrays_are_memory_mapped(inkml_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "shards"
    prepare_shards(inkml_root, out, split="train", shard_size=16)
    shard = next((out / "train").glob("*.npz"))

    with ShardReader(shard) as reader:
        points = reader.array("points")
        assert isinstance(points, np.memmap)
        assert points.dtype == np.float32 and points.ndim == 2
        # a plain read gives exactly the same values
        with np.load(shard) as npz:
            np.testing.assert_array_equal(np.asarray(points), npz["points"])

    with ShardReader(shard, mmap=False) as reader:
        assert not isinstance(reader.array("points"), np.memmap)


def test_prepare_shards_renders_images_when_requested(inkml_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "shards"
    manifest = prepare_shards(
        inkml_root, out, split="valid", shard_size=16, render=True, image_size=(32, 96)
    )
    assert manifest.has_images
    assert manifest.image_size == (32, 96)

    with ShardReader(out / "valid" / manifest.shards[0].name) as reader:
        sample = reader.sample(0)
        assert sample.image is not None
        assert sample.image.shape == (32, 96)
        assert sample.image.dtype == np.uint8


def test_prepare_shards_marks_the_synthetic_split(inkml_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "shards"
    manifest = prepare_shards(inkml_root, out, split="synthetic", shard_size=16)
    with ShardReader(out / "synthetic" / manifest.shards[0].name) as reader:
        assert reader.sample(0).is_synthetic is True


def test_prepare_shards_reads_a_split_subdirectory(tmp_path: Path, inkml_root: Path) -> None:
    root = tmp_path / "mathwriting" / "inkml"
    (root / "test").mkdir(parents=True)
    for path in sorted(inkml_root.glob("*.inkml"))[:4]:
        (root / "test" / path.name).write_bytes(path.read_bytes())

    manifest = prepare_shards(root, tmp_path / "shards", split="test")
    assert manifest.n_samples == 4
    assert manifest.source_root.endswith("test")


def test_build_vocab_from_shards_writes_vocab_json(inkml_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "shards"
    prepare_shards(inkml_root, out, split="train", shard_size=16)
    vocab_path = build_vocab_from_shards(out, tmp_path / "vocab.json", splits=("train",))

    from aicls.ocr import LatexTokenizer

    tokenizer = LatexTokenizer.from_file(vocab_path)
    assert tokenizer.vocab_size > 4
    with ShardReader(next((out / "train").glob("*.npz"))) as reader:
        label = reader.sample(0).label
    assert tokenizer.decode(tokenizer.encode(label)) == label


def test_prepare_shards_rejects_missing_root_and_bad_shard_size(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError):
        prepare_shards(tmp_path / "nope", tmp_path / "out", split="train")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(InputValidationError):
        prepare_shards(empty, tmp_path / "out", split="train")
    with pytest.raises(InputValidationError):
        prepare_shards(empty, tmp_path / "out", split="train", shard_size=0)


def test_load_shard_manifest_rejects_unknown_split(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError):
        load_shard_manifest(tmp_path, "train")


def test_prepare_shards_streams_a_tar_archive_without_extracting_it(tmp_path: Path) -> None:
    """The whole official layout is sharded from the archive in one pass; no tree is written."""
    archive = _official_archive(tmp_path)
    out = tmp_path / "shards"

    manifests = prepare_shards_from_source(
        TarInkmlSource(archive), out, splits=list(ARCHIVE_LAYOUT), shard_size=4
    )

    assert set(manifests) == set(ARCHIVE_LAYOUT)
    assert {name: m.n_samples for name, m in manifests.items()} == ARCHIVE_LAYOUT
    for split, manifest in manifests.items():
        assert manifest.split == split
        assert manifest.fingerprint.startswith("sha256:")
        assert manifest.source_root == f"tar:{archive}!{split}"
        assert (out / split / SHARD_MANIFEST_FILENAME).is_file()
        assert load_shard_manifest(out, split).fingerprint == manifest.fingerprint
    # the synthetic split keeps its flag through the streaming path
    with ShardReader(out / "synthetic" / manifests["synthetic"].shards[0].name) as reader:
        assert reader.sample(0).is_synthetic is True
    with ShardReader(out / "train" / manifests["train"].shards[0].name) as reader:
        assert reader.sample(0).is_synthetic is False
    # nothing was extracted next to the archive
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "mathwriting-2024.tgz",
        "shards",
        "src",
    ]


def test_streamed_shards_carry_the_same_samples_as_the_extracted_tree(tmp_path: Path) -> None:
    """Same corpus, two sources: identical ids, labels and stroke data.

    ``tar`` writes a sorted directory walk, so the archive order matches the directory order and
    the shards come out identical — including the fingerprint, which is asserted below because a
    report cites it. That is a property of *this* archive's layout, not a guarantee about every
    archive, which is why ``source_root`` still records which source produced the shards.
    """
    archive = _official_archive(tmp_path)
    corpus = tmp_path / "src" / "mathwriting-2024"

    from_dir = prepare_shards(corpus, tmp_path / "a", split="train", shard_size=4)
    from_tar = prepare_shards_from_source(
        TarInkmlSource(archive), tmp_path / "b", splits=["train"], shard_size=4
    )["train"]

    def samples(root: Path, manifest: ShardManifest) -> dict[str, tuple[str, int]]:
        collected: dict[str, tuple[str, int]] = {}
        for entry in manifest.shards:
            with ShardReader(root / "train" / entry.name) as reader:
                for index in range(len(reader)):
                    sample = reader.sample(index)
                    collected[sample.sample_id] = (sample.label, len(sample.strokes))
        return collected

    assert from_dir.n_samples == from_tar.n_samples == ARCHIVE_LAYOUT["train"]
    assert samples(tmp_path / "a", from_dir) == samples(tmp_path / "b", from_tar)
    assert from_dir.fingerprint == from_tar.fingerprint
    assert [e.sha256 for e in from_dir.shards] == [e.sha256 for e in from_tar.shards]
    assert from_dir.source_root == str(corpus / "train")
    assert from_tar.source_root.startswith("tar:")


def test_shard_order_follows_the_archive_so_the_fingerprint_can_change(tmp_path: Path) -> None:
    """A differently ordered archive gives different shard boundaries and a different fingerprint.

    This is the reason ``source_root`` exists. Two archives holding the *same* documents in a
    different member order are prepared here; the samples match, the fingerprints do not.
    """
    corpus = tmp_path / "corpus" / "train"
    synth_inkml_dataset(corpus, 6, 1234, prefix="train")
    files = sorted(corpus.glob("*.inkml"))

    def pack(name: str, order: list[Path]) -> Path:
        archive = tmp_path / name
        with tarfile.open(archive, "w") as bundle:
            for path in order:
                bundle.add(path, arcname=f"corpus/train/{path.name}")
        return archive

    forward = prepare_shards_from_source(
        TarInkmlSource(pack("a.tar", files)), tmp_path / "a", splits=["train"], shard_size=2
    )["train"]
    backward = prepare_shards_from_source(
        TarInkmlSource(pack("b.tar", list(reversed(files)))),
        tmp_path / "b",
        splits=["train"],
        shard_size=2,
    )["train"]

    assert forward.n_samples == backward.n_samples == 6
    assert forward.fingerprint != backward.fingerprint

    def ids(root: Path, manifest: ShardManifest) -> set[str]:
        found: set[str] = set()
        for entry in manifest.shards:
            with ShardReader(root / "train" / entry.name) as reader:
                found |= {reader.sample(i).sample_id for i in range(len(reader))}
        return found

    assert ids(tmp_path / "a", forward) == ids(tmp_path / "b", backward)


def test_prepare_shards_from_source_renders_and_limits_per_split(tmp_path: Path) -> None:
    archive = _official_archive(tmp_path)

    manifests = prepare_shards_from_source(
        TarInkmlSource(archive),
        tmp_path / "shards",
        splits=["train", "valid"],
        shard_size=8,
        render=True,
        image_size=(32, 96),
        limit=2,
    )

    assert {name: m.n_samples for name, m in manifests.items()} == {"train": 2, "valid": 2}
    assert manifests["train"].has_images and manifests["train"].image_size == (32, 96)
    shard = tmp_path / "shards" / "train" / manifests["train"].shards[0].name
    with ShardReader(shard) as reader:
        image = reader.sample(0).image
        assert image is not None and image.shape == (32, 96)


def test_prepare_shards_from_source_writes_an_empty_manifest_for_a_missing_split(
    tmp_path: Path,
) -> None:
    """A split the archive does not hold is reported as zero samples, not silently dropped."""
    corpus = tmp_path / "corpus"
    synth_inkml_dataset(corpus / "train", 3, 1234, prefix="train")
    archive = tmp_path / "partial.tgz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(corpus, arcname="partial")

    manifests = prepare_shards_from_source(
        TarInkmlSource(archive), tmp_path / "shards", splits=["train", "valid"]
    )

    assert manifests["train"].n_samples == 3
    assert manifests["valid"].n_samples == 0
    assert manifests["valid"].shards == []
    assert load_shard_manifest(tmp_path / "shards", "valid").n_samples == 0


def test_prepare_shards_from_source_rejects_an_archive_without_any_inkml(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "README.txt").write_text("nothing here\n", encoding="utf-8", newline="\n")
    archive = tmp_path / "empty.tgz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(empty, arcname="empty")

    with pytest.raises(InputValidationError) as excinfo:
        prepare_shards_from_source(TarInkmlSource(archive), tmp_path / "out", splits=["train"])
    assert excinfo.value.details["reason"] == "no_inkml_files"

    with pytest.raises(InputValidationError):
        prepare_shards_from_source(TarInkmlSource(archive), tmp_path / "out", splits=[])


def test_prepare_shards_skip_invalid_keeps_the_run_alive_and_counts_the_loss(
    tmp_path: Path,
) -> None:
    """One corrupt document in 630 000 must not throw away the shards already written."""
    corpus = tmp_path / "corpus"
    synth_inkml_dataset(corpus / "train", 3, 1234, prefix="train")
    (corpus / "train" / "broken.inkml").write_text("<ink><trace>oops", encoding="utf-8")

    with pytest.raises(OcrInputError):
        prepare_shards(corpus, tmp_path / "strict", split="train")

    manifest = prepare_shards(corpus, tmp_path / "lenient", split="train", skip_invalid=True)
    assert manifest.n_samples == 3
    assert manifest.n_skipped == 1


def test_prepare_shards_from_source_accepts_a_directory_source(inkml_root: Path) -> None:
    """The directory path goes through exactly the same engine, so the two cannot drift."""
    out = inkml_root.parent / "shards"
    manifests = prepare_shards_from_source(
        DirectoryInkmlSource(inkml_root), out, splits=["train"], shard_size=5
    )

    assert manifests["train"].n_samples == 12
    assert manifests["train"].source_root == str(inkml_root)


def test_normalize_label_is_idempotent_and_canonical() -> None:
    raw = r"\frac{1}{2}  +   x^{2}"
    once = normalize_label(raw)
    assert once == normalize_label(once)
    assert "  " not in once
    assert normalize_label("") == ""


# ------------------------------------------------------------------ dataset identity of a root
def test_discover_shard_splits_finds_every_prepared_split(mixed_shards_root: Path) -> None:
    """The layout knowledge lives here, so a run manifest never has to guess a file name."""
    assert discover_shard_splits(mixed_shards_root) == ("synthetic", "train")
    assert discover_shard_splits(mixed_shards_root / "train") == ("train",)
    assert discover_shard_splits(mixed_shards_root / "nope") == ()


def test_shard_fingerprints_reads_what_preparation_wrote(mixed_shards_root: Path) -> None:
    """Each value is the fingerprint the preparation computed, not a re-hash of the bytes."""
    fingerprints = shard_fingerprints(mixed_shards_root)

    assert set(fingerprints) == {"synthetic", "train"}
    for split, value in fingerprints.items():
        assert value == load_shard_manifest(mixed_shards_root, split).fingerprint
        assert value.startswith("sha256:")
    assert list(fingerprints) == sorted(fingerprints), "split order is deterministic"


def test_shard_fingerprints_honours_an_explicit_split_list(mixed_shards_root: Path) -> None:
    """A run records the splits it opened, not everything that happens to be on disk."""
    assert set(shard_fingerprints(mixed_shards_root, ["train"])) == {"train"}
    assert shard_fingerprints(mixed_shards_root, []) == {}


def test_shard_fingerprints_never_labels_one_split_with_another_name(
    shards_root: Path, tmp_path: Path
) -> None:
    """``load_shard_manifest`` falls back to the root manifest; a fingerprint must not.

    Copying ``train``'s manifest to the root makes the fallback resolve for *every* requested
    split. Publishing ``valid -> <train fingerprint>`` from that would be a provenance lie.
    """
    root_manifest = shards_root / SHARD_MANIFEST_FILENAME
    root_manifest.write_text(
        (shards_root / "train" / SHARD_MANIFEST_FILENAME).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    assert shard_fingerprints(shards_root, ["valid"]) == {}
    assert set(shard_fingerprints(shards_root, ["train"])) == {"train"}
    # the root manifest still names its own split, so discovery reports it once
    assert discover_shard_splits(shards_root) == ("train",)


def test_shard_fingerprints_skips_an_unreadable_manifest(shards_root: Path) -> None:
    """A corrupt manifest must not stop a training run from writing its provenance record."""
    (shards_root / "train" / SHARD_MANIFEST_FILENAME).write_text("{oops", encoding="utf-8")
    assert shard_fingerprints(shards_root) == {}
    assert discover_shard_splits(shards_root) == ("train",)


def test_shards_root_fingerprint_is_order_independent_and_sensitive(
    mixed_shards_root: Path,
) -> None:
    per_split = shard_fingerprints(mixed_shards_root)
    aggregate = shards_root_fingerprint(per_split)

    assert aggregate.startswith("sha256:")
    assert aggregate == shards_root_fingerprint(dict(reversed(list(per_split.items()))))
    assert aggregate != shards_root_fingerprint({"train": per_split["train"]})
    assert shards_root_fingerprint({}) == ""


def test_changing_a_shard_changes_the_fingerprints(inkml_root: Path, tmp_path: Path) -> None:
    """The link is only worth having if it breaks when the data changes."""
    from aicls.data import synth_inkml_dataset

    first = tmp_path / "shards-a"
    prepare_shards(inkml_root, first, split="train", shard_size=5)
    before = shard_fingerprints(first)

    other = synth_inkml_dataset(tmp_path / "inkml-b", 12, 4321)
    second = tmp_path / "shards-b"
    prepare_shards(other, second, split="train", shard_size=5)
    after = shard_fingerprints(second)

    assert before["train"] != after["train"]
    assert shards_root_fingerprint(before) != shards_root_fingerprint(after)


def test_read_shard_manifest_rejects_a_malformed_file(tmp_path: Path) -> None:
    broken = tmp_path / SHARD_MANIFEST_FILENAME
    broken.write_text("[]", encoding="utf-8")
    with pytest.raises(InputValidationError):
        read_shard_manifest(broken)

    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(InputValidationError):
        read_shard_manifest(broken)

    with pytest.raises(InputValidationError):
        read_shard_manifest(tmp_path / "absent.json")


def test_the_recorded_fingerprint_is_not_a_re_hash_unless_asked(shards_root: Path) -> None:
    """State the limit of the cheap read instead of letting a reader assume it away.

    Re-preparing the corpus changes the fingerprint (that is the case that matters, and
    :func:`test_changing_a_shard_changes_the_fingerprints` pins it). Editing a shard file behind
    the manifest's back does not, because nothing re-hashed it — until ``verify=True``, which
    costs a full read of the split and is what ``aicls data verify`` is for.
    """
    manifest = load_shard_manifest(shards_root, "train")
    shard = shards_root / "train" / manifest.shards[0].name
    shard.write_bytes(shard.read_bytes() + b"\x00")

    assert shard_fingerprints(shards_root)["train"] == manifest.fingerprint
    assert shard_fingerprints(shards_root, verify=True) == {}


def test_verify_drops_a_split_whose_shard_disappeared(shards_root: Path) -> None:
    manifest = load_shard_manifest(shards_root, "train")
    (shards_root / "train" / manifest.shards[0].name).unlink()

    assert shard_fingerprints(shards_root, verify=True) == {}
    assert shard_fingerprints(shards_root) == {"train": manifest.fingerprint}
