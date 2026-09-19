"""``training/scripts/prepare_mathwriting.py``: the ungated InkML -> shards entry point.

The archive used here is built in the test from synthetic InkML fixtures; the released
``mathwriting-2024.tgz`` sits behind the ``DATASET_DOWNLOAD`` gate and is never fetched. What is
proven is the *shape*: the official ``<prefix>/<split>/*.inkml`` layout, one pass over the
stream, no extracted tree, and a plan that can be read before anything is written.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import tarfile
import types

import pytest

from aicls.data import synth_inkml_dataset
from aicls.data.shards import SHARD_MANIFEST_FILENAME, load_shard_manifest

Loader = Callable[[str], types.ModuleType]

SCRIPT = "training/scripts/prepare_mathwriting.py"

#: Split layout of the fixture archive, mirroring the released corpus.
LAYOUT: dict[str, int] = {"train": 4, "valid": 2, "test": 2, "symbols": 2, "synthetic": 3}


def summary_of(stdout: str) -> dict[str, object]:
    """Parse the ``--json`` summary out of stdout (structured log lines share the stream).

    The summary is the only multi-line JSON document printed: ``json.dumps(..., indent=2)``
    starts it on a line that is exactly ``{``, while every log line is single-line JSON.
    """
    lines = stdout.splitlines()
    start = max(index for index, line in enumerate(lines) if line == "{")
    return dict(json.loads(chr(10).join(lines[start:])))


@pytest.fixture
def script(load_script: Loader) -> types.ModuleType:
    """The preparation script, imported by path."""
    return load_script(SCRIPT)


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    """A ``.tgz`` with the official ``mathwriting-2024/<split>/*.inkml`` layout."""
    corpus = tmp_path / "src" / "mathwriting-2024"
    for index, (split, count) in enumerate(LAYOUT.items()):
        synth_inkml_dataset(corpus / split, count, 1234 + 100 * index, prefix=split)
    path = tmp_path / "mathwriting-2024.tgz"
    with tarfile.open(path, "w:gz") as bundle:
        bundle.add(corpus, arcname=corpus.name)
    return path


def test_prepare_streams_every_official_split_from_the_archive_in_one_pass(
    script: types.ModuleType,
    archive: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "shards"

    code = script.main(["--archive", str(archive), "--out", str(out), "--shard-size", "3"])

    assert code == 0
    assert "converting" in capsys.readouterr().out
    for split, count in LAYOUT.items():
        manifest = load_shard_manifest(out, split)
        assert manifest.n_samples == count, split
        assert manifest.source_root == f"tar:{archive}!{split}"
        assert (out / split / SHARD_MANIFEST_FILENAME).is_file()
    # nothing was extracted: the only new directory under tmp_path is the shards root
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "mathwriting-2024.tgz",
        "shards",
        "src",
    ]


def test_prepare_json_summary_names_the_source_and_every_fingerprint(
    script: types.ModuleType,
    archive: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "shards"

    code = script.main(
        ["--archive", str(archive), "--out", str(out), "--split", "train", "valid", "--json"]
    )
    assert code == 0

    summary = summary_of(capsys.readouterr().out)
    assert summary["source_kind"] == "tar"
    assert summary["source"] == str(archive)
    splits = dict(summary["splits"])  # type: ignore[call-overload]
    assert set(splits) == {"train", "valid"}
    assert splits["train"]["samples"] == LAYOUT["train"]
    assert splits["train"]["fingerprint"].startswith("sha256:")
    assert splits["train"]["source"].startswith("tar:")


def test_prepare_dry_run_counts_archive_members_and_writes_nothing(
    script: types.ModuleType,
    archive: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "shards"

    assert script.main(["--archive", str(archive), "--out", str(out), "--dry-run"]) == 0

    printed = capsys.readouterr().out
    assert "archive stream (nothing extracted)" in printed
    assert "dry run: nothing was written." in printed
    for split, count in LAYOUT.items():
        assert f"  {split:10s} {count:>8d} inkml" in printed
    assert not out.exists()


def test_prepare_dry_run_reports_a_missing_source_and_names_the_gate(
    script: types.ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dry run against absent data is still gate-free, and still says what is missing."""
    absent = tmp_path / "not-downloaded.tgz"

    assert script.main(["--archive", str(absent), "--out", str(tmp_path / "s"), "--dry-run"]) == 0
    assert "(MISSING)" in capsys.readouterr().out

    code = script.main(["--archive", str(absent), "--out", str(tmp_path / "s")])
    err = capsys.readouterr().err
    assert code == 1
    assert "inkml_source_missing" in err
    assert script.SOURCE_GATE in err
    assert "download_mathwriting.py" in err


def test_prepare_archive_without_a_value_uses_the_catalog_archive_name(
    script: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bare ``--archive`` resolves to AICLS_DATA_ROOT/mathwriting/<catalog archive name>."""
    from aicls.core.settings import get_settings

    monkeypatch.setenv("AICLS_DATA_ROOT", str(tmp_path))
    get_settings.cache_clear()
    try:
        args = script.build_parser().parse_args(["--archive"])
        path, is_archive = script.resolve_source_path(args)
        assert is_archive is True
        assert path == tmp_path / "mathwriting" / "mathwriting-2024.tgz"

        args = script.build_parser().parse_args([])
        path, is_archive = script.resolve_source_path(args)
        assert is_archive is False
        assert path.as_posix().endswith("mathwriting/raw")
    finally:
        get_settings.cache_clear()


def test_prepare_still_reads_an_extracted_directory(
    script: types.ModuleType, tmp_path: Path
) -> None:
    """The historical ``--src`` path is untouched by the streaming addition."""
    corpus = tmp_path / "raw"
    synth_inkml_dataset(corpus / "train", 5, 1234, prefix="train")
    out = tmp_path / "shards"

    assert script.main(["--src", str(corpus), "--out", str(out), "--split", "train"]) == 0

    manifest = load_shard_manifest(out, "train")
    assert manifest.n_samples == 5
    assert manifest.source_root == str(corpus / "train")


def test_prepare_manifest_records_the_corrected_licence(
    script: types.ModuleType, archive: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--manifest`` must quote the catalog licence, not a hard-coded Apache-2.0 string."""
    out = tmp_path / "shards"

    code = script.main(
        ["--archive", str(archive), "--out", str(out), "--split", "train", "--manifest", "--json"]
    )
    assert code == 0

    summary = summary_of(capsys.readouterr().out)
    assert summary["license"] == "CC-BY-NC-SA-4.0"
    written = json.loads(Path(str(summary["manifest"])).read_text(encoding="utf-8"))
    assert written["license"] == "CC-BY-NC-SA-4.0"


def test_prepare_limit_and_skip_invalid_reach_the_data_layer(
    script: types.ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = tmp_path / "corpus"
    synth_inkml_dataset(corpus / "train", 5, 1234, prefix="train")
    (corpus / "train" / "broken.inkml").write_text("<ink><trace>oops", encoding="utf-8")
    out = tmp_path / "shards"

    assert script.main(["--src", str(corpus), "--out", str(out), "--split", "train"]) == 1
    assert "xml_malformed" in capsys.readouterr().err

    code = script.main(
        [
            "--src",
            str(corpus),
            "--out",
            str(out),
            "--split",
            "train",
            "--skip-invalid",
            "--limit",
            "3",
        ]
    )
    assert code == 0
    manifest = load_shard_manifest(out, "train")
    assert manifest.n_samples + manifest.n_skipped == 3


def test_prepare_reports_an_unreadable_archive_as_such(
    script: types.ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    junk = tmp_path / "corrupt.tgz"
    junk.write_bytes(b"not a tar archive")

    code = script.main(["--archive", str(junk), "--out", str(tmp_path / "s")])

    assert code == 1
    assert "inkml_archive_unreadable" in capsys.readouterr().err
