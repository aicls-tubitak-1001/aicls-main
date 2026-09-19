"""Data manifests: build, verify, tamper detection and fingerprint stability (ADR-0008)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from aicls.core.errors import InputValidationError, NotFoundError
from aicls.data import DataManifest
from aicls.data.manifest import MANIFEST_FILENAME, load_manifest

LICENSE = "Apache-2.0"
SOURCE = "https://example.org/mathwriting"


def make_tree(root: Path) -> Path:
    """Create a small deterministic file tree and return its root."""
    (root / "inkml" / "train").mkdir(parents=True)
    (root / "inkml" / "train" / "a.inkml").write_text("<ink/>\n", encoding="utf-8")
    (root / "inkml" / "train" / "b.inkml").write_text("<ink></ink>\n", encoding="utf-8")
    (root / "vocab.json").write_text('{"tokens": []}\n', encoding="utf-8")
    return root


def test_data_manifest_build_and_verify(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "mathwriting")
    manifest = DataManifest.build(
        root,
        "mathwriting",
        license=LICENSE,
        source_url=SOURCE,
        retrieved_at=datetime(2026, 9, 7, 12, 0, tzinfo=UTC),
    )

    assert manifest.name == "mathwriting"
    assert manifest.license == LICENSE
    assert manifest.source_url == SOURCE
    assert manifest.retrieved_at.year == 2026
    assert manifest.version.startswith("mathwriting-")
    assert manifest.fingerprint.startswith("sha256:")
    assert [f.path for f in manifest.files] == [
        "inkml/train/a.inkml",
        "inkml/train/b.inkml",
        "vocab.json",
    ]
    assert all(len(f.sha256) == 64 for f in manifest.files)
    assert all(f.size > 0 for f in manifest.files)
    assert manifest.total_bytes == sum(f.size for f in manifest.files)

    report = manifest.verify(root)
    assert report.ok
    assert report.checked == 3
    assert report.missing == [] and report.changed == [] and report.extra == []
    assert report.fingerprint_matches

    # the manifest survives a JSON round-trip through the file the CLI writes
    path = manifest.save(root)
    assert path.name == MANIFEST_FILENAME
    reloaded = load_manifest(path)
    assert reloaded.fingerprint == manifest.fingerprint
    assert reloaded.verify(root).ok


def test_data_manifest_verify_detects_tampered_sha(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "mathwriting")
    manifest = DataManifest.build(root, "mathwriting", license=LICENSE, source_url=SOURCE)

    (root / "inkml" / "train" / "a.inkml").write_text("<ink>tampered</ink>\n", encoding="utf-8")
    report = manifest.verify(root)

    assert not report.ok
    assert report.changed == ["inkml/train/a.inkml"]
    assert report.missing == [] and report.extra == []
    assert not report.fingerprint_matches
    assert "1" in report.summary()


def test_data_manifest_verify_reports_missing_and_extra(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "mathwriting")
    manifest = DataManifest.build(root, "mathwriting", license=LICENSE, source_url=SOURCE)

    (root / "vocab.json").unlink()
    (root / "inkml" / "train" / "c.inkml").write_text("<ink/>\n", encoding="utf-8")
    report = manifest.verify(root)

    assert not report.ok
    assert report.missing == ["vocab.json"]
    assert report.extra == ["inkml/train/c.inkml"]
    assert report.changed == []


def test_manifest_fingerprint_depends_on_content_and_not_on_walk_order(tmp_path: Path) -> None:
    first = DataManifest.build(make_tree(tmp_path / "one"), "d", license=LICENSE, source_url=SOURCE)
    second = DataManifest.build(
        make_tree(tmp_path / "two"), "d", license=LICENSE, source_url=SOURCE
    )
    assert first.fingerprint == second.fingerprint

    (tmp_path / "two" / "vocab.json").write_text('{"tokens": ["x"]}\n', encoding="utf-8")
    third = DataManifest.build(tmp_path / "two", "d", license=LICENSE, source_url=SOURCE)
    assert third.fingerprint != first.fingerprint


def test_manifest_build_skips_its_own_file_and_caches(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "mathwriting")
    (root / MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "x.pyc").write_bytes(b"\x00")

    manifest = DataManifest.build(root, "mathwriting", license=LICENSE, source_url=SOURCE)
    paths = [f.path for f in manifest.files]
    assert MANIFEST_FILENAME not in paths
    assert not any(p.startswith("__pycache__") for p in paths)


def test_manifest_build_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError):
        DataManifest.build(tmp_path / "nope", "x", license=LICENSE, source_url=SOURCE)


def test_load_manifest_rejects_malformed_file(tmp_path: Path) -> None:
    bad = tmp_path / "manifest.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(InputValidationError):
        load_manifest(bad)


def test_load_manifest_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError):
        load_manifest(tmp_path / "manifest.json")
