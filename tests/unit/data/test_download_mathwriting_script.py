"""``training/scripts/download_mathwriting.py``: the transfer itself, with no network.

The gate refusals live in ``tests/gated/test_gates.py``; this module is about what the script
does with the bytes **once the gate has already accepted** — and it never touches the network to
find out. ``urllib.request.urlopen`` is replaced by a fake transport that serves a local byte
string and understands ``Range``, so resume, atomicity, the content-length check and the tar
extraction are all exercised offline and deterministically.

The archive served here is a tar this module builds; the real MathWriting corpus is behind the
``DATASET_DOWNLOAD`` gate and is not fetched by any test.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import io
from pathlib import Path
import tarfile
import types
from typing import Any
import urllib.request

import pytest

from aicls.core.errors import InputValidationError
from aicls.core.settings import get_settings

Loader = Callable[[str], types.ModuleType]

SCRIPT = "training/scripts/download_mathwriting.py"


class FakeResponse(io.BytesIO):
    """Minimal stand-in for ``http.client.HTTPResponse``."""

    def __init__(self, payload: bytes, *, status: int, headers: dict[str, str]) -> None:
        super().__init__(payload)
        self.status = status
        self.headers = headers

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class FakeTransport:
    """Serves ``payload`` over the ``urlopen`` interface, honouring a ``Range`` header.

    Attributes:
        calls: One ``(url, range_header)`` tuple per request, so a test can prove that a resumed
            run really asked for the remainder instead of the whole file.
    """

    def __init__(
        self,
        payload: bytes,
        *,
        announce: int | None = None,
        honour_range: bool = True,
        truncate_to: int | None = None,
    ) -> None:
        self.payload = payload
        self.announce = len(payload) if announce is None else announce
        self.honour_range = honour_range
        self.truncate_to = truncate_to
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, request: Any, timeout: float | None = None) -> FakeResponse:
        rng = request.get_header("Range")
        self.calls.append((request.full_url, rng))
        start = 0
        status = 200
        headers: dict[str, str] = {}
        if rng and self.honour_range:
            start = int(str(rng).removeprefix("bytes=").rstrip("-"))
            status = 206
            headers["Content-Range"] = f"bytes {start}-{self.announce - 1}/{self.announce}"
        body = self.payload[start:]
        if self.truncate_to is not None:
            body = body[: max(0, self.truncate_to - start)]
        headers["Content-Length"] = str(self.announce - start)
        return FakeResponse(body, status=status, headers=headers)


@pytest.fixture
def script(load_script: Loader) -> types.ModuleType:
    """The download script, imported by path."""
    return load_script(SCRIPT)


@pytest.fixture
def approved(
    script: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> Iterator[types.ModuleType]:
    """The script with ``require_gate`` neutralised — the gate itself is tested elsewhere.

    This does **not** open a gate: it replaces the check inside an imported module object for the
    duration of one test, exactly as ``tests/gated`` replaces the download function. No
    environment variable is set and no ledger row is written.
    """
    monkeypatch.setattr(script, "require_gate", lambda *a, **k: None)
    yield script


@pytest.fixture
def corpus_tgz(tmp_path: Path) -> bytes:
    """A small ``.tgz`` with the official split layout, as raw bytes to be 'served'."""
    from aicls.data import synth_inkml_dataset

    corpus = tmp_path / "payload" / "mathwriting-2024"
    for index, (split, count) in enumerate({"train": 3, "valid": 2}.items()):
        synth_inkml_dataset(corpus / split, count, 1234 + index, prefix=split)
    archive = tmp_path / "payload.tgz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(corpus, arcname=corpus.name)
    return archive.read_bytes()


def _catalog_bytes(
    script: types.ModuleType, monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> None:
    """Point the plan at a catalog entry that describes the fixture archive.

    Both the size **and the digest** are overridden. Until 2026-09-08 the catalogue carried
    ``sha256: null`` for MathWriting and only the size had to be patched; when the real digests
    were filled in from the approved download, every test using this helper started failing with
    ``sha256_mismatch`` — the script was right and the fixture was lying about its identity.
    Patching the digest to the fixture's own keeps the verification step *live* in these tests
    (``test_approved_run_refuses_a_digest_mismatch`` is the other half) instead of disabling it
    by writing ``None``, which would have made the failure go away without proving anything.
    """
    from aicls.core.hashing import sha256_bytes
    from aicls.data import catalog_entry

    update = {"expected_bytes": len(payload), "sha256": sha256_bytes(payload)}
    real = catalog_entry(script.CATALOG_NAME)
    excerpt = catalog_entry(script.EXCERPT_CATALOG_NAME)
    patched = {
        script.CATALOG_NAME: real.model_copy(update=update),
        script.EXCERPT_CATALOG_NAME: excerpt.model_copy(update=update),
    }
    monkeypatch.setattr(script, "catalog_entry", lambda name: patched[name])


# ------------------------------------------------------------------------------------ dry run
def test_dry_run_prints_the_corrected_size_licence_and_target(
    script: types.ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The gate-free preview is the approval dossier; every corrected fact has to be in it."""
    dest = tmp_path / "mathwriting"

    assert script.main(["--dest", str(dest), "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "CC-BY-NC-SA-4.0" in out
    assert "NonCommercial" in out
    assert "3096141721 B (2.88 GiB)" in out
    assert "mathwriting-2024.tgz" in out
    assert str(dest) in out
    assert "gate status    : closed" in out
    assert not dest.exists()


def test_dry_run_excerpt_plans_the_15_mb_sample_instead(
    script: types.ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "mathwriting"

    assert script.main(["--dest", str(dest), "--excerpt", "--dry-run", "--json"]) == 0

    import json

    payload = json.loads(capsys.readouterr().out.split("dry run:")[0])
    assert script.human_size(1_620_634) == "1.55 MiB"  # not "0.00 GiB"
    assert payload["dataset"] == "mathwriting_excerpt"
    assert payload["expected_bytes"] == 1_620_634
    assert payload["license"] == "CC-BY-NC-SA-4.0"
    assert payload["archive"].endswith("mathwriting-2024-excerpt.tgz")
    assert payload["extract"] is False
    assert payload["gate"] == "DATASET_DOWNLOAD"
    assert payload["gate_open"] is False


def test_plan_needs_only_the_archive_unless_extraction_is_asked_for(
    script: types.ModuleType, tmp_path: Path
) -> None:
    """Streaming preparation removed the ~20 GB estimate from the default disk requirement."""
    plain = script.build_plan(tmp_path / "d")
    unpacked = script.build_plan(tmp_path / "d", extract=True, extract_factor=7.0)

    assert plain.needed_bytes == plain.entry.expected_bytes
    assert unpacked.needed_bytes == int(plain.entry.expected_bytes * 8.0)


# ------------------------------------------------------------------------- the transfer itself
def test_download_writes_through_a_part_file_and_publishes_atomically(
    script: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"mathwriting-bytes" * 100
    transport = FakeTransport(payload)
    monkeypatch.setattr(urllib.request, "urlopen", transport)
    target = tmp_path / "mathwriting-2024.tgz"

    written = script.download_archive(
        "https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz",
        target,
        expected_bytes=len(payload),
    )

    assert written == target
    assert target.read_bytes() == payload
    assert not target.with_name(target.name + script.PARTIAL_SUFFIX).exists()
    assert transport.calls == [
        ("https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz", None)
    ]


def test_download_resumes_from_a_partial_file_instead_of_restarting(
    script: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 3 GB transfer that dropped at 90 % must cost the remaining 10 %, not 3 GB again."""
    payload = bytes(range(256)) * 40
    target = tmp_path / "mathwriting-2024.tgz"
    partial = target.with_name(target.name + script.PARTIAL_SUFFIX)
    partial.write_bytes(payload[:9000])
    transport = FakeTransport(payload)
    monkeypatch.setattr(urllib.request, "urlopen", transport)

    script.download_archive(
        "https://storage.googleapis.com/x.tgz", target, expected_bytes=len(payload)
    )

    assert target.read_bytes() == payload
    assert transport.calls[0][1] == "bytes=9000-"


def test_download_restarts_when_the_server_ignores_the_range_request(
    script: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"abcdefghij" * 500
    target = tmp_path / "a.tgz"
    target.with_name(target.name + script.PARTIAL_SUFFIX).write_bytes(payload[:1000])
    monkeypatch.setattr(urllib.request, "urlopen", FakeTransport(payload, honour_range=False))

    script.download_archive(
        "https://storage.googleapis.com/a.tgz", target, expected_bytes=len(payload)
    )

    assert target.read_bytes() == payload  # not the 1000 stale bytes plus a full body


def test_download_refuses_a_length_the_catalogue_does_not_expect_before_writing(
    script: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The size check happens on the response headers, so no byte reaches the disk."""
    payload = b"x" * 512
    monkeypatch.setattr(urllib.request, "urlopen", FakeTransport(payload))
    target = tmp_path / "a.tgz"

    with pytest.raises(InputValidationError) as excinfo:
        script.download_archive("https://storage.googleapis.com/a.tgz", target, expected_bytes=999)

    assert excinfo.value.details["reason"] == "content_length_mismatch"
    assert excinfo.value.details["announced"] == 512
    assert not target.exists()
    assert not target.with_name(target.name + script.PARTIAL_SUFFIX).exists()


def test_download_keeps_the_part_file_when_the_transfer_ends_short(
    script: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting the partial file would turn a dropped 3 GB transfer into a restart from zero."""
    payload = b"y" * 4000
    monkeypatch.setattr(urllib.request, "urlopen", FakeTransport(payload, truncate_to=1500))
    target = tmp_path / "a.tgz"

    with pytest.raises(InputValidationError) as excinfo:
        script.download_archive(
            "https://storage.googleapis.com/a.tgz", target, expected_bytes=len(payload)
        )

    assert excinfo.value.details["reason"] == "short_download"
    partial = target.with_name(target.name + script.PARTIAL_SUFFIX)
    assert not target.exists()
    assert partial.is_file() and partial.stat().st_size == 1500


def test_download_discards_a_partial_larger_than_the_expected_archive(
    script: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"z" * 300
    target = tmp_path / "a.tgz"
    target.with_name(target.name + script.PARTIAL_SUFFIX).write_bytes(b"q" * 5000)
    transport = FakeTransport(payload)
    monkeypatch.setattr(urllib.request, "urlopen", transport)

    script.download_archive(
        "https://storage.googleapis.com/a.tgz", target, expected_bytes=len(payload)
    )

    assert target.read_bytes() == payload
    assert transport.calls[0][1] is None  # started over, no Range header


def test_download_rejects_a_url_outside_the_allow_list(script: types.ModuleType) -> None:
    for bad in ("http://storage.googleapis.com/a.tgz", "https://example.com/a.tgz", None):
        with pytest.raises(InputValidationError):
            script.validate_url(bad)


def test_announced_total_understands_content_range_and_content_length(
    script: types.ModuleType,
) -> None:
    ranged = FakeResponse(b"", status=206, headers={"Content-Range": "bytes 10-99/100"})
    plain = FakeResponse(b"", status=200, headers={"Content-Length": "90"})
    silent = FakeResponse(b"", status=200, headers={})

    assert script.announced_total(ranged, 10) == 100
    assert script.announced_total(plain, 10) == 100
    assert script.announced_total(silent, 10) == 0


# ------------------------------------------------------------------------------- tar handling
def test_extract_archive_unpacks_a_tgz_and_refuses_escaping_members(
    script: types.ModuleType, tmp_path: Path, corpus_tgz: bytes
) -> None:
    archive = tmp_path / "corpus.tgz"
    archive.write_bytes(corpus_tgz)

    count = script.extract_archive(archive, tmp_path / "raw")

    assert count == 5
    assert sorted(p.name for p in (tmp_path / "raw" / "mathwriting-2024").iterdir()) == [
        "train",
        "valid",
    ]

    evil = tmp_path / "evil.tgz"
    with tarfile.open(evil, "w:gz") as bundle:
        info = tarfile.TarInfo("../escaped.txt")
        info.size = 3
        bundle.addfile(info, io.BytesIO(b"bad"))
    with pytest.raises(InputValidationError) as excinfo:
        script.extract_archive(evil, tmp_path / "raw2")
    assert excinfo.value.details["reason"] == "tar_member_escapes_root"
    assert not (tmp_path / "escaped.txt").exists()


def test_extract_archive_refuses_a_symlink_member(script: types.ModuleType, tmp_path: Path) -> None:
    """Nothing but plain files and directories belongs in an InkML corpus."""
    archive = tmp_path / "link.tgz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        bundle.addfile(info)

    with pytest.raises(InputValidationError) as excinfo:
        script.extract_archive(archive, tmp_path / "raw")
    assert excinfo.value.details["reason"] == "tar_member_not_a_regular_file"


# ---------------------------------------------------------------------------- the whole script
def test_approved_run_downloads_verifies_and_writes_a_manifest_without_extracting(
    approved: types.ModuleType,
    tmp_path: Path,
    corpus_tgz: bytes,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End to end with the gate neutralised: .tgz on disk, manifest written, nothing unpacked."""
    _catalog_bytes(approved, monkeypatch, corpus_tgz)
    monkeypatch.setattr(urllib.request, "urlopen", FakeTransport(corpus_tgz))
    dest = tmp_path / "mathwriting"

    code = approved.main(["--dest", str(dest), "--i-have-approval"])

    assert code == 0
    archive = dest / "mathwriting-2024.tgz"
    assert archive.read_bytes() == corpus_tgz
    assert (dest / "manifest.json").is_file()
    assert not (dest / "raw").exists()
    out = capsys.readouterr().out
    assert "CC-BY-NC-SA-4.0" in out
    assert "--archive" in out  # the follow-up command streams the archive


def test_approved_run_with_extract_unpacks_and_keeps_the_archive_when_asked(
    approved: types.ModuleType,
    tmp_path: Path,
    corpus_tgz: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _catalog_bytes(approved, monkeypatch, corpus_tgz)
    monkeypatch.setattr(urllib.request, "urlopen", FakeTransport(corpus_tgz))
    dest = tmp_path / "mathwriting"

    code = approved.main(["--dest", str(dest), "--i-have-approval", "--extract", "--keep-archive"])

    assert code == 0
    assert (dest / "raw" / "mathwriting-2024" / "train").is_dir()
    assert (dest / "mathwriting-2024.tgz").is_file()


def test_approved_excerpt_run_writes_the_small_archive_under_its_own_name(
    approved: types.ModuleType,
    tmp_path: Path,
    corpus_tgz: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _catalog_bytes(approved, monkeypatch, corpus_tgz)
    monkeypatch.setattr(urllib.request, "urlopen", FakeTransport(corpus_tgz))
    dest = tmp_path / "mathwriting"

    assert approved.main(["--dest", str(dest), "--excerpt", "--i-have-approval"]) == 0

    assert (dest / "mathwriting-2024-excerpt.tgz").is_file()
    assert not (dest / "mathwriting-2024.tgz").exists()


def test_approved_run_refuses_a_digest_mismatch(
    approved: types.ModuleType,
    tmp_path: Path,
    corpus_tgz: bytes,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A byte-perfect size with the wrong content must still be refused.

    ``expected_bytes`` only proves the transfer was complete; the digest is what proves it is the
    corpus a reported CER was measured on. Both MathWriting entries carry a real ``sha256`` since
    2026-09-08, so this path is now live in production and has to be pinned.
    """
    from aicls.data import catalog_entry

    wrong = "0" * 64
    real = catalog_entry(approved.CATALOG_NAME)
    update = {"expected_bytes": len(corpus_tgz), "sha256": wrong}
    monkeypatch.setattr(
        approved, "catalog_entry", lambda name: real.model_copy(update=update)
    )
    monkeypatch.setattr(urllib.request, "urlopen", FakeTransport(corpus_tgz))
    dest = tmp_path / "mathwriting"

    code = approved.main(["--dest", str(dest), "--i-have-approval"])

    assert code == 1
    assert "sha256_mismatch" in capsys.readouterr().err


def test_approved_run_refuses_when_the_volume_is_too_small(
    approved: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The disk check is the last inert step before the network; it must still bite."""
    monkeypatch.setattr(approved, "free_bytes_for", lambda path: 1024)

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("the disk check must run before any transfer")

    monkeypatch.setattr(approved, "download_archive", explode)

    assert approved.main(["--dest", str(tmp_path / "d"), "--i-have-approval"]) == 1


def test_script_default_destination_follows_the_data_root(
    script: types.ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AICLS_DATA_ROOT", str(tmp_path))
    get_settings.cache_clear()
    try:
        plan = script.build_plan()
        assert plan.dest == tmp_path / "mathwriting"
        assert plan.archive == tmp_path / "mathwriting" / "mathwriting-2024.tgz"
    finally:
        get_settings.cache_clear()
