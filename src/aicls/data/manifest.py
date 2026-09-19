"""SHA-256 manifests for large data roots (ADR-0008, docs/ml/veri-yonetimi.md §3).

Large data never enters git and never enters OneDrive; a data root is represented in the
repository by its ``manifest.json`` alone: every file with its size and SHA-256, plus the licence,
the source URL, the retrieval date and a **fingerprint** — the SHA-256 of the whole file list.
That fingerprint is what a report cites (``EvalReport.dataset_fingerprint``), so a number in the
2027-05 progress report can always be traced back to the exact bytes it was measured on.

:meth:`DataManifest.verify` never raises on a mismatch: it returns a :class:`VerifyReport`
listing missing, changed and unexpected files, so the CLI (``aicls data verify``) can print all
problems at once and the ``data``-marked tests can skip instead of failing.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from aicls.core.hashing import FINGERPRINT_PREFIX, sha256_file, sha256_json
from aicls.core.logging import get_logger
from aicls.data.errors import invalid_data, missing_path

_log = get_logger(__name__)

#: Name of the manifest file written into every data root.
MANIFEST_FILENAME = "manifest.json"

#: Directory names never walked when building a manifest.
EXCLUDED_DIRS: frozenset[str] = frozenset(
    {"__pycache__", ".git", ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
)

#: File names never recorded in a manifest (the manifest itself and OS clutter).
EXCLUDED_NAMES: frozenset[str] = frozenset({MANIFEST_FILENAME, "Thumbs.db", ".DS_Store"})


class FileEntry(BaseModel):
    """One file of a data root: repo-relative POSIX path, size in bytes and SHA-256."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    size: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)


class VerifyReport(BaseModel):
    """Result of :meth:`DataManifest.verify` — what a data root is missing or has changed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    checked: int
    missing: list[str] = Field(default_factory=list)
    changed: list[str] = Field(default_factory=list)
    extra: list[str] = Field(default_factory=list)
    fingerprint_matches: bool = True

    def summary(self) -> str:
        """One-line English summary for logs and the CLI."""
        return (
            f"checked={self.checked} missing={len(self.missing)} "
            f"changed={len(self.changed)} extra={len(self.extra)} ok={self.ok}"
        )


class DataManifest(BaseModel):
    """Identity of a data root: files, licence, source and fingerprint.

    Attributes:
        name: Dataset name (``"mathwriting"``, ``"audio_commands"``, ...).
        version: ``<name>-<YYYY.MM>`` version label (docs/ml/veri-yonetimi.md §3).
        license: SPDX identifier or free text (``"Apache-2.0"``, ``"internal"``).
        source_url: Where the data came from; ``repo://`` for data produced here.
        retrieved_at: When the root was captured (UTC).
        files: Every file below the root, sorted by path.
        fingerprint: ``"sha256:<hex>"`` over the file list — the dataset identity in reports.
    """

    model_config = ConfigDict(extra="forbid")

    #: Schema version of the manifest document itself.
    SCHEMA_VERSION: ClassVar[int] = 1

    name: str
    version: str
    license: str
    source_url: str
    retrieved_at: datetime
    files: list[FileEntry] = Field(default_factory=list)
    fingerprint: str = ""
    schema_version: int = SCHEMA_VERSION

    @property
    def total_bytes(self) -> int:
        """Sum of all file sizes."""
        return sum(entry.size for entry in self.files)

    @property
    def n_files(self) -> int:
        """Number of recorded files."""
        return len(self.files)

    # ------------------------------------------------------------------ construction
    @classmethod
    def build(
        cls,
        root: Path,
        name: str,
        *,
        license: str,
        source_url: str,
        version: str | None = None,
        retrieved_at: datetime | None = None,
    ) -> DataManifest:
        """Hash every file below ``root`` and return the manifest.

        Args:
            root: Data root to walk (must exist).
            name: Dataset name recorded in the manifest.
            license: Licence of the data.
            source_url: Origin of the data.
            version: Version label; defaults to ``<name>-<YYYY.MM>`` of ``retrieved_at``.
            retrieved_at: Capture timestamp; defaults to "now" in UTC.

        Returns:
            The manifest, with files sorted by POSIX path and the fingerprint filled in.

        Raises:
            NotFoundError: If ``root`` is not an existing directory.
        """
        root = Path(root)
        if not root.is_dir():
            raise missing_path(root, "data_root_missing")
        stamp = retrieved_at or datetime.now(UTC)
        entries = [
            FileEntry(
                path=path.relative_to(root).as_posix(),
                size=path.stat().st_size,
                sha256=sha256_file(path),
            )
            for path in _iter_files(root)
        ]
        entries.sort(key=lambda entry: entry.path)
        manifest = cls(
            name=name,
            version=version or f"{name}-{stamp:%Y.%m}",
            license=license,
            source_url=source_url,
            retrieved_at=stamp,
            files=entries,
            fingerprint=fingerprint_of(entries),
        )
        _log.info(
            "data.manifest.built",
            dataset=name,
            files=manifest.n_files,
            bytes=manifest.total_bytes,
            fingerprint=manifest.fingerprint,
        )
        return manifest

    # ------------------------------------------------------------------ verification
    def verify(self, root: Path) -> VerifyReport:
        """Re-hash ``root`` and compare it with this manifest.

        Args:
            root: The data root to check.

        Returns:
            A :class:`VerifyReport`; ``ok`` is true only when nothing is missing, changed or
            unexpected and the recomputed fingerprint matches.

        Raises:
            NotFoundError: If ``root`` is not an existing directory.
        """
        root = Path(root)
        if not root.is_dir():
            raise missing_path(root, "data_root_missing")
        on_disk = {
            path.relative_to(root).as_posix(): FileEntry(
                path=path.relative_to(root).as_posix(),
                size=path.stat().st_size,
                sha256=sha256_file(path),
            )
            for path in _iter_files(root)
        }
        recomputed = fingerprint_of(list(on_disk.values()))
        missing: list[str] = []
        changed: list[str] = []
        seen: set[str] = set()
        for entry in self.files:
            found = on_disk.get(entry.path)
            if found is None:
                missing.append(entry.path)
                continue
            seen.add(entry.path)
            if found.size != entry.size or found.sha256 != entry.sha256:
                changed.append(entry.path)
        extra = sorted(set(on_disk) - seen)
        report = VerifyReport(
            ok=not (missing or changed or extra) and recomputed == self.fingerprint,
            checked=len(self.files),
            missing=missing,
            changed=changed,
            extra=extra,
            fingerprint_matches=recomputed == self.fingerprint,
        )
        if not report.ok:
            _log.warning("data.manifest.verify_failed", dataset=self.name, summary=report.summary())
        return report

    # ------------------------------------------------------------------ persistence
    def to_json(self) -> str:
        """Serialise to pretty JSON with a trailing newline (LF, UTF-8)."""
        payload: dict[str, Any] = json.loads(self.model_dump_json())
        return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"

    def save(self, target: Path) -> Path:
        """Write ``manifest.json``.

        Args:
            target: A directory (the file is written as ``<target>/manifest.json``) or the
                target file path itself.

        Returns:
            The path that was written.
        """
        target = Path(target)
        path = target / MANIFEST_FILENAME if target.is_dir() else target
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(self.to_json())
        return path


def fingerprint_of(files: list[FileEntry]) -> str:
    """``"sha256:<hex>"`` over the (path, size, sha256) triples, in path order."""
    ordered = sorted(files, key=lambda entry: entry.path)
    return FINGERPRINT_PREFIX + sha256_json(
        [[entry.path, entry.size, entry.sha256] for entry in ordered]
    )


def load_manifest(path: Path) -> DataManifest:
    """Read a ``manifest.json`` written by :meth:`DataManifest.save`.

    Args:
        path: The manifest file, or a directory containing one.

    Returns:
        The parsed manifest.

    Raises:
        NotFoundError: If the file does not exist.
        InputValidationError: If it is not valid JSON or not a valid manifest document.
    """
    path = Path(path)
    if path.is_dir():
        path = path / MANIFEST_FILENAME
    if not path.is_file():
        raise missing_path(path, "manifest_missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise invalid_data("manifest_not_json", path=str(path), error=str(exc)) from exc
    try:
        return DataManifest.model_validate(payload)
    except ValueError as exc:
        raise invalid_data("manifest_invalid", path=str(path), error=str(exc)) from exc


def _iter_files(root: Path) -> list[Path]:
    """Every file below ``root``, excluding caches and the manifest itself."""
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if any(part in EXCLUDED_DIRS for part in parts[:-1]):
            continue
        if path.name in EXCLUDED_NAMES:
            continue
        found.append(path)
    return found


__all__ = [
    "EXCLUDED_DIRS",
    "EXCLUDED_NAMES",
    "MANIFEST_FILENAME",
    "DataManifest",
    "FileEntry",
    "VerifyReport",
    "fingerprint_of",
    "load_manifest",
]
