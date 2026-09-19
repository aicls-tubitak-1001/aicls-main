"""Where a contributed recording lives on disk, and how it is written and removed.

Layout (docs/ml/veri-yonetimi.md §2, outside OneDrive — ADR-0011)::

    AICLS_DATA_ROOT/audio/raw/<speaker_pseudonym>/<ulid>.wav   accepted recordings
    AICLS_DATA_ROOT/audio/features/<speaker_pseudonym>/        derived features (KVKK: deleted
                                                               together with the audio)

Every directory component is derived from a validated pseudonym (``P00001``) and a generated
ULID, never from client input, so no request can escape the audio root. Writes go to a temporary
file in the same directory and are then renamed, so a crash mid-upload can never leave a
half-written WAV that the manifest would happily export.
"""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
from pathlib import Path
import re

from aicls.core.errors import InputValidationError
from aicls.core.ids import ulid
from aicls.core.logging import get_logger
from aicls.core.messages_tr import msg
from aicls.core.settings import get_settings

#: Sub-directory of ``AICLS_DATA_ROOT`` holding accepted recordings.
AUDIO_RAW_DIR = "audio/raw"

#: Sub-directory holding features derived from the recordings (removed on withdrawal).
AUDIO_FEATURES_DIR = "audio/features"

#: Suffix of a WAV file that is still being written.
TEMP_SUFFIX = ".part"

#: A speaker pseudonym: ``P`` plus five digits (``aicls.db.repositories.audio``).
_PSEUDONYM_RE = re.compile(r"^P\d{5}$")

_log = get_logger(__name__)


def resolve_data_root(data_root: Path | None = None) -> Path:
    """Return ``data_root`` or the configured ``AICLS_DATA_ROOT``."""
    return Path(data_root) if data_root is not None else get_settings().data_root


def check_pseudonym(pseudonym: str) -> str:
    """Return ``pseudonym`` if it is a well-formed ``P00001`` label.

    Raises:
        InputValidationError: If it is not — a malformed value must never become a path
            component.
    """
    if not _PSEUDONYM_RE.match(pseudonym or ""):
        raise InputValidationError(
            msg("VALIDATION_ERROR"), details={"field": "pseudonym", "value": pseudonym}
        )
    return pseudonym


def audio_raw_root(*, data_root: Path | None = None) -> Path:
    """``AICLS_DATA_ROOT/audio/raw``."""
    return resolve_data_root(data_root) / "audio" / "raw"


def speaker_audio_dir(pseudonym: str, *, data_root: Path | None = None) -> Path:
    """Directory holding one speaker's accepted recordings."""
    return audio_raw_root(data_root=data_root) / check_pseudonym(pseudonym)


def speaker_feature_dir(pseudonym: str, *, data_root: Path | None = None) -> Path:
    """Directory holding features derived from one speaker's recordings."""
    return resolve_data_root(data_root) / "audio" / "features" / check_pseudonym(pseudonym)


def new_recording_path(pseudonym: str, *, data_root: Path | None = None) -> tuple[Path, str]:
    """Return the absolute path and the ``AICLS_DATA_ROOT``-relative POSIX path of a new file.

    The file name is a fresh ULID, so recordings sort by arrival time and no client-supplied
    string ever reaches the filesystem.
    """
    name = f"{ulid()}.wav"
    directory = speaker_audio_dir(pseudonym, data_root=data_root)
    return directory / name, f"{AUDIO_RAW_DIR}/{check_pseudonym(pseudonym)}/{name}"


def write_bytes_atomic(path: Path, payload: bytes) -> int:
    """Write ``payload`` to ``path`` through a temporary file in the same directory.

    Returns:
        The number of bytes written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + TEMP_SUFFIX)
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    except OSError:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise
    return len(payload)


def _files_under(directory: Path) -> Iterator[Path]:
    if not directory.is_dir():
        return
    for candidate in sorted(directory.rglob("*")):
        if candidate.is_file():
            yield candidate


def delete_speaker_files(pseudonym: str, *, data_root: Path | None = None) -> tuple[int, int]:
    """Delete every audio and derived-feature file of one speaker (KVKK withdrawal).

    Both speaker directories are removed afterwards, so nothing is left that could hint at how
    much a participant contributed.

    Args:
        pseudonym: The speaker's pseudonym (``P00012``).
        data_root: Root to clean; defaults to ``AICLS_DATA_ROOT``.

    Returns:
        ``(files_deleted, bytes_deleted)``.
    """
    files = 0
    total = 0
    for directory in (
        speaker_audio_dir(pseudonym, data_root=data_root),
        speaker_feature_dir(pseudonym, data_root=data_root),
    ):
        for candidate in _files_under(directory):
            size = candidate.stat().st_size
            candidate.unlink()
            files += 1
            total += size
        if directory.is_dir():
            _remove_empty_tree(directory)
    if files:
        _log.info("collect_withdraw_files_deleted", files=files, bytes=total)
    return files, total


def _remove_empty_tree(directory: Path) -> None:
    """Remove ``directory`` and any now-empty sub-directories, ignoring what is still in use."""
    for candidate in sorted(directory.rglob("*"), reverse=True):
        if candidate.is_dir():
            with contextlib.suppress(OSError):
                candidate.rmdir()
    with contextlib.suppress(OSError):
        directory.rmdir()


__all__ = [
    "AUDIO_FEATURES_DIR",
    "AUDIO_RAW_DIR",
    "TEMP_SUFFIX",
    "audio_raw_root",
    "check_pseudonym",
    "delete_speaker_files",
    "new_recording_path",
    "resolve_data_root",
    "speaker_audio_dir",
    "speaker_feature_dir",
    "write_bytes_atomic",
]
