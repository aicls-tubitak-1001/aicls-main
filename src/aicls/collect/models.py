"""Value types of the collection flow: demographics in, decisions and reports out.

:class:`Demographics` is the only structure a participant fills in, and it is deliberately
band-only: ``extra="forbid"`` makes a request carrying a name, a birth date or an e-mail address
fail validation instead of silently storing it (KVKK data minimisation, docs/guvenlik-gizlilik.md
§2). The remaining records are plain frozen dataclasses returned to the API and CLI layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from aicls.asr.qc import QcReport
from aicls.db.enums import AgeBand, Gender, University

#: Longest accepted free-text accent region (``speakers.accent_region`` is ``VARCHAR(40)``).
MAX_ACCENT_REGION_LENGTH = 40

#: Dataset split names, in the order splits are filled (ADR-0008).
SPLIT_NAMES: tuple[str, ...] = ("train", "dev", "test")


class Demographics(BaseModel):
    """Coarse, optional demographic bands of a voice contributor.

    No field identifies a person: age is a band, gender is self-declared with a "not answered"
    default, university is one of the partner institutions, and the accent region is a coarse
    free-text label. Unknown keys are rejected so an over-eager client cannot smuggle a name or
    an e-mail address into the speaker row.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    age_band: AgeBand = AgeBand.B25_34
    gender: Gender = Gender.NA
    university: University = University.OTHER
    accent_region: str | None = None

    @field_validator("accent_region", mode="after")
    @classmethod
    def _check_accent_region(cls, value: str | None) -> str | None:
        """Trim the label, map blanks to ``None`` and bound its length."""
        if value is None:
            return None
        trimmed = value.strip()
        if not trimmed:
            return None
        if len(trimmed) > MAX_ACCENT_REGION_LENGTH:
            raise ValueError(f"accent_region is longer than {MAX_ACCENT_REGION_LENGTH} characters")
        return trimmed


@dataclass(frozen=True, slots=True)
class RecordingDecision:
    """Outcome of ``POST /contribute/recordings`` (docs/mimari/api.md §3.8).

    A rejected upload carries the quality report and nothing else: no file was written and no row
    was inserted, which is what ``test_recording_rejected_by_qc_writes_no_file`` proves.

    Attributes:
        accepted: Whether the recording passed consent and quality control.
        qc: The full quality report, including the Turkish rejection sentences.
        recording_id: ULID of the stored ``recordings`` row, or ``None`` when rejected.
        path: Absolute path of the stored WAV file, or ``None`` when rejected.
        rel_path: The same file relative to ``AICLS_DATA_ROOT`` (POSIX), as stored in the
            database, or ``None`` when rejected.
        sha256: Hex digest of the stored bytes, or ``""`` when rejected.
        duration_s: Length of the decoded utterance in seconds.
    """

    accepted: bool
    qc: QcReport
    recording_id: str | None = None
    path: Path | None = None
    rel_path: str | None = None
    sha256: str = ""
    duration_s: float = 0.0

    @property
    def reasons_tr(self) -> list[str]:
        """Turkish rejection sentences (empty when the recording was accepted)."""
        return list(self.qc.reasons_tr)

    def as_dict(self) -> dict[str, Any]:
        """``RecordingOut`` payload — deliberately without the filesystem path."""
        return {
            "id": self.recording_id,
            "accepted": self.accepted,
            "qc": self.qc.as_dict(),
            "duration_s": self.duration_s,
        }


@dataclass(frozen=True, slots=True)
class WithdrawReport:
    """Outcome of a KVKK withdrawal (``POST /contribute/withdraw``).

    Attributes:
        speaker_pseudonym: The pseudonym whose data was removed (``P00012``).
        recordings_deleted: Number of ``recordings`` rows soft-deleted by this call.
        files_deleted: Number of audio and derived-feature files removed from disk.
        bytes_deleted: Total size of those files.
        consent_kept: Always ``True`` — the consent acceptance row is the legal evidence and is
            never removed.
        withdrawn_at: Timestamp recorded on the speaker row.
    """

    speaker_pseudonym: str
    recordings_deleted: int = 0
    files_deleted: int = 0
    bytes_deleted: int = 0
    consent_kept: bool = True
    withdrawn_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form for the CLI and the audit log."""
        return {
            "speaker_pseudonym": self.speaker_pseudonym,
            "recordings_deleted": self.recordings_deleted,
            "files_deleted": self.files_deleted,
            "bytes_deleted": self.bytes_deleted,
            "consent_kept": self.consent_kept,
            "withdrawn_at": None if self.withdrawn_at is None else self.withdrawn_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ManifestReport:
    """Summary of one training-manifest export (``datasets/manifests/audio_commands.json``).

    Attributes:
        path: The JSONL manifest that was written.
        splits_path: The speaker-disjoint split file that was read and rewritten.
        rows: Number of manifest rows.
        speakers: Number of distinct speakers in the manifest.
        seconds: Total accepted audio duration.
        by_split: Row count per split.
        by_split_speakers: Speaker count per split.
        by_university / by_gender / by_age_band: Row counts per demographic band.
        consent_versions: Every consent version represented in the export.
        fingerprint: ``"sha256:<hex>"`` of the manifest file (order-sensitive).
        excluded_qc_failed: Recordings skipped because quality control rejected them.
        excluded_withdrawn: Recordings skipped because their speaker withdrew consent.
        missing_files: Recordings skipped because their audio file is no longer on disk.
        frozen_speakers: Speakers whose split came from the existing split file.
        seed: Split seed used for the newly assigned speakers.
        source: Provenance label written into every row (``"contrib"``, ``"internal"``,
            ``"tts"``).
    """

    path: Path
    splits_path: Path
    rows: int = 0
    speakers: int = 0
    seconds: float = 0.0
    by_split: dict[str, int] = field(default_factory=dict)
    by_split_speakers: dict[str, int] = field(default_factory=dict)
    by_university: dict[str, int] = field(default_factory=dict)
    by_gender: dict[str, int] = field(default_factory=dict)
    by_age_band: dict[str, int] = field(default_factory=dict)
    consent_versions: tuple[str, ...] = ()
    fingerprint: str = ""
    excluded_qc_failed: int = 0
    excluded_withdrawn: int = 0
    missing_files: int = 0
    frozen_speakers: int = 0
    seed: int = 1234
    source: str = "contrib"

    @property
    def hours(self) -> float:
        """Total accepted audio in hours (the İP-3 "50 saat" progress figure)."""
        return self.seconds / 3600.0

    def as_dict(self) -> dict[str, Any]:
        """Summary document written next to the manifest (docs/ml/veri-yonetimi.md §3)."""
        return {
            "path": self.path.as_posix(),
            "splits_path": self.splits_path.as_posix(),
            "rows": self.rows,
            "speakers": self.speakers,
            "seconds": round(self.seconds, 3),
            "hours": round(self.hours, 4),
            "by_split": dict(self.by_split),
            "by_split_speakers": dict(self.by_split_speakers),
            "by_university": dict(self.by_university),
            "by_gender": dict(self.by_gender),
            "by_age_band": dict(self.by_age_band),
            "consent_versions": list(self.consent_versions),
            "fingerprint": self.fingerprint,
            "excluded_qc_failed": self.excluded_qc_failed,
            "excluded_withdrawn": self.excluded_withdrawn,
            "missing_files": self.missing_files,
            "frozen_speakers": self.frozen_speakers,
            "seed": self.seed,
            "source": self.source,
        }


__all__ = [
    "MAX_ACCENT_REGION_LENGTH",
    "SPLIT_NAMES",
    "Demographics",
    "ManifestReport",
    "RecordingDecision",
    "WithdrawReport",
]
