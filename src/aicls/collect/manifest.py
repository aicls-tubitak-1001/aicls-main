"""Training-manifest export with a frozen, speaker-disjoint, stratified split (ADR-0008).

Two files are produced next to each other:

``<out>`` (JSONL)
    One row per exportable recording, carrying **only** :data:`MANIFEST_FIELDS` — the audio path,
    the reference text, the pseudonym and the demographic bands. Nothing here identifies a
    participant (``test_export_manifest_has_no_identity_fields``), and a recording is exportable
    only when quality control passed, the row is not soft-deleted and the speaker has not
    withdrawn (``test_export_manifest_excludes_withdrawn_and_qc_failed``).

``<out>.parent/audio_commands_splits.json``
    The speaker → split map. It is read before the split is computed and written back afterwards,
    so a speaker keeps the split they were first assigned to. That is what makes WER comparable
    between the 2027 progress report and the final report: test speakers of an earlier export can
    never drift into training data later (``test_split_freezes_previous_test_speakers``).

New speakers are stratified by ``(university, gender, age_band)``: inside each stratum the
assignment order is a keyed SHA-256 of the seed and the pseudonym (deterministic, no ``random``
module state), and each speaker goes to whichever split is furthest below its quota *for that
stratum*, so every split sees every band.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from aicls.collect.models import SPLIT_NAMES, ManifestReport
from aicls.collect.storage import resolve_data_root
from aicls.core.errors import InputValidationError
from aicls.core.hashing import fingerprint_jsonl
from aicls.core.logging import get_logger
from aicls.core.messages_tr import msg
from aicls.db.models import Asset, ConsentVersion, PromptSentence, Recording, Speaker
from aicls.db.repositories.audio import RecordingRepository

#: Every key of a manifest row, in the order they are written.
MANIFEST_FIELDS: tuple[str, ...] = (
    "path",
    "text",
    "speaker_pseudonym",
    "university",
    "gender",
    "age_band",
    "split",
    "duration_s",
    "sample_rate",
    "sha256",
    "scenario_tag",
    "consent_version",
    "source",
    "recording_id",
)

#: Default train/dev/test shares (docs/ml/veri-yonetimi.md §3).
DEFAULT_RATIOS: tuple[float, float, float] = (0.8, 0.1, 0.1)

#: Default seed of the split assignment (repo-wide convention).
DEFAULT_SPLIT_SEED = 1234

#: File name of the split map written next to a manifest.
SPLITS_FILENAME = "audio_commands_splits.json"

#: Provenance labels a manifest row may carry (docs/ml/veri-yonetimi.md §4 and §5).
SOURCE_LABELS: tuple[str, ...] = ("contrib", "internal", "tts")

#: Schema version of the split file.
SPLITS_SCHEMA_VERSION = 1

_SUFFIX = ".jsonl"
_log = get_logger(__name__)

#: One exportable recording with the rows it joins to.
Triple = tuple[Recording, Speaker, PromptSentence]


@dataclass(frozen=True, slots=True)
class SplitFile:
    """The persisted speaker → split map (``audio_commands_splits.json``).

    Attributes:
        by_speaker: Pseudonym to split name.
        seed: Seed that produced the assignment of the newest speakers.
        ratios: Train/dev/test shares in force when it was written.
        updated_at: When it was last rewritten (provenance only).
    """

    by_speaker: dict[str, str] = field(default_factory=dict)
    seed: int = DEFAULT_SPLIT_SEED
    ratios: tuple[float, float, float] = DEFAULT_RATIOS
    updated_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON document written to disk."""
        return {
            "version": SPLITS_SCHEMA_VERSION,
            "seed": self.seed,
            "ratios": dict(zip(SPLIT_NAMES, self.ratios, strict=True)),
            "updated_at": self.updated_at,
            "speakers": dict(sorted(self.by_speaker.items())),
        }


def load_splits(path: Path) -> SplitFile:
    """Read a split file; a missing file yields an empty map (the first export).

    Raises:
        InputValidationError: If the file exists but is not a valid split document. Guessing here
            would silently reshuffle a frozen test set, so it is a hard error.
    """
    if not path.is_file():
        return SplitFile()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputValidationError(
            msg("VALIDATION_ERROR"),
            details={"file": str(path), "reason": "unreadable_split_file"},
        ) from exc
    speakers = document.get("speakers") if isinstance(document, dict) else None
    if not isinstance(speakers, dict):
        raise InputValidationError(
            msg("VALIDATION_ERROR"), details={"file": str(path), "reason": "no_speakers_map"}
        )
    by_speaker: dict[str, str] = {}
    for pseudonym, split in speakers.items():
        if split not in SPLIT_NAMES:
            raise InputValidationError(
                msg("VALIDATION_ERROR"),
                details={"file": str(path), "speaker": str(pseudonym), "split": str(split)},
            )
        by_speaker[str(pseudonym)] = str(split)
    stored = document.get("ratios") or {}
    ratios = (
        float(stored.get("train", DEFAULT_RATIOS[0])),
        float(stored.get("dev", DEFAULT_RATIOS[1])),
        float(stored.get("test", DEFAULT_RATIOS[2])),
    )
    return SplitFile(
        by_speaker=by_speaker,
        seed=int(document.get("seed", DEFAULT_SPLIT_SEED)),
        ratios=ratios,
        updated_at=str(document.get("updated_at", "")),
    )


def save_splits(path: Path, splits: SplitFile) -> None:
    """Write the split map as pretty, sorted JSON (a reviewable, diff-able artefact)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(splits.as_dict(), ensure_ascii=False, indent=2, sort_keys=False)
    path.write_text(payload + "\n", encoding="utf-8")


def _order_key(seed: int, pseudonym: str) -> str:
    """Deterministic, seed-dependent ordering key for a speaker (no global RNG state)."""
    return hashlib.sha256(f"{seed}:{pseudonym}".encode()).hexdigest()


def _neediest_split(targets: Mapping[str, float], current: Mapping[str, int]) -> str:
    """Return the split furthest below its quota (ties resolved train → dev → test)."""
    best = SPLIT_NAMES[0]
    best_deficit = targets[best] - current[best]
    for name in SPLIT_NAMES[1:]:
        deficit = targets[name] - current[name]
        if deficit > best_deficit:
            best, best_deficit = name, deficit
    return best


def assign_splits(
    strata: Mapping[str, Sequence[str]],
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    frozen: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Assign every speaker to a split, keeping the assignments of ``frozen`` unchanged.

    Args:
        strata: Pseudonym to the stratum key it belongs to (university, gender, age band).
        seed: Seed of the deterministic ordering inside a stratum.
        ratios: Train/dev/test shares.
        frozen: Assignments from a previous export; these are never recomputed, which is how the
            earlier test speakers stay in the test split.

    Returns:
        Pseudonym to split name for every speaker in ``strata``.
    """
    kept = dict(frozen or {})
    assignment: dict[str, str] = {p: kept[p] for p in strata if p in kept}
    keys = {pseudonym: tuple(stratum) for pseudonym, stratum in strata.items()}

    newcomers_by_stratum: dict[tuple[str, ...], list[str]] = {}
    for pseudonym, stratum in keys.items():
        if pseudonym not in assignment:
            newcomers_by_stratum.setdefault(stratum, []).append(pseudonym)

    for stratum, newcomers in sorted(newcomers_by_stratum.items()):
        current = dict.fromkeys(SPLIT_NAMES, 0)
        for pseudonym, split in assignment.items():
            if keys[pseudonym] == stratum:
                current[split] += 1
        total = sum(current.values()) + len(newcomers)
        targets = {name: ratio * total for name, ratio in zip(SPLIT_NAMES, ratios, strict=True)}
        for pseudonym in sorted(newcomers, key=lambda p: _order_key(seed, p)):
            chosen = _neediest_split(targets, current)
            assignment[pseudonym] = chosen
            current[chosen] += 1
    return assignment


def _stratum(speaker: Speaker) -> tuple[str, ...]:
    """Stratification key of a speaker: the demographic bands, never an identifier."""
    return (
        str(speaker.university.value),
        str(speaker.gender.value),
        str(speaker.age_band.value),
    )


def _count(values: Iterable[str]) -> dict[str, int]:
    """Count occurrences, preserving first-seen key order (stable JSON summaries)."""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _asset_paths(repo: RecordingRepository, triples: Sequence[Triple]) -> dict[str, str]:
    """Map recording id to the audio path stored on its asset row."""
    paths: dict[str, str] = {}
    for recording, _, _ in triples:
        asset = repo.session.get(Asset, recording.asset_id)
        paths[recording.id] = "" if asset is None else str(asset.path)
    return paths


def _consent_versions(repo: RecordingRepository, speakers: Iterable[Speaker]) -> dict[str, str]:
    """Map consent-version id to its label for the speakers being exported."""
    versions: dict[str, str] = {}
    for speaker in speakers:
        key = speaker.consent_version_id
        if key not in versions:
            row = repo.session.get(ConsentVersion, key)
            versions[key] = "" if row is None else row.version
    return versions


# The two counters below are reporting figures the repository has no query for; they are written
# with the SQLAlchemy 2.0 typed API (never raw SQL) and should move into
# ``RecordingRepository.export_exclusions()`` when ``aicls.db`` next changes.
def _count_qc_failed(repo: RecordingRepository) -> int:
    """Live recordings that automatic quality control rejected."""
    return int(
        repo.session.scalar(
            select(func.count(Recording.id)).where(
                Recording.qc_passed.is_(False), Recording.deleted_at.is_(None)
            )
        )
        or 0
    )


def _count_withdrawn(repo: RecordingRepository) -> int:
    """Recordings belonging to speakers who withdrew their consent."""
    return int(
        repo.session.scalar(
            select(func.count(Recording.id))
            .select_from(Recording)
            .join(Speaker, Speaker.id == Recording.speaker_id)
            .where(Speaker.withdrawn_at.is_not(None))
        )
        or 0
    )


def export_manifest(
    repo: RecordingRepository,
    *,
    out: Path,
    split_seed: int = DEFAULT_SPLIT_SEED,
    data_root: Path | None = None,
    splits_path: Path | None = None,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    source: str = "contrib",
    verify_files: bool = True,
) -> ManifestReport:
    """Write the ASR training manifest and refresh the frozen split file.

    Args:
        repo: The voice data-collection repository.
        out: Destination JSONL file (``audio/manifest.jsonl`` by convention).
        split_seed: Seed used to order newly assigned speakers.
        data_root: Root the ``path`` fields are relative to; defaults to ``AICLS_DATA_ROOT``.
        splits_path: Split map to freeze against; defaults to
            ``out.parent/audio_commands_splits.json``.
        ratios: Train/dev/test shares.
        source: Provenance label written into every row (``contrib``/``internal``/``tts``).
        verify_files: Skip rows whose audio file is missing on disk (and count them in the
            report) rather than exporting a manifest a trainer cannot read.

    Returns:
        A :class:`~aicls.collect.models.ManifestReport` — the summary that goes into
        ``datasets/manifests/audio_commands.json`` and the progress report.

    Raises:
        InputValidationError: If ``out`` is not a ``.jsonl`` path, ``source`` is unknown, the
            ratios are not three non-negative shares, or the split file is unreadable.
    """
    shares = _validate_export_arguments(out=out, source=source, ratios=ratios)
    root = resolve_data_root(data_root)
    destination = splits_path if splits_path is not None else out.parent / SPLITS_FILENAME
    previous = load_splits(destination)

    triples = repo.exportable()
    paths = _asset_paths(repo, triples)
    kept: list[Triple] = []
    missing = 0
    for recording, speaker, prompt in triples:
        relative = paths[recording.id]
        if verify_files and not (root / relative).is_file():
            missing += 1
            _log.warning(
                "collect_manifest_missing_audio", pseudonym=speaker.pseudonym, path=relative
            )
            continue
        kept.append((recording, speaker, prompt))

    speakers = {speaker.pseudonym: speaker for _, speaker, _ in kept}
    strata = {pseudonym: _stratum(speaker) for pseudonym, speaker in speakers.items()}
    assignment = assign_splits(strata, seed=split_seed, ratios=shares, frozen=previous.by_speaker)
    frozen_speakers = sum(1 for pseudonym in speakers if pseudonym in previous.by_speaker)
    versions = _consent_versions(repo, speakers.values())

    rows: list[dict[str, Any]] = [
        {
            "path": paths[recording.id],
            "text": recording.transcript_final or prompt.text_tr,
            "speaker_pseudonym": speaker.pseudonym,
            "university": str(speaker.university.value),
            "gender": str(speaker.gender.value),
            "age_band": str(speaker.age_band.value),
            "split": assignment[speaker.pseudonym],
            "duration_s": round(float(recording.duration_s), 3),
            "sample_rate": int(recording.sample_rate),
            "sha256": recording.sha256,
            "scenario_tag": prompt.scenario_tag,
            "consent_version": versions.get(speaker.consent_version_id, ""),
            "source": source,
            "recording_id": recording.id,
        }
        for recording, speaker, prompt in kept
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            ordered = {key: row[key] for key in MANIFEST_FIELDS}
            handle.write(json.dumps(ordered, ensure_ascii=False, sort_keys=False) + "\n")

    merged = dict(previous.by_speaker)
    merged.update(assignment)
    save_splits(
        destination,
        SplitFile(
            by_speaker=merged,
            seed=split_seed,
            ratios=shares,
            updated_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        ),
    )

    report = ManifestReport(
        path=out,
        splits_path=destination,
        rows=len(rows),
        speakers=len(speakers),
        seconds=round(sum(float(row["duration_s"]) for row in rows), 3),
        by_split=_count(str(row["split"]) for row in rows),
        by_split_speakers=_count(assignment[pseudonym] for pseudonym in speakers),
        by_university=_count(str(row["university"]) for row in rows),
        by_gender=_count(str(row["gender"]) for row in rows),
        by_age_band=_count(str(row["age_band"]) for row in rows),
        consent_versions=tuple(sorted({str(row["consent_version"]) for row in rows})),
        fingerprint=fingerprint_jsonl(out),
        excluded_qc_failed=_count_qc_failed(repo),
        excluded_withdrawn=_count_withdrawn(repo),
        missing_files=missing,
        frozen_speakers=frozen_speakers,
        seed=split_seed,
        source=source,
    )
    _log.info(
        "collect_manifest_exported",
        rows=report.rows,
        speakers=report.speakers,
        hours=round(report.hours, 3),
        fingerprint=report.fingerprint,
    )
    return report


def _validate_export_arguments(
    *, out: Path, source: str, ratios: Sequence[float]
) -> tuple[float, float, float]:
    """Validate the destination, the provenance label and the split shares.

    Raises:
        InputValidationError: On any of the three.
    """
    if out.suffix != _SUFFIX:
        raise InputValidationError(
            msg("VALIDATION_ERROR"),
            details={"field": "out", "expected_suffix": _SUFFIX, "value": out.name},
        )
    if source not in SOURCE_LABELS:
        raise InputValidationError(
            msg("VALIDATION_ERROR"),
            details={"field": "source", "value": source, "allowed": list(SOURCE_LABELS)},
        )
    shares = tuple(float(value) for value in ratios)
    if len(shares) != len(SPLIT_NAMES) or any(value < 0.0 for value in shares) or not sum(shares):
        raise InputValidationError(
            msg("VALIDATION_ERROR"), details={"field": "ratios", "value": list(shares)}
        )
    return shares[0], shares[1], shares[2]


__all__ = [
    "DEFAULT_RATIOS",
    "DEFAULT_SPLIT_SEED",
    "MANIFEST_FIELDS",
    "SOURCE_LABELS",
    "SPLITS_FILENAME",
    "SplitFile",
    "assign_splits",
    "export_manifest",
    "load_splits",
    "save_splits",
]
