"""Turn a contribution export into the ASR training manifest (**ungated**).

Input is the JSONL export of the voice data-collection tab (``aicls contrib export-manifest``,
:func:`aicls.collect.manifest.export_manifest`), which is already pseudonymised. This script does
the part the trainer depends on and the part reviewers ask about:

* it **refuses** any row carrying an identity field (``user_id``, ``email``, ``name``,
  ``birth_date``, ``ip`` — :data:`aicls.data.audio_commands.FORBIDDEN_FIELDS`). KVKK is enforced
  here as a hard error, not as a convention;
* it builds a **speaker-disjoint**, stratified split with
  :func:`aicls.data.splits.make_splits` (``group_key=speaker_pseudonym``,
  ``stratify=(university, gender)``, ratios 0.8/0.1/0.1, seed 1234). A speaker can never appear
  in two splits — that property is what makes the WER < 10 % target meaningful;
* it **freezes** the previous assignment by default, so a re-export after new recordings never
  moves an old speaker out of the test set (a moved test speaker silently invalidates every
  earlier number).

Ungated: it reads a manifest and writes a manifest. It downloads nothing and never opens the
audio files unless ``--verify-files`` is used.

Usage (from the repository root, project venv)::

    python training/scripts/prepare_audio_commands.py --export <export.jsonl> --dry-run
    python training/scripts/prepare_audio_commands.py --export <export.jsonl> --out <manifest.jsonl>

Exit codes: ``0`` success or dry run, ``1`` a typed AI-CLS error (identity field, malformed row,
inconsistent stratum for one speaker).
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:  # plain checkout without an editable install
    sys.path.insert(0, str(_SRC))

from aicls.core.errors import AiclsError, InputValidationError, NotFoundError  # noqa: E402
from aicls.core.hashing import fingerprint_jsonl  # noqa: E402
from aicls.core.logging import get_logger  # noqa: E402
from aicls.core.messages_tr import msg  # noqa: E402
from aicls.core.settings import get_settings  # noqa: E402
from aicls.data import SPLITS_FILENAME, Splits, make_splits, save_splits  # noqa: E402
from aicls.data.audio_commands import FORBIDDEN_FIELDS  # noqa: E402
from aicls.data.splits import CORE_SPLITS  # noqa: E402

_log = get_logger(__name__)

#: Default export produced by the collection service.
DEFAULT_EXPORT_SUBDIR = "audio/manifest.jsonl"

#: Default training manifest location (what the ASR configs point ``data.manifest`` at).
DEFAULT_OUT_SUBDIR = "manifests/audio_commands.jsonl"

#: Default train/valid/test shares (docs/ml/veri-yonetimi.md §3).
DEFAULT_RATIOS: tuple[float, float, float] = (0.8, 0.1, 0.1)

#: Repo-wide seed.
DEFAULT_SEED = 1234

#: Stratum tolerance the data layer's own test uses; reported, not enforced, here.
STRATUM_TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class ExportRow:
    """One validated export row.

    Attributes:
        sample_id: Row id (``id``/``recording_id``/the file stem).
        speaker: Speaker pseudonym — the group that must never be split.
        stratum: ``(university, gender)``, the stratification key.
        duration_s: Recording length, for the hour counts in the summary.
        payload: The original row, written back out with ``split`` filled in.
    """

    sample_id: str
    speaker: str
    stratum: tuple[str, str]
    duration_s: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)


def read_export(path: Path) -> list[ExportRow]:
    """Read and validate the contribution export.

    Args:
        path: JSONL export file.

    Returns:
        The rows, in file order.

    Raises:
        NotFoundError: If ``path`` does not exist.
        InputValidationError: On a row that is not JSON, carries an identity field, has no
            speaker pseudonym, or is missing ``path``/``text``. ``details["reason"]`` names which.
    """
    if not path.is_file():
        raise NotFoundError(
            msg("FILE_NOT_FOUND", path=str(path)),
            details={"path": str(path), "reason": "export_missing"},
        )
    rows: list[ExportRow] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputValidationError(
                msg("VALIDATION_ERROR"),
                details={"reason": "export_row_not_json", "line": line_no, "path": str(path)},
            ) from exc
        if not isinstance(payload, dict):
            raise InputValidationError(
                msg("VALIDATION_ERROR"),
                details={"reason": "export_row_not_an_object", "line": line_no},
            )
        forbidden = sorted(FORBIDDEN_FIELDS & set(payload))
        if forbidden:
            raise InputValidationError(
                msg("VALIDATION_ERROR"),
                details={
                    "reason": "export_row_has_identity_fields",
                    "line": line_no,
                    "fields": forbidden,
                },
            )
        rows.append(_row_from(payload, line_no, seen))
    if not rows:
        raise InputValidationError(
            msg("VALIDATION_ERROR"), details={"reason": "export_empty", "path": str(path)}
        )
    return rows


def _row_from(payload: dict[str, Any], line_no: int, seen: set[str]) -> ExportRow:
    """Validate one already-parsed export row.

    Raises:
        InputValidationError: On a missing field or a duplicate id.
    """
    sample_id = str(
        payload.get("id") or payload.get("recording_id") or Path(str(payload.get("path", ""))).stem
    )
    speaker = str(payload.get("speaker_pseudonym") or payload.get("speaker") or "")
    text = payload.get("text")
    missing = [
        key
        for key, ok in (
            ("id", bool(sample_id)),
            ("speaker_pseudonym", bool(speaker)),
            ("path", bool(payload.get("path"))),
            ("text", isinstance(text, str) and bool(text.strip())),
        )
        if not ok
    ]
    if missing:
        raise InputValidationError(
            msg("VALIDATION_ERROR"),
            details={"reason": "export_row_incomplete", "line": line_no, "missing": missing},
        )
    if sample_id in seen:
        raise InputValidationError(
            msg("VALIDATION_ERROR"),
            details={"reason": "duplicate_sample_id", "line": line_no, "id": sample_id},
        )
    seen.add(sample_id)
    return ExportRow(
        sample_id=sample_id,
        speaker=speaker,
        stratum=(str(payload.get("university") or ""), str(payload.get("gender") or "")),
        duration_s=float(payload.get("duration_s") or 0.0),
        payload=dict(payload),
    )


def filter_present(
    rows: list[ExportRow], audio_root: Path
) -> tuple[list[ExportRow], list[ExportRow]]:
    """Split ``rows`` into those whose audio file exists under ``audio_root`` and those missing.

    A manifest the trainer cannot read is worse than a smaller one, so missing rows are dropped
    by default and counted in the summary (same policy as ``aicls.collect.export_manifest``).
    """
    present: list[ExportRow] = []
    missing: list[ExportRow] = []
    for row in rows:
        relative = str(row.payload.get("path", ""))
        target = Path(relative)
        candidate = target if target.is_absolute() else audio_root / relative
        (present if candidate.is_file() else missing).append(row)
    return present, missing


def assign_splits(
    rows: list[ExportRow],
    *,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
    frozen: Path | Splits | None = None,
) -> Splits:
    """Speaker-disjoint, stratified split over the export rows.

    Args:
        rows: Validated export rows.
        ratios: ``(train, valid, test)`` item shares.
        seed: Shuffle seed.
        frozen: Previous split file (or object) whose assignment is kept.

    Returns:
        Splits of **sample ids**; the speaker of a sample never appears in another split.

    Raises:
        InputValidationError: If one speaker's rows disagree about ``(university, gender)``.
    """
    speaker_of = {row.sample_id: row.speaker for row in rows}
    stratum_of = {row.sample_id: row.stratum for row in rows}
    splits = make_splits(
        [row.sample_id for row in rows],
        group_key=lambda sample_id: speaker_of[sample_id],
        stratify=lambda sample_id: stratum_of[sample_id],
        ratios=ratios,
        seed=seed,
        frozen=frozen,
    )
    return splits


def write_manifest(rows: list[ExportRow], splits: Splits, out: Path) -> Path:
    """Write the training manifest, one JSON object per line, with ``split`` filled in.

    Rows whose id is in no split are dropped: an unassigned row would be invisible to the
    dataset anyway and silently shrink the corpus.
    """
    index = splits.index()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            split = index.get(row.sample_id)
            if split is None:  # pragma: no cover - make_splits assigns every id
                continue
            payload = dict(row.payload)
            payload["split"] = split
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return out


def summarize(
    rows: list[ExportRow], splits: Splits, *, missing: int = 0, out: Path | None = None
) -> dict[str, Any]:
    """Build the summary document (counts, hours, stratum shares, fingerprints)."""
    index = splits.index()
    by_split_rows: Counter[str] = Counter()
    by_split_seconds: dict[str, float] = dict.fromkeys(CORE_SPLITS, 0.0)
    by_split_speakers: dict[str, set[str]] = {name: set() for name in CORE_SPLITS}
    strata: Counter[tuple[str, str]] = Counter()
    strata_by_split: dict[str, Counter[tuple[str, str]]] = {name: Counter() for name in CORE_SPLITS}
    for row in rows:
        split = index.get(row.sample_id)
        if split is None:  # pragma: no cover - every id is assigned
            continue
        by_split_rows[split] += 1
        by_split_seconds[split] += row.duration_s
        by_split_speakers[split].add(row.speaker)
        strata[row.stratum] += 1
        strata_by_split[split][row.stratum] += 1
    total = sum(by_split_rows.values()) or 1
    deviations = {
        f"{stratum[0]}|{stratum[1]}": {
            name: round(
                (strata_by_split[name][stratum] / (by_split_rows[name] or 1)) - (count / total), 4
            )
            for name in CORE_SPLITS
        }
        for stratum, count in sorted(strata.items())
    }
    speakers = {name: sorted(values) for name, values in by_split_speakers.items()}
    overlap = sorted(
        set(speakers["train"]) & set(speakers["valid"])
        | set(speakers["train"]) & set(speakers["test"])
        | set(speakers["valid"]) & set(speakers["test"])
    )
    return {
        "rows": sum(by_split_rows.values()),
        "speakers": len({row.speaker for row in rows}),
        "hours": round(sum(by_split_seconds.values()) / 3600.0, 3),
        "missing_files": missing,
        "by_split": {name: by_split_rows[name] for name in CORE_SPLITS},
        "by_split_speakers": {name: len(speakers[name]) for name in CORE_SPLITS},
        "by_split_hours": {name: round(by_split_seconds[name] / 3600.0, 3) for name in CORE_SPLITS},
        "speaker_overlap": overlap,
        "stratum_share_deviation": deviations,
        "stratum_tolerance": STRATUM_TOLERANCE,
        "splits_fingerprint": splits.fingerprint,
        "manifest_fingerprint": fingerprint_jsonl(out) if out is not None and out.is_file() else "",
        "seed": splits.seed,
        "ratios": list(splits.ratios or DEFAULT_RATIOS),
    }


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the script."""
    settings = get_settings()
    out_default = settings.data_root / DEFAULT_OUT_SUBDIR
    parser = argparse.ArgumentParser(
        prog="prepare_audio_commands.py",
        description="Build the speaker-disjoint ASR manifest from a contribution export.",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=settings.data_root / DEFAULT_EXPORT_SUBDIR,
        help="contribution export JSONL (default AICLS_DATA_ROOT/audio/manifest.jsonl)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=out_default,
        help="training manifest to write (default AICLS_DATA_ROOT/manifests/audio_commands.jsonl)",
    )
    parser.add_argument(
        "--splits-path",
        type=Path,
        default=None,
        help=f"split file to write (default <out dir>/{SPLITS_FILENAME})",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VALID", "TEST"),
        default=list(DEFAULT_RATIOS),
        help="item shares; must sum to 1",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="split shuffle seed")
    parser.add_argument(
        "--no-freeze",
        action="store_true",
        help="do NOT keep the previous assignment (re-splits everyone; invalidates old numbers)",
    )
    parser.add_argument(
        "--verify-files",
        action="store_true",
        help="drop rows whose audio file is not on disk under --audio-root",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=settings.data_root,
        help="root the manifest paths are relative to (default AICLS_DATA_ROOT)",
    )
    parser.add_argument("--summary", type=Path, default=None, help="also write the summary JSON")
    parser.add_argument(
        "--dry-run", action="store_true", help="validate and report, writing nothing"
    )
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command line (``sys.argv[1:]`` when omitted).

    Returns:
        Process exit code: 0 ok/dry run, 1 typed error.
    """
    args = build_parser().parse_args(argv)
    export, out = Path(args.export), Path(args.out)
    splits_path = Path(args.splits_path) if args.splits_path else out.parent / SPLITS_FILENAME
    ratios = (float(args.ratios[0]), float(args.ratios[1]), float(args.ratios[2]))

    print("plan: prepare audio_commands manifest (ungated)")
    print(f"  export      : {export}{'' if export.is_file() else '   (MISSING)'}")
    print(f"  manifest    : {out}")
    print(f"  splits      : {splits_path}")
    print(f"  ratios      : {ratios}   seed: {args.seed}   freeze: {not args.no_freeze}")

    try:
        rows = read_export(export)
        missing_rows: list[ExportRow] = []
        if args.verify_files:
            rows, missing_rows = filter_present(rows, Path(args.audio_root))
            if not rows:
                raise InputValidationError(
                    msg("VALIDATION_ERROR"),
                    details={"reason": "no_audio_files_found", "root": str(args.audio_root)},
                )
        frozen = splits_path if (not args.no_freeze and splits_path.is_file()) else None
        splits = assign_splits(rows, ratios=ratios, seed=int(args.seed), frozen=frozen)
    except AiclsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(f"details: {exc.details}", file=sys.stderr)
        return int(exc.exit_code)

    if args.dry_run:
        summary = summarize(rows, splits, missing=len(missing_rows))
        print(_render(summary, wrote=False))
        print("dry run: nothing was written.")
        return 0

    write_manifest(rows, splits, out)
    save_splits(splits, splits_path)
    summary = summarize(rows, splits, missing=len(missing_rows), out=out)
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
    _log.info(
        "training.prepare_audio_commands.done",
        rows=summary["rows"],
        speakers=summary["speakers"],
        hours=summary["hours"],
        splits_fingerprint=summary["splits_fingerprint"],
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False) if args.json else _render(summary))
    return 0


def _render(summary: dict[str, Any], *, wrote: bool = True) -> str:
    """Human-readable summary."""
    lines = [
        f"{'wrote' if wrote else 'would write'} {summary['rows']} rows "
        f"({summary['hours']} h) from {summary['speakers']} speakers",
    ]
    for name in CORE_SPLITS:
        lines.append(
            f"  {name:6s} {summary['by_split'][name]:>7d} rows  "
            f"{summary['by_split_speakers'][name]:>4d} speakers  "
            f"{summary['by_split_hours'][name]:>8.3f} h"
        )
    if summary["missing_files"]:
        lines.append(f"  dropped {summary['missing_files']} rows with no audio file on disk")
    overlap = summary["speaker_overlap"]
    lines.append(
        "  speaker disjoint: yes"
        if not overlap
        else f"  SPEAKER OVERLAP (bug): {', '.join(overlap[:5])}"
    )
    lines.append(f"  splits fingerprint  : {summary['splits_fingerprint']}")
    if summary["manifest_fingerprint"]:
        lines.append(f"  manifest fingerprint: {summary['manifest_fingerprint']}")
    lines.append(
        "  note: quote both fingerprints in any ASR report (ADR-0015); the split seed is "
        f"{summary['seed']}."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
