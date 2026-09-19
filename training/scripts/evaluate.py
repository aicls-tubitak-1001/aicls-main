"""Measure a checkpoint on a held-out split and write an :class:`aicls.eval.EvalReport`
(**ungated**).

Ungated because it neither downloads nor trains: it reads data that is already on disk and a
checkpoint that is already in ``AICLS_MODELS_ROOT``. The numbers it produces are the ones a
TÜBİTAK progress report quotes, so the script is strict about provenance:

* the metric definitions come from :mod:`aicls.eval` — one definition, one implementation
  (ADR-0015). This script computes nothing itself;
* every report records the dataset fingerprint, and the ASR runner **refuses** to produce a WER
  without a speaker-disjoint split assignment;
* a run without a real checkpoint (``--stub``) or on a synthetic source is marked ``stub=true``
  in the report. That flag makes every renderer say "wiring check", because a stub number is not
  evidence for the CER < 10 % / WER < 10 % targets and must never be quoted as one.

Usage (from the repository root, project venv)::

    python training/scripts/evaluate.py --component ocr --synthetic 32 --stub
    python training/scripts/evaluate.py --component ocr --shards <root> --split test \\
        --model-dir <checkpoint>
    python training/scripts/evaluate.py --component asr --manifest <manifest.jsonl> \\
        --split test --model-dir <checkpoint>

Exit codes: ``0`` success or dry run, ``1`` a typed AI-CLS error (missing shards, missing split
assignment, unreadable checkpoint).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:  # plain checkout without an editable install
    sys.path.insert(0, str(_SRC))

from aicls.core.errors import AiclsError  # noqa: E402
from aicls.core.logging import get_logger  # noqa: E402
from aicls.core.settings import get_settings  # noqa: E402
from aicls.eval import (  # noqa: E402
    EvalReport,
    render_markdown,
    run_asr_eval,
    run_ocr_eval,
    write_report,
)
from aicls.eval.provenance import git_info, git_status_output  # noqa: E402
from aicls.eval.report import DEFAULT_OUT_DIR  # noqa: E402

_log = get_logger(__name__)

#: Components this script can evaluate.
COMPONENTS: tuple[str, ...] = ("ocr", "asr")

#: Dataset-identity string used when the source is generated rather than read from disk.
SYNTHETIC_FINGERPRINT = "synthetic:{kind}:n={n},seed={seed}"

#: What the ASR runner is told when the source has no speakers at all. It is deliberately not a
#: hash: nobody may mistake a synthetic wiring check for a speaker-disjoint measurement.
SYNTHETIC_SPLITS_FINGERPRINT = "synthetic:no-speaker-disjoint-split"

#: Printed under any report that is a wiring check rather than evidence.
WIRING_CHECK_NOTE = (
    "WIRING CHECK, NOT A RESULT: this report was produced with a stub model and/or a synthetic "
    "source. It shows that the evaluation path runs, not that anything was recognised. Real "
    "CER/WER evidence needs the DATASET_DOWNLOAD, WEIGHTS_DOWNLOAD, CUDA_INSTALL and TRAINING "
    "gates open (ADR-0012)."
)

#: Repo-wide seed.
DEFAULT_SEED = 1234


def blind_ink(ink: Any) -> Any:
    """Return a copy of ``ink`` with the ground-truth label removed.

    :class:`aicls.ocr.types.Ink` carries its own ``label``, and
    :class:`aicls.ocr.stub.StubRecognizer` **returns that label when it is present**. Handing a
    labelled sample to a recogniser would therefore turn any stub evaluation into an oracle with
    a CER of exactly 0 — a number that looks like a result and is nothing of the sort. The
    evaluation source blinds every sample, so the model only ever sees strokes.
    """
    import dataclasses

    try:
        return dataclasses.replace(ink, label=None)
    except (TypeError, ValueError):  # pragma: no cover - not a dataclass; nothing to hide
        return ink


def ocr_pairs_from_shards(root: Path, split: str, max_n: int | None = None) -> list[Any]:
    """Read ``(Ink, label)`` pairs out of prepared shards.

    Args:
        root: Shards root produced by ``prepare_mathwriting.py``.
        split: Split to read (``test`` for a report).
        max_n: Stop after this many samples.

    Returns:
        A list of ``(ink, label)`` tuples, the shape :func:`aicls.eval.run_ocr_eval` expects.

    Raises:
        NotFoundError: If the split has no shard manifest.
    """
    from aicls.data import ShardReader, load_shard_manifest

    manifest = load_shard_manifest(root, split)
    pairs: list[Any] = []
    for entry in manifest.shards:
        with ShardReader(Path(root) / split / entry.name) as reader:
            for index in range(len(reader)):
                sample = reader.sample(index)
                pairs.append((blind_ink(sample.to_ink()), sample.label))
                if max_n is not None and len(pairs) >= max_n:
                    return pairs
    return pairs


def ocr_pairs_synthetic(n: int, seed: int = DEFAULT_SEED) -> list[Any]:
    """Generate ``(Ink, label)`` pairs from the curriculum templates (a wiring-check source)."""
    from aicls.data import synth_ink
    from aicls.ocr import CURRICULUM_LATEX

    return [
        (
            blind_ink(synth_ink(seed + index, CURRICULUM_LATEX[index % len(CURRICULUM_LATEX)])),
            CURRICULUM_LATEX[index % len(CURRICULUM_LATEX)],
        )
        for index in range(int(n))
    ]


def asr_dataset(manifest: Path, split: str, *, splits_path: Path | None = None) -> Any:
    """Open the audio-command split with the **waveform** as the sample's features.

    :func:`aicls.eval.run_asr_eval` hands whatever the sample carries to the transcriber as a
    waveform, so the feature function here is the identity: feature extraction belongs to the
    model wrapper, not to the evaluation source.
    """
    from aicls.data import AudioCommandDataset

    return AudioCommandDataset(
        manifest, split, feature_fn=lambda audio: audio.samples, splits=splits_path
    )


def asr_dataset_synthetic(n: int, seed: int = DEFAULT_SEED, seconds: float = 1.0) -> Any:
    """Generate synthetic utterances (a wiring-check source; there are no real speakers)."""
    from aicls.data import SyntheticAudioDataset

    return SyntheticAudioDataset(
        int(n), seed=seed, feature_fn=lambda audio: audio.samples, seconds=seconds
    )


def build_recognizer(model_dir: Path | None, *, device: str = "cpu") -> tuple[Any, str, bool]:
    """Return ``(recognizer, model_version, is_stub)`` for the OCR path."""
    if model_dir is None:
        from aicls.ocr.stub import StubRecognizer

        recognizer = StubRecognizer()
        return recognizer, str(recognizer.model_version), True
    from aicls.ocr.torch_recognizer import TorchRecognizer

    loaded = TorchRecognizer.load(model_dir, device)
    return loaded, str(loaded.model_version), False


def build_transcriber(model_dir: Path | None, *, device: str = "cpu") -> tuple[Any, str, bool]:
    """Return ``(transcriber, model_version, is_stub)`` for the ASR path."""
    if model_dir is None:
        from aicls.asr.stub import StubTranscriber

        transcriber = StubTranscriber()
        return transcriber, str(transcriber.model_version), True
    from aicls.asr.whisper_transcriber import WhisperTranscriber

    loaded = WhisperTranscriber.load(model_dir, device)
    return loaded, str(loaded.model_version), False


def _provenance(args: argparse.Namespace) -> Any:
    """The run's git state, dirty unless the caller asked for the tree to be checked.

    ``git_info`` never runs git itself: an unknown tree counts as dirty, which is the right
    default because claiming a clean tree nobody verified is how an unreproducible number gets
    quoted. The consequence, until 2026-09-09, was that this script could not produce a citable
    report at all -- it never passed ``status_output``, so every number it wrote was marked
    non-citable, including the ones its own docstring calls "the numbers a TÜBİTAK progress
    report quotes". ``--use-git-status`` is the same opt-in escape hatch the harness already had.
    """
    root = Path(__file__).resolve().parents[2]
    status = git_status_output(root) if getattr(args, "use_git_status", False) else None
    return git_info(root, status_output=status)


def evaluate_ocr(args: argparse.Namespace) -> EvalReport:
    """Run the OCR evaluation and return the report (not yet written).

    Raises:
        InputValidationError: If neither ``--shards`` nor ``--synthetic`` names a source.
    """
    recognizer, model_version, is_stub = build_recognizer(args.model_dir, device=args.device)
    if args.synthetic:
        pairs = ocr_pairs_synthetic(int(args.synthetic), int(args.seed))
        fingerprint = SYNTHETIC_FINGERPRINT.format(
            kind="ink", n=int(args.synthetic), seed=int(args.seed)
        )
        corpus = None
        version = "synthetic"
    else:
        from aicls.data import load_shard_manifest

        root = Path(args.shards)
        pairs = ocr_pairs_from_shards(root, args.split, args.max_n)
        manifest = load_shard_manifest(root, args.split)
        fingerprint = manifest.fingerprint
        corpus = root / args.split
        version = manifest.dataset
    return run_ocr_eval(
        recognizer,
        pairs,
        git=_provenance(args),
        max_n=args.max_n,
        corpus=corpus,
        dataset_fingerprint=fingerprint,
        dataset_version=version,
        model_version=model_version,
        failures_dir=args.failures_dir,
        name=args.name or f"ocr_{args.split}",
        # Two different reasons a number may not be quoted, recorded separately: `stub` says
        # there was no real model, `synthetic` says there was no real corpus. Folding the
        # second into the first made a real model on generated data report as a stub run.
        stub=is_stub,
        synthetic=bool(args.synthetic),
    )


def evaluate_asr(args: argparse.Namespace) -> EvalReport:
    """Run the ASR evaluation and return the report (not yet written)."""
    transcriber, model_version, is_stub = build_transcriber(args.model_dir, device=args.device)
    if args.synthetic:
        dataset = asr_dataset_synthetic(int(args.synthetic), int(args.seed))
        fingerprint = SYNTHETIC_FINGERPRINT.format(
            kind="audio", n=int(args.synthetic), seed=int(args.seed)
        )
        splits_fingerprint = SYNTHETIC_SPLITS_FINGERPRINT
        corpus = None
        version = "synthetic"
    else:
        dataset = asr_dataset(Path(args.manifest), args.split, splits_path=args.splits_path)
        fingerprint = None
        splits_fingerprint = None
        corpus = Path(args.manifest)
        version = "audio_commands"
    return run_asr_eval(
        transcriber,
        dataset,
        git=_provenance(args),
        splits_fingerprint=splits_fingerprint,
        corpus=corpus,
        dataset_fingerprint=fingerprint,
        dataset_version=version,
        model_version=model_version,
        failures_dir=args.failures_dir,
        name=args.name or f"asr_{args.split}",
        # Two different reasons a number may not be quoted, recorded separately: `stub` says
        # there was no real model, `synthetic` says there was no real corpus. Folding the
        # second into the first made a real model on generated data report as a stub run.
        stub=is_stub,
        synthetic=bool(args.synthetic),
    )


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the script."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Evaluate a checkpoint and write an EvalReport (ungated; local data only).",
    )
    parser.add_argument("--component", choices=COMPONENTS, required=True, help="what to evaluate")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="local checkpoint directory; omit (or pass --stub) for the stub model",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="force the stub model; the report is marked stub=true (a wiring check)",
    )
    parser.add_argument("--device", default="cpu", help="torch device for a real checkpoint")
    parser.add_argument(
        "--shards",
        type=Path,
        default=settings.data_root / "mathwriting/shards",
        help="OCR source: shards root (default AICLS_DATA_ROOT/mathwriting/shards)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=settings.data_root / "manifests/audio_commands.jsonl",
        help="ASR source: manifest JSONL",
    )
    parser.add_argument("--splits-path", type=Path, default=None, help="ASR split file override")
    parser.add_argument("--split", default="test", help="split to measure (default test)")
    parser.add_argument(
        "--synthetic",
        type=int,
        default=0,
        metavar="N",
        help="generate N synthetic samples instead of reading a corpus (wiring check)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="synthetic-source seed")
    parser.add_argument("--max-n", type=int, default=None, help="evaluate at most N samples")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="where the report JSON is written (default reports/metrics)",
    )
    parser.add_argument("--failures-dir", type=Path, default=None, help="where to write failures")
    parser.add_argument("--name", default=None, help="report name")
    parser.add_argument(
        "--use-git-status",
        action="store_true",
        help="run `git status --porcelain` to prove the tree clean, so the report can be cited",
    )
    parser.add_argument("--no-write", action="store_true", help="print the report, write nothing")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan and exit 0, measuring nothing"
    )
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser


def _plan(args: argparse.Namespace) -> str:
    """Human-readable description of what would be measured."""
    if args.synthetic:
        source = f"synthetic ({args.synthetic} samples, seed {args.seed})"
    else:
        source = str(args.shards if args.component == "ocr" else args.manifest)
    model = "stub" if args.model_dir is None else str(args.model_dir)
    wiring = args.model_dir is None or bool(args.synthetic)
    lines = [
        f"plan: evaluate {args.component} (ungated)",
        f"  source      : {source}   split {args.split}",
        f"  model       : {model}   device {args.device}",
        f"  max n       : {args.max_n if args.max_n is not None else 'all'}",
        f"  out dir     : {args.out_dir}{'  (--no-write)' if args.no_write else ''}",
    ]
    if wiring:
        lines.append(f"  HONESTY     : {WIRING_CHECK_NOTE}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command line (``sys.argv[1:]`` when omitted).

    Returns:
        Process exit code: 0 ok/dry run, 1 typed error.
    """
    args = build_parser().parse_args(argv)
    if args.stub:
        args.model_dir = None
    print(_plan(args))
    if args.dry_run:
        print("dry run: nothing was measured and nothing was written.")
        return 0

    try:
        report = evaluate_ocr(args) if args.component == "ocr" else evaluate_asr(args)
    except AiclsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(f"details: {exc.details}", file=sys.stderr)
        return int(exc.exit_code)

    if not args.no_write:
        path = write_report(report, args.out_dir)
        print(f"report: {path}")
    document: dict[str, Any] = report.model_dump(mode="json")
    if args.json:
        print(json.dumps(document, indent=2, ensure_ascii=False))
    else:
        print(render_markdown([report]))
    if document.get("breakdown", {}).get("stub") or args.synthetic:
        print(WIRING_CHECK_NOTE)
    _log.info(
        "training.evaluate.done",
        component=args.component,
        n=document.get("n"),
        metrics=document.get("metrics"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
