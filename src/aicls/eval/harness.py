"""The evaluation harness: run every component over the committed corpora, once (ADR-0015 §5).

``run_all`` is what produces ``reports/metrics/*.json``, the Markdown summary and the
``latest.json`` pointer the MkDocs macros read. It exists so that the seven contractual numbers
are produced by **one** command, from **one** set of definitions, with the provenance attached
automatically — never by seven ad-hoc scripts whose results are pasted into a document.

What it runs, and what each number is worth:

* ``parser`` and ``cas`` — real components over real committed corpora. These are regression
  measurements on **project-authored** sets (see the provenance note each report carries), not
  independent evidence.
* ``nlu`` — the rule-based layer over the Turkish gold set, same caveat.
* ``book`` — the İP-8 resolver over the sample chapters, loaded through the real migrations.
* ``ocr`` and ``asr`` — the **stub** backends. Their numbers are wiring checks: both stubs pick
  an answer by hashing their input, from a fixed table (OCR) or a sentence pool (ASR). They are
  marked ``stub`` in the report, ``promote`` publishes ``null`` for them in ``latest.json``, and
  they must never be quoted as evidence for the CER/WER targets, which need the MathWriting
  download and the training gate.
* ``latency`` — the deterministic text pipeline (NLU -> CAS), server side only.

The OCR run is **blinded**: :func:`aicls.eval.runners.run_ocr_eval` strips the ground-truth
label from every sample before the recogniser sees it, because the stub returns that label when
it finds one. Until 2026-09-08 it did not, and the harness published ``ocr.cer = 0.0`` /
``ocr.exprate = 1.0`` — a stub marking its own exam against the answer key it had been handed.
``breakdown["label_leak_control"]`` re-runs the unblinded version on purpose and reports what it
scores, so the size of that mistake stays visible in every report.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from aicls.asr.stub import StubTranscriber
from aicls.asr.types import Audio
from aicls.core.hashing import fingerprint_jsonl, sha256_json
from aicls.core.logging import get_logger
from aicls.core.paths import find_repo_root
from aicls.data.splits import load_splits, make_splits, save_splits
from aicls.data.synthetic import synth_audio, synth_ink
from aicls.eval.metrics import corpus_cer, exact_match
from aicls.eval.pipeline import LocalPipelineClient
from aicls.eval.provenance import GitInfo, git_info, git_status_output
from aicls.eval.report import EvalReport, promote, render_markdown, write_report
from aicls.eval.runners import (
    read_jsonl,
    run_asr_eval,
    run_book_eval,
    run_cas_eval,
    run_latency_eval,
    run_nlu_eval,
    run_ocr_eval,
    run_parser_eval,
)
from aicls.mathengine import InProcessExecutor, MathEngine
from aicls.ocr import CURRICULUM_LATEX, Ink, StubRecognizer

_logger = get_logger(__name__)

#: Evaluation kinds ``run_all`` can produce, in run order.
KINDS: tuple[str, ...] = ("parser", "cas", "nlu", "ocr", "asr", "book", "latency")

#: Corpus versions of the committed sets that do not carry a header row.
DATASET_VERSIONS: dict[str, str] = {
    "parser": "latex-valid-inputs-2026.09",
    "nlu": "nlu-commands-2026.09",
    "book": "book-fetch-2026.09",
    "latency": "nlu-commands-2026.09",
    "ocr": "synthetic-curriculum-ink-v1",
    "asr": "synthetic-command-audio-v1",
}

#: Default corpus sizes of the synthetic stub runs.
DEFAULT_OCR_REPEATS = 4
DEFAULT_ASR_ROWS = 300
DEFAULT_LATENCY_N = 200

#: Duration of one synthetic utterance, seconds.
SYNTH_AUDIO_SECONDS = 2.0

#: How many synthetic speakers the audio corpus is spread over.
SYNTH_SPEAKERS = 12


@dataclass(frozen=True, slots=True)
class HarnessOutput:
    """What one ``run_all`` produced.

    Attributes:
        reports: The reports, in run order.
        files: The written per-report JSON files.
        latest: The promoted ``latest.json``.
        summary: The generated Markdown summary.
    """

    reports: tuple[EvalReport, ...]
    files: tuple[Path, ...]
    latest: Path
    summary: Path


@dataclass(frozen=True, slots=True)
class _AsrSample:
    """One synthetic audio-command sample (the shape :func:`run_asr_eval` consumes)."""

    audio: Audio
    text: str
    speaker: str
    sample_id: str


@dataclass(frozen=True, slots=True)
class _MemoryAsrDataset:
    """An in-memory speaker-disjoint audio split, built from a real split file."""

    samples: tuple[_AsrSample, ...]
    splits_fingerprint: str
    split: str = "test"
    rows: tuple[_AsrSample, ...] = field(default_factory=tuple)

    def __iter__(self) -> Iterator[_AsrSample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)


def default_paths(root: Path | None = None) -> dict[str, Path]:
    """Committed corpus paths, relative to the repository root."""
    base = Path(root) if root is not None else find_repo_root()
    return {
        "root": base,
        "valid_inputs": base / "datasets" / "latex" / "valid_inputs.jsonl",
        "problems": base / "datasets" / "curriculum" / "problems.jsonl",
        "commands": base / "datasets" / "nlu" / "commands_test.jsonl",
        "fetch_test": base / "datasets" / "book" / "fetch_test.jsonl",
        "book_sample": base / "datasets" / "book" / "sample",
        "out_dir": base / "reports" / "metrics",
    }


# --------------------------------------------------------------------------------------------
# synthetic corpora for the stub backends
# --------------------------------------------------------------------------------------------
def build_ocr_samples(*, repeats: int = DEFAULT_OCR_REPEATS, seed: int = 1234) -> list[Ink]:
    """Deterministic synthetic handwriting for the curriculum expressions.

    Every sample carries its label, because the label is the answer key the run marks against.
    It never reaches the recogniser: :func:`aicls.eval.runners.run_ocr_eval` blinds each sample
    on the way in (:func:`aicls.eval.runners.blind_sample`).
    """
    samples: list[Ink] = []
    for repeat in range(repeats):
        for index, latex in enumerate(CURRICULUM_LATEX):
            samples.append(synth_ink(np.random.default_rng(seed + repeat * 1000 + index), latex))
    return samples


def strip_labels(samples: Sequence[Ink]) -> list[Ink]:
    """The same ink without ground truth — a corpus that is blind at the source.

    Since 2026-09-08 the runner blinds every sample itself, so this is defence in depth rather
    than the only guard: it is the right shape for a caller that wants to hand a backend a
    corpus which never carried an answer in the first place (the shape
    ``training/scripts/evaluate.py`` builds with its own ``blind_ink``).
    """
    return [
        Ink(strokes=list(sample.strokes), label=None, writer_id=sample.writer_id, meta={})
        for sample in samples
    ]


def build_asr_dataset(
    commands: Path, *, n: int = DEFAULT_ASR_ROWS, seed: int = 1234, work_dir: Path | None = None
) -> tuple[_MemoryAsrDataset, tuple[str, ...]]:
    """Build a speaker-disjoint synthetic audio corpus from the Turkish command set.

    The waveforms are synthetic (no recording is ever committed, docs/ml/veri-yonetimi.md) and
    the split is produced by the **real** :func:`aicls.data.make_splits`, written to a real
    split file and read back, so its fingerprint is a genuine one and
    :func:`aicls.eval.runners.run_asr_eval` exercises its speaker-disjointness contract.

    Args:
        commands: ``datasets/nlu/commands_test.jsonl``.
        n: How many rows to synthesise.
        seed: Determinism seed.
        work_dir: Where the split file is written; a temporary directory when omitted.

    Returns:
        The ``test`` split as a dataset, and the full sentence pool (the stub's vocabulary).
    """
    rows = [row for row in read_jsonl(commands) if str(row.get("text", "")).strip()][:n]
    texts = tuple(str(row["text"]) for row in rows)
    ids = [f"synth-{index:04d}" for index in range(len(rows))]
    speaker_of = {
        sample_id: f"SPK{index % SYNTH_SPEAKERS:02d}" for index, sample_id in enumerate(ids)
    }
    splits = make_splits(ids, group_key=lambda sample_id: speaker_of[sample_id], seed=seed)
    directory = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp())
    path = save_splits(splits, directory / "audio_commands_splits.json")
    stored = load_splits(path)
    test_ids = set(stored.test)
    samples = tuple(
        _AsrSample(
            audio=synth_audio(np.random.default_rng(seed + index), SYNTH_AUDIO_SECONDS),
            text=texts[index],
            speaker=speaker_of[sample_id],
            sample_id=sample_id,
        )
        for index, sample_id in enumerate(ids)
        if sample_id in test_ids
    )
    return _MemoryAsrDataset(samples=samples, splits_fingerprint=stored.fingerprint), texts


@contextmanager
def book_resolver(sample_dir: Path) -> Iterator[tuple[Any, float]]:
    """Yield ``(resolver, tagging_coverage)`` over a migrated in-memory book database.

    The schema comes from ``alembic upgrade head``, not ``metadata.create_all``, so the İP-8
    numbers are measured against the real tables, indexes and foreign keys.
    """
    from aicls.book import BookResolver, load_units, tag_markdown_dir
    from aicls.book.ingest.validators import require_full_coverage
    from aicls.db import BookRepository, make_engine, session_scope, upgrade_head

    engine = make_engine("sqlite+pysqlite:///:memory:")
    try:
        upgrade_head(engine)
        with session_scope(engine) as session:
            results = tag_markdown_dir(Path(sample_dir))
            coverage = require_full_coverage(*results)
            repo = BookRepository(session)
            load_units(
                [node for result in results for node in result.nodes],
                [triple for result in results for triple in result.triples],
                repo,
                tagged_ratio=coverage,
            )
            yield BookResolver(repo), coverage
    finally:
        engine.dispose()


# --------------------------------------------------------------------------------------------
# the harness
# --------------------------------------------------------------------------------------------
def run_all(
    *,
    root: Path | None = None,
    out_dir: Path | None = None,
    kinds: Sequence[str] = KINDS,
    git: GitInfo | None = None,
    ocr_repeats: int = DEFAULT_OCR_REPEATS,
    asr_rows: int = DEFAULT_ASR_ROWS,
    latency_n: int = DEFAULT_LATENCY_N,
    write: bool = True,
) -> HarnessOutput:
    """Run the requested evaluations and write the report files.

    Args:
        root: Repository root (the committed corpora are found under it).
        out_dir: Where the reports go; ``reports/metrics`` by default.
        kinds: Which runners to execute, from :data:`KINDS`.
        git: Provenance of the run; read from ``.git`` when omitted, which means "unknown tree
            state" and therefore ``citable=False``.
        ocr_repeats: How many synthetic ink samples per curriculum expression.
        asr_rows: How many synthetic utterances to build before splitting.
        latency_n: How many commands to time.
        write: Write the files (set ``False`` in tests that only want the reports).

    Returns:
        The :class:`HarnessOutput`.
    """
    paths = default_paths(root)
    target = Path(out_dir) if out_dir is not None else paths["out_dir"]
    provenance = git if git is not None else git_info(paths["root"])
    failures_dir = target if write else None
    selected = [kind for kind in KINDS if kind in set(kinds)]
    reports: list[EvalReport] = []

    for kind in selected:
        _logger.info("eval.harness.running", kind=kind)
        reports.append(
            _run_one(
                kind,
                paths=paths,
                git=provenance,
                failures_dir=failures_dir,
                ocr_repeats=ocr_repeats,
                asr_rows=asr_rows,
                latency_n=latency_n,
            )
        )

    files: list[Path] = []
    latest = target / "latest.json"
    summary = target / "summary.md"
    if write:
        files = [write_report(report, target) for report in reports]
        latest = promote(reports, target)
        target.mkdir(parents=True, exist_ok=True)
        with summary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(render_markdown(reports))
    return HarnessOutput(reports=tuple(reports), files=tuple(files), latest=latest, summary=summary)


def _run_one(
    kind: str,
    *,
    paths: dict[str, Path],
    git: GitInfo,
    failures_dir: Path | None,
    ocr_repeats: int,
    asr_rows: int,
    latency_n: int,
) -> EvalReport:
    """Run one evaluation kind (see :func:`run_all`)."""
    if kind == "parser":
        return run_parser_eval(
            paths["valid_inputs"],
            git=git,
            dataset_version=DATASET_VERSIONS["parser"],
            failures_dir=failures_dir,
        )
    if kind == "cas":
        engine = MathEngine(InProcessExecutor())
        return run_cas_eval(paths["problems"], engine, git=git, failures_dir=failures_dir)
    if kind == "nlu":
        return run_nlu_eval(
            paths["commands"],
            git=git,
            dataset_version=DATASET_VERSIONS["nlu"],
            failures_dir=failures_dir,
        )
    if kind == "ocr":
        return _run_ocr(git=git, failures_dir=failures_dir, repeats=ocr_repeats)
    if kind == "asr":
        return _run_asr(paths["commands"], git=git, failures_dir=failures_dir, rows=asr_rows)
    if kind == "book":
        with book_resolver(paths["book_sample"]) as (resolver, coverage):
            return run_book_eval(
                resolver,
                paths["fetch_test"],
                tagging_coverage=coverage,
                git=git,
                dataset_version=DATASET_VERSIONS["book"],
                failures_dir=failures_dir,
            )
    client = LocalPipelineClient()
    return run_latency_eval(
        client,
        paths["commands"],
        n=latency_n,
        git=git,
        dataset_version=DATASET_VERSIONS["latency"],
    )


def synthetic_fingerprint(descriptor: dict[str, Any]) -> str:
    """Identity of a generated corpus: the hash of the parameters that generate it.

    A synthetic corpus has no file to fingerprint, but it is fully determined by its generator
    and seed, so the hash of that description identifies it exactly as a JSONL fingerprint
    identifies a committed set.
    """
    return "sha256:" + sha256_json(descriptor)


def oracle_control(recognizer: Any, samples: Sequence[Ink]) -> dict[str, Any]:
    """Score the recogniser **without** blinding, to show what the blinding is worth.

    :func:`aicls.eval.runners.run_ocr_eval` hides every label before the recogniser sees the
    sample. This helper deliberately does not: it calls the backend on the labelled ink and
    scores the answers the same way. For the stub the two numbers are worlds apart (a perfect
    ``0.0`` here against roughly ``0.97`` in the blinded run), which is the evidence that the
    blinding is load-bearing rather than decorative. For a real recogniser they would be
    identical, because a real recogniser has no use for the field.

    Args:
        recognizer: The backend to probe.
        samples: Labelled ink; samples without a label are skipped.

    Returns:
        ``unblinded_cer``, ``unblinded_exprate`` and ``n``.
    """
    pairs = [
        (str(sample.label), str(recognizer.recognize(sample).latex))
        for sample in samples
        if sample.label
    ]
    total = len(pairs)
    return {
        "unblinded_cer": corpus_cer(pairs) if pairs else 0.0,
        "unblinded_exprate": (
            sum(exact_match(ref, hyp) for ref, hyp in pairs) / total if total else 0.0
        ),
        "n": total,
    }


def _run_ocr(*, git: GitInfo, failures_dir: Path | None, repeats: int) -> EvalReport:
    """OCR wiring check: the stub over blinded ink, plus the label-leak control."""
    recognizer = StubRecognizer()
    labelled = build_ocr_samples(repeats=repeats)
    fingerprint = synthetic_fingerprint(
        {
            "generator": "aicls.eval.harness.build_ocr_samples",
            "labels": list(CURRICULUM_LATEX),
            "repeats": repeats,
            "seed": 1234,
        }
    )
    report = run_ocr_eval(
        recognizer,
        labelled,
        git=git,
        dataset_fingerprint=fingerprint,
        dataset_version=DATASET_VERSIONS["ocr"],
        model_version=getattr(recognizer, "model_version", None),
        failures_dir=failures_dir,
        stub=True,
        synthetic=True,
    )
    breakdown = dict(report.breakdown)
    breakdown["blinded"] = True
    breakdown["label_leak_control"] = {
        **oracle_control(recognizer, labelled),
        "blinded_cer": report.metrics["cer"],
        "note": (
            "the identical stub over the identical samples with their labels left in place: it "
            "hands back the answer it was given and scores a perfect 0. The measured run above "
            "blinds every sample, so the gap between these two lines is exactly the amount of "
            "credit the stub used to take for work it never did."
        ),
    }
    breakdown["notes"] = (
        "WIRING CHECK, NOT A RESULT: the stub recogniser guesses from a fixed curriculum table "
        "and every sample is blinded before it reaches the backend, so this CER is the score of "
        "a lookup and NOT evidence for the < 10 % CER target (that needs MathWriting and the "
        "training gate). The promoted latest.json therefore publishes null for it."
    )
    return report.model_copy(update={"breakdown": breakdown})


def _run_asr(commands: Path, *, git: GitInfo, failures_dir: Path | None, rows: int) -> EvalReport:
    """ASR wiring check: the hash-driven stub over a speaker-disjoint synthetic split."""
    with tempfile.TemporaryDirectory(prefix="aicls-eval-") as work:
        dataset, texts = build_asr_dataset(commands, n=rows, work_dir=Path(work))
        transcriber = StubTranscriber(sentences=texts)
        report = run_asr_eval(
            transcriber,
            dataset,
            git=git,
            corpus=commands,
            dataset_fingerprint=synthetic_fingerprint(
                {
                    "generator": "aicls.eval.harness.build_asr_dataset",
                    "texts_from": commands.name,
                    "texts_fingerprint": fingerprint_jsonl(commands),
                    "rows": rows,
                    "seconds": SYNTH_AUDIO_SECONDS,
                    "speakers": SYNTH_SPEAKERS,
                    "splits": dataset.splits_fingerprint,
                    "seed": 1234,
                }
            ),
            dataset_version=DATASET_VERSIONS["asr"],
            model_version=getattr(transcriber, "model_version", None),
            failures_dir=failures_dir,
            stub=True,
            synthetic=True,
        )
    breakdown = dict(report.breakdown)
    breakdown["blinded"] = True
    breakdown["notes"] = (
        "WIRING CHECK, NOT A RESULT: the stub transcriber picks a sentence by hashing the "
        "waveform (the runner hands it the waveform and nothing else), so its WER is the WER of "
        "a random draw from the sentence pool. It proves the speaker-disjoint split, the two WER "
        "columns and the command-intent path are wired; it is NOT evidence for the < 10 % WER "
        "target, and the promoted latest.json therefore publishes null for it."
    )
    return report.model_copy(update={"breakdown": breakdown})


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point (``python -m aicls.eval``).

    Args:
        argv: Arguments; ``sys.argv[1:]`` when omitted.

    Returns:
        Process exit code (0).
    """
    parser = argparse.ArgumentParser(prog="aicls-eval", description="Run the evaluation harness")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--kind", action="append", choices=KINDS, help="run only these kinds")
    parser.add_argument("--latency-n", type=int, default=DEFAULT_LATENCY_N)
    parser.add_argument("--asr-rows", type=int, default=DEFAULT_ASR_ROWS)
    parser.add_argument("--ocr-repeats", type=int, default=DEFAULT_OCR_REPEATS)
    parser.add_argument(
        "--use-git-status",
        action="store_true",
        help="run `git status --porcelain` to prove the tree clean (CI only)",
    )
    args = parser.parse_args(argv)
    root = find_repo_root()
    status = git_status_output(root) if args.use_git_status else None
    output = run_all(
        root=root,
        out_dir=args.out,
        kinds=args.kind or KINDS,
        git=git_info(root, status_output=status),
        ocr_repeats=args.ocr_repeats,
        asr_rows=args.asr_rows,
        latency_n=args.latency_n,
    )
    for report in output.reports:
        _logger.info(
            "eval.harness.report", kind=report.kind, n=report.n, metrics=dict(report.metrics)
        )
    return 0


__all__ = [
    "DATASET_VERSIONS",
    "DEFAULT_ASR_ROWS",
    "DEFAULT_LATENCY_N",
    "DEFAULT_OCR_REPEATS",
    "KINDS",
    "HarnessOutput",
    "book_resolver",
    "build_asr_dataset",
    "build_ocr_samples",
    "default_paths",
    "main",
    "oracle_control",
    "run_all",
    "strip_labels",
    "synthetic_fingerprint",
]
