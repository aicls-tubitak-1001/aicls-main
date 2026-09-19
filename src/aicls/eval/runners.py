"""Per-component evaluation runs (docs/ml/degerlendirme.md §5).

Each runner drives a **real** component over a committed corpus and returns an
:class:`aicls.eval.report.EvalReport`. Nothing here defines a metric: every number comes from
:mod:`aicls.eval.metrics`, so the parser runner and the OCR runner cannot disagree about what a
rate is. What a runner owns is the *corpus contract*: which rows are in the denominator, what
counts as an abstention, which provenance fields must accompany the number.

Three rules are enforced rather than documented:

* a runner never invents provenance — the dataset fingerprint comes from
  :func:`aicls.core.hashing.fingerprint_jsonl` and the commit from :mod:`aicls.eval.provenance`;
* a runner refuses to produce a number it cannot defend — :func:`run_asr_eval` will not measure
  a WER without a speaker-disjoint split, because a WER measured on a speaker the model was
  trained on is not a WER;
* **a runner never lets a backend see the answer.** The sample handed to a recogniser is
  blinded first (:func:`blind_sample`), so the marking is always done against an answer key the
  candidate never held. This is not a theoretical precaution: ``aicls.ocr.StubRecognizer``
  returns ``Ink.label`` verbatim when it finds one, so an unblinded stub run scored ``cer =
  0.0`` and that number was promoted into ``reports/metrics/latest.json`` — a perfect score
  produced by an oracle marking its own exam. The ASR runner is blind by construction: it hands
  the transcriber a bare waveform and never the sample that carries the transcript.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import dataclasses
import json
from pathlib import Path
from typing import Any, Protocol

from aicls.core import messages_tr
from aicls.core.errors import AiclsError, InputValidationError
from aicls.core.hashing import fingerprint_jsonl
from aicls.core.logging import get_logger
from aicls.data.curriculum import Problem, load_curriculum
from aicls.eval import checkers
from aicls.eval.metrics import (
    DEFAULT_MAX_FAILURES,
    bootstrap_ci,
    cer,
    corpus_cer,
    corpus_ter,
    corpus_wer,
    edit_distance,
    exact_match,
    latency_stats,
    parse_success_rate,
    symbolic_equivalence,
)
from aicls.eval.normalizers import latex_v1, tr_words
from aicls.eval.provenance import GitInfo, git_info
from aicls.eval.report import (
    WIRING_CHECK_KEY,
    EvalReport,
    assess_wiring_check,
    wiring_check_reason_text,
    write_failures,
)
from aicls.mathengine import MathEngine, Result, UnsupportedProblem
from aicls.nlu import Clarification, Matched, ParseContext, parse_command

_logger = get_logger(__name__)

#: Header row id used by the curriculum set to carry its provenance.
HEADER_ID = "_header"

#: How many worst cases a runner keeps.
WORST_CASES = 100

#: Attribute names that carry ground truth on an evaluation sample. :func:`blind_sample` clears
#: every one of them before the sample is handed to a backend.
ANSWER_FIELDS: tuple[str, ...] = ("label",)

#: Label-length buckets of the OCR breakdown.
LENGTH_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("0-10", 0, 10),
    ("11-25", 11, 25),
    ("26-50", 26, 50),
    ("51+", 51, 10**9),
)


class Recognizer(Protocol):
    """What :func:`run_ocr_eval` needs from a handwriting recogniser."""

    def recognize(self, inp: Any, *, beam: int = 4) -> Any:
        """Recognise one sample and return an object carrying ``latex`` and ``elapsed_ms``."""
        ...


class Transcriber(Protocol):
    """What :func:`run_asr_eval` needs from a speech backend."""

    def transcribe(self, audio: Any) -> Any:
        """Transcribe one waveform and return an object carrying ``text`` and ``elapsed_ms``."""
        ...


class BookResolverLike(Protocol):
    """What :func:`run_book_eval` needs from the İP-8 resolver."""

    def resolve(self, query: str, *, unit_type: Any = None, k: int = 5) -> Sequence[Any]:
        """Return the best units for ``query``, best first."""
        ...


class PipelineClient(Protocol):
    """What :func:`run_latency_eval` needs: one call, one set of stage timings."""

    def run(self, text: str) -> Mapping[str, float]:
        """Execute one command end to end and return ``{stage: milliseconds}`` incl. ``total``."""
        ...


# --------------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL corpus into dictionaries.

    Args:
        path: The file.

    Returns:
        One dict per non-empty line, in file order.

    Raises:
        InputValidationError: If a line is not a JSON object.
    """
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputValidationError(
                    messages_tr.MSG_VALIDATION_ERROR,
                    details={"reason": "row_not_json", "path": str(path), "line": number},
                ) from exc
            if not isinstance(row, dict):
                raise InputValidationError(
                    messages_tr.MSG_VALIDATION_ERROR,
                    details={"reason": "row_not_an_object", "path": str(path), "line": number},
                )
            rows.append(row)
    return rows


def blind_sample(sample: Any, *, fields: Sequence[str] = ANSWER_FIELDS) -> Any:
    """Return a copy of ``sample`` with its ground-truth fields cleared.

    An evaluation sample usually carries its own answer (:attr:`aicls.ocr.types.Ink.label`),
    because the same object is a training example, a fixture and a replayed recording. A backend
    must never receive that copy: :class:`aicls.ocr.stub.StubRecognizer` returns the label when
    it finds one, and :class:`aicls.ocr.types.Ink`'s own docstring already promises that
    "recognisers never read it during inference". Blinding turns that promise into something the
    evaluation path enforces, whatever the backend does.

    Args:
        sample: The dataset sample about to be handed to a backend.
        fields: Attribute names that hold ground truth; :data:`ANSWER_FIELDS` by default.

    Returns:
        ``sample`` itself when it carries no answer, otherwise a copy with those fields set to
        ``None``. Everything else (geometry, ``sample_id``, ``writer_id``, ``meta``) survives.

    Raises:
        InputValidationError: If the sample carries an answer that cannot be removed — it is
            neither a dataclass nor a pydantic model, or the copy kept the field. Refusing is
            the point: a number measured by a backend that could read the answer key is not a
            measurement, so the runner stops instead of publishing one.
    """
    present = [name for name in fields if getattr(sample, name, None) is not None]
    if not present:
        return sample
    cleared: dict[str, None] = dict.fromkeys(present)
    if dataclasses.is_dataclass(sample) and not isinstance(sample, type):
        blinded: Any = dataclasses.replace(sample, **cleared)
    elif callable(getattr(sample, "model_copy", None)):
        blinded = sample.model_copy(update=cleared)
    else:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={
                "reason": "sample_cannot_be_blinded",
                "type": type(sample).__name__,
                "fields": present,
                "hint": "pass (model_input, label) pairs whose input carries no ground truth",
            },
        )
    leaked = [name for name in fields if getattr(blinded, name, None) is not None]
    if leaked:  # pragma: no cover - defensive: a copy that refuses to drop its answer
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={
                "reason": "sample_still_carries_the_answer",
                "type": type(sample).__name__,
                "fields": leaked,
            },
        )
    return blinded


def _provenance(git: GitInfo | None) -> GitInfo:
    """The run's git state; read from ``.git`` when the caller did not supply one."""
    return git if git is not None else git_info()


def _build_report(
    *,
    name: str,
    kind: Any,
    git: GitInfo,
    corpus: Path | None,
    dataset_version: str,
    n: int,
    metrics: Mapping[str, float],
    ci95: Mapping[str, tuple[float, float]] | None = None,
    breakdown: Mapping[str, Any] | None = None,
    model_version: str | None = None,
    failures_path: str | None = None,
    dataset_fingerprint: str | None = None,
    synthetic: bool = False,
) -> EvalReport:
    """Assemble a report with the provenance every runner must supply.

    ``synthetic`` records that the corpus was generated in-process rather than being the corpus
    the metric names. It is a fact only the runner knows -- the report can see a dirty tree and a
    sample count for itself, but not where its samples came from -- so it has to be carried across
    the boundary as data, under :data:`aicls.eval.report.WIRING_CHECK_KEY`.

    It matters most exactly when nothing else fires. A real recogniser, on a clean tree, over
    thousands of generated curriculum expressions, clears every other bar and publishes a CER as a
    measurement; but a CER over ink this repository drew is not evidence for a target about
    handwriting. The stub flag does not cover this, because the model really was the model.
    """
    fingerprint = (
        dataset_fingerprint
        if dataset_fingerprint is not None
        else (fingerprint_jsonl(corpus) if corpus is not None else "")
    )
    merged = dict(breakdown or {})
    if synthetic:
        verdict = assess_wiring_check(synthetic_data=True)
        merged[WIRING_CHECK_KEY] = {
            "reason": wiring_check_reason_text(verdict),
            "codes": list(verdict.codes),
            "reasons": [reason.to_dict() for reason in verdict.reasons],
        }
    breakdown = merged
    return EvalReport(
        name=name,
        kind=kind,
        git_sha=git.sha,
        dirty=git.dirty,
        citable=git.citable,
        dataset_fingerprint=fingerprint,
        dataset_version=dataset_version,
        model_version=model_version,
        n=n,
        metrics={key: float(value) for key, value in metrics.items()},
        ci95=dict(ci95 or {}),
        breakdown=dict(breakdown or {}),
        failures_path=failures_path,
    )


def _write_worst(
    rows: Sequence[Mapping[str, Any]], failures_dir: Path | None, stem: str
) -> str | None:
    """Write the worst cases next to the report and return the relative path, if asked to."""
    if failures_dir is None or not rows:
        return None
    path = write_failures(rows[:WORST_CASES], Path(failures_dir) / f"{stem}_worst.jsonl")
    return path.as_posix()


# --------------------------------------------------------------------------------------------
# parser (IP-4)
# --------------------------------------------------------------------------------------------
def run_parser_eval(
    path: Path,
    *,
    git: GitInfo | None = None,
    dataset_version: str = "",
    failures_dir: Path | None = None,
    name: str = "parser_valid",
) -> EvalReport:
    """Strict parse success over ``datasets/latex/valid_inputs.jsonl`` (İP-4, ``> 99 %``).

    Every row must parse, convert to SymPy **and** survive the round trip; a printed answer the
    grammar cannot read back counts as a failure (:func:`aicls.eval.metrics.parse_success_rate`).

    Args:
        path: The JSONL corpus (``id``, ``latex``, ``source`` per row).
        git: Provenance of the run.
        dataset_version: Corpus version for the report.
        failures_dir: Where to write the failing rows, when wanted.
        name: Report name.

    Returns:
        The report: ``success_rate`` with its interval, plus a per-source breakdown.
    """
    provenance = _provenance(git)
    rows = [row for row in read_jsonl(Path(path)) if row.get("id") != HEADER_ID]
    pairs = [(str(row.get("id", index)), str(row["latex"])) for index, row in enumerate(rows)]
    result = parse_success_rate(pairs, max_failures=DEFAULT_MAX_FAILURES)

    failed_ids = {row_id for row_id, _ in result.failures}
    by_source: dict[str, dict[str, float]] = {}
    flags: list[float] = []
    for row, (row_id, _) in zip(rows, pairs, strict=True):
        source = str(row.get("source", ""))
        bucket = by_source.setdefault(source, {"n": 0.0, "ok": 0.0})
        bucket["n"] += 1.0
        ok = row_id not in failed_ids
        bucket["ok"] += float(ok)
        flags.append(float(ok))
    for bucket in by_source.values():
        bucket["rate"] = bucket["ok"] / bucket["n"] if bucket["n"] else 0.0

    ci95 = {"success_rate": bootstrap_ci(flags)} if flags else {}
    failures = [{"id": row_id, "reason": reason} for row_id, reason in result.failures]
    return _build_report(
        name=name,
        kind="parser",
        git=provenance,
        corpus=Path(path),
        dataset_version=dataset_version,
        n=result.total,
        metrics={"success_rate": result.rate, "failures": float(result.total - result.ok)},
        ci95=ci95,
        breakdown={"by_source": by_source, "failures": failures[:WORST_CASES]},
        failures_path=_write_worst(failures, failures_dir, name),
    )


# --------------------------------------------------------------------------------------------
# CAS (IP-4)
# --------------------------------------------------------------------------------------------
def _cas_outcome(problem: Problem, result: Result) -> tuple[str, str]:
    """Judge one engine answer with the row's checker; returns ``(outcome, reason)``."""
    checker = problem.checker
    args = dict(problem.checker_args)
    if checker == "exact_latex":
        ok, reason = checkers.compare_latex(result.latex, problem.expected.latex)
        return ("correct" if ok else "wrong"), reason
    try:
        expected = checkers.rebuild_srepr(problem.expected.srepr)
        actual = checkers.rebuild_srepr(result.srepr)
    except InputValidationError as exc:
        return "error", f"srepr: {exc.details.get('reason')}"
    if checker == "plot_ok" and (result.kind != "plot" or not result.plot_png):
        return "wrong", f"no PNG produced (kind={result.kind})"
    strategy = "symbolic" if checker == "plot_ok" else checker
    ok, reason = checkers.compare(strategy, actual, expected, args)
    if not ok:
        return "wrong", reason
    if checker == "symbolic" and args.get("up_to_constant"):
        problem_text = checkers.check_antiderivative(
            actual, str(problem.command.get("expr", "")), args
        )
        if problem_text:
            return "wrong", problem_text
    return "correct", ""


def run_cas_eval(
    problems: Path,
    engine: MathEngine,
    *,
    git: GitInfo | None = None,
    failures_dir: Path | None = None,
    name: str = "cas_curriculum",
) -> EvalReport:
    """CAS accuracy over the İP-4 curriculum set (``> 99 %``).

    The denominator is every non-header row. ``unsupported`` rows stay in it and are reported
    as their own column (docs/ml/degerlendirme.md §2): refusing to answer is not the same as
    answering wrongly, but it is not a success either.

    Parse success is measured on the same pass, over the LaTeX inputs only (a ``text_tr`` row is
    a sentence for the NLU layer, not a parser input), with the strict round-trip definition.

    Args:
        problems: ``datasets/curriculum/problems.jsonl``.
        engine: The engine to run the commands on (``InProcessExecutor`` or the sandbox pool).
        git: Provenance of the run.
        failures_dir: Where to write the mismatching rows.
        name: Report name.

    Returns:
        The report: ``correct_rate``, ``unsupported_rate``, ``parse_success_rate``, engine
        timings, and a per-topic breakdown. The corpus provenance sentence from the header row
        is copied into ``breakdown["notes"]`` so no report can quote the number without it.
    """
    provenance = _provenance(git)
    path = Path(problems)
    rows = load_curriculum(path)
    header = next((row for row in rows if row.id == HEADER_ID), None)
    dataset_version = ""
    notes = ""
    if header is not None:
        extra = header.model_extra or {}
        dataset_version = str(extra.get("dataset_version", ""))
        notes = str(extra.get("provenance_en", ""))
    problems_rows = [row for row in rows if row.id != HEADER_ID]

    counts = {"correct": 0, "wrong": 0, "unsupported": 0, "error": 0}
    by_topic: dict[str, dict[str, float]] = {}
    durations: list[float] = []
    flags: list[float] = []
    parse_inputs: list[tuple[str, str]] = []
    failures: list[dict[str, Any]] = []

    for problem in problems_rows:
        if problem.input.kind == "latex":
            parse_inputs.append((problem.id, problem.input.value))
        try:
            result = engine.execute(dict(problem.command))
        except AiclsError as exc:
            outcome = "unsupported" if isinstance(exc, UnsupportedProblem) else "error"
            reason = f"{type(exc).__name__}: {exc.details.get('reason', exc.code.value)}"
        else:
            durations.append(float(result.elapsed_ms))
            outcome, reason = _cas_outcome(problem, result)
        counts[outcome] += 1
        flags.append(float(outcome == "correct"))
        topic = by_topic.setdefault(problem.topic, {"n": 0.0, "correct": 0.0})
        topic["n"] += 1.0
        topic["correct"] += float(outcome == "correct")
        if outcome != "correct" and len(failures) < WORST_CASES:
            failures.append(
                {"id": problem.id, "topic": problem.topic, "outcome": outcome, "reason": reason}
            )
    for topic_stats in by_topic.values():
        topic_stats["rate"] = topic_stats["correct"] / topic_stats["n"] if topic_stats["n"] else 0.0

    total = len(problems_rows)
    parse_result = parse_success_rate(parse_inputs, max_failures=WORST_CASES)
    timings = latency_stats(durations) if durations else None
    metrics = {
        "correct_rate": counts["correct"] / total if total else 0.0,
        "unsupported_rate": counts["unsupported"] / total if total else 0.0,
        "error_rate": counts["error"] / total if total else 0.0,
        "parse_success_rate": parse_result.rate,
        "parse_n": float(parse_result.total),
    }
    if timings is not None:
        metrics.update({"mean_ms": timings.mean, "p50_ms": timings.p50, "p95_ms": timings.p95})
    ci95 = {"correct_rate": bootstrap_ci(flags)} if flags else {}
    return _build_report(
        name=name,
        kind="cas",
        git=provenance,
        corpus=path,
        dataset_version=dataset_version,
        n=total,
        metrics=metrics,
        ci95=ci95,
        breakdown={
            "counts": counts,
            "by_topic": by_topic,
            "parse_failures": [
                {"id": row_id, "reason": reason} for row_id, reason in parse_result.failures
            ],
            "failures": failures,
            "notes": notes,
        },
        failures_path=_write_worst(failures, failures_dir, name),
    )


# --------------------------------------------------------------------------------------------
# NLU (IP-4)
# --------------------------------------------------------------------------------------------
def _slot_counts(
    gold: Mapping[str, Any], predicted: Mapping[str, Any], *, strict: bool
) -> tuple[int, int, int]:
    """True/false positives and false negatives for one row's slots.

    ``strict`` decides what a slot the layer resolved but the gold row does not declare means.
    Under the lenient definition it is ignored; under the strict one it is a false positive.
    The lenient definition is what let a wrong ``order`` slot ride along invisibly with a
    perfect slot F1 (KNOWN_ISSUES, priority 1), so the strict number is the reported one and
    the lenient one is kept only for comparability with the earlier gate report.
    """
    true_positive = false_positive = false_negative = 0
    for name, value in gold.items():
        if name not in predicted:
            false_negative += 1
        elif predicted[name] == value:
            true_positive += 1
        else:
            false_positive += 1
            false_negative += 1
    if strict:
        false_positive += sum(1 for name in predicted if name not in gold)
    return true_positive, false_positive, false_negative


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    """F1 from raw counts; 1.0 when there was nothing to find and nothing was claimed."""
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 1.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 1.0
    )
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_nlu_eval(
    path: Path,
    *,
    inventory: Any = None,
    git: GitInfo | None = None,
    dataset_version: str = "",
    failures_dir: Path | None = None,
    name: str = "nlu_commands",
) -> EvalReport:
    """Text-to-command accuracy over a Turkish gold set (İP-4, ``>= 0.95``).

    Definitions (frozen with the metric, because the numbers are quoted in reports):

    * **intent_accuracy** — the resolved intent equals the gold intent. A row whose gold intent
      is empty is out of scope and is only correct when the layer returns ``NoMatch``; a
      clarification is never correct.
    * **slot_f1** — over ``(name, value)`` pairs, counting a slot the layer resolved but the
      gold row does not declare as a **false positive** (the strict definition of ADR-0015 and
      KNOWN_ISSUES priority 1). ``slot_f1_declared_only`` is the older, lenient definition,
      reported next to it so the two can be compared.
    * **command_exact** — intent correct *and* every declared gold slot resolved to the gold
      value: the command the platform would execute is the one the gold row describes.
    * **clarification_rate** — share of rows that ended in a question.

    Args:
        path: Gold JSONL (``id``, ``text``, ``intent``, ``slots``, optional ``context``).
        inventory: Inventory to resolve against; the packaged one when omitted.
        git: Provenance of the run.
        dataset_version: Corpus version.
        failures_dir: Where to write the mistakes.
        name: Report name.

    Returns:
        The report.
    """
    provenance = _provenance(git)
    rows = [row for row in read_jsonl(Path(path)) if row.get("id") != HEADER_ID]
    total = len(rows)
    correct = matched = clarifications = no_matches = command_exact = 0
    strict_counts = [0, 0, 0]
    lenient_counts = [0, 0, 0]
    flags: list[float] = []
    failures: list[dict[str, Any]] = []

    for row in rows:
        gold_intent = str(row.get("intent", ""))
        gold_slots = dict(row.get("slots") or {})
        context_map = row.get("context") or {}
        context = ParseContext(**context_map) if context_map else None
        outcome = parse_command(str(row["text"]), context=context, inventory=inventory)
        if isinstance(outcome, Matched):
            matched += 1
            predicted_intent = outcome.intent
            predicted_slots: dict[str, Any] = dict(outcome.slots)
        elif isinstance(outcome, Clarification):
            clarifications += 1
            predicted_intent = f"<clarification:{outcome.intent or 'ambiguous'}>"
            predicted_slots = {}
        else:
            no_matches += 1
            predicted_intent = ""
            predicted_slots = {}

        intent_ok = predicted_intent == gold_intent
        correct += int(intent_ok)
        flags.append(float(intent_ok))
        for counts, strict in ((strict_counts, True), (lenient_counts, False)):
            if gold_slots or (strict and predicted_slots):
                row_counts = _slot_counts(gold_slots, predicted_slots, strict=strict)
                for index in range(3):
                    counts[index] += row_counts[index]
        slots_ok = all(predicted_slots.get(key) == value for key, value in gold_slots.items())
        command_exact += int(intent_ok and slots_ok)
        if not intent_ok and len(failures) < WORST_CASES:
            failures.append(
                {
                    "id": str(row.get("id", "")),
                    "text": str(row["text"]),
                    "gold": gold_intent,
                    "predicted": predicted_intent,
                }
            )

    metrics = {
        "intent_accuracy": correct / total if total else 0.0,
        "slot_f1": _f1(*strict_counts),
        "slot_f1_declared_only": _f1(*lenient_counts),
        "command_exact": command_exact / total if total else 0.0,
        "clarification_rate": clarifications / total if total else 0.0,
        "no_match_rate": no_matches / total if total else 0.0,
    }
    ci95 = {"intent_accuracy": bootstrap_ci(flags)} if flags else {}
    return _build_report(
        name=name,
        kind="nlu",
        git=provenance,
        corpus=Path(path),
        dataset_version=dataset_version,
        n=total,
        metrics=metrics,
        ci95=ci95,
        breakdown={
            "matched": matched,
            "clarifications": clarifications,
            "no_matches": no_matches,
            "slot_counts_strict": {
                "tp": strict_counts[0],
                "fp": strict_counts[1],
                "fn": strict_counts[2],
            },
            "failures": failures,
        },
        failures_path=_write_worst(failures, failures_dir, name),
    )


# --------------------------------------------------------------------------------------------
# OCR (IP-3)
# --------------------------------------------------------------------------------------------
def _ocr_items(ds: Iterable[Any], max_n: int | None) -> list[tuple[str, str, Any]]:
    """Normalise a dataset into ``(sample_id, reference_label, blinded_model_input)`` triples.

    The reference is taken off the sample **here**, and only a :func:`blind_sample` copy travels
    on to the recogniser. The two halves of a marked exam are therefore separated at the one
    place every OCR run passes through, rather than in each caller: a backend that reads
    ``label`` — the stub does — finds ``None`` and has to answer from the strokes.
    """
    items: list[tuple[str, str, Any]] = []
    for index, sample in enumerate(ds):
        if max_n is not None and index >= max_n:
            break
        if isinstance(sample, tuple) and len(sample) == 2:
            model_input, label = sample
            sample_id = str(getattr(model_input, "sample_id", index))
        else:
            model_input = sample
            label = getattr(sample, "label", None)
            sample_id = str(getattr(sample, "sample_id", index))
        reference = "" if label is None else str(label)
        items.append((sample_id, reference, blind_sample(model_input)))
    return items


def _length_bucket(length: int) -> str:
    """Name of the label-length bucket ``length`` falls into."""
    for label, low, high in LENGTH_BUCKETS:
        if low <= length <= high:
            return label
    return LENGTH_BUCKETS[-1][0]  # pragma: no cover - the last bucket is unbounded


def run_ocr_eval(
    recognizer: Recognizer,
    ds: Iterable[Any],
    *,
    max_n: int | None = None,
    git: GitInfo | None = None,
    corpus: Path | None = None,
    dataset_fingerprint: str | None = None,
    dataset_version: str = "",
    model_version: str | None = None,
    failures_dir: Path | None = None,
    name: str = "ocr_test",
    stub: bool = False,
    synthetic: bool = False,
) -> EvalReport:
    """Handwriting recognition quality (İP-3, CER ``< 10 %``).

    Every sample is blinded before it reaches ``recognizer`` (:func:`blind_sample`), so the
    hypothesis is always produced without sight of the reference it is scored against. Without
    that step the deterministic stub — which echoes ``Ink.label`` — scores a perfect ``cer = 0``
    and the harness publishes a fake headline metric.

    Args:
        recognizer: Anything with ``recognize(sample)``; the stub, or a trained model.
        ds: Samples carrying a ``label`` (or ``(input, label)`` pairs). The label is read here
            and removed from the copy the recogniser is given.
        max_n: Evaluate at most this many samples.
        git: Provenance of the run.
        corpus: Corpus file whose fingerprint identifies the data, when there is one.
        dataset_fingerprint: Explicit fingerprint, for a corpus that is generated rather than
            read from a file (the generator's parameters are its identity).
        dataset_version: Corpus version.
        model_version: Model registry version.
        failures_dir: Where to write the worst cases.
        name: Report name.
        stub: Mark the run as a wiring check produced with a stub model. A stub number is
            **not evidence** for the CER target and the flag makes every renderer say so.
        synthetic: The corpus was generated in-process rather than read from the dataset the
            metric names. Independent of ``stub``: a real model on generated ink is still not
            evidence for a target about handwriting.

    Returns:
        The report: ``cer`` (with its interval), ``ter``, ``exprate``, the three symbolic
        verdict shares and the recogniser's own timing.
    """
    provenance = _provenance(git)
    items = _ocr_items(ds, max_n)
    pairs: list[tuple[str, str]] = []
    rows: list[tuple[float, float]] = []
    durations: list[float] = []
    verdicts = {"equal": 0, "different": 0, "unparsable": 0}
    by_length: dict[str, dict[str, float]] = {}
    worst: list[dict[str, Any]] = []

    for sample_id, reference, model_input in items:
        recognition = recognizer.recognize(model_input)
        hypothesis = str(getattr(recognition, "latex", ""))
        durations.append(float(getattr(recognition, "elapsed_ms", 0.0)))
        pairs.append((reference, hypothesis))
        reference_chars = "".join(latex_v1(reference).split())
        errors = edit_distance(reference_chars, "".join(latex_v1(hypothesis).split()))
        rows.append((float(errors), float(len(reference_chars))))
        verdicts[symbolic_equivalence(reference, hypothesis)] += 1
        bucket = by_length.setdefault(
            _length_bucket(len(reference_chars)), {"n": 0.0, "errors": 0.0, "ref_chars": 0.0}
        )
        bucket["n"] += 1.0
        bucket["errors"] += float(errors)
        bucket["ref_chars"] += float(len(reference_chars))
        if errors:
            worst.append(
                {
                    "id": sample_id,
                    "ref": reference,
                    "hyp": hypothesis,
                    "cer": cer(reference, hypothesis),
                }
            )
    for bucket in by_length.values():
        bucket["cer"] = bucket["errors"] / bucket["ref_chars"] if bucket["ref_chars"] else 0.0
    worst.sort(key=lambda row: float(row["cer"]), reverse=True)

    total = len(items)
    metrics = {
        "cer": corpus_cer(pairs),
        "ter": corpus_ter(pairs),
        "exprate": (sum(exact_match(ref, hyp) for ref, hyp in pairs) / total if total else 0.0),
        "symeq_equal": verdicts["equal"] / total if total else 0.0,
        "symeq_different": verdicts["different"] / total if total else 0.0,
        "symeq_unparsable": verdicts["unparsable"] / total if total else 0.0,
    }
    if durations:
        timings = latency_stats(durations)
        metrics.update({"mean_ms": timings.mean, "p95_ms": timings.p95})
    ci95 = {"cer": bootstrap_ci(rows)} if rows else {}
    return _build_report(
        name=name,
        kind="ocr",
        git=provenance,
        corpus=corpus,
        dataset_fingerprint=dataset_fingerprint,
        dataset_version=dataset_version,
        n=total,
        metrics=metrics,
        ci95=ci95,
        breakdown={
            "by_label_length": by_length,
            "symbolic_verdicts": verdicts,
            "stub": stub,
            "worst": worst[:WORST_CASES],
        },
        model_version=model_version,
        failures_path=_write_worst(worst, failures_dir, name),
        synthetic=synthetic,
    )


# --------------------------------------------------------------------------------------------
# ASR (IP-3)
# --------------------------------------------------------------------------------------------
def _audio_of(sample: Any) -> Any:
    """The waveform of an ASR sample: an ``audio`` attribute, or features plus a sample rate.

    This is also where the ASR path is blinded: the transcriber receives an
    :class:`aicls.asr.types.Audio` built here and never the sample object, which is what carries
    ``text``, ``intent`` and ``speaker``. The OCR path needs :func:`blind_sample` because a
    recogniser is handed the sample itself; here the answer simply never leaves this function.
    """
    audio = getattr(sample, "audio", None)
    if audio is not None:
        return audio
    from aicls.asr.types import Audio

    features = getattr(sample, "features", None)
    if features is None:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "asr_sample_without_audio", "type": type(sample).__name__},
        )
    return Audio(features, int(getattr(sample, "sample_rate", 16_000)))


def _intent_of(outcome: Any) -> str:
    """Intent of a parse outcome (``""`` for a refusal, ``<clarification:…>`` for a question)."""
    if isinstance(outcome, Matched):
        return outcome.intent
    if isinstance(outcome, Clarification):
        return f"<clarification:{outcome.intent or 'ambiguous'}>"
    return ""


def run_asr_eval(
    transcriber: Transcriber,
    ds: Any,
    *,
    splits_fingerprint: str | None = None,
    git: GitInfo | None = None,
    corpus: Path | None = None,
    dataset_fingerprint: str | None = None,
    dataset_version: str = "",
    model_version: str | None = None,
    inventory: Any = None,
    failures_dir: Path | None = None,
    name: str = "asr_test",
    stub: bool = False,
    synthetic: bool = False,
) -> EvalReport:
    """Speech recognition quality on a **speaker-disjoint** split (İP-3, WER ``< 10 %``).

    Both WER columns of ADR-0015 are reported: ``wer_raw`` (no normalisation) and
    ``wer_tr_v1``. ``command_intent_accuracy`` closes the loop the contract actually cares
    about (İP-2 "komutların iletilmesi"): the transcript is pushed through
    :func:`aicls.nlu.parse_command` and compared with the intent of the reference transcript
    (or with the row's own gold intent when it carries one), so a recognition error only counts
    when it changes the command.

    Args:
        transcriber: Anything with ``transcribe(audio)``.
        ds: Dataset of samples with ``text``, ``speaker`` and a waveform. It must expose a
            ``splits_fingerprint`` (:class:`aicls.data.AudioCommandDataset` does).
        splits_fingerprint: Explicit override when the dataset does not carry one.
        git: Provenance of the run.
        corpus: Manifest file whose fingerprint identifies the data.
        dataset_fingerprint: Explicit fingerprint, for a generated corpus.
        dataset_version: Corpus version.
        model_version: Model registry version.
        inventory: NLU inventory for the command metrics.
        failures_dir: Where to write the worst cases.
        name: Report name.
        stub: Mark the run as a wiring check produced with a stub model.
        synthetic: The corpus was generated in-process rather than recorded from speakers.

    Returns:
        The report.

    Raises:
        InputValidationError: If no speaker-disjoint split assignment is available. A WER
            measured on a speaker the model has already heard is not a WER, so the runner
            refuses to produce the number at all rather than produce a flattering one.
    """
    provenance = _provenance(git)
    fingerprint = splits_fingerprint or getattr(ds, "splits_fingerprint", None)
    if not fingerprint:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={
                "reason": "speaker_split_required",
                "hint": "pass a dataset carrying splits_fingerprint (audio_commands_splits.json)",
            },
        )
    samples = list(ds)
    pairs: list[tuple[str, str]] = []
    rows: list[tuple[float, float]] = []
    durations: list[float] = []
    speakers: set[str] = set()
    intent_ok = command_ok = 0
    worst: list[dict[str, Any]] = []

    for index, sample in enumerate(samples):
        reference = str(getattr(sample, "text", ""))
        transcript = transcriber.transcribe(_audio_of(sample))
        hypothesis = str(getattr(transcript, "text", ""))
        durations.append(float(getattr(transcript, "elapsed_ms", 0.0)))
        pairs.append((reference, hypothesis))
        reference_words = tr_words(reference)
        errors = edit_distance(reference_words, tr_words(hypothesis))
        rows.append((float(errors), float(len(reference_words))))
        speakers.add(str(getattr(sample, "speaker", "")))

        gold_intent = getattr(sample, "intent", None)
        reference_outcome = parse_command(reference, inventory=inventory)
        gold = str(gold_intent) if gold_intent is not None else _intent_of(reference_outcome)
        hypothesis_outcome = parse_command(hypothesis, inventory=inventory)
        same_intent = _intent_of(hypothesis_outcome) == gold
        intent_ok += int(same_intent)
        same_command = same_intent and (
            isinstance(hypothesis_outcome, Matched)
            and isinstance(reference_outcome, Matched)
            and hypothesis_outcome.command == reference_outcome.command
        )
        command_ok += int(same_command)
        if errors or not same_intent:
            worst.append(
                {
                    "id": str(getattr(sample, "sample_id", index)),
                    "ref": reference,
                    "hyp": hypothesis,
                    "wer": errors / len(reference_words) if reference_words else 1.0,
                    "intent_ok": same_intent,
                }
            )
    worst.sort(key=lambda row: float(row["wer"]), reverse=True)

    total = len(samples)
    metrics = {
        "wer_tr_v1": corpus_wer(pairs, norm="tr_v1"),
        "wer_raw": corpus_wer(pairs, norm="none"),
        "cer": corpus_cer(pairs, norm="tr_v1"),
        "command_intent_accuracy": intent_ok / total if total else 0.0,
        "command_exact": command_ok / total if total else 0.0,
    }
    if durations:
        timings = latency_stats(durations)
        metrics.update({"mean_ms": timings.mean, "p95_ms": timings.p95})
    ci95 = {"wer_tr_v1": bootstrap_ci(rows)} if rows else {}
    return _build_report(
        name=name,
        kind="asr",
        git=provenance,
        corpus=corpus,
        dataset_fingerprint=dataset_fingerprint,
        dataset_version=dataset_version,
        n=total,
        metrics=metrics,
        ci95=ci95,
        breakdown={
            "speaker_disjoint": True,
            "splits_fingerprint": fingerprint,
            "speakers": len(speakers),
            "stub": stub,
            "worst": worst[:WORST_CASES],
        },
        model_version=model_version,
        failures_path=_write_worst(worst, failures_dir, name),
        synthetic=synthetic,
    )


# --------------------------------------------------------------------------------------------
# book (IP-8)
# --------------------------------------------------------------------------------------------
def run_book_eval(
    resolver: BookResolverLike,
    path: Path,
    *,
    tagging_coverage: float | None = None,
    git: GitInfo | None = None,
    dataset_version: str = "",
    failures_dir: Path | None = None,
    name: str = "book_fetch",
) -> EvalReport:
    """Textbook fetch accuracy (İP-8, top-1 ``> 95 %``) and tagging coverage (``100 %``).

    Args:
        resolver: A loaded :class:`aicls.book.BookResolver`.
        path: ``datasets/book/fetch_test.jsonl``.
        tagging_coverage: Coverage of the loaded chapters (``aicls.book.coverage``); reported
            as its own contractual column when given.
        git: Provenance of the run.
        dataset_version: Corpus version.
        failures_dir: Where to write the missed queries.
        name: Report name.

    Returns:
        The report: ``fetch_top1``, ``fetch_top3``, optional ``tagging_coverage`` and a
        per-phrasing-kind breakdown.
    """
    provenance = _provenance(git)
    from aicls.book import UnitType

    rows = [row for row in read_jsonl(Path(path)) if row.get("id") != HEADER_ID]
    top1 = top3 = 0
    flags: list[float] = []
    by_kind: dict[str, dict[str, float]] = {}
    misses: list[dict[str, Any]] = []

    for row in rows:
        unit_type = UnitType(row["unit_type"]) if row.get("unit_type") else None
        hits = list(resolver.resolve(str(row["query"]), unit_type=unit_type, k=3))
        found = [str(hit.unit.id) for hit in hits]
        expected = str(row["expect_unit"])
        hit_1 = bool(found) and found[0] == expected
        top1 += int(hit_1)
        top3 += int(expected in found[:3])
        flags.append(float(hit_1))
        bucket = by_kind.setdefault(str(row.get("kind", "")), {"n": 0.0, "top1": 0.0})
        bucket["n"] += 1.0
        bucket["top1"] += float(hit_1)
        if not hit_1:
            misses.append(
                {
                    "id": str(row.get("id", "")),
                    "query": str(row["query"]),
                    "expected": expected,
                    "got": found[0] if found else "",
                }
            )
    for bucket in by_kind.values():
        bucket["rate"] = bucket["top1"] / bucket["n"] if bucket["n"] else 0.0

    total = len(rows)
    metrics = {
        "fetch_top1": top1 / total if total else 0.0,
        "fetch_top3": top3 / total if total else 0.0,
    }
    if tagging_coverage is not None:
        metrics["tagging_coverage"] = float(tagging_coverage)
    ci95 = {"fetch_top1": bootstrap_ci(flags)} if flags else {}
    return _build_report(
        name=name,
        kind="book",
        git=provenance,
        corpus=Path(path),
        dataset_version=dataset_version,
        n=total,
        metrics=metrics,
        ci95=ci95,
        breakdown={"by_kind": by_kind, "misses": misses[:WORST_CASES]},
        failures_path=_write_worst(misses, failures_dir, name),
    )


# --------------------------------------------------------------------------------------------
# latency (IP-2 / IP-4 / IP-9)
# --------------------------------------------------------------------------------------------
def run_latency_eval(
    client: PipelineClient,
    commands: Path,
    *,
    n: int = 200,
    git: GitInfo | None = None,
    dataset_version: str = "",
    tag: str | None = None,
    name: str = "latency_local",
    stub: bool = True,
) -> EvalReport:
    """End-to-end latency over a command corpus (contract: mean ``< 3 s``).

    The contractual figure is measured **in the browser** (command sent -> KaTeX rendered,
    ADR-0014); this runner measures the server-side pipeline with stub models, which is the
    part CI can reproduce. The report therefore carries ``stub=True`` and the per-stage
    breakdown, and the browser figure replaces it once the pilot runs.

    Args:
        client: Anything with ``run(text) -> {stage: ms}`` (see
            :class:`aicls.eval.pipeline.LocalPipelineClient`).
        commands: Command corpus (``datasets/nlu/commands_test.jsonl``).
        n: How many commands to time; the corpus is cycled when it is shorter.
        git: Provenance of the run.
        dataset_version: Corpus version.
        tag: Only use rows carrying this tag (``"latency"``).
        name: Report name.
        stub: Whether stub models were used (almost always True in CI).

    Returns:
        The report: ``mean_ms``, ``p50_ms``, ``p95_ms`` and a per-stage table.
    """
    provenance = _provenance(git)
    rows = [row for row in read_jsonl(Path(commands)) if row.get("id") != HEADER_ID]
    if tag is not None:
        rows = [row for row in rows if tag in (row.get("tags") or [])]
    if not rows:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "no_latency_commands", "path": str(commands), "tag": tag},
        )
    totals: list[float] = []
    stages: dict[str, list[float]] = {}
    for index in range(max(1, n)):
        row = rows[index % len(rows)]
        timings = client.run(str(row["text"]))
        totals.append(float(timings.get("total", 0.0)))
        for stage, milliseconds in timings.items():
            stages.setdefault(stage, []).append(float(milliseconds))

    summary = latency_stats(totals)
    by_stage = {
        stage: {
            "mean_ms": latency_stats(values).mean,
            "p50_ms": latency_stats(values).p50,
            "p95_ms": latency_stats(values).p95,
        }
        for stage, values in sorted(stages.items())
    }
    return _build_report(
        name=name,
        kind="latency",
        git=provenance,
        corpus=Path(commands),
        dataset_version=dataset_version,
        n=summary.n,
        metrics={"mean_ms": summary.mean, "p50_ms": summary.p50, "p95_ms": summary.p95},
        ci95={"mean_ms": bootstrap_ci(totals)},
        breakdown={"by_stage": by_stage, "stub": stub, "commands": len(rows)},
    )


__all__ = [
    "ANSWER_FIELDS",
    "HEADER_ID",
    "LENGTH_BUCKETS",
    "WORST_CASES",
    "BookResolverLike",
    "PipelineClient",
    "Recognizer",
    "Transcriber",
    "blind_sample",
    "read_jsonl",
    "run_asr_eval",
    "run_book_eval",
    "run_cas_eval",
    "run_latency_eval",
    "run_nlu_eval",
    "run_ocr_eval",
    "run_parser_eval",
]
