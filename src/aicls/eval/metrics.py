"""Frozen metric definitions (ADR-0015, docs/ml/degerlendirme.md §2).

Every number in a TÜBİTAK progress report comes from this module, and every metric is defined
**once**: the OCR runner, the ASR runner, the CLI and the docs macros all call the same
function, so "CER" cannot quietly mean two things in two tables.

The definitions, in one place:

* **CER** — corpus level ``Σ(S+D+I) / Σ len(ref)`` over the *characters* of the ``latex_v1``
  form, spaces removed (the spaces are an artefact of the normaliser's tokenisation).
* **TER** — the same accounting over the ``latex_v1`` *tokens*.
* **ExpRate** — ``exact_match`` after normalisation.
* **WER** — corpus level over the words of the ``tr_v1`` form; every report also prints the raw
  column (``norm="none"``).
* **Symbolic equivalence** — three valued: ``equal`` / ``different`` / ``unparsable``. An answer
  the grammar cannot read is *never* counted as a wrong answer.
* **Parse success** — parse **and** ``to_sympy`` **and** round trip; a silent misparse is a
  failure.
* **Latency** — mean (the contractual wording), p50, p95, n.
* **SUS** — the standard 10-item scoring.
* **bootstrap_ci** — deterministic percentile bootstrap, 1000 resamples, seed 1234.

Rates are ratios of sums, never means of per-row ratios: a 40-character expression must weigh
more than a 3-character one, or the reported CER would depend on how the corpus was chopped up.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import random
from typing import Any, Literal, TypeVar, cast

from aicls.core import messages_tr
from aicls.core.errors import InputValidationError
from aicls.eval.normalizers import apply_norm, latex_v1, tr_words
from aicls.latex2sympy import roundtrip_check
from aicls.mathengine import symbolic_equivalent

_T = TypeVar("_T")

#: Accepted normaliser names for the character/expression metrics.
LatexNorm = Literal["latex_v1", "tr_v1", "none"]

#: Accepted normaliser names for the word metrics.
TextNorm = Literal["tr_v1", "none"]

#: Result of :func:`symbolic_equivalence` (mirrors :data:`aicls.mathengine.Equivalence`).
Equivalence = Literal["equal", "different", "unparsable"]

#: Default number of bootstrap resamples (ADR-0015).
DEFAULT_RESAMPLES = 1000

#: Repo-wide deterministic seed.
DEFAULT_SEED = 1234

#: Number of SUS items and the scale bounds.
SUS_ITEMS = 10
SUS_MIN = 1
SUS_MAX = 5

#: How many failing rows :func:`parse_success_rate` keeps by default.
DEFAULT_MAX_FAILURES = 100


@dataclass(frozen=True, slots=True)
class RateResult:
    """Outcome of a pass/fail corpus run (parse success, fetch accuracy, ...).

    Attributes:
        ok: Number of rows that passed.
        total: Number of rows examined.
        rate: ``ok / total``; ``0.0`` for an empty corpus — never ``1.0``, because no evidence
            is not perfect evidence.
        failures: ``(id, reason)`` for the failing rows, truncated to ``max_failures``.
    """

    ok: int
    total: int
    rate: float
    failures: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """Latency summary as the contract words it: mean first, then the percentiles.

    Attributes:
        mean: Arithmetic mean in milliseconds (the "< 3 s ortalama" figure).
        p50: Median in milliseconds.
        p95: 95th percentile in milliseconds.
        n: Number of samples the three values were computed from.
    """

    mean: float
    p50: float
    p95: float
    n: int


# --------------------------------------------------------------------------------------------
# edit distance
# --------------------------------------------------------------------------------------------
def edit_distance(ref: Sequence[_T], hyp: Sequence[_T]) -> int:
    """Levenshtein distance between two sequences (substitution/deletion/insertion cost 1).

    Args:
        ref: Reference sequence (characters or tokens).
        hyp: Hypothesis sequence.

    Returns:
        The minimum number of edits turning ``ref`` into ``hyp``.

    Example:
        >>> edit_distance("kitten", "sitting")
        3
    """
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    previous = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, start=1):
        current = [i]
        for j, hyp_item in enumerate(hyp, start=1):
            substitution = previous[j - 1] + (0 if ref_item == hyp_item else 1)
            current.append(min(previous[j] + 1, current[j - 1] + 1, substitution))
        previous = current
    return previous[-1]


def _rate(errors: float, reference_length: float) -> float:
    """``errors / reference_length`` with the frozen empty-reference convention.

    An empty reference has no denominator: the result is ``0.0`` when the hypothesis is empty
    too and ``1.0`` otherwise ("everything the model said is wrong"), never a ``ZeroDivisionError``
    and never a silent ``0.0`` that would flatter the model.
    """
    if reference_length <= 0:
        return 0.0 if errors == 0 else 1.0
    return errors / reference_length


# --------------------------------------------------------------------------------------------
# character / token metrics
# --------------------------------------------------------------------------------------------
def _characters(text: str, norm: str) -> str:
    """The character sequence CER compares: normalised text with every space removed."""
    return "".join(apply_norm(text, norm).split())


def _tokens(text: str, norm: str) -> list[str]:
    """The token sequence TER compares: the normaliser's own tokenisation."""
    return apply_norm(text, norm).split()


def cer(ref: str, hyp: str, *, norm: LatexNorm = "latex_v1") -> float:
    """Character error rate of one pair (İP-3 OCR metric).

    Args:
        ref: Reference string (MathWriting's ``normalized_label``, never the raw ``label``).
        hyp: Hypothesis string.
        norm: Normaliser version; ``"none"`` produces the raw column.

    Returns:
        ``(S+D+I) / len(ref)`` over non-space characters. The value is **not capped at 1**:
        insertions can push it above 1, and a report that clipped it would hide a failure mode.
    """
    reference, hypothesis = _characters(ref, norm), _characters(hyp, norm)
    return _rate(edit_distance(reference, hypothesis), len(reference))


def corpus_cer(pairs: Iterable[tuple[str, str]], *, norm: LatexNorm = "latex_v1") -> float:
    """Corpus character error rate ``Σ(S+D+I) / Σ len(ref)`` (the reported CER).

    Args:
        pairs: ``(reference, hypothesis)`` pairs.
        norm: Normaliser version.

    Returns:
        The pooled rate; ``0.0`` for an empty corpus (``n`` in the report says it was empty).
    """
    errors = length = 0
    for ref, hyp in pairs:
        reference, hypothesis = _characters(ref, norm), _characters(hyp, norm)
        errors += edit_distance(reference, hypothesis)
        length += len(reference)
    return _rate(errors, length)


def ter(ref: str, hyp: str, *, norm: LatexNorm = "latex_v1") -> float:
    """Token error rate of one pair — the same accounting as :func:`cer` over tokens."""
    reference, hypothesis = _tokens(ref, norm), _tokens(hyp, norm)
    return _rate(edit_distance(reference, hypothesis), len(reference))


def corpus_ter(pairs: Iterable[tuple[str, str]], *, norm: LatexNorm = "latex_v1") -> float:
    """Corpus token error rate ``Σ(S+D+I) / Σ tokens(ref)`` (the secondary OCR column)."""
    errors = length = 0
    for ref, hyp in pairs:
        reference, hypothesis = _tokens(ref, norm), _tokens(hyp, norm)
        errors += edit_distance(reference, hypothesis)
        length += len(reference)
    return _rate(errors, length)


def wer(ref: str, hyp: str, *, norm: TextNorm = "tr_v1") -> float:
    """Word error rate of one utterance (İP-3 ASR metric).

    Args:
        ref: Reference transcript.
        hyp: ASR hypothesis.
        norm: ``"tr_v1"`` (the reported column) or ``"none"`` (the raw column).

    Returns:
        ``(S+D+I) / words(ref)``.
    """
    reference, hypothesis = _words(ref, norm), _words(hyp, norm)
    return _rate(edit_distance(reference, hypothesis), len(reference))


def corpus_wer(pairs: Iterable[tuple[str, str]], *, norm: TextNorm = "tr_v1") -> float:
    """Corpus word error rate ``Σ(S+D+I) / Σ words(ref)``.

    This is the figure the ``< 10 %`` target refers to, measured on a **speaker-disjoint** test
    split (the split is the runner's responsibility, see
    :func:`aicls.eval.runners.run_asr_eval`).
    """
    errors = length = 0
    for ref, hyp in pairs:
        reference, hypothesis = _words(ref, norm), _words(hyp, norm)
        errors += edit_distance(reference, hypothesis)
        length += len(reference)
    return _rate(errors, length)


def _words(text: str, norm: str) -> list[str]:
    """Words of ``text`` under ``norm`` (``tr_v1`` words, or whitespace splitting for raw)."""
    if norm == "tr_v1":
        return tr_words(text)
    return apply_norm(text, norm).split()


def exact_match(ref: str, hyp: str, *, norm: LatexNorm = "latex_v1") -> bool:
    """Whether the two strings are identical after normalisation (ExpRate at row level)."""
    return apply_norm(ref, norm) == apply_norm(hyp, norm)


def exact_match_rate(pairs: Iterable[tuple[str, str]], *, norm: LatexNorm = "latex_v1") -> float:
    """ExpRate: the share of rows whose normalised hypothesis equals the reference."""
    total = matched = 0
    for ref, hyp in pairs:
        total += 1
        matched += int(exact_match(ref, hyp, norm=norm))
    return matched / total if total else 0.0


# --------------------------------------------------------------------------------------------
# symbolic metrics
# --------------------------------------------------------------------------------------------
def symbolic_equivalence(ref_latex: str, hyp_latex: str) -> Equivalence:
    r"""Tri-state symbolic comparison of two LaTeX answers (CAS accuracy, İP-4).

    Two answers that normalise to the same string are ``"equal"`` by definition; anything else
    is decided by :func:`aicls.mathengine.symbolic_equivalent`, which proves equality by
    simplification or by deterministic numeric probes.

    The identity shortcut is not an optimisation, it is a correctness rule. ``simplify(a - a)``
    is ``nan`` when ``a`` is infinite, so the CAS comparison answers ``"different"`` for two
    *identical* answers such as ``\lim_{x\to0^{+}}\ln x=-\infty`` — a defect that would score
    a perfect recognition as wrong (the same trap ``datasets/curriculum`` documents in its
    ``compare()``). A metric must never call two identical strings different.

    Args:
        ref_latex: Reference answer.
        hyp_latex: Answer to judge.

    Returns:
        ``"equal"``, ``"different"``, or ``"unparsable"`` when the two differ and at least one
        side is outside the grammar (or no probe could be evaluated). ``"unparsable"`` is
        reported as its own column and never folded into ``"different"``: coverage and accuracy
        are different claims.
    """
    if ref_latex == hyp_latex or latex_v1(ref_latex) == latex_v1(hyp_latex):
        return "equal"
    return symbolic_equivalent(ref_latex, hyp_latex)


def parse_success_rate(
    inputs: Iterable[str | tuple[str, str]],
    *,
    numeric_samples: int = 3,
    max_failures: int = DEFAULT_MAX_FAILURES,
) -> RateResult:
    """Strict parse success over a corpus of valid LaTeX inputs (İP-4, ``> 99 %``).

    A row counts as a success only when it parses, converts to SymPy **and** survives the round
    trip (``parse -> SymPy -> print -> parse``). A silent misparse — the platform printing an
    answer it cannot itself read back — is therefore a failure, not a success.

    Args:
        inputs: LaTeX strings, or ``(id, latex)`` pairs so failures can be named. A bare string
            gets its 0-based index as id.
        numeric_samples: Points the round trip's weakest level may sample.
        max_failures: How many failing rows to keep in :attr:`RateResult.failures`.

    Returns:
        The :class:`RateResult`.
    """
    ok = total = 0
    failures: list[tuple[str, str]] = []
    for index, item in enumerate(inputs):
        row_id, latex = item if isinstance(item, tuple) else (str(index), item)
        total += 1
        result = roundtrip_check(latex, numeric_samples=numeric_samples)
        if result.ok:
            ok += 1
        elif len(failures) < max_failures:
            failures.append((row_id, result.reason))
    return RateResult(
        ok=ok, total=total, rate=(ok / total if total else 0.0), failures=tuple(failures)
    )


# --------------------------------------------------------------------------------------------
# latency and questionnaire
# --------------------------------------------------------------------------------------------
def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of ``values`` (the frozen definition of p50/p95).

    Args:
        values: Samples (need not be sorted; must be non-empty).
        q: Quantile in ``[0, 1]``.

    Returns:
        The interpolated value at rank ``(n-1) * q``, the same convention as
        ``numpy.percentile``'s default.

    Raises:
        InputValidationError: If ``values`` is empty or ``q`` is outside ``[0, 1]``.
    """
    if not values:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR, details={"reason": "empty_samples"}
        )
    if not 0.0 <= q <= 1.0:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR, details={"reason": "quantile_out_of_range", "q": q}
        )
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def latency_stats(samples: Sequence[float]) -> LatencyStats:
    """Summarise measured latencies in milliseconds.

    Args:
        samples: Per-request durations in milliseconds.

    Returns:
        Mean, p50, p95 and n.

    Raises:
        InputValidationError: If ``samples`` is empty (a latency figure without samples is not
            a number) or holds a negative duration.
    """
    if not samples:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR, details={"reason": "empty_samples"}
        )
    values = [float(sample) for sample in samples]
    negative = [value for value in values if value < 0.0]
    if negative:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "negative_duration", "value": negative[0]},
        )
    return LatencyStats(
        mean=sum(values) / len(values),
        p50=percentile(values, 0.5),
        p95=percentile(values, 0.95),
        n=len(values),
    )


def sus_score(answers: Sequence[int]) -> float:
    """System Usability Scale score of one questionnaire (İP-2/İP-9, target ``> 75``).

    Args:
        answers: Ten answers on the 1..5 Likert scale, in item order.

    Returns:
        The score in ``[0, 100]``: odd items contribute ``x-1``, even items ``5-x``, and the sum
        is multiplied by 2.5.

    Raises:
        InputValidationError: If there are not exactly ten answers or one is outside 1..5.
    """
    values = list(answers)
    if len(values) != SUS_ITEMS:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "sus_item_count", "n": len(values), "expected": SUS_ITEMS},
        )
    for value in values:
        if not isinstance(value, int) or not SUS_MIN <= value <= SUS_MAX:
            raise InputValidationError(
                messages_tr.MSG_VALIDATION_ERROR,
                details={"reason": "sus_value_out_of_range", "value": value},
            )
    contributions = [
        (value - SUS_MIN) if index % 2 == 0 else (SUS_MAX - value)
        for index, value in enumerate(values)
    ]
    return float(sum(contributions)) * 2.5


# --------------------------------------------------------------------------------------------
# confidence intervals
# --------------------------------------------------------------------------------------------
def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _ratio_of_sums(rows: Sequence[tuple[float, float]]) -> float:
    numerator = sum(row[0] for row in rows)
    denominator = sum(row[1] for row in rows)
    return _rate(numerator, denominator)


def bootstrap_ci(
    values: Sequence[float] | Sequence[tuple[float, float]],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    statistic: Callable[[Sequence[Any]], float] | None = None,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval (ADR-0015: 1000 resamples, seed 1234).

    Args:
        values: Either per-row values (the statistic defaults to the mean) or ``(numerator,
            denominator)`` rows for a pooled rate such as CER, where the statistic defaults to
            ``Σnum / Σden`` — resampling *rows*, which is what makes the interval of a corpus
            rate honest.
        n_resamples: Number of bootstrap resamples (``> 0``).
        seed: Seed of the resampler; the same data and seed always give the same interval.
        statistic: Override for the statistic to bootstrap.
        alpha: ``0.05`` gives the reported 95 % interval.

    Returns:
        ``(low, high)``.

    Raises:
        InputValidationError: If ``values`` is empty, ``n_resamples`` is not positive or
            ``alpha`` is outside ``(0, 1)``.
    """
    rows: list[Any] = list(values)
    if not rows:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR, details={"reason": "empty_sample"}
        )
    if n_resamples <= 0:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "n_resamples_not_positive", "n_resamples": n_resamples},
        )
    if not 0.0 < alpha < 1.0:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR, details={"reason": "alpha_out_of_range"}
        )
    estimator = statistic if statistic is not None else _default_statistic(rows)
    rng = random.Random(seed)
    size = len(rows)
    estimates: list[float] = []
    for _ in range(n_resamples):
        sample = [rows[rng.randrange(size)] for _ in range(size)]
        estimates.append(float(estimator(sample)))
    return percentile(estimates, alpha / 2.0), percentile(estimates, 1.0 - alpha / 2.0)


def _default_statistic(rows: Sequence[Any]) -> Callable[[Sequence[Any]], float]:
    """Mean for scalar rows, pooled ratio for ``(numerator, denominator)`` rows."""
    if isinstance(rows[0], tuple):
        return cast("Callable[[Sequence[Any]], float]", _ratio_of_sums)
    return cast("Callable[[Sequence[Any]], float]", _mean)


__all__ = [
    "DEFAULT_MAX_FAILURES",
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "SUS_ITEMS",
    "Equivalence",
    "LatencyStats",
    "LatexNorm",
    "RateResult",
    "TextNorm",
    "bootstrap_ci",
    "cer",
    "corpus_cer",
    "corpus_ter",
    "corpus_wer",
    "edit_distance",
    "exact_match",
    "exact_match_rate",
    "latency_stats",
    "parse_success_rate",
    "percentile",
    "sus_score",
    "symbolic_equivalence",
    "ter",
    "wer",
]
