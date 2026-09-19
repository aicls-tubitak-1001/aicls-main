"""Frozen metric definitions (ADR-0015, docs/ml/degerlendirme.md §2).

The reference values below are **hand-computed from the formula**, not read off the
implementation: each one names the substitutions, deletions and insertions it counts and the
denominator it divides by. If the implementation and one of these numbers disagree, the
implementation is wrong — that is the whole point of freezing a metric.
"""

from __future__ import annotations

import math

import pytest

from aicls.core.errors import InputValidationError
from aicls.eval.metrics import (
    LatencyStats,
    RateResult,
    bootstrap_ci,
    cer,
    corpus_cer,
    corpus_ter,
    corpus_wer,
    edit_distance,
    exact_match,
    exact_match_rate,
    latency_stats,
    parse_success_rate,
    percentile,
    sus_score,
    symbolic_equivalence,
    ter,
    wer,
)


# ------------------------------------------------------------------------------ edit distance
def test_edit_distance_reference_values() -> None:
    """Levenshtein on sequences: the primitive every rate metric divides."""
    assert edit_distance("kitten", "sitting") == 3  # k->s, e->i, +g
    assert edit_distance("", "abc") == 3  # three insertions
    assert edit_distance("abc", "") == 3  # three deletions
    assert edit_distance("abc", "abc") == 0
    assert edit_distance(["a", "bb"], ["a", "cc"]) == 1  # tokens, not characters


# ---------------------------------------------------------------------------------------- CER
def test_cer_reference_values_hand_computed() -> None:
    """Three CER values computed by hand from ``Σ(S+D+I) / Σ len(ref)``.

    1. ``x^{2}`` vs ``x^{3}``: ``latex_v1`` gives ``x ^ { 2 }`` / ``x ^ { 3 }``; CER counts
       characters with the spaces removed, so ``x^{2}`` vs ``x^{3}`` — 5 reference characters,
       one substitution (``2`` -> ``3``) => 1/5 = 0.2.
    2. ``\\alpha + 1`` vs ``\\alpha + 12``: normalised ``\\alpha + 1`` / ``\\alpha + 12``,
       spaces removed => ``\\alpha+1`` (8 characters) and ``\\alpha+12``; one insertion
       => 1/8 = 0.125.
    3. ``abc`` vs ``ac``: three single-character tokens against two; one deletion => 1/3.
    """
    assert cer("x^{2}", "x^{3}") == pytest.approx(0.2)
    assert cer(r"\alpha + 1", r"\alpha + 12") == pytest.approx(1 / 8)
    assert cer("abc", "ac") == pytest.approx(1 / 3)


def test_cer_identical_zero_and_bounds() -> None:
    """CER is 0 for identical strings, never negative, and 1.0 when everything is wrong."""
    assert cer("x^{2}+1", "x^{2}+1") == 0.0
    assert cer("", "") == 0.0
    assert cer("abc", "") == 1.0  # three deletions over three reference characters
    # an empty reference with a non-empty hypothesis has no denominator: the frozen convention
    # is 1.0 (fully wrong), never a division by zero.
    assert cer("", "abc") == 1.0


def test_cer_can_exceed_one_when_the_hypothesis_is_much_longer() -> None:
    """Insertions are counted, so CER is not capped at 1 — reports must not clip it silently."""
    assert cer("a", "abcdef") == pytest.approx(5.0)


def test_cer_uses_normalized_label_not_raw() -> None:
    """The metric compares ``latex_v1`` forms, so surface spelling never counts as an error.

    ``\\left(x\\right)`` and ``(x)`` are the same expression written two ways; a CER that
    scored them as different would measure LaTeX style, not recognition quality (ADR-0015:
    MathWriting's ``normalized_label`` is the reference, never the raw ``label``).
    """
    assert cer(r"\left(x\right)", "(x)") == 0.0
    assert cer(r"\dfrac{1}{2}", r"\frac{1}{2}") == 0.0
    assert cer("x^2", "x^{2}") == 0.0
    # ... while the raw column does see the difference
    assert cer(r"\left(x\right)", "(x)", norm="none") > 0.0


def test_corpus_cer_is_a_ratio_of_sums_not_a_mean_of_ratios() -> None:
    """Corpus CER pools errors and reference lengths (ADR-0015), so long rows weigh more.

    Pairs ``("abc", "abd")`` (1 error / 3) and ``("z", "")`` (1 error / 1): the ratio of sums is
    2/4 = 0.5, while the mean of the two row ratios would be (1/3 + 1)/2 = 0.667.
    """
    pairs = [("abc", "abd"), ("z", "")]
    assert corpus_cer(pairs) == pytest.approx(0.5)
    assert corpus_cer([]) == 0.0


# ---------------------------------------------------------------------------------------- TER
def test_ter_vs_cer_differ_on_brace_case() -> None:
    """TER counts tokens, CER counts characters — the two must not be interchangeable.

    ``x^{12}`` vs ``x^{13}``: the tokens are ``x ^ { 12 }`` and ``x ^ { 13 }``, one substitution
    out of five tokens => TER 0.2, while the characters ``x^{12}`` / ``x^{13}`` give one
    substitution out of six => CER 1/6.
    """
    assert ter("x^{12}", "x^{13}") == pytest.approx(0.2)
    assert cer("x^{12}", "x^{13}") == pytest.approx(1 / 6)

    # the brace case: a redundant group is normalised away for both metrics ...
    assert ter("{x}", "x") == 0.0
    # ... but a *script* brace is kept, so a missing exponent brace is not free in TER
    assert ter("x^{12}", "x^{1}2") == pytest.approx(0.4)


def test_corpus_ter_pools_tokens() -> None:
    """Corpus TER is Σ token errors / Σ reference tokens."""
    pairs = [("x^{12}", "x^{13}"), ("a", "a")]
    assert corpus_ter(pairs) == pytest.approx(1 / 6)


# ---------------------------------------------------------------------------------------- WER
def test_wer_reference_values_hand_computed() -> None:
    """WER on ``tr_v1`` words: one substitution, one deletion, one insertion."""
    # "x karenin türevini al" vs "x karenin türevini bul": 1 substitution / 4 words
    assert wer("x karenin türevini al", "x karenin türevini bul") == pytest.approx(0.25)
    # a deletion: 4 reference words, the hypothesis drops one
    assert wer("x karenin türevini al", "x karenin türevini") == pytest.approx(0.25)
    # an insertion: the hypothesis adds a word
    assert wer("x karenin türevini al", "x karenin türevini hemen al") == pytest.approx(0.25)


def test_wer_raw_and_tr_v1_both_reported() -> None:
    """Casing, punctuation and digits change the raw WER but not the ``tr_v1`` WER.

    Both columns appear in every ASR report (ADR-0015): the raw one is comparable with the
    literature's "no normalisation" figure, the ``tr_v1`` one with our own target.
    """
    reference = "0'dan 1'e kadar integral"
    hypothesis = "0'dan 1'e kadar İNTEGRAL,"
    assert wer(reference, hypothesis, norm="none") == pytest.approx(0.25)
    assert wer(reference, hypothesis, norm="tr_v1") == 0.0
    # the tr_v1 column counts the digit-expanded words: "sıfır dan bir e kadar integral"
    assert corpus_wer([(reference, hypothesis)], norm="tr_v1") == 0.0
    assert corpus_wer([(reference, hypothesis)], norm="none") == pytest.approx(0.25)


def test_wer_corpus_matches_jiwer_reference() -> None:
    """Cross-check the frozen implementation against ``jiwer`` (ADR-0015: reference only).

    ``jiwer`` is a dev dependency and is never used to produce a reported number; it is used
    here so that a bug in our pooled Levenshtein cannot survive unnoticed.
    """
    jiwer = pytest.importorskip("jiwer")
    references = [
        "x karenin türevini al",
        "sinüs x in integralini hesapla",
        "limit x sıfıra giderken sinüs x bölü x",
        "üç x artı beş eşittir on denklemini çöz",
        "",
    ]
    hypotheses = [
        "x karenin türevini bul",
        "sinüs x in integralini hesapla",
        "limit x sıfıra giderken sinüs bölü x",
        "üç x artı beş eşittir on denklemi çöz sen",
        "",
    ]
    pairs = list(zip(references, hypotheses, strict=True))
    non_empty = [(r, h) for r, h in pairs if r]
    expected = jiwer.wer([r for r, _ in non_empty], [h for _, h in non_empty])
    assert corpus_wer(non_empty, norm="none") == pytest.approx(expected)


def test_corpus_wer_empty_corpus_is_zero() -> None:
    """An empty corpus has no errors; the report's ``n`` is what tells the reader it is empty."""
    assert corpus_wer([]) == 0.0


# -------------------------------------------------------------------------- exact match / CAS
def test_exact_match_is_normalized_equality() -> None:
    """ExpRate is expression-level exact match *after* normalisation."""
    assert exact_match("x^2", "x^{2}") is True
    assert exact_match("x^2", "x^{3}") is False
    assert exact_match_rate([("x^2", "x^{2}"), ("y", "z")]) == pytest.approx(0.5)
    assert exact_match_rate([]) == 0.0


def test_symbolic_equivalence_accepts_factored_vs_expanded() -> None:
    """A factored answer and its expansion are the same answer (CAS accuracy, İP-4)."""
    assert symbolic_equivalence("(x+1)^{2}", "x^{2}+2x+1") == "equal"
    assert symbolic_equivalence(r"\frac{1}{2}", "0.5") == "equal"
    assert symbolic_equivalence("2x", "x+x") == "equal"
    assert symbolic_equivalence("x^{2}", "x^{3}") == "different"


def test_symbolic_equivalence_returns_unparsable_not_different() -> None:
    """An answer outside the grammar abstains; it is never scored as a wrong answer.

    ``\\binom{n}{k}`` is documented as out of scope for the v1 grammar. Counting it as
    "different" would turn a *coverage* gap into an *accuracy* claim, which is exactly the
    conflation ADR-0015 forbids.
    """
    assert symbolic_equivalence(r"\binom{n}{k}", "x") == "unparsable"
    assert symbolic_equivalence("x", r"\binom{n}{k}") == "unparsable"
    assert symbolic_equivalence("", "x") == "unparsable"


# ------------------------------------------------------------------------- parse success rate
def test_parse_success_rate_counts_silent_misparse_as_failure() -> None:
    """Strict definition: parse + to_sympy + round trip, all three (ADR-0015).

    ``\\frac{1}{0}`` is the natural example on the committed grammar: it parses, it converts to
    SymPy (``zoo``) — and the printed answer ``\\tilde{\\infty}`` cannot be read back, so the
    platform would show the user something it cannot itself re-read. A metric that only asked
    "did it parse?" would score this row as a success.
    """
    result = parse_success_rate(["x^{2}", r"\frac{1}{0}"])
    assert isinstance(result, RateResult)
    assert (result.ok, result.total) == (1, 2)
    assert result.rate == pytest.approx(0.5)
    failed_ids = [failure_id for failure_id, _ in result.failures]
    assert failed_ids == ["1"]
    assert "reparse_failed" in result.failures[0][1]


def test_parse_success_rate_counts_a_parse_error_as_failure_with_its_id() -> None:
    """Ids come from ``(id, latex)`` pairs so a corpus run can name the failing rows."""
    result = parse_success_rate([("ok-1", "x + 1"), ("bad-1", r"\binom{n}{k}")])
    assert (result.ok, result.total, result.rate) == (1, 2, 0.5)
    assert result.failures[0][0] == "bad-1"
    assert "parse_failed" in result.failures[0][1]


def test_parse_success_rate_of_an_empty_corpus_is_zero_not_one() -> None:
    """No inputs means no evidence; a rate of 1.0 would be a false claim."""
    result = parse_success_rate([])
    assert (result.ok, result.total, result.rate) == (0, 0, 0.0)


def test_parse_success_rate_keeps_at_most_max_failures() -> None:
    """The failure list is bounded so a broken corpus cannot blow up a report file."""
    result = parse_success_rate([r"\binom{n}{k}"] * 10, max_failures=3)
    assert result.ok == 0
    assert len(result.failures) == 3


# ------------------------------------------------------------------------------------ latency
def test_latency_stats_mean_p50_p95() -> None:
    """Mean, median and p95 with linear interpolation on the sorted sample (frozen).

    For ``1..10`` ms: mean = 5.5; p50 sits at index (10-1)*0.5 = 4.5 => (5+6)/2 = 5.5;
    p95 sits at index (10-1)*0.95 = 8.55 => 9 + 0.55*(10-9) = 9.55.
    """
    stats = latency_stats([float(i) for i in range(1, 11)])
    assert isinstance(stats, LatencyStats)
    assert stats.n == 10
    assert stats.mean == pytest.approx(5.5)
    assert stats.p50 == pytest.approx(5.5)
    assert stats.p95 == pytest.approx(9.55)


def test_latency_stats_single_sample_and_unsorted_input() -> None:
    """One sample is its own mean/p50/p95; the input order never matters."""
    assert latency_stats([42.0]) == LatencyStats(mean=42.0, p50=42.0, p95=42.0, n=1)
    unsorted_stats = latency_stats([10.0, 1.0, 5.0])
    assert unsorted_stats.p50 == pytest.approx(5.0)


def test_latency_stats_refuses_an_empty_or_negative_sample() -> None:
    """A latency figure without samples is not a number, and time is never negative."""
    with pytest.raises(InputValidationError):
        latency_stats([])
    with pytest.raises(InputValidationError):
        latency_stats([1.0, -2.0])


def test_percentile_reference_values() -> None:
    """The percentile helper is the single definition both p50 and p95 use."""
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)


# ---------------------------------------------------------------------------------------- SUS
def test_sus_score_reference_values() -> None:
    """Standard SUS scoring: odd items ``x-1``, even items ``5-x``, sum × 2.5.

    * every item 1  -> odd 0×5 + even 4×5 = 20 -> 50
    * every item 3  -> 2×10 = 20               -> 50 (the neutral answer is the midpoint)
    * odd 5 / even 1 -> 4×10 = 40              -> 100 (best possible)
    * odd 1 / even 5 -> 0                      -> 0   (worst possible)
    """
    assert sus_score([1] * 10) == pytest.approx(50.0)
    assert sus_score([3] * 10) == pytest.approx(50.0)
    assert sus_score([5, 1] * 5) == pytest.approx(100.0)
    assert sus_score([1, 5] * 5) == pytest.approx(0.0)
    # a realistic questionnaire: 4,2,4,2,4,2,4,2,4,2 -> odd 3×5 + even 3×5 = 30 -> 75
    assert sus_score([4, 2] * 5) == pytest.approx(75.0)


def test_sus_score_validates_its_input() -> None:
    """Ten items, each 1..5 — anything else is a data error, not a 0 score."""
    with pytest.raises(InputValidationError):
        sus_score([1] * 9)
    with pytest.raises(InputValidationError):
        sus_score([0] + [3] * 9)
    with pytest.raises(InputValidationError):
        sus_score([6] + [3] * 9)


# --------------------------------------------------------------------------------- bootstrap
def test_bootstrap_ci_contains_point_estimate() -> None:
    """The 95 % interval brackets the statistic it was computed from."""
    values = [float(i) for i in range(1, 51)]
    point = sum(values) / len(values)
    low, high = bootstrap_ci(values)
    assert low <= point <= high
    assert low < high


def test_bootstrap_ci_is_deterministic_for_a_seed() -> None:
    """Same data, same seed, same interval — a report must be reproducible."""
    values = [0.0, 1.0] * 20
    assert bootstrap_ci(values, seed=1234) == bootstrap_ci(values, seed=1234)
    assert bootstrap_ci(values, seed=1234) != bootstrap_ci(values, seed=7)


def test_bootstrap_ci_of_a_constant_sample_is_a_point() -> None:
    """No variation in, no width out."""
    assert bootstrap_ci([0.5] * 20) == (0.5, 0.5)


def test_bootstrap_ci_supports_ratio_of_sums_pairs() -> None:
    """Corpus rates resample ``(errors, length)`` rows, not per-row ratios (ADR-0015)."""
    rows = [(1.0, 10.0)] * 30 + [(0.0, 10.0)] * 70
    low, high = bootstrap_ci(rows)
    assert low <= 0.03 <= high
    assert 0.0 <= low < high <= 1.0


def test_bootstrap_ci_refuses_an_empty_sample() -> None:
    """An interval over nothing is meaningless."""
    with pytest.raises(InputValidationError):
        bootstrap_ci([])
    with pytest.raises(InputValidationError):
        bootstrap_ci([1.0], n_resamples=0)


def test_bootstrap_ci_bounds_are_finite() -> None:
    """Guard against NaN leaking into a report file."""
    low, high = bootstrap_ci([0.0, 1.0, 0.0, 1.0])
    assert math.isfinite(low) and math.isfinite(high)


# --------------------------------------------------------------------- input-validation edges
def test_percentile_validates_its_arguments() -> None:
    """No samples and an out-of-range quantile are data errors, never a made-up number."""
    with pytest.raises(InputValidationError):
        percentile([], 0.5)
    with pytest.raises(InputValidationError):
        percentile([1.0], 1.5)
    with pytest.raises(InputValidationError):
        percentile([1.0], -0.1)


def test_bootstrap_ci_validates_alpha() -> None:
    """``alpha`` outside ``(0, 1)`` does not describe a confidence level."""
    with pytest.raises(InputValidationError):
        bootstrap_ci([1.0, 2.0], alpha=1.5)
    with pytest.raises(InputValidationError):
        bootstrap_ci([1.0, 2.0], alpha=0.0)


def test_bootstrap_ci_accepts_a_custom_statistic() -> None:
    """A metric that is not a mean (a median, a p95) can be bootstrapped too."""
    values = [float(i) for i in range(1, 21)]
    low, high = bootstrap_ci(values, statistic=lambda sample: percentile(list(sample), 0.5))
    assert low <= percentile(values, 0.5) <= high
