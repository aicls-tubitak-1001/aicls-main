"""Property tests for the frozen normalisers and the rate metrics (ADR-0015).

Idempotence is the property that lets a report say "computed with ``latex_v1``" without also
saying how many times it ran; the metric properties (identity, symmetry of the denominator,
bounds) are what make a reported rate readable at all.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
import pytest

from aicls.eval.metrics import cer, corpus_cer, corpus_wer, edit_distance, wer
from aicls.eval.normalizers import latex_v1, tr_v1, tr_words, turkish_number_words

pytestmark = pytest.mark.property

#: Characters a LaTeX hypothesis is built from (commands, braces, scripts, digits, letters).
_LATEX_PIECES = st.sampled_from(
    [
        "x",
        "y",
        "1",
        "23",
        "+",
        "-",
        "^",
        "_",
        "{",
        "}",
        "(",
        ")",
        r"\frac",
        r"\sin",
        r"\left",
        r"\right",
        r"\alpha",
        r"\,",
        r"\dfrac",
        " ",
    ]
)

_LATEX = st.lists(_LATEX_PIECES, min_size=0, max_size=24).map("".join)

#: Turkish-ish text: words, digits, punctuation, mixed case.
_TR_PIECES = st.sampled_from(
    [
        "türev",
        "İntegral",
        "SINIR",
        "x",
        "0",
        "12",
        "2026",
        "'",
        ",",
        ".",
        "-",
        " ",
        "sıfır",
        "bir",
        "ıslak",
    ]
)

_TR = st.lists(_TR_PIECES, min_size=0, max_size=24).map("".join)


@settings(max_examples=200, deadline=None)
@given(_LATEX)
def test_latex_v1_is_idempotent(text: str) -> None:
    """``latex_v1(latex_v1(s)) == latex_v1(s)`` for any input."""
    once = latex_v1(text)
    assert latex_v1(once) == once


@settings(max_examples=200, deadline=None)
@given(_TR)
def test_tr_v1_is_idempotent(text: str) -> None:
    """``tr_v1(tr_v1(s)) == tr_v1(s)`` for any input."""
    once = tr_v1(text)
    assert tr_v1(once) == once


@settings(max_examples=100, deadline=None)
@given(_TR)
def test_tr_v1_output_carries_no_digits_and_no_punctuation(text: str) -> None:
    """Rules 3 and 4 are total: nothing digit-like or punctuation-like survives."""
    normalised = tr_v1(text)
    assert not any(character.isdigit() for character in normalised)
    assert "'" not in normalised and "," not in normalised and "-" not in normalised


@settings(max_examples=100, deadline=None)
@given(_TR)
def test_tr_v1_never_produces_empty_words_or_double_spaces(text: str) -> None:
    """Rule 5: the output is a single-space join of non-empty words."""
    normalised = tr_v1(text)
    assert "  " not in normalised
    assert normalised == normalised.strip()
    assert all(word for word in normalised.split(" ") if normalised)


@settings(max_examples=100, deadline=None)
@given(st.integers(min_value=0, max_value=10**12 - 1))
def test_turkish_number_words_are_pure_words(value: int) -> None:
    """The digit rule may only ever emit words, or it would not be idempotent."""
    words = turkish_number_words(value)
    assert words
    assert all(word.isalpha() for word in words)


@settings(max_examples=100, deadline=None)
@given(_LATEX)
def test_cer_of_a_string_with_itself_is_zero(text: str) -> None:
    """A metric that scored an identical hypothesis above zero would be unusable."""
    assert cer(text, text) == 0.0
    assert wer(text, text) == 0.0


@settings(max_examples=100, deadline=None)
@given(_LATEX, _LATEX)
def test_cer_is_non_negative_and_zero_only_for_equal_normalised_forms(ref: str, hyp: str) -> None:
    """CER >= 0, and CER == 0 exactly when the normalised forms agree."""
    value = cer(ref, hyp)
    assert value >= 0.0
    same = "".join(latex_v1(ref).split()) == "".join(latex_v1(hyp).split())
    assert (value == 0.0) is same


@settings(max_examples=100, deadline=None)
@given(_TR, _TR)
def test_edit_distance_is_symmetric_and_bounded(ref: str, hyp: str) -> None:
    """Levenshtein is a metric: symmetric, and never longer than the longer sequence."""
    left, right = tr_words(ref), tr_words(hyp)
    distance = edit_distance(left, right)
    assert distance == edit_distance(right, left)
    assert distance <= max(len(left), len(right))


@settings(max_examples=50, deadline=None)
@given(st.lists(st.tuples(_TR, _TR), min_size=1, max_size=6))
def test_corpus_rate_is_between_the_extremes_of_its_rows(pairs: list[tuple[str, str]]) -> None:
    """A pooled rate can never be below every row's rate nor above all of them."""
    rates = [wer(ref, hyp) for ref, hyp in pairs if tr_words(ref)]
    if not rates:
        return
    pooled = corpus_wer([(ref, hyp) for ref, hyp in pairs if tr_words(ref)])
    assert min(rates) - 1e-9 <= pooled <= max(rates) + 1e-9


@settings(max_examples=50, deadline=None)
@given(st.lists(st.tuples(_LATEX, _LATEX), min_size=1, max_size=6))
def test_corpus_cer_of_identical_pairs_is_zero(pairs: list[tuple[str, str]]) -> None:
    """Pooling identical rows cannot invent an error."""
    assert corpus_cer([(ref, ref) for ref, _ in pairs]) == 0.0
