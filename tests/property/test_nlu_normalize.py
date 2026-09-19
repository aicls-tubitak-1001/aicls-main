"""Property tests for the Turkish normaliser and the spoken-math grammar.

Two invariants hold for *every* input, which is what makes the layer safe to put behind a
microphone: the normaliser is idempotent on its own rendering, and nothing that leaves the
package is anything but an :class:`aicls.core.errors.AiclsError`.
"""

from __future__ import annotations

import contextlib
import string

from hypothesis import given, settings
from hypothesis import strategies as st
import pytest

from aicls.core.errors import AiclsError
from aicls.core.turkish import tr_casefold
from aicls.nlu import normalize_text, parse_command, spoken_to_latex
from aicls.nlu.errors import SpokenMathError

pytestmark = pytest.mark.property

#: The alphabet real input is drawn from: Turkish letters, digits, operators and separators.
TURKISH_LETTERS = "abcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ"
ALPHABET = TURKISH_LETTERS + string.digits + " '’.,+-*/=^()"

#: Words the layer's own vocabulary is made of; mixing them with noise exercises real paths.
COMMAND_WORDS = (
    "türev",
    "türevini",
    "integral",
    "integralini",
    "limit",
    "sıfırdan",
    "bire",
    "kadar",
    "x",
    "kare",
    "karenin",
    "artı",
    "eksi",
    "iki",
    "sinüs",
    "al",
    "hesapla",
    "tanımını",
    "getir",
    "temizle",
    "grafiğini",
    "çiz",
)

text_strategy = st.text(alphabet=ALPHABET, min_size=0, max_size=60)
word_strategy = st.lists(st.sampled_from(COMMAND_WORDS), min_size=0, max_size=8).map(" ".join)
utterances = st.one_of(text_strategy, word_strategy)


class TestNormalizerProperties:
    @settings(max_examples=200)
    @given(utterances)
    def test_property_normalizer_idempotent(self, text: str) -> None:
        """``normalize_text(n.text).text == n.text`` for every input."""
        once = normalize_text(text)
        twice = normalize_text(once.text)
        assert twice.text == once.text

    @settings(max_examples=200)
    @given(utterances)
    def test_property_normalizer_never_raises(self, text: str) -> None:
        normalize_text(text)

    @settings(max_examples=200)
    @given(utterances)
    def test_property_tokens_are_casefolded_and_span_bounded(self, text: str) -> None:
        analysed = normalize_text(text)
        assert len(analysed.tokens) == len(analysed.spans) == len(analysed.kinds)
        for token, (start, end) in zip(analysed.tokens, analysed.spans, strict=True):
            assert token == tr_casefold(token) or token.isdigit() or not token.isalpha()
            assert 0 <= start < end <= len(analysed.original)

    @settings(max_examples=100)
    @given(utterances)
    def test_property_token_range_is_within_bounds(self, text: str) -> None:
        analysed = normalize_text(text)
        first, last = analysed.token_range(0, len(analysed.text))
        assert 0 <= first <= last <= len(analysed.tokens)


class TestParseCommandProperties:
    @settings(max_examples=150, deadline=None)
    @given(utterances)
    def test_property_parse_command_never_raises_non_aicls_error(self, text: str) -> None:
        """Only ``AiclsError`` may leave the package; a pydantic or lark error must not."""
        with contextlib.suppress(AiclsError):
            parse_command(text)

    @settings(max_examples=100, deadline=None)
    @given(word_strategy)
    def test_property_outcome_is_always_one_of_three(self, text: str) -> None:
        from aicls.nlu import Clarification, Matched, NoMatch

        try:
            outcome = parse_command(text)
        except AiclsError:
            return
        assert isinstance(outcome, Matched | Clarification | NoMatch)
        assert isinstance(outcome.trace, list)


class TestSpokenMathProperties:
    @settings(max_examples=150, deadline=None)
    @given(utterances)
    def test_property_spoken_to_latex_only_raises_spoken_math_error(self, text: str) -> None:
        with contextlib.suppress(SpokenMathError):
            spoken_to_latex(text)

    @settings(max_examples=100, deadline=None)
    @given(
        st.lists(
            st.sampled_from(["x", "y", "kare", "küp", "artı", "eksi", "çarpı", "iki", "sinüs"]),
            min_size=1,
            max_size=6,
        ).map(" ".join)
    )
    def test_property_output_is_reparsable(self, text: str) -> None:
        """Whatever the grammar accepts, ``parse_latex`` accepts the LaTeX it produces."""
        from aicls.latex2sympy import parse_latex

        try:
            latex = spoken_to_latex(text)
        except SpokenMathError:
            return
        parse_latex(latex)
