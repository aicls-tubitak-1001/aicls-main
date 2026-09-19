"""Word-boundary recovery (KNOWN_ISSUES PRIORITY 2).

A Whisper-class decoder loses word boundaries constantly on fast Turkish dictation:
``"türev tanımını"`` comes back as ``"türevtanımını"``. Before this pass the layer scored
``0.276`` on the ``join_words`` perturbation while every other perturbation was ``>= 0.99``
(docs/mimari/metin-komut-katmani.md §9.4), which made word-boundary loss the layer's single
report-relevant weakness.

:meth:`aicls.nlu.lexicon.Lexicon.resegment` is a longest-match pass over the lexicon vocabulary,
the numeral words and the suffix inventory. It only ever fires on a token that is **neither a
known word nor a known stem plus one inflectional suffix**, so it cannot re-cut a word the
lexicon already understands, and every recovered piece must itself be recognisable -- otherwise
the token is left exactly as the decoder wrote it.
"""

from __future__ import annotations

import pytest

from aicls.mathengine.models import Diff, Integrate
from aicls.nlu import Matched, load_lexicon, normalize_text, parse_command
from aicls.nlu.lexicon import MAX_RESEGMENT_PIECES, Lexicon


@pytest.fixture(scope="module")
def lexicon() -> Lexicon:
    return load_lexicon()


class TestResegment:
    @pytest.mark.parametrize(
        ("joined", "pieces"),
        [
            ("türevtanımını", ("türev", "tanımını")),
            ("xkarenin", ("x", "karenin")),
            ("birbölü", ("bir", "bölü")),
            ("integralinial", ("integralini", "al")),
            ("sıfırdanbire", ("sıfırdan", "bire")),
            ("tahtayıtemizle", ("tahtayı", "temizle")),
            ("grafiğiniçiz", ("grafiğini", "çiz")),
        ],
    )
    def test_recovers_a_lost_boundary(
        self, lexicon: Lexicon, joined: str, pieces: tuple[str, ...]
    ) -> None:
        assert lexicon.resegment(joined) == pieces

    @pytest.mark.parametrize(
        "token",
        ["türev", "türevini", "kadar", "sıfırdan", "x", "karekök", "sadeleştir"],
    )
    def test_never_re_cuts_a_word_the_lexicon_knows(self, lexicon: Lexicon, token: str) -> None:
        assert lexicon.resegment(token) is None

    @pytest.mark.parametrize(
        "token",
        ["bilgisayarı", "yemekhane", "menüsü", "zırzop", "nerede", "şarkı", "kütüphanede"],
    )
    def test_leaves_a_genuinely_unknown_word_alone(self, lexicon: Lexicon, token: str) -> None:
        """Segmenting an out-of-scope word into command words would invent a command."""
        assert lexicon.resegment(token) is None

    def test_refuses_a_segmentation_of_only_single_letters(self, lexicon: Lexicon) -> None:
        """Otherwise every string would resegment into a product of variables."""
        assert lexicon.resegment("xyz") is None

    def test_piece_count_is_bounded(self, lexicon: Lexicon) -> None:
        result = lexicon.resegment("türevtanımınıgetir")
        assert result is None or len(result) <= MAX_RESEGMENT_PIECES

    def test_a_run_longer_than_the_piece_budget_is_refused(self, lexicon: Lexicon) -> None:
        """Five words run together is the search finding a coincidence, not a boundary."""
        assert lexicon.resegment("türevtanımınıekranagetirbul") is None

    def test_a_token_longer_than_the_search_limit_is_left_alone(self, lexicon: Lexicon) -> None:
        assert lexicon.resegment("türev" * 20) is None

    def test_understands_accepts_a_stem_plus_one_suffix(self, lexicon: Lexicon) -> None:
        assert lexicon.understands("türev")
        assert lexicon.understands("türevini")
        assert not lexicon.understands("türevtanımını")

    def test_head_token_reduces_to_the_stem(self, lexicon: Lexicon) -> None:
        assert lexicon.head_token("karenin") == "kare"
        assert lexicon.head_token("sinüs") == "sin"
        assert lexicon.head_token("zırzop") == "zırzop"


class TestNormaliserUsesResegmentation:
    @pytest.mark.parametrize(
        ("text", "normalised"),
        [
            ("türevtanımını getir", "türev tanımını getir"),
            ("xkarenin türevini al", "x kare nin türev ini al"),
            ("x karenin türeviniAl", "x kare nin türev ini al"),
        ],
    )
    def test_normalisation_recovers_the_boundary(self, text: str, normalised: str) -> None:
        assert normalise_tail(text) == normalise_tail(normalised)

    def test_normalisation_stays_idempotent(self) -> None:
        once = normalize_text("türevtanımınıgetir xkarenin")
        twice = normalize_text(once.text)
        assert twice.text == once.text

    def test_spans_stay_inside_the_original_text(self) -> None:
        analysed = normalize_text("xkarenin türevini al")
        for start, end in analysed.spans:
            assert 0 <= start < end <= len(analysed.original)


def normalise_tail(text: str) -> str:
    """The normalised token string (helper: the two spellings must agree token for token)."""
    return normalize_text(text).text


class TestJoinedCommandsResolve:
    @pytest.mark.parametrize(
        ("text", "command"),
        [
            ("xkarenin türevini al", Diff(expr="x^{2}", var="x", order=1)),
            ("x karenin türeviniAl", Diff(expr="x^{2}", var="x", order=1)),
            ("x karenin integralinial", Integrate(expr="x^{2}", var="x")),
        ],
    )
    def test_a_joined_utterance_still_produces_its_command(
        self, text: str, command: object
    ) -> None:
        outcome = parse_command(text)
        assert isinstance(outcome, Matched), outcome
        assert outcome.command == command

    def test_a_joined_book_lookup_still_resolves(self) -> None:
        outcome = parse_command("türevtanımını getir")
        assert isinstance(outcome, Matched), outcome
        assert outcome.intent == "book_fetch"
