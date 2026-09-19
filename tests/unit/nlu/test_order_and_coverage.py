"""The wrong-answer defects of KNOWN_ISSUES PRIORITY 1, turned into regression tests.

Two independent failures produced a *confidently rendered, SymPy-computed answer to the wrong
question* on a platform whose headline claim is "no hallucination":

1. **The exponent numeral was consumed twice.** ``"x üzeri üçün türevini al"`` filled ``expr``
   with ``x^{3}`` **and** ``order`` with ``3``, so the platform answered ``d³/dx³ x³ = 6``
   instead of ``3x²``. The rule is now: a token already consumed by another slot of the adopted
   reading may not fill ``ORDER``, and an order above one must be *said* -- an ordinal word
   (``ikinci``) or an explicit ``mertebe``/``derece`` phrase.
2. **A token vanished inside the numeral folder.** ``"x kare eksi dört sıfır denklemini çöz"``
   folded ``dört sıfır`` into the single numeral ``4``, silently dropping the utterance's last
   content word and solving ``x² − 4`` -- a different equation from the one that was said.

Every case listed in the issue is a named test here; the sibling families (integrate, limit,
solve, plot, series) are tested too so that the fix cannot overreach into them.
"""

from __future__ import annotations

import pytest

from aicls.mathengine.models import Diff, Integrate, Limit, Series, Solve
from aicls.nlu import (
    Clarification,
    Matched,
    NoMatch,
    ParseContext,
    normalize_text,
    parse_command,
)
from aicls.nlu.slots import ORDER_ANCHORS, scan_order


def _matched(text: str, **kwargs: object) -> Matched:
    outcome = parse_command(text, **kwargs)  # type: ignore[arg-type]  # kwargs are typed above
    assert isinstance(outcome, Matched), outcome
    return outcome


class TestExponentIsNotAnOrder:
    """KNOWN_ISSUES PRIORITY 1: the exponent numeral must not also fill ``ORDER``."""

    def test_nlu_x_uzeri_ucun_turevini_al_is_first_order(self) -> None:
        """The headline reproduction: it must be ``3x²``, never ``6``."""
        outcome = _matched("x üzeri üçün türevini al")
        assert outcome.command == Diff(expr="x^{3}", var="x", order=1)

    def test_nlu_x_uzeri_besin_turevini_al_is_first_order(self) -> None:
        """``5x⁴``, never ``120``."""
        outcome = _matched("x üzeri beşin türevini al")
        assert outcome.command == Diff(expr="x^{5}", var="x", order=1)

    def test_nlu_x_uzeri_uc_turevini_al_is_first_order(self) -> None:
        outcome = _matched("x üzeri üç türevini al")
        assert outcome.command == Diff(expr="x^{3}", var="x", order=1)

    def test_nlu_x_uzeri_uc_ifadesinin_turevini_al_is_first_order(self) -> None:
        """An object noun already blocked the bug; it must keep doing so."""
        outcome = _matched("x üzeri üç ifadesinin türevini al")
        assert outcome.command == Diff(expr="x^{3}", var="x", order=1)

    def test_nlu_x_uzeri_ucun_ikinci_turevini_al_keeps_order_two(self) -> None:
        """An explicit ordinal outside the expression still wins: ``6x``."""
        outcome = _matched("x üzeri üçün ikinci türevini al")
        assert outcome.command == Diff(expr="x^{3}", var="x", order=2)

    def test_nlu_x_karenin_turevini_al_is_first_order(self) -> None:
        outcome = _matched("x karenin türevini al")
        assert outcome.command == Diff(expr="x^{2}", var="x", order=1)

    def test_nlu_x_kupunun_ikinci_turevini_al_keeps_order_two(self) -> None:
        outcome = _matched("x küpünün ikinci türevini al")
        assert outcome.command == Diff(expr="x^{3}", var="x", order=2)

    @pytest.mark.parametrize(
        ("text", "expr"),
        [
            ("x üzeri dördün türevini al", "x^{4}"),
            ("x üzeri altının türevini hesapla", "x^{6}"),
            ("iki x üzeri üç ifadesinin türevini bul", "2 x^{3}"),
            ("x üzeri on ikinin türevini al", "x^{12}"),
        ],
    )
    def test_no_exponent_ever_becomes_an_order(self, text: str, expr: str) -> None:
        outcome = _matched(text)
        assert outcome.command == Diff(expr=expr, var="x", order=1)

    @pytest.mark.parametrize(
        ("text", "order"),
        [
            ("x üzeri üçün üçüncü türevini al", 3),
            ("x üzeri beşin dördüncü türevini al", 4),
            ("x üzeri üçün ikinci mertebeden türevini al", 2),
            ("x üzeri üçün iki mertebeden türevini al", 2),
        ],
    )
    def test_an_explicit_order_still_reaches_the_command(self, text: str, order: int) -> None:
        """The fix must not make higher orders unreachable when they are actually said."""
        outcome = _matched(text)
        assert isinstance(outcome.command, Diff)
        assert outcome.command.expr.startswith("x^{")
        assert outcome.command.order == order

    def test_context_derivative_keeps_reading_the_ordinal(self) -> None:
        outcome = _matched(
            "üçüncü mertebeden türevini al", context=ParseContext(last_latex="x^{5}")
        )
        assert outcome.command == Diff(expr="x^{5}", var="x", order=3)


class TestOrderAnchors:
    """The ``ORDER`` scan rule itself."""

    def test_turev_is_no_longer_a_digit_anchor(self) -> None:
        """``türev`` anchored a bare digit; that is exactly what read the exponent as an order."""
        assert set(ORDER_ANCHORS) == {"mertebe", "derece"}

    @pytest.mark.parametrize(
        ("text", "order"),
        [
            ("ikinci türevini al", 2),
            ("üçüncü mertebeden türevini al", 3),
            ("2 mertebeden türevini al", 2),
            ("3 dereceden türevini al", 3),
            ("iki x in türevini al", None),
            ("türevini al", None),
            ("x üzeri üç türevini al", None),
            ("x üzeri beşin türevini al", None),
        ],
    )
    def test_scan_order_reads_only_explicit_orders(self, text: str, order: int | None) -> None:
        assert scan_order(normalize_text(text)) == order

    def test_a_skipped_index_cannot_supply_the_order(self) -> None:
        """An ordinal word inside a span another slot claimed does not fill ``ORDER``."""
        analysed = normalize_text("ikinci türevini al")
        assert scan_order(analysed) == 2
        assert scan_order(analysed, skip=frozenset({0})) is None

    def test_an_anchored_digit_survives_a_claimed_span(self) -> None:
        """``mertebe`` is decisive evidence: the digit is an order wherever it was captured."""
        analysed = normalize_text("2 mertebeden türevini al")
        assert scan_order(analysed, skip=frozenset({0})) == 2

    def test_a_digit_right_after_uzeri_is_an_exponent_not_an_order(self) -> None:
        analysed = normalize_text("x üzeri iki mertebeden türevini al")
        assert scan_order(analysed) is None


class TestConsumedIndices:
    """``consumed_indices``: which tokens the adopted reading already claimed."""

    def test_no_match_claims_nothing(self) -> None:
        from aicls.nlu.slots import consumed_indices

        assert consumed_indices(normalize_text("x karenin türevini al"), None) == frozenset()

    def test_the_expression_capture_claims_its_tokens(self) -> None:
        from aicls.nlu import load_inventory
        from aicls.nlu.matcher import IntentMatcher
        from aicls.nlu.slots import consumed_indices

        inventory = load_inventory()
        analysed = normalize_text("x üzeri üçün türevini al")
        candidate = next(
            item
            for item in IntentMatcher(inventory).rank(analysed)
            if item.id == "differentiate" and item.matches
        )
        claimed = consumed_indices(analysed, candidate.matches[0])
        assert claimed, candidate.matches[0].pattern.source
        assert analysed.tokens.index("3") in claimed

    def test_an_unfilled_optional_group_claims_nothing(self) -> None:
        """A pattern slot that did not participate must not block the order scan."""
        from aicls.nlu.matcher import PatternMatch
        from aicls.nlu.patterns import compile_pattern
        from aicls.nlu.slots import consumed_indices

        analysed = normalize_text("ikinci türevini al")
        pattern = compile_pattern("ikinci {expr}? türev", suffixes=(), slot_types={"expr": "EXPR"})
        match = pattern.search(analysed.text)
        assert match is not None
        found = PatternMatch(pattern=pattern, match=match, full=False, coverage=0.5)
        assert consumed_indices(analysed, found) == frozenset()


class TestTheFixDoesNotOverreach:
    """The same ``x üzeri <sayı>`` family for every other CAS intent."""

    @pytest.mark.parametrize(
        ("text", "command"),
        [
            (
                "x üzeri üçün integralini al",
                Integrate(expr="x^{3}", var="x", lower=None, upper=None),
            ),
            (
                "x üzeri beşin integralini hesapla",
                Integrate(expr="x^{5}", var="x", lower=None, upper=None),
            ),
            (
                "sıfırdan bire kadar x üzeri üçün integralini hesapla",
                Integrate(expr="x^{3}", var="x", lower="0", upper="1"),
            ),
            (
                "x sıfıra giderken x üzeri üçün limitini bul",
                Limit(expr="x^{3}", var="x", to="0", direction="+-"),
            ),
            (
                "x üzeri üç eşittir sekiz denklemini çöz",
                Solve(equation="x^{3} = 8", var="x"),
            ),
        ],
    )
    def test_sibling_intents_are_unchanged(self, text: str, command: object) -> None:
        assert _matched(text).command == command

    def test_series_order_still_comes_from_the_ordinal(self) -> None:
        outcome = _matched("sinüs x'in beşinci mertebeden seri açılımını hesapla")
        assert outcome.command == Series(expr=r"\sin\left(x\right)", var="x", point="0", order=5)

    def test_series_default_order_survives_an_exponent(self) -> None:
        """``x üzeri üç`` must not become a fifth-order Taylor expansion either."""
        outcome = _matched("x üzeri üç ifadesinin seri açılımını yap")
        assert isinstance(outcome.command, Series)
        assert outcome.command.order == 6


class TestNumeralPhraseIsNotSilentlyMerged:
    """KNOWN_ISSUES PRIORITY 1, second half: a swallowed trailing token."""

    def test_nlu_x_kare_eksi_dort_sifir_denklemini_coz_is_not_silently_truncated(self) -> None:
        """``dört sıfır`` is not a Turkish numeral; the layer must ask, not solve ``x²−4``."""
        outcome = parse_command("x kare eksi dört sıfır denklemini çöz")
        assert isinstance(outcome, Clarification | NoMatch), outcome
        if isinstance(outcome, Clarification):
            assert outcome.missing_slot == "equation"
            assert outcome.question_tr == "Hangi denklemi çözeyim?"

    def test_the_intended_sentence_still_resolves(self) -> None:
        """The same sentence *with* ``eşittir`` is a valid command and must stay one."""
        outcome = _matched("x kare eksi dört eşittir sıfır denklemini çöz")
        assert outcome.command == Solve(equation="x^{2} - 4 = 0", var="x")

    def test_the_dropped_token_survives_normalisation(self) -> None:
        analysed = normalize_text("x kare eksi dört sıfır denklemini çöz")
        assert analysed.tokens.count("0") == 1
        assert analysed.tokens.count("4") == 1

    @pytest.mark.parametrize(
        "text",
        [
            "x kare eksi dört sıfır denklemini çöz",
            "iki x eksi altı sıfır denklemini çöz",
            "x kare üç beş ifadesinin türevini al",
        ],
    )
    def test_an_unreadable_numeral_run_never_becomes_a_command(self, text: str) -> None:
        assert not isinstance(parse_command(text), Matched)


class TestUnaccountedTokens:
    """A math-bearing token left outside the adopted reading must not be ignored."""

    @pytest.mark.parametrize(
        "text",
        [
            "x karenin türevini al iki y küp",
            "tahtayı temizle x kare artı bir",
        ],
    )
    def test_a_math_bearing_leftover_refuses_the_command(self, text: str) -> None:
        outcome = parse_command(text)
        assert not isinstance(outcome, Matched), outcome

    @pytest.mark.parametrize(
        "text",
        [
            "sinüs x'in üçüncü mertebeden seri açılımını al",
            "tekrar yap",
            "hepsini temizle",
        ],
    )
    def test_a_harmless_leftover_is_still_tolerated(self, text: str) -> None:
        """Only *math-bearing* leftovers refuse: a stray verb or filler must not."""
        assert isinstance(parse_command(text), Matched)
