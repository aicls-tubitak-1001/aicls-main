"""``parse_command``: the contractual behaviours of the text-to-command layer.

Most test names here are fixed by the module spec and by
``docs/mimari/metin-komut-katmani.md`` §8: they are the proposal's own examples turned into
regression tests.
"""

from __future__ import annotations

import contextlib
from typing import Any

import pytest

from aicls.core.errors import AiclsError
from aicls.mathengine.models import Diff, Evaluate, Factor, Integrate, Limit, Plot, Series, Solve
from aicls.nlu import (
    BookFetch,
    Clarification,
    Matched,
    NoMatch,
    ParseContext,
    UiAction,
    UnitType,
    load_inventory,
    parse_command,
)
from aicls.nlu.resolver import explain


def matched(text: str, **kwargs: Any) -> Matched:
    """Parse ``text`` and assert it produced a :class:`Matched`."""
    outcome = parse_command(text, **kwargs)
    assert isinstance(outcome, Matched), outcome
    return outcome


class TestProposalExamples:
    def test_nlu_x_karenin_turevini_al_maps_to_diff(self) -> None:
        """The proposal's own example: "x karenin türevini al" -> ``diff(x**2, x)``."""
        outcome = matched("x karenin türevini al")
        assert outcome.intent == "differentiate"
        assert outcome.command == Diff(expr="x^{2}", var="x", order=1)
        assert outcome.source == "rules"

    def test_nlu_sifirdan_bire_x_kare_integrali_maps_to_integrate_bounds(self) -> None:
        outcome = matched("sıfırdan bire kadar x karenin integralini hesapla")
        assert outcome.intent == "integrate_definite"
        assert outcome.command == Integrate(expr="x^{2}", var="x", lower="0", upper="1")

    def test_nlu_x_sonsuza_giderken_limit_maps_to_limit_infty(self) -> None:
        outcome = matched("x sonsuza giderken bir bölü x'in limiti")
        assert outcome.intent == "limit"
        assert outcome.command == Limit(expr=r"\frac{1}{x}", var="x", to=r"\infty", direction="+-")

    def test_nlu_sagdan_yaklasirken_sets_direction_plus(self) -> None:
        outcome = matched("x sıfıra sağdan yaklaşırken bir bölü x'in limitini bul")
        assert isinstance(outcome.command, Limit)
        assert outcome.command.direction == "+"
        assert outcome.slots["direction"] == "+"

    def test_soldan_yaklasirken_sets_direction_minus(self) -> None:
        outcome = matched("x sıfıra soldan yaklaşırken bir bölü x'in limiti")
        assert isinstance(outcome.command, Limit)
        assert outcome.command.direction == "-"

    def test_nlu_turev_tanimini_getir_maps_to_book_fetch_definition(self) -> None:
        outcome = matched("türev tanımını getir")
        assert outcome.intent == "book_fetch"
        assert outcome.command == BookFetch(unit_type=UnitType.DEFINITION, query="türev")

    def test_nlu_turevini_al_uses_last_latex_context(self) -> None:
        outcome = matched("türevini al", context=ParseContext(last_latex="x^{3}"))
        assert outcome.command == Diff(expr="x^{3}", var="x", order=1)
        assert outcome.slots["expr"] == "x^{3}"


class TestEveryIntent:
    @pytest.mark.parametrize(
        ("text", "intent", "command"),
        [
            ("x karenin integralini al", "integrate_indefinite", Integrate(expr="x^{2}")),
            (
                "üç x artı beş eşittir on denklemini çöz",
                "solve",
                Solve(equation="3 x + 5 = 10", var="x"),
            ),
            (
                "x kare eksi dört ifadesini çarpanlarına ayır",
                "factor",
                Factor(expr="x^{2} - 4"),
            ),
            (
                "e üzeri x fonksiyonunun seri açılımını hesapla",
                "series",
                Series(expr="e^{x}", var="x", point="0", order=6),
            ),
            (
                "x eşittir iki için x kare artı bir ifadesinin değerini hesapla",
                "evaluate_at",
                Evaluate(expr="x^{2} + 1", subs={"x": "2"}),
            ),
            (
                "iki x üzeri üç fonksiyonunun grafiğini çiz",
                "plot",
                Plot(expr="2 x^{3}", var="x"),
            ),
            ("tahtayı temizle", "ui_clear", UiAction(action="clear")),
            ("geri al", "ui_undo", UiAction(action="undo")),
            ("son komutu tekrar et", "ui_repeat_last", UiAction(action="repeat_last")),
            ("sonucu dışa aktar", "ui_export", UiAction(action="export")),
            ("kitapta türev ara", "book_search", BookFetch(unit_type=None, query="türev")),
        ],
    )
    def test_maps_to_the_expected_command(self, text: str, intent: str, command: Any) -> None:
        outcome = matched(text)
        assert outcome.intent == intent
        assert outcome.command == command

    def test_second_order_derivative_reads_the_ordinal(self) -> None:
        outcome = matched("sinüs x'in ikinci türevi")
        assert outcome.command == Diff(expr=r"\sin\left(x\right)", var="x", order=2)

    def test_every_v1_intent_is_reachable(self) -> None:
        """All seventeen intents exist and every one of them has at least one worked example."""
        inventory = load_inventory()
        assert len(inventory.intents) == 17
        reached = {
            matched(example.text, context=_context(example)).intent
            for _, example in inventory.examples()
        }
        assert reached == set(inventory.intent_ids)


def _context(example: Any) -> ParseContext | None:
    return ParseContext(**example.context) if example.context else None


class TestClarification:
    def test_nlu_missing_variable_returns_clarification_tr(self) -> None:
        """Two free symbols and neither is the default: ask, do not guess."""
        outcome = parse_command("a karenin çarpı b nin türevini al")
        assert isinstance(outcome, Clarification)
        assert outcome.missing_slot == "var"
        assert outcome.question_tr == "Hangi değişkene göre türev alayım?"
        assert outcome.intent == "differentiate"

    def test_missing_expression_returns_clarification_tr(self) -> None:
        outcome = parse_command("türevini al")
        assert isinstance(outcome, Clarification)
        assert outcome.missing_slot == "expr"
        assert outcome.question_tr == "Hangi ifadenin türevini alayım?"

    def test_nlu_ambiguous_returns_clarification_with_candidates(self) -> None:
        """ "açılım" is both an algebraic expansion and a series expansion in Turkish."""
        outcome = parse_command("x karenin açılımını yap")
        assert isinstance(outcome, Clarification)
        assert outcome.missing_slot is None
        assert {item.intent for item in outcome.candidates} == {"expand", "series"}
        assert all(item.label_tr for item in outcome.candidates)
        assert outcome.question_tr

    def test_clarification_candidates_are_capped(self) -> None:
        outcome = parse_command("x karenin açılımını yap")
        assert isinstance(outcome, Clarification)
        assert len(outcome.candidates) <= load_inventory().defaults.max_candidates


class TestNoMatch:
    def test_nlu_unknown_phrase_returns_nomatch_with_suggestions(self) -> None:
        outcome = parse_command("bugün hava nasıl")
        assert isinstance(outcome, NoMatch)
        assert outcome.reason == "no_intent"
        assert len(outcome.suggestions_tr) >= 3
        assert "türev" in " ".join(outcome.suggestions_tr)

    @pytest.mark.parametrize(
        "text",
        [
            "en yakın kafe nerede",
            "bana bir şarkı söyle",
            "yemekhane menüsü nedir",
            "ders programını göster",
            "bilgisayarı kapat",
        ],
    )
    def test_out_of_scope_requests_are_filtered(self, text: str) -> None:
        """İP-4 risk plan: out-of-scope requests get a guiding message, never a wrong command."""
        assert isinstance(parse_command(text), NoMatch)

    def test_empty_input_is_a_nomatch(self) -> None:
        outcome = parse_command("   ")
        assert isinstance(outcome, NoMatch)
        assert outcome.reason == "empty_input"


class TestTrace:
    def test_matched_carries_a_trace(self) -> None:
        outcome = matched("x karenin türevini al")
        joined = "\n".join(outcome.trace)
        assert "normalized: x kare nin türev ini al" in joined
        assert "intent: differentiate" in joined
        assert "pattern: " in joined
        assert "command: diff" in joined

    def test_explain_returns_the_trace(self) -> None:
        assert explain("x karenin türevini al")[0].startswith("normalized:")

    def test_clarification_and_nomatch_carry_a_trace(self) -> None:
        assert parse_command("türevini al").trace
        assert parse_command("bugün hava nasıl").trace


class TestAlternatives:
    def test_alternatives_are_ranked_below_the_winner(self) -> None:
        outcome = matched("x karenin türevini al")
        assert all(item.score <= outcome.score for item in outcome.alternatives)
        assert all(item.intent != outcome.intent for item in outcome.alternatives)


class TestContext:
    def test_active_var_wins_when_it_occurs_in_the_expression(self) -> None:
        outcome = matched(
            "a kare artı t kare ifadesinin türevini al",
            context=ParseContext(active_var="t"),
        )
        assert isinstance(outcome.command, Diff)
        assert outcome.command.var == "t"

    def test_unparsable_context_is_ignored(self) -> None:
        outcome = parse_command("türevini al", context=ParseContext(last_latex=r"\begin{cases}"))
        assert isinstance(outcome, Clarification)

    def test_voice_modality_is_accepted(self) -> None:
        outcome = matched("türevini al", context=ParseContext(last_latex="x^{2}", modality="voice"))
        assert isinstance(outcome.command, Diff)


class TestRobustness:
    @pytest.mark.parametrize(
        "text",
        ["", "?????", "\x00\x01", "x" * 3000, "türev " * 200, "((((", "\\frac{1}{"],
    )
    def test_only_aicls_errors_escape(self, text: str) -> None:
        with contextlib.suppress(AiclsError):
            parse_command(text)

    def test_a_custom_inventory_can_be_passed(self) -> None:
        inventory = load_inventory()
        outcome = parse_command("x karenin türevini al", inventory=inventory)
        assert isinstance(outcome, Matched)
