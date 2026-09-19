"""Slot extraction and the intent -> typed command mapper."""

from __future__ import annotations

import pytest

from aicls.latex2sympy import parse_latex
from aicls.mathengine.models import Diff, Simplify
from aicls.nlu import load_inventory, normalize_text
from aicls.nlu.errors import CommandBuildError
from aicls.nlu.inventory.schema import Inventory
from aicls.nlu.mapper import CAS_BUILDERS, build_command
from aicls.nlu.matcher import IntentMatcher, score_candidate
from aicls.nlu.slots import (
    SlotExtractor,
    canonical_latex,
    free_symbols,
    iter_nodes,
    scan_order,
)
from aicls.nlu.types import BookFetch, ParseContext, UiAction, UnitType


@pytest.fixture(scope="module")
def inventory() -> Inventory:
    return load_inventory()


class TestFreeSymbols:
    @pytest.mark.parametrize(
        ("latex", "symbols"),
        [
            ("x^{2}", ("x",)),
            (r"\sin\left(x\right)", ("x",)),
            ("a^{2} b", ("a", "b")),
            ("e^{x}", ("x",)),
            (r"\pi", ()),
            ("2 + 3", ()),
        ],
    )
    def test_collects_symbols(self, latex: str, symbols: tuple[str, ...]) -> None:
        assert free_symbols(parse_latex(latex)) == symbols

    def test_iter_nodes_walks_the_whole_tree(self) -> None:
        node = parse_latex("x^{2} + 2 x")
        assert len(list(iter_nodes(node))) > 3


class TestCanonicalLatex:
    def test_reprints_valid_latex(self) -> None:
        assert canonical_latex("x^2") == "x^{2}"

    def test_returns_none_for_invalid_latex(self) -> None:
        assert canonical_latex(r"\begin{cases}") is None


class TestScanOrder:
    @pytest.mark.parametrize(
        ("text", "order"),
        [
            ("ikinci türevini al", 2),
            ("üçüncü mertebeden türevini al", 3),
            ("iki x in türevini al", None),
            ("türevini al", None),
        ],
    )
    def test_reads_only_real_orders(self, text: str, order: int | None) -> None:
        assert scan_order(normalize_text(text)) == order

    def test_a_digit_needs_an_anchor(self) -> None:
        assert scan_order(normalize_text("2 mertebeden türevini al")) == 2


class TestSlotExtractor:
    def test_resolves_from_the_pattern(self, inventory: Inventory) -> None:
        extractor = SlotExtractor(inventory)
        analysed = normalize_text("x karenin türevini al")
        candidate = next(
            item for item in IntentMatcher(inventory).rank(analysed) if item.id == "differentiate"
        )
        resolution = extractor.best(candidate.intent, analysed, candidate.matches)
        assert resolution.complete
        assert resolution.values["expr"] == "x^{2}"
        assert resolution.sources["expr"] == "pattern"

    def test_resolves_from_the_context(self, inventory: Inventory) -> None:
        extractor = SlotExtractor(inventory)
        analysed = normalize_text("türevini al")
        candidate = next(
            item for item in IntentMatcher(inventory).rank(analysed) if item.id == "differentiate"
        )
        resolution = extractor.best(
            candidate.intent,
            analysed,
            candidate.matches,
            context=ParseContext(last_latex="x^{3}"),
        )
        assert resolution.values["expr"] == "x^{3}"
        assert resolution.sources["expr"] == "context"

    def test_reports_missing_required_slots(self, inventory: Inventory) -> None:
        extractor = SlotExtractor(inventory)
        analysed = normalize_text("türevini al")
        resolution = extractor.extract(inventory.by_id["differentiate"], analysed)
        assert resolution.missing == ("expr",)
        assert not resolution.complete
        assert "slot expr: unresolved" in resolution.trace

    def test_extractor_without_a_match_still_uses_defaults(self, inventory: Inventory) -> None:
        extractor = SlotExtractor(inventory)
        analysed = normalize_text("tahtayı temizle")
        resolution = extractor.best(inventory.by_id["ui_clear"], analysed, ())
        assert resolution.complete
        assert resolution.values == {}


class TestScoring:
    def test_score_is_bounded(self, inventory: Inventory) -> None:
        analysed = normalize_text("x karenin türevini al")
        for candidate in IntentMatcher(inventory).rank(analysed):
            assert 0.0 <= candidate.base_score <= 1.0

    def test_conflicting_strong_keyword_lowers_the_score(self, inventory: Inventory) -> None:
        analysed = normalize_text("türev tanımını getir")
        ranked = {item.id: item for item in IntentMatcher(inventory).rank(analysed)}
        assert "türev" in ranked["book_fetch"].conflicts
        assert ranked["book_fetch"].base_score > ranked["differentiate"].base_score

    def test_score_of_no_match_is_zero(self, inventory: Inventory) -> None:
        analysed = normalize_text("x karenin türevini al")
        candidate = next(iter(IntentMatcher(inventory).rank(analysed)))
        assert score_candidate(candidate, None, inventory.defaults.weights) < candidate.base_score

    def test_ranking_an_empty_utterance_is_empty(self, inventory: Inventory) -> None:
        assert IntentMatcher(inventory).rank(normalize_text("")) == []

    def test_matcher_exposes_its_inventory(self, inventory: Inventory) -> None:
        assert IntentMatcher(inventory).inventory is inventory


class TestMapper:
    def test_every_cas_op_has_a_builder(self, inventory: Inventory) -> None:
        ops = {intent.cas_op for intent in inventory.intents if intent.kind == "cas"}
        assert ops <= set(CAS_BUILDERS)

    def test_builds_a_cas_command(self, inventory: Inventory) -> None:
        command = build_command(
            inventory.by_id["differentiate"], {"expr": "x^{2}", "var": "x", "order": 2}
        )
        assert command == Diff(expr="x^{2}", var="x", order=2)

    def test_builds_a_book_fetch(self, inventory: Inventory) -> None:
        command = build_command(
            inventory.by_id["book_fetch"], {"unit_type": "Theorem", "query": "rolle"}
        )
        assert command == BookFetch(unit_type=UnitType.THEOREM, query="rolle")

    def test_builds_a_book_search_without_a_unit_type(self, inventory: Inventory) -> None:
        command = build_command(inventory.by_id["book_search"], {"query": "türev"})
        assert command == BookFetch(unit_type=None, query="türev")

    def test_builds_a_ui_action(self, inventory: Inventory) -> None:
        assert build_command(inventory.by_id["ui_clear"], {}) == UiAction(action="clear")

    def test_defaults_are_applied(self, inventory: Inventory) -> None:
        assert build_command(inventory.by_id["simplify"], {"expr": "x"}) == Simplify(expr="x")

    @pytest.mark.parametrize(
        "slots",
        [{}, {"expr": "x^{2}", "var": "not a var"}, {"expr": "x^{2}", "order": 99}],
    )
    def test_model_rejection_becomes_a_command_build_error(
        self, inventory: Inventory, slots: dict[str, object]
    ) -> None:
        with pytest.raises(CommandBuildError):
            build_command(inventory.by_id["differentiate"], slots)

    def test_unknown_cas_op_is_refused(self, inventory: Inventory) -> None:
        broken = inventory.by_id["differentiate"].model_copy(update={"cas_op": "nope"})
        with pytest.raises(CommandBuildError):
            build_command(broken, {"expr": "x"})
