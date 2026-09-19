"""Slot types that the shipped inventory fills by scanning, exercised through patterns.

The v1 inventory reads ``ORDER``, ``DIRECTION`` and ``UNIT_TYPE`` with lexical scans, because
Turkish puts them anywhere in the sentence. The extractor also supports capturing them from a
pattern, which is what a larger inventory (or a domain-specific one) will use, so the code path
is pinned here against a small synthetic inventory rather than left untested.
"""

from __future__ import annotations

from typing import Any

import pytest

from aicls.nlu import normalize_text
from aicls.nlu.inventory import build_inventory
from aicls.nlu.inventory.schema import Inventory
from aicls.nlu.matcher import IntentMatcher
from aicls.nlu.slots import SlotExtractor, is_variable_name
from aicls.nlu.types import ParseContext

SLOT_TYPES: dict[str, dict[str, Any]] = {
    "EXPR": {"kind": "latex_expr"},
    "VAR": {"kind": "symbol"},
    "POINT": {"kind": "expr_or_infty"},
    "BOUNDS": {"kind": "pair", "of": "EXPR"},
    "ORDER": {"kind": "int"},
    "DIRECTION": {"kind": "enum", "values": ["+", "-", "+-"]},
    "UNIT_TYPE": {"kind": "enum", "values": ["Definition", "Theorem"]},
    "QUERY": {"kind": "text"},
}


def build(**intent: Any) -> Inventory:
    """A one-intent inventory around ``intent``."""
    base = {
        "id": "probe",
        "kind": "ui",
        "ui_action": "clear",
        "label_tr": "deneme",
        "keywords": {"temizle": 3},
    }
    base.update(intent)
    return build_inventory({"version": "1.0", "slot_types": SLOT_TYPES, "intents": [base]})


def resolve(inventory: Inventory, text: str, context: ParseContext | None = None) -> Any:
    analysed = normalize_text(text)
    candidates = IntentMatcher(inventory).rank(analysed)
    matches = candidates[0].matches if candidates else ()
    return SlotExtractor(inventory).best(inventory.intents[0], analysed, matches, context=context)


class TestPatternCapturedSlots:
    def test_order_from_a_pattern(self) -> None:
        inventory = build(
            slots={"order": {"type": "ORDER"}},
            patterns=["{order} mertebe temizle"],
        )
        assert resolve(inventory, "ikinci mertebeden temizle").values["order"] == 2

    def test_an_order_that_is_not_a_number_stays_unresolved(self) -> None:
        inventory = build(
            slots={"order": {"type": "ORDER", "required": True}},
            clarify_tr={"order": "Kaçıncı?"},
            patterns=["{order} mertebe temizle"],
        )
        assert resolve(inventory, "türev mertebeden temizle").missing == ("order",)

    def test_direction_from_a_pattern(self) -> None:
        inventory = build(
            slots={"direction": {"type": "DIRECTION"}},
            patterns=["{direction} temizle"],
        )
        assert resolve(inventory, "sağdan temizle").values["direction"] == "sağdan"

    def test_unit_type_from_a_pattern(self) -> None:
        inventory = build(
            slots={"unit_type": {"type": "UNIT_TYPE"}},
            patterns=["{unit_type} temizle"],
        )
        assert resolve(inventory, "tanım temizle").values["unit_type"] == "Definition"

    def test_unit_type_from_a_pattern_accepts_the_enum_value(self) -> None:
        inventory = build(
            slots={"unit_type": {"type": "UNIT_TYPE"}},
            patterns=["{unit_type} temizle"],
        )
        assert resolve(inventory, "theorem temizle").values["unit_type"] == "Theorem"

    def test_an_unknown_unit_type_stays_unresolved(self) -> None:
        inventory = build(
            slots={"unit_type": {"type": "UNIT_TYPE", "required": True}},
            clarify_tr={"unit_type": "Hangi tür?"},
            patterns=["{unit_type} temizle"],
        )
        assert resolve(inventory, "grafik temizle").missing == ("unit_type",)

    def test_an_optional_slot_that_did_not_match_is_not_captured(self) -> None:
        inventory = build(
            slots={"expr": {"type": "EXPR", "from_context": "last_latex"}},
            patterns=["temizle {expr}?"],
        )
        resolution = resolve(inventory, "temizle", ParseContext(last_latex="x^{2}"))
        assert resolution.values["expr"] == "x^{2}"
        assert resolution.sources["expr"] == "context"

    def test_a_query_slot_that_caught_nothing_but_suffixes(self) -> None:
        inventory = build(
            slots={"query": {"type": "QUERY", "required": True}},
            clarify_tr={"query": "Ne arayayım?"},
            patterns=["temizle {query}?"],
        )
        assert resolve(inventory, "tahtayı temizle").missing == ("query",)


class TestLatexFallback:
    def test_a_latex_fragment_that_is_not_spoken_math_is_parsed_as_latex(self) -> None:
        """Typed LaTeX reaches the same slot: the spoken grammar fails, the LaTeX parser wins."""
        from aicls.nlu import Matched, parse_command

        outcome = parse_command(r"x \cdot y ifadesinin türevini al")
        assert isinstance(outcome, Matched)
        assert outcome.slots["expr"] == "x y"


class TestVariableNames:
    @pytest.mark.parametrize("token", ["x", "t", "y", "alpha", "theta", "x_{1}"])
    def test_accepts_real_variable_names(self, token: str) -> None:
        assert is_variable_name(token)

    @pytest.mark.parametrize("token", ["kare", "türev", "", "x y", "1", "ifade"])
    def test_rejects_everything_else(self, token: str) -> None:
        assert not is_variable_name(token)


class TestNumberFolding:
    def test_a_lone_decimal_marker_is_left_as_written(self) -> None:
        """ "virgül" alone is not a numeral; the run must survive rather than blow up."""
        assert "virgül" in normalize_text("virgül x").tokens
