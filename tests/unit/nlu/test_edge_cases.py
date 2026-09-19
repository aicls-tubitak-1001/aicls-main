"""Failure paths and less-travelled branches of the layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from aicls.core.errors import AiclsError, InternalError
from aicls.nlu import (
    Clarification,
    Matched,
    NoMatch,
    ParseContext,
    load_inventory,
    normalize_text,
    parse_command,
)
from aicls.nlu.errors import CommandBuildError, InventoryError, NluError, SpokenMathError
from aicls.nlu.inventory import build_inventory
from aicls.nlu.inventory.schema import Inventory
from aicls.nlu.lexicon import Lexicon, load_lexicon
from aicls.nlu.matcher import IntentMatcher
from aicls.nlu.resources import LEXICON_RESOURCE, resource_path
from aicls.nlu.slots import SlotExtractor
from aicls.nlu.types import ParseContext as Context


@pytest.fixture(scope="module")
def inventory() -> Inventory:
    return load_inventory()


class TestErrors:
    def test_every_error_is_an_aicls_error(self) -> None:
        for error in (
            NluError(),
            InventoryError("bad"),
            SpokenMathError("x", reason="parse_error"),
            CommandBuildError("differentiate"),
        ):
            assert isinstance(error, AiclsError)
            assert error.message_tr
            assert error.code

    def test_errors_carry_structured_details(self) -> None:
        error = SpokenMathError("x kare artı", reason="parse_error", position=7)
        assert error.details["reason"] == "parse_error"
        assert error.details["position"] == 7
        assert error.text == "x kare artı"

    def test_inventory_error_merges_extra_details(self) -> None:
        error = InventoryError("bad_pattern", details={"intent": "x"})
        assert error.details == {"reason": "bad_pattern", "intent": "x"}

    def test_command_build_error_merges_extra_details(self) -> None:
        error = CommandBuildError("differentiate", details={"error": "boom"})
        assert error.details["intent"] == "differentiate"
        assert error.details["error"] == "boom"


class TestResolverFailurePaths:
    def test_a_command_the_model_refuses_becomes_a_nomatch(
        self, inventory: Inventory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model rejection must never surface as an exception in the pilot classroom."""

        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise CommandBuildError("differentiate", reason="model_rejected")

        monkeypatch.setattr("aicls.nlu.resolver.build_command", refuse)
        outcome = parse_command("x karenin türevini al")
        assert isinstance(outcome, NoMatch)
        assert outcome.reason == "command_invalid"

    def test_an_unexpected_failure_becomes_an_internal_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(*args: Any, **kwargs: Any) -> Any:
            raise ZeroDivisionError("boom")

        monkeypatch.setattr("aicls.nlu.resolver.normalize_text", explode)
        with pytest.raises(InternalError):
            parse_command("x karenin türevini al")

    def test_an_aicls_error_is_re_raised_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise InventoryError("boom")

        monkeypatch.setattr("aicls.nlu.resolver.normalize_text", refuse)
        with pytest.raises(InventoryError):
            parse_command("x karenin türevini al")


class TestSlotEdgeCases:
    def test_invalid_active_var_is_ignored(self, inventory: Inventory) -> None:
        outcome = parse_command(
            "türevini al",
            context=ParseContext(last_latex="x^{2}", active_var="not a var"),
        )
        assert isinstance(outcome, Matched)
        assert outcome.slots["var"] == "x"

    def test_empty_last_latex_is_ignored(self) -> None:
        outcome = parse_command("türevini al", context=ParseContext(last_latex=""))
        assert isinstance(outcome, Clarification)

    def test_a_variable_slot_that_is_not_a_symbol_is_refused(self, inventory: Inventory) -> None:
        """ "x kare ye göre" names something that is not a variable: ask, do not guess."""
        outcome = parse_command("x küpün x kare ye göre türevini al")
        assert isinstance(outcome, Clarification | NoMatch)

    def test_unit_type_from_a_pattern_capture(self, inventory: Inventory) -> None:
        extractor = SlotExtractor(inventory)
        analysed = normalize_text("türev tanımını getir")
        candidate = next(
            item for item in IntentMatcher(inventory).rank(analysed) if item.id == "book_fetch"
        )
        resolution = extractor.best(candidate.intent, analysed, candidate.matches)
        assert resolution.values["unit_type"] == "Definition"

    def test_extractor_accepts_an_explicit_lexicon(self, inventory: Inventory) -> None:
        extractor = SlotExtractor(inventory, lexicon=load_lexicon())
        analysed = normalize_text("tahtayı temizle")
        assert extractor.extract(inventory.by_id["ui_clear"], analysed).complete

    def test_expression_with_no_symbols_falls_back_to_the_default_var(self) -> None:
        outcome = parse_command("iki artı üç ifadesinin türevini al")
        assert isinstance(outcome, Matched)
        assert outcome.slots["var"] == "x"

    def test_single_symbol_expression_uses_that_symbol(self) -> None:
        outcome = parse_command("t karenin türevini al")
        assert isinstance(outcome, Matched)
        assert outcome.slots["var"] == "t"


class TestInventoryValidationReporting:
    def _tiny(self, example_text: str) -> Inventory:
        return build_inventory(
            {
                "version": "1.0",
                "defaults": {"suggestions_tr": ["x karenin türevini al"]},
                "slot_types": {"EXPR": {"kind": "latex_expr"}},
                "intents": [
                    {
                        "id": "differentiate",
                        "kind": "cas",
                        "cas_op": "diff",
                        "label_tr": "türev",
                        "slots": {"expr": {"type": "EXPR", "required": True}},
                        "clarify_tr": {"expr": "Hangi ifade?"},
                        "keywords": {"türev": 3},
                        "patterns": ["{expr} türev al"],
                        "examples": [{"text": example_text, "slots": {"expr": "x^{2}"}}],
                    }
                ],
            }
        )

    def test_reports_an_example_that_does_not_resolve(self) -> None:
        problems = self._tiny("bugün hava nasıl").validate_examples()
        assert len(problems) == 1
        assert "expected Matched" in problems[0]

    def test_reports_an_example_with_a_wrong_slot(self) -> None:
        problems = self._tiny("x küpün türevini al").validate_examples()
        assert len(problems) == 1
        assert "slot expr=" in problems[0]

    def test_reports_intents_with_too_few_examples(self) -> None:
        assert self._tiny("x karenin türevini al").check_example_counts() == [
            "differentiate: 1 examples (minimum 10)"
        ]

    def test_reports_an_example_resolving_to_another_intent(self) -> None:
        inventory = build_inventory(
            {
                "version": "1.0",
                "slot_types": {},
                "intents": [
                    {
                        "id": "ui_clear",
                        "kind": "ui",
                        "ui_action": "clear",
                        "label_tr": "temizle",
                        "keywords": {"temizle": 3},
                        "patterns": ["temizle"],
                        "examples": [{"text": "temizle"}],
                    },
                    {
                        "id": "ui_undo",
                        "kind": "ui",
                        "ui_action": "undo",
                        "label_tr": "geri al",
                        "keywords": {"geri": 3},
                        "patterns": ["geri al"],
                        "examples": [{"text": "temizle"}],
                    },
                ],
            }
        )
        problems = inventory.validate_examples()
        assert any("intent 'ui_clear'" in line for line in problems)


class TestLexiconValidation:
    def _lexicon(self, tmp_path: Path, **overrides: Any) -> Lexicon:
        from aicls.nlu.lexicon import load_lexicon as loader

        data = yaml.safe_load(resource_path(LEXICON_RESOURCE).read_text(encoding="utf-8"))
        data.update(overrides)
        path = tmp_path / "lexicon.yaml"
        path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        return loader(str(path))

    @pytest.mark.parametrize(
        "overrides",
        [
            {"variants": ["not", "a", "mapping"]},
            {"vocabulary": {"not": "a list"}},
            {"scan": ["not a mapping"]},
            {"scan": {"direction": "not a list"}},
            {"scan": {"direction": [{"words": [], "value": "+"}]}},
            {"scan": {"direction": [{"value": "+"}]}},
            {"phrases": {"tek": "kelime"}},
        ],
    )
    def test_malformed_sections_are_refused(
        self, tmp_path: Path, overrides: dict[str, Any]
    ) -> None:
        with pytest.raises(InventoryError):
            self._lexicon(tmp_path, **overrides)

    def test_scan_rules_are_longest_first(self) -> None:
        rules = load_lexicon().scan["direction"]
        assert [len(rule.words) for rule in rules] == sorted(
            (len(rule.words) for rule in rules), reverse=True
        )

    def test_split_suffix_leaves_an_unknown_word_alone(self) -> None:
        assert load_lexicon().split_suffix("zırzop") == ("zırzop", None)

    def test_split_suffix_refuses_a_too_short_stem(self) -> None:
        assert load_lexicon().split_suffix("ini") == ("ini", None)

    def test_fold_phrases_leaves_single_tokens_alone(self) -> None:
        lexicon = load_lexicon()
        assert lexicon.fold_phrases(["x", "kare"]) == [("x", 0, 0), ("kare", 1, 1)]


class TestContextDataclass:
    def test_defaults(self) -> None:
        context = Context()
        assert context.last_latex is None
        assert context.active_var is None
        assert context.modality == "text"
