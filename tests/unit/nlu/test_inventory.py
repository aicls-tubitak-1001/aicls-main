"""The packaged command inventory: schema, self-consistency and worked examples."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aicls.nlu.errors import InventoryError
from aicls.nlu.inventory import build_inventory, load_inventory
from aicls.nlu.inventory.schema import (
    MIN_EXAMPLES_PER_INTENT,
    Inventory,
    example_texts,
    slot_type_of,
)
from aicls.nlu.lexicon import load_lexicon
from aicls.nlu.resources import COMMANDS_RESOURCE, load_yaml, resource_path

#: The seventeen v1 intents, exactly as ADR-0005 lists them.
EXPECTED_INTENTS = (
    "differentiate",
    "integrate_indefinite",
    "integrate_definite",
    "limit",
    "solve",
    "simplify",
    "expand",
    "factor",
    "series",
    "evaluate_at",
    "plot",
    "book_fetch",
    "book_search",
    "ui_clear",
    "ui_undo",
    "ui_repeat_last",
    "ui_export",
)


@pytest.fixture(scope="module")
def inventory() -> Inventory:
    return load_inventory()


class TestStructure:
    def test_ships_as_a_package_resource(self) -> None:
        path = resource_path(COMMANDS_RESOURCE)
        assert path.is_file()
        assert path.parent.name == "inventory"

    def test_holds_the_seventeen_v1_intents(self, inventory: Inventory) -> None:
        assert inventory.intent_ids == EXPECTED_INTENTS

    def test_inventory_each_intent_has_at_least_10_examples(self, inventory: Inventory) -> None:
        assert inventory.check_example_counts() == []
        for intent in inventory.intents:
            assert len(intent.examples) >= MIN_EXAMPLES_PER_INTENT, intent.id

    def test_every_intent_has_patterns_and_a_label(self, inventory: Inventory) -> None:
        for intent in inventory.intents:
            assert intent.patterns, intent.id
            assert intent.label_tr, intent.id

    def test_slot_types_cover_the_documented_set(self, inventory: Inventory) -> None:
        assert set(inventory.slot_types) == {
            "EXPR",
            "VAR",
            "BOUNDS",
            "POINT",
            "DIRECTION",
            "ORDER",
            "UNIT_TYPE",
            "QUERY",
        }

    def test_cas_intents_name_a_real_mathengine_op(self, inventory: Inventory) -> None:
        from aicls.mathengine.ops import SUPPORTED_OPS

        for intent in inventory.intents:
            if intent.kind == "cas":
                assert intent.cas_op in SUPPORTED_OPS, intent.id

    def test_every_required_slot_has_a_turkish_question(self, inventory: Inventory) -> None:
        for intent in inventory.intents:
            for name in intent.required_slots:
                assert intent.clarify_tr.get(name), f"{intent.id}.{name}"

    def test_slot_type_lookup(self, inventory: Inventory) -> None:
        differentiate = inventory.by_id["differentiate"]
        assert slot_type_of(inventory, differentiate, "expr").kind == "latex_expr"

    def test_core_tagged_subset_exists(self, inventory: Inventory) -> None:
        """İP-3 risk plan B: a ``core`` subset of the most-used commands must be identifiable."""
        core = [intent for intent in inventory.intents if "core" in intent.tags]
        assert len(core) >= 10
        assert example_texts(inventory, ["core"])


class TestExamples:
    def test_inventory_every_example_resolves_to_own_intent(self, inventory: Inventory) -> None:
        """Every worked example must come back as its own intent with its declared slots."""
        assert inventory.validate_examples() == []

    def test_examples_are_unique(self, inventory: Inventory) -> None:
        seen: set[tuple[str, str]] = set()
        for intent_id, example in inventory.examples():
            key = (example.text.casefold(), str(sorted(example.context.items())))
            assert key not in seen, f"{intent_id}: duplicate example {example.text!r}"
            seen.add(key)


class TestLexiconConsistency:
    def test_every_keyword_stem_is_in_the_lexicon_vocabulary(self, inventory: Inventory) -> None:
        """A keyword the normaliser cannot produce would never fire."""
        vocabulary = load_lexicon().vocabulary
        missing = {
            keyword
            for intent in inventory.intents
            for keyword in intent.keywords
            if keyword not in vocabulary
        }
        assert missing == set()

    def test_scan_slots_name_an_existing_scan_rule(self, inventory: Inventory) -> None:
        scan = load_lexicon().scan
        for intent in inventory.intents:
            for spec in intent.slots.values():
                if spec.fill == "scan" and spec.scan_key:
                    assert spec.scan_key in scan or spec.type == "ORDER"


class TestValidation:
    def test_unknown_slot_type_is_refused(self) -> None:
        data = {
            "version": "1.0",
            "slot_types": {"EXPR": {"kind": "latex_expr"}},
            "intents": [
                {
                    "id": "x",
                    "kind": "cas",
                    "cas_op": "diff",
                    "label_tr": "x",
                    "slots": {"expr": {"type": "QUERY"}},
                }
            ],
        }
        with pytest.raises(InventoryError):
            build_inventory(data)

    def test_cas_intent_without_op_is_refused(self) -> None:
        data = {
            "version": "1.0",
            "slot_types": {},
            "intents": [{"id": "x", "kind": "cas", "label_tr": "x"}],
        }
        with pytest.raises(InventoryError):
            build_inventory(data)

    def test_required_slot_without_a_question_is_refused(self) -> None:
        data = {
            "version": "1.0",
            "slot_types": {"EXPR": {"kind": "latex_expr"}},
            "intents": [
                {
                    "id": "x",
                    "kind": "cas",
                    "cas_op": "diff",
                    "label_tr": "x",
                    "slots": {"expr": {"type": "EXPR", "required": True}},
                }
            ],
        }
        with pytest.raises(InventoryError):
            build_inventory(data)

    def test_duplicate_intent_id_is_refused(self) -> None:
        intent = {"id": "x", "kind": "ui", "ui_action": "clear", "label_tr": "x"}
        with pytest.raises(InventoryError):
            build_inventory({"version": "1.0", "slot_types": {}, "intents": [intent, intent]})

    def test_broken_pattern_is_refused_at_load_time(self) -> None:
        data = {
            "version": "1.0",
            "slot_types": {},
            "intents": [
                {
                    "id": "x",
                    "kind": "ui",
                    "ui_action": "clear",
                    "label_tr": "x",
                    "patterns": ["<NOPE> temizle"],
                }
            ],
        }
        with pytest.raises(InventoryError):
            build_inventory(data)

    def test_non_mapping_yaml_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "commands.yaml"
        path.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(InventoryError):
            load_yaml(path)

    def test_invalid_yaml_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "commands.yaml"
        path.write_text("a: [1,\n", encoding="utf-8")
        with pytest.raises(InventoryError):
            load_yaml(path)

    def test_missing_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(InventoryError):
            load_yaml(tmp_path / "absent.yaml")


class TestAlternativeInventory:
    def test_a_file_path_can_be_loaded(self, tmp_path: Path) -> None:
        source = resource_path(COMMANDS_RESOURCE).read_text(encoding="utf-8")
        data = yaml.safe_load(source)
        data["intents"] = data["intents"][:1]
        path = tmp_path / "small.yaml"
        path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        small = load_inventory(path)
        assert small.intent_ids == ("differentiate",)
