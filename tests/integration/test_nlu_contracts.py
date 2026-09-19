"""Contract tests for every boundary ``aicls.nlu`` introduces.

Wave 1 lost a payload between two packages that each passed their own tests
(KNOWN_ISSUES.md, "Cross-module contract testing"), so the boundaries this package adds are
pinned here rather than in either side's unit tests:

* **nlu -> mathengine** -- a command the layer builds must be *executable*. It is not enough that
  the pydantic model validates: the LaTeX the spoken-math grammar produced has to survive
  ``parse_latex`` inside the engine and reach SymPy. This is the ASR → CAS handoff of the
  proposal's pipeline, and the one place a Turkish-side mistake becomes a wrong answer.
* **nlu -> latex2sympy** -- every LaTeX-bearing slot value must be re-parsable and unchanged by
  ``sanitize_latex``; the layer must never hand the CAS a string outside the KaTeX allow-list.
* **nlu -> book** -- ``aicls.nlu.types.UnitType`` is a *copy* of ``aicls.book.schema.UnitType``
  (ADR-0002 forbids the import). The two must not drift, or a ``BookFetch`` will name a unit type
  the book resolver does not know.
* **nlu -> api** -- ``Matched.command`` is what ``POST /compute/latex`` receives; the CAS members
  must round-trip through ``COMMAND_ADAPTER``, and ``BookFetch``/``UiAction`` must serialise to
  their documented shapes.
* **nlu -> asr / collect** -- ``datasets/nlu/reading_scenarios.yaml`` is the reading-scenario
  library. Its rows must fit ``aicls.db.models.audio.PromptSentence`` and must cover the twelve
  sentences currently hard-coded in ``aicls.asr.stub.STUB_SENTENCES``, which is the migration
  this module unblocks.
* **nlu -> layering** -- the package must not reach ``aicls.db`` or ``aicls.book`` at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from aicls.book.schema import UnitType as BookUnitType
from aicls.core.errors import AiclsError
from aicls.core.latex_safety import sanitize_latex
from aicls.latex2sympy import parse_latex
from aicls.mathengine import COMMAND_ADAPTER, InProcessExecutor, MathEngine
from aicls.mathengine.models import Plot
from aicls.nlu import (
    BookFetch,
    Matched,
    ParseContext,
    UiAction,
    UnitType,
    load_inventory,
    parse_command,
)
from aicls.nlu.inventory.schema import Inventory

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
READING_SCENARIOS = REPO_ROOT / "datasets" / "nlu" / "reading_scenarios.yaml"
COMMANDS_TEST = REPO_ROOT / "datasets" / "nlu" / "commands_test.jsonl"

#: Columns of ``aicls.db.models.audio.PromptSentence`` the library must be able to fill.
PROMPT_COLUMNS = ("text_tr", "scenario_tag", "intent", "difficulty")


@pytest.fixture(scope="module")
def inventory() -> Inventory:
    return load_inventory()


@pytest.fixture(scope="module")
def engine() -> MathEngine:
    return MathEngine(executor=InProcessExecutor(), timeout_s=20.0)


def _first_example(inventory: Inventory, intent_id: str) -> tuple[str, ParseContext | None]:
    example = inventory.by_id[intent_id].examples[0]
    return example.text, (ParseContext(**example.context) if example.context else None)


class TestMathEngineBoundary:
    @pytest.mark.parametrize(
        "intent_id",
        [
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
        ],
    )
    def test_every_cas_intent_produces_an_executable_command(
        self, inventory: Inventory, engine: MathEngine, intent_id: str
    ) -> None:
        """Turkish in, a real SymPy answer out -- the whole point of the layer."""
        text, context = _first_example(inventory, intent_id)
        outcome = parse_command(text, context=context)
        assert isinstance(outcome, Matched), (intent_id, outcome)
        result = engine.execute(outcome.command)  # type: ignore[arg-type]  # CAS intents only
        assert result.latex
        assert result.kind in {"expr", "number", "set", "bool", "plot", "steps"}

    def test_the_proposal_example_computes(self, engine: MathEngine) -> None:
        """ "x karenin türevini al" must end at ``2 x``, computed by SymPy."""
        outcome = parse_command("x karenin türevini al")
        assert isinstance(outcome, Matched)
        assert engine.execute(outcome.command).plain == "2*x"  # type: ignore[arg-type]

    def test_a_definite_integral_computes_its_bounds(self, engine: MathEngine) -> None:
        outcome = parse_command("sıfırdan bire kadar x karenin integralini hesapla")
        assert isinstance(outcome, Matched)
        assert engine.execute(outcome.command).plain == "1/3"  # type: ignore[arg-type]

    def test_a_limit_at_infinity_computes(self, engine: MathEngine) -> None:
        outcome = parse_command("x sonsuza giderken bir bölü x'in limiti")
        assert isinstance(outcome, Matched)
        assert engine.execute(outcome.command).plain == "0"  # type: ignore[arg-type]

    def test_plot_commands_stay_inside_the_engine_contract(self, inventory: Inventory) -> None:
        outcome = parse_command("x karenin grafiğini çiz")
        assert isinstance(outcome, Matched)
        assert isinstance(outcome.command, Plot)
        assert outcome.command.xmin < outcome.command.xmax

    def test_every_inventory_example_either_computes_or_fails_by_name(
        self, inventory: Inventory, engine: MathEngine
    ) -> None:
        """No raw exception may cross the boundary: a 500 in the pilot classroom is a failure."""
        for intent_id, example in inventory.examples():
            if inventory.by_id[intent_id].kind != "cas" or intent_id == "plot":
                continue
            context = ParseContext(**example.context) if example.context else None
            outcome = parse_command(example.text, context=context)
            assert isinstance(outcome, Matched)
            try:
                engine.execute(outcome.command)  # type: ignore[arg-type]
            except AiclsError:
                continue


class TestDerivativeOrderBoundary:
    """The KNOWN_ISSUES PRIORITY 1 boundary, asserted where it actually hurts: the answer.

    A slot value can look plausible and still be the wrong question. These cases pin the number
    SymPy returns end to end, because that number is what the classroom sees, and "3x²" versus
    "6" is the whole difference between a working platform and a hallucinating one.
    """

    @pytest.mark.parametrize(
        ("text", "plain"),
        [
            ("x üzeri üçün türevini al", "3*x**2"),
            ("x üzeri beşin türevini al", "5*x**4"),
            ("x üzeri üç türevini al", "3*x**2"),
            ("x üzeri üç ifadesinin türevini al", "3*x**2"),
            ("x üzeri üçün ikinci türevini al", "6*x"),
            ("x karenin türevini al", "2*x"),
            ("x küpünün ikinci türevini al", "6*x"),
        ],
    )
    def test_the_exponent_never_becomes_the_derivative_order(
        self, engine: MathEngine, text: str, plain: str
    ) -> None:
        outcome = parse_command(text)
        assert isinstance(outcome, Matched), outcome
        assert engine.execute(outcome.command).plain == plain  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("text", "plain"),
        [
            ("x üzeri üçün integralini al", "x**4/4"),
            ("sıfırdan bire kadar x üzeri üçün integralini hesapla", "1/4"),
            ("x sıfıra giderken x üzeri üçün limitini bul", "0"),
            ("x üzeri üç eşittir sekiz denklemini çöz", "{2}"),
        ],
    )
    def test_the_fix_does_not_reach_into_the_sibling_intents(
        self, engine: MathEngine, text: str, plain: str
    ) -> None:
        outcome = parse_command(text)
        assert isinstance(outcome, Matched), outcome
        assert engine.execute(outcome.command).plain == plain  # type: ignore[arg-type]

    def test_a_dropped_token_never_reaches_the_engine(self) -> None:
        """``"x kare eksi dört sıfır denklemini çöz"`` must ask, not solve ``x² − 4``."""
        outcome = parse_command("x kare eksi dört sıfır denklemini çöz")
        assert not isinstance(outcome, Matched), outcome

    def test_every_gold_row_that_declares_an_order_computes_that_order(
        self, engine: MathEngine
    ) -> None:
        """Every ``differentiate`` gold row, executed: the order in the row is the order run."""
        import sympy

        from aicls.latex2sympy import latex_to_sympy

        with COMMANDS_TEST.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        checked = 0
        for row in rows:
            if row["intent"] != "differentiate" or "order" not in (row.get("slots") or {}):
                continue
            context = ParseContext(**row["context"]) if row.get("context") else None
            outcome = parse_command(row["text"], context=context)
            assert isinstance(outcome, Matched), row["text"]
            expected = sympy.diff(
                latex_to_sympy(row["slots"]["expr"]),
                sympy.Symbol(row["slots"].get("var", "x"), real=True),
                row["slots"]["order"],
            )
            result = engine.execute(outcome.command)  # type: ignore[arg-type]
            # Re-sympify through ``str`` so both sides use assumption-free symbols: the engine
            # declares ``Symbol('x', real=True)`` and ``x_real - x`` does not cancel.
            difference = sympy.sympify(result.plain) - sympy.sympify(str(expected))
            assert sympy.simplify(difference) == 0, row["text"]
            checked += 1
        assert checked >= 20, checked


class TestLatexBoundary:
    def test_every_latex_slot_is_reparsable_and_katex_safe(self, inventory: Inventory) -> None:
        for intent_id, example in inventory.examples():
            context = ParseContext(**example.context) if example.context else None
            outcome = parse_command(example.text, context=context)
            assert isinstance(outcome, Matched), (intent_id, example.text)
            for name, value in outcome.slots.items():
                if not isinstance(value, str) or name in {"var", "query", "unit_type"}:
                    continue
                if not any(
                    character in value for character in "\\^{}0123456789abcdefghijklmnopqrstuvwxyz"
                ):
                    continue
                parse_latex(value)
                assert sanitize_latex(value) == value, (intent_id, name, value)


class TestBookBoundary:
    def test_unit_type_mirrors_the_book_schema(self) -> None:
        """ADR-0002 forbids importing ``aicls.book``; the copy must not drift."""
        assert {item.value for item in UnitType} == {item.value for item in BookUnitType}
        assert [item.name for item in UnitType] == [item.name for item in BookUnitType]

    def test_book_fetch_names_a_unit_type_the_book_package_knows(self) -> None:
        outcome = parse_command("türev tanımını getir")
        assert isinstance(outcome, Matched)
        command = outcome.command
        assert isinstance(command, BookFetch)
        assert command.unit_type is not None
        assert BookUnitType(command.unit_type.value) is BookUnitType.DEFINITION

    def test_book_search_leaves_the_unit_type_open(self) -> None:
        outcome = parse_command("kitapta türev ara")
        assert isinstance(outcome, Matched)
        assert isinstance(outcome.command, BookFetch)
        assert outcome.command.unit_type is None


class TestApiBoundary:
    def test_cas_commands_round_trip_through_the_api_adapter(self, inventory: Inventory) -> None:
        for intent_id, example in inventory.examples():
            if inventory.by_id[intent_id].kind != "cas":
                continue
            context = ParseContext(**example.context) if example.context else None
            outcome = parse_command(example.text, context=context)
            assert isinstance(outcome, Matched)
            payload = outcome.command.model_dump(mode="json")
            assert COMMAND_ADAPTER.validate_python(payload) == outcome.command

    def test_book_fetch_serialises_to_the_documented_shape(self) -> None:
        payload = BookFetch(unit_type=UnitType.THEOREM, query="rolle").model_dump(mode="json")
        assert payload == {"op": "book_fetch", "unit_type": "Theorem", "query": "rolle"}
        assert json.loads(json.dumps(payload)) == payload

    def test_ui_action_serialises_to_the_documented_shape(self) -> None:
        payload = UiAction(action="clear").model_dump(mode="json")
        assert payload == {"op": "ui", "action": "clear"}

    def test_every_outcome_is_json_serialisable(self, inventory: Inventory) -> None:
        outcome = parse_command("x karenin türevini al")
        assert isinstance(outcome, Matched)
        body = {
            "intent": outcome.intent,
            "score": outcome.score,
            "slots": outcome.slots,
            "trace": outcome.trace,
            "command": outcome.command.model_dump(mode="json"),
        }
        assert json.loads(json.dumps(body, ensure_ascii=False))["intent"] == "differentiate"


class TestReadingScenarioBoundary:
    @pytest.fixture(scope="class")
    def library(self) -> dict[str, Any]:
        return yaml.safe_load(READING_SCENARIOS.read_text(encoding="utf-8"))

    def test_rows_fit_the_prompt_sentence_table(self, library: dict[str, Any]) -> None:
        from aicls.db.models.audio import PromptSentence

        columns = {column.name for column in PromptSentence.__table__.columns}
        assert set(PROMPT_COLUMNS) <= columns
        for scenario in library["scenarios"]:
            assert isinstance(scenario["tag"], str) and len(scenario["tag"]) <= 40
            for sentence in scenario["sentences"]:
                assert 0 < len(sentence["text"]) <= 500
                assert 1 <= int(sentence.get("difficulty", 1)) <= 5

    def test_intents_are_inventory_intents(
        self, library: dict[str, Any], inventory: Inventory
    ) -> None:
        allowed = {*inventory.intent_ids, None, "none"}
        assert {scenario["intent"] for scenario in library["scenarios"]} <= allowed

    def test_library_covers_the_hard_coded_asr_stub_sentences(
        self, library: dict[str, Any]
    ) -> None:
        """``aicls.asr.stub.STUB_SENTENCES`` must be replaceable by this file (KNOWN_ISSUES)."""
        from aicls.asr.stub import STUB_SENTENCES

        library_texts = {
            sentence["text"]
            for scenario in library["scenarios"]
            for sentence in scenario["sentences"]
        }
        missing = [text for text in STUB_SENTENCES if text not in library_texts]
        assert missing == [], missing

    def test_library_is_large_enough_to_replace_the_stub_pool(
        self, library: dict[str, Any]
    ) -> None:
        from aicls.asr.stub import STUB_SENTENCES

        total = sum(len(scenario["sentences"]) for scenario in library["scenarios"])
        assert total > len(STUB_SENTENCES) * 10

    def test_gold_set_rows_reference_real_intents(self, inventory: Inventory) -> None:
        allowed = {*inventory.intent_ids, ""}
        with COMMANDS_TEST.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    assert json.loads(line)["intent"] in allowed


class TestLayering:
    def test_nlu_never_reaches_db_or_book(self) -> None:
        """ADR-0002: the book lookup is returned as a command, never executed here."""
        import grimp

        graph = grimp.build_graph("aicls", include_external_packages=False)
        for forbidden in ("aicls.db", "aicls.book"):
            assert graph.find_shortest_chain("aicls.nlu", forbidden) is None

    def test_nlu_never_reaches_torch(self) -> None:
        import grimp

        graph = grimp.build_graph("aicls", include_external_packages=True)
        assert graph.find_shortest_chain("aicls.nlu", "torch") is None
