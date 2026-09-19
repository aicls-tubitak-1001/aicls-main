"""Contract tests for every boundary ``aicls.mathengine`` introduces.

Wave 1 lost a payload between two packages that each passed their own tests (KNOWN_ISSUES.md,
"Cross-module contract testing"), so the boundaries this package adds are pinned here rather
than in either side's unit tests:

* **mathengine -> latex2sympy** -- every ``Result.latex`` must be re-parsable, because the UI
  lets the user send a result straight back as the next input.
* **ocr -> mathengine** -- ``aicls.ocr.CURRICULUM_LATEX`` is what the handwriting model emits;
  each string must either compute or fail with a *named* ``AiclsError``. A raw exception here
  is a 500 in the pilot classroom.
* **mathengine -> api** -- the JSON body of ``POST /compute/latex`` and the ``ComputeOut.result``
  object of docs/mimari/api.md §3.4, plus the ``operations.result_kind`` enum of
  docs/mimari/veri-modeli.md.
* **mathengine -> core** -- the sanitiser is the *only* gate on LaTeX input, and the İP-4
  contract sentence is byte-for-byte the one in ``aicls.core.messages_tr``.
* **mathengine -> sandbox worker** -- what crosses the process pipe must survive the round trip
  unchanged, results and errors alike.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import sympy

from aicls.core import messages_tr
from aicls.core.errors import AiclsError, ErrorCode
from aicls.core.latex_safety import sanitize_latex
from aicls.latex2sympy import latex_to_sympy, sympy_to_latex
from aicls.mathengine import (
    COMMAND_ADAPTER,
    MAX_INPUT_CHARS,
    CasCommand,
    Diff,
    Integrate,
    MathEngine,
    Plot,
    Result,
    Solve,
    Step,
    UnsupportedProblem,
    symbolic_equivalent,
)
from aicls.mathengine.errors import error_to_payload, payload_to_error
from aicls.mathengine.ops import SUPPORTED_OPS
from aicls.ocr import CURRICULUM_LATEX

pytestmark = pytest.mark.integration

#: ``ComputeOut.result`` in docs/mimari/api.md §3.4, verbatim.
API_RESULT_EXAMPLE: dict[str, Any] = {
    "kind": "expr",
    "latex": "2 x",
    "plain": "2*x",
    "srepr": "Mul(Integer(2), Symbol('x'))",
    "steps": [],
    "plot_url": None,
    "cache_hit": False,
}

#: ``ComputeLatexIn.command`` in docs/mimari/api.md §3.4, verbatim.
API_COMMAND_EXAMPLE: dict[str, Any] = {"op": "diff", "expr": "x^{2}", "var": "x", "order": 1}

#: ``operations.result_kind ENUM(...)`` in docs/mimari/veri-modeli.md §2.
DB_RESULT_KINDS = ("expr", "number", "set", "bool", "plot", "steps")


# ---------------------------------------------------------------------------------------------
# mathengine -> api  (the payload shapes two agents are coding against in parallel)
# ---------------------------------------------------------------------------------------------
def test_the_documented_request_body_validates_as_a_command() -> None:
    command = COMMAND_ADAPTER.validate_python(API_COMMAND_EXAMPLE)
    assert isinstance(command, Diff)
    assert command.model_dump(mode="json") == API_COMMAND_EXAMPLE


def test_the_result_carries_every_field_the_documented_response_needs() -> None:
    payload = MathEngine().execute(Diff.model_validate(API_COMMAND_EXAMPLE)).model_dump(mode="json")
    for field in API_RESULT_EXAMPLE:
        if field == "plot_url":
            # The API derives ``plot_url`` from an asset it stores; the engine hands it bytes.
            assert "plot_png" in payload
            continue
        assert field in payload, field
    assert payload["kind"] == API_RESULT_EXAMPLE["kind"]
    assert payload["plain"] == API_RESULT_EXAMPLE["plain"]
    assert payload["cache_hit"] is False


def test_every_result_kind_is_a_value_the_database_enum_accepts() -> None:
    kinds = Result.model_fields["kind"].annotation
    assert set(getattr(kinds, "__args__", ())) == set(DB_RESULT_KINDS)


def test_the_command_union_covers_exactly_the_documented_ops() -> None:
    schema = COMMAND_ADAPTER.json_schema()
    mapping = schema["discriminator"]["mapping"]
    assert set(mapping) == set(SUPPORTED_OPS)
    assert schema["discriminator"]["propertyName"] == "op"


def test_a_command_survives_a_json_round_trip_through_the_wire() -> None:
    for command in (
        Diff(op="diff", expr="x^{2}"),
        Integrate(op="integrate", expr="x^{2}", lower="0", upper=r"\infty"),
        Solve(op="solve", equation="x^{2}-4=0", domain="complexes"),
    ):
        wire = json.dumps(command.model_dump(mode="json"))
        restored: CasCommand = COMMAND_ADAPTER.validate_python(json.loads(wire))
        assert restored == command


def test_the_step_shape_the_web_client_assumes_is_the_one_we_emit() -> None:
    """KNOWN_ISSUES: ``web/`` provisionally reads ``{title_tr, latex}``; mathengine owns Step."""
    result = MathEngine().execute(Diff(op="diff", expr=r"x\sin(x)"))
    assert result.steps
    for step in result.steps:
        payload = step.model_dump(mode="json")
        assert set(payload) >= {"title_tr", "latex"}
        assert isinstance(payload["title_tr"], str) and payload["title_tr"]
        assert isinstance(payload["latex"], str) and payload["latex"]
    assert set(Step.model_fields) == {"title_tr", "latex", "rule", "detail_tr"}


# ---------------------------------------------------------------------------------------------
# mathengine -> latex2sympy  (results must re-enter the grammar they came from)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "command",
    [
        Diff(op="diff", expr="x^{2}"),
        Diff(op="diff", expr=r"x\sin(x)"),
        Diff(op="diff", expr=r"e^{x}\cos(x)"),
        Integrate(op="integrate", expr="x^{2}"),
        Integrate(op="integrate", expr="x^{2}", lower="0", upper="1"),
        Integrate(op="integrate", expr=r"\frac{1}{x}"),
    ],
    ids=lambda command: command.op + ":" + getattr(command, "expr", command.op),
)
def test_expression_results_are_parsable_again(command: CasCommand) -> None:
    result = MathEngine().execute(command)
    assert result.kind in {"expr", "number"}
    reparsed = latex_to_sympy(result.latex)
    assert symbolic_equivalent(reparsed, sympy.sympify(result.srepr)) == "equal"


def test_set_valued_results_are_display_only_in_v1() -> None:
    """A known, deliberate gap between the printer and the grammar -- pinned, not hidden.

    ``sympy_to_latex`` renders a solution set as ``\\left\\{-2, 2\\right\\}``, and the v1 Lark
    grammar has no set literal, so a set answer cannot be fed back in as the next input the way
    an expression can. Two consequences the other packages must honour, which is why this test
    exists rather than a comment:

    * ``aicls.api`` / ``web`` must not offer "use this result as the next input" for
      ``kind == "set"``;
    * ``aicls.eval`` must compare set answers through ``srepr``, not by re-parsing ``latex``.

    Closing the gap means adding set literals to ``grammar/latex_math.lark`` (owner:
    latex2sympy); when that lands, this test should be replaced by the parametrised one above.
    """
    result = MathEngine().execute(Solve(op="solve", equation="x^{2}-4=0"))
    assert result.kind == "set"
    assert result.latex == r"\left\{-2, 2\right\}"
    with pytest.raises(AiclsError) as excinfo:
        latex_to_sympy(result.latex)
    assert excinfo.value.code in {ErrorCode.PARSE_ERROR, ErrorCode.UNSUPPORTED_CONSTRUCT}
    # ``srepr`` is the machine-readable channel that *does* round-trip for every kind.
    assert sympy.sympify(result.srepr) == sympy.FiniteSet(-2, 2)


def test_result_latex_is_already_sanitised() -> None:
    """The API renders ``Result.latex`` through KaTeX without another pass over the sanitiser."""
    for command in (Diff(op="diff", expr="x^{2}"), Solve(op="solve", equation="x^{2}-4=0")):
        latex = MathEngine().execute(command).latex
        assert sanitize_latex(latex) == latex


def test_steps_latex_is_also_sanitised() -> None:
    result = MathEngine().execute(Diff(op="diff", expr=r"x\cos(x)"))
    for step in result.steps:
        assert sanitize_latex(step.latex) == step.latex


# ---------------------------------------------------------------------------------------------
# ocr -> mathengine  (what the handwriting model emits must never crash the CAS)
# ---------------------------------------------------------------------------------------------
def test_every_ocr_curriculum_string_yields_a_result_or_a_named_error() -> None:
    """A raw exception on this path is a 500; only ``AiclsError`` is an acceptable failure."""
    engine = MathEngine()
    outcomes: dict[str, str] = {}
    for latex in CURRICULUM_LATEX:
        expression = latex.split("=")[0] if "=" in latex else latex
        try:
            engine.execute(Diff(op="diff", expr=expression))
            outcomes[latex] = "computed"
        except AiclsError as exc:
            outcomes[latex] = exc.code.value
    assert len(outcomes) == len(CURRICULUM_LATEX)
    assert all(
        value == "computed"
        or value
        in {
            ErrorCode.PARSE_ERROR.value,
            ErrorCode.UNSUPPORTED_CONSTRUCT.value,
            ErrorCode.UNSUPPORTED_PROBLEM.value,
            ErrorCode.VALIDATION_ERROR.value,
        }
        for value in outcomes.values()
    ), outcomes
    assert sum(value == "computed" for value in outcomes.values()) >= len(CURRICULUM_LATEX) // 2


def test_the_known_parser_gap_is_a_named_error_not_a_crash() -> None:
    """KNOWN_ISSUES #1: ``\\frac{\\partial f}{\\partial x}`` does not parse yet (latex2sympy)."""
    engine = MathEngine()
    try:
        engine.execute(Diff(op="diff", expr=r"\frac{\partial f}{\partial x}"))
    except AiclsError as exc:
        assert exc.code in {ErrorCode.PARSE_ERROR, ErrorCode.UNSUPPORTED_CONSTRUCT}
        assert exc.http_status == 422
        assert exc.message_tr  # the user is told something in Turkish, not shown a stack trace


# ---------------------------------------------------------------------------------------------
# mathengine -> core  (sanitiser and the frozen İP-4 sentence)
# ---------------------------------------------------------------------------------------------
def test_command_fields_go_through_the_core_sanitiser() -> None:
    hostile = r"x^{2}\href{javascript:alert(1)}{y}\def\z{bad}"
    assert Diff(op="diff", expr=hostile).expr == sanitize_latex(hostile)


def test_the_input_limit_is_the_core_latex_limit() -> None:
    from aicls.core.latex_safety import MAX_LATEX_LENGTH

    assert MAX_INPUT_CHARS == MAX_LATEX_LENGTH


def test_the_unsupported_sentence_is_the_one_core_owns() -> None:
    with pytest.raises(UnsupportedProblem) as excinfo:
        MathEngine().execute(Solve(op="solve", equation=r"\cos(x)=x"))
    assert excinfo.value.message_tr is messages_tr.MSG_UNSUPPORTED_PROBLEM
    assert excinfo.value.to_problem()["type"].endswith("/UNSUPPORTED_PROBLEM")


# ---------------------------------------------------------------------------------------------
# mathengine -> sandbox worker  (what crosses the process pipe)
# ---------------------------------------------------------------------------------------------
def test_a_result_survives_the_worker_serialisation_round_trip() -> None:
    """``PoolExecutor`` ships ``model_dump()`` over a pipe and rebuilds with ``model_validate``."""
    original = MathEngine().execute(Plot(op="plot", expr="x^{2}", xmin=-2.0, xmax=2.0))
    restored = Result.model_validate(original.model_dump())
    assert restored == original
    assert restored.plot_png == original.plot_png


def test_an_error_survives_the_worker_serialisation_round_trip() -> None:
    try:
        MathEngine().execute(Solve(op="solve", equation=r"\cos(x)=x"))
    except UnsupportedProblem as exc:
        restored = payload_to_error(error_to_payload(exc))
        assert isinstance(restored, UnsupportedProblem)
        assert restored.message_tr == exc.message_tr
        assert restored.details == exc.details
        assert restored.http_status == exc.http_status
    else:  # pragma: no cover - the command above always fails
        pytest.fail("expected UnsupportedProblem")


def test_the_printer_contract_holds_for_every_result_kind() -> None:
    """``plain``/``srepr``/``latex`` are three renderings of one value, never of three."""
    engine = MathEngine()
    for command in (
        Diff(op="diff", expr="x^{2}"),
        Integrate(op="integrate", expr="x^{2}", lower="0", upper="1"),
        Solve(op="solve", equation="x^{2}-4=0"),
    ):
        result = engine.execute(command)
        value = sympy.sympify(result.srepr)
        assert sympy.sstr(value) == result.plain
        assert sympy_to_latex(value) == result.latex
