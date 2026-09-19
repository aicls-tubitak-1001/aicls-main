"""Typed command union, ``LatexStr`` sanitising and the ``Result``/``Step`` payloads.

The union is a contract: ``aicls.nlu`` builds these models and ``aicls.api`` serialises them
straight into ``ComputeOut`` (docs/mimari/api.md §3.4), so the field names, defaults and the
``op`` discriminator are pinned here.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import TypeAdapter, ValidationError
import pytest

from aicls.core.errors import ErrorCode, InputValidationError
from aicls.mathengine import (
    MAX_INPUT_CHARS,
    CasCommand,
    Diff,
    Evaluate,
    Expand,
    Factor,
    Integrate,
    Limit,
    Plot,
    Result,
    Series,
    Simplify,
    Solve,
    Step,
    canonical_key,
    command_input_chars,
)

ADAPTER: TypeAdapter[CasCommand] = TypeAdapter(CasCommand)


# ---------------------------------------------------------------------------------------------
# discriminated union
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"op": "diff", "expr": "x^{2}"}, Diff),
        ({"op": "integrate", "expr": "x^{2}", "lower": "0", "upper": "1"}, Integrate),
        ({"op": "limit", "expr": r"\frac{\sin x}{x}", "to": "0"}, Limit),
        ({"op": "solve", "equation": "x^{2}-4=0"}, Solve),
        ({"op": "simplify", "expr": "x+x"}, Simplify),
        ({"op": "expand", "expr": "(x+1)^{2}"}, Expand),
        ({"op": "factor", "expr": "x^{2}-1"}, Factor),
        ({"op": "series", "expr": "e^{x}"}, Series),
        ({"op": "evaluate", "expr": "x^{2}", "subs": {"x": "3"}}, Evaluate),
        ({"op": "plot", "expr": "x^{2}"}, Plot),
    ],
)
def test_discriminator_selects_the_right_model(payload: dict[str, Any], expected: type) -> None:
    command = ADAPTER.validate_python(payload)
    assert isinstance(command, expected)
    assert command.op == payload["op"]


def test_defaults_match_the_documented_contract() -> None:
    assert Diff(op="diff", expr="x^{2}").model_dump() == {
        "op": "diff",
        "expr": "x^{2}",
        "var": "x",
        "order": 1,
    }
    integrate = Integrate(op="integrate", expr="x^{2}")
    assert (integrate.var, integrate.lower, integrate.upper) == ("x", None, None)
    limit = Limit(op="limit", expr="x", to="0")
    assert (limit.var, limit.direction) == ("x", "+-")
    solve = Solve(op="solve", equation="x=1")
    assert (solve.var, solve.domain) == ("x", "reals")
    series = Series(op="series", expr="e^{x}")
    assert (series.var, series.point, series.order) == ("x", "0", 6)
    plot = Plot(op="plot", expr="x")
    assert (plot.var, plot.xmin, plot.xmax) == ("x", -10.0, 10.0)


def test_unknown_op_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ADAPTER.validate_python({"op": "integrate_by_parts", "expr": "x"})


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Diff.model_validate({"op": "diff", "expr": "x", "eval": "__import__"})


def test_commands_are_frozen() -> None:
    command = Diff(op="diff", expr="x^{2}")
    with pytest.raises(ValidationError):
        command.expr = "y"  # type: ignore[misc]  # frozen model, checked at runtime


# ---------------------------------------------------------------------------------------------
# LatexStr sanitising and the 2048-character contract limit
# ---------------------------------------------------------------------------------------------
def test_latex_fields_are_sanitised_with_core_policy() -> None:
    command = Diff(op="diff", expr=r"$x^{2}\href{javascript:alert(1)}{y}$")
    assert "href" not in command.expr
    assert command.expr.startswith("x^{2}")


def test_empty_latex_is_rejected() -> None:
    with pytest.raises(InputValidationError):
        Diff(op="diff", expr="   ")


def test_input_over_2048_chars_rejected() -> None:
    """ADR-0016: a single LaTeX field above ``MAX_INPUT_CHARS`` never reaches SymPy."""
    too_long = "x+" * (MAX_INPUT_CHARS // 2) + "x"
    assert len(too_long) > MAX_INPUT_CHARS
    with pytest.raises(InputValidationError) as excinfo:
        Diff(op="diff", expr=too_long)
    assert excinfo.value.code is ErrorCode.VALIDATION_ERROR
    assert str(MAX_INPUT_CHARS) in excinfo.value.message_tr
    assert excinfo.value.details["max_len"] == MAX_INPUT_CHARS


def test_command_input_chars_sums_every_latex_field() -> None:
    command = Integrate(op="integrate", expr="x^{2}", lower="0", upper="1")
    assert command_input_chars(command) == len("x^{2}") + 1 + 1
    subs = Evaluate(op="evaluate", expr="x+y", subs={"x": "1", "y": "2"})
    assert command_input_chars(subs) == len("x+y") + 2


def test_a_non_string_latex_field_is_rejected_by_pydantic() -> None:
    with pytest.raises(ValidationError):
        Diff.model_validate({"op": "diff", "expr": 42})


def test_integration_bounds_come_in_pairs() -> None:
    with pytest.raises(ValidationError):
        Integrate(op="integrate", expr="x", lower="0")
    with pytest.raises(ValidationError):
        Integrate(op="integrate", expr="x", upper="1")
    assert Integrate(op="integrate", expr="x", lower="0", upper="1").lower == "0"


def test_is_valid_var_matches_the_field_constraint() -> None:
    from aicls.mathengine.models import is_valid_var

    assert is_valid_var("x") is True
    assert is_valid_var("theta") is True
    assert is_valid_var("x_{12}") is True
    assert is_valid_var("2x") is False
    assert is_valid_var("x; drop") is False


def test_variable_names_are_restricted() -> None:
    assert Diff(op="diff", expr="x", var="theta").var == "theta"
    assert Diff(op="diff", expr="x", var="x_{1}").var == "x_{1}"
    with pytest.raises(ValidationError):
        Diff(op="diff", expr="x", var="x; import os")


def test_diff_order_and_series_order_are_bounded() -> None:
    with pytest.raises(ValidationError):
        Diff(op="diff", expr="x", order=0)
    with pytest.raises(ValidationError):
        Series(op="series", expr="x", order=0)


def test_plot_range_must_be_increasing_and_finite() -> None:
    with pytest.raises(ValidationError):
        Plot(op="plot", expr="x", xmin=5.0, xmax=-5.0)
    with pytest.raises(ValidationError):
        Plot(op="plot", expr="x", xmin=float("nan"), xmax=1.0)


def test_evaluate_subs_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Evaluate(op="evaluate", expr="x", subs={f"x{i}": "1" for i in range(64)})
    with pytest.raises(ValidationError):
        Evaluate(op="evaluate", expr="x", subs={})


# ---------------------------------------------------------------------------------------------
# canonical cache key
# ---------------------------------------------------------------------------------------------
def test_canonical_key_is_stable_and_order_independent() -> None:
    first = Evaluate(op="evaluate", expr="x+y", subs={"x": "1", "y": "2"})
    second = Evaluate(op="evaluate", expr="x+y", subs={"y": "2", "x": "1"})
    assert canonical_key(first) == canonical_key(second)
    assert json.loads(canonical_key(first))["op"] == "evaluate"


def test_canonical_key_separates_defaults_from_explicit_values() -> None:
    assert canonical_key(Diff(op="diff", expr="x^{2}")) == canonical_key(
        Diff(op="diff", expr="x^{2}", var="x", order=1)
    )
    assert canonical_key(Diff(op="diff", expr="x^{2}", order=2)) != canonical_key(
        Diff(op="diff", expr="x^{2}")
    )


# ---------------------------------------------------------------------------------------------
# Result / Step
# ---------------------------------------------------------------------------------------------
def test_result_json_shape_matches_api_documentation() -> None:
    result = Result(kind="expr", latex=r"2 \cdot x", plain="2*x", srepr="Mul(Integer(2), x)")
    payload = result.model_dump(mode="json")
    for field in ("kind", "latex", "plain", "srepr", "steps", "plot_png", "cache_hit", "timings"):
        assert field in payload
    assert payload["steps"] == []
    assert payload["plot_png"] is None
    assert payload["cache_hit"] is False


def test_step_keeps_the_shape_the_web_client_assumes() -> None:
    step = Step(title_tr="Çarpım kuralı", latex="u'v + uv'")
    payload = step.model_dump(mode="json")
    assert payload["title_tr"] == "Çarpım kuralı"
    assert payload["latex"] == "u'v + uv'"
    assert payload["rule"] == ""


def test_result_is_immutable() -> None:
    result = Result(kind="number", latex=r"\frac{1}{3}", plain="1/3", srepr="Rational(1, 3)")
    with pytest.raises(ValidationError):
        result.latex = "0"  # type: ignore[misc]  # frozen model, checked at runtime
