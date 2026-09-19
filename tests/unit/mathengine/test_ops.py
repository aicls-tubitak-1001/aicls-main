"""The ten CAS operations, starting with the four worked examples of the proposal.

``test_diff_x_squared_equals_2x``, ``test_integrate_x_squared_0_1_equals_one_third``,
``test_solve_x_squared_minus_4_returns_minus2_and_2`` and
``test_limit_sin_x_over_x_at_0_is_1`` are the proposal's own examples (A.2.2) and are
contractual: they must keep passing under that name.
"""

from __future__ import annotations

import pytest
import sympy

from aicls.core import messages_tr
from aicls.core.errors import ErrorCode
from aicls.latex2sympy import ParseError
from aicls.mathengine import (
    Diff,
    Evaluate,
    Expand,
    Factor,
    Integrate,
    Limit,
    Plot,
    Series,
    Simplify,
    Solve,
    UnsupportedProblem,
    compute,
    symbolic_equivalent,
)
from aicls.mathengine.ops.common import build_context, classify, reject_unevaluated, sympy_guard


def _value(latex_result: str) -> sympy.Basic:
    from aicls.latex2sympy import latex_to_sympy

    return latex_to_sympy(latex_result)


# ---------------------------------------------------------------------------------------------
# the four proposal examples
# ---------------------------------------------------------------------------------------------
def test_diff_x_squared_equals_2x() -> None:
    result = compute(Diff(op="diff", expr="x^{2}", var="x"))
    assert result.kind == "expr"
    assert result.plain == "2*x"
    assert symbolic_equivalent(_value(result.latex), 2 * sympy.Symbol("x", real=True)) == "equal"
    assert "Symbol('x'" in result.srepr


def test_integrate_x_squared_0_1_equals_one_third() -> None:
    result = compute(Integrate(op="integrate", expr="x^{2}", var="x", lower="0", upper="1"))
    assert result.kind == "number"
    assert result.plain == "1/3"
    assert result.latex == r"\frac{1}{3}"
    assert sympy.sympify(result.plain) == sympy.Rational(1, 3)


def test_solve_x_squared_minus_4_returns_minus2_and_2() -> None:
    result = compute(Solve(op="solve", equation="x^{2}-4=0", var="x"))
    assert result.kind == "set"
    solutions = sympy.sympify(result.srepr.replace("Symbol('x', real=True)", "Symbol('x')"))
    assert set(solutions) == {sympy.Integer(-2), sympy.Integer(2)}
    assert result.plain == "{-2, 2}"


def test_limit_sin_x_over_x_at_0_is_1() -> None:
    result = compute(Limit(op="limit", expr=r"\frac{\sin x}{x}", var="x", to="0"))
    assert result.kind == "number"
    assert result.plain == "1"


# ---------------------------------------------------------------------------------------------
# differentiation
# ---------------------------------------------------------------------------------------------
def test_higher_order_derivative() -> None:
    result = compute(Diff(op="diff", expr="x^{4}", order=3))
    assert result.plain == "24*x"
    assert result.steps == []  # narration is first order only


def test_derivative_of_a_product_is_narrated() -> None:
    result = compute(Diff(op="diff", expr=r"x\sin(x)"))
    assert [step.rule for step in result.steps] == [
        "product_rule",
        "product_rule_applied",
        "result",
    ]
    assert result.steps[0].title_tr == "Çarpım kuralı"


def test_derivative_in_another_variable() -> None:
    result = compute(Diff(op="diff", expr="t^{3}", var="t"))
    assert result.plain == "3*t**2"


# ---------------------------------------------------------------------------------------------
# integration
# ---------------------------------------------------------------------------------------------
def test_indefinite_integral_has_no_constant_of_integration() -> None:
    result = compute(Integrate(op="integrate", expr="x^{2}"))
    assert result.plain == "x**3/3"
    assert "C" not in result.plain


def test_definite_integral_with_infinite_upper_bound() -> None:
    result = compute(Integrate(op="integrate", expr="e^{-x}", lower="0", upper=r"\infty"))
    assert result.plain == "1"


def test_definite_integral_between_two_infinities() -> None:
    result = compute(
        Integrate(op="integrate", expr="e^{-x^{2}}", lower=r"-\infty", upper=r"\infty")
    )
    assert result.latex == r"\sqrt{\pi}"


def test_unevaluated_integral_is_an_unsupported_problem() -> None:
    with pytest.raises(UnsupportedProblem) as excinfo:
        compute(Integrate(op="integrate", expr=r"\frac{\sin(x)}{\ln(x)}"))
    assert excinfo.value.details["reason"] == "unevaluated_integral"
    assert excinfo.value.message_tr == messages_tr.MSG_UNSUPPORTED_PROBLEM


# ---------------------------------------------------------------------------------------------
# limits
# ---------------------------------------------------------------------------------------------
def test_one_sided_limits_differ() -> None:
    right = compute(Limit(op="limit", expr=r"\frac{1}{x}", to="0", direction="+"))
    left = compute(Limit(op="limit", expr=r"\frac{1}{x}", to="0", direction="-"))
    assert right.plain == "oo"
    assert left.plain == "-oo"


def test_two_sided_limit_that_does_not_exist_is_unsupported() -> None:
    with pytest.raises(UnsupportedProblem) as excinfo:
        compute(Limit(op="limit", expr=r"\frac{\lvert x \rvert}{x}", to="0", direction="+-"))
    assert excinfo.value.details["reason"] == "limit_does_not_exist"


def test_a_value_error_that_is_not_a_missing_limit_stays_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only SymPy's "does not exist" wording gets the dedicated reason; the rest are generic."""

    def boom(*args: object, **kwargs: object) -> sympy.Basic:
        raise ValueError("something else went wrong")

    monkeypatch.setattr("sympy.limit", boom)
    with pytest.raises(UnsupportedProblem) as excinfo:
        compute(Limit(op="limit", expr="x", to="0"))
    assert excinfo.value.details["reason"] == "limit_failed"


def test_limit_at_infinity() -> None:
    result = compute(Limit(op="limit", expr=r"\frac{1}{x}", to=r"\infty"))
    assert result.plain == "0"


# ---------------------------------------------------------------------------------------------
# solving
# ---------------------------------------------------------------------------------------------
def test_solve_over_the_complexes_finds_imaginary_roots() -> None:
    real = compute(Solve(op="solve", equation="x^{2}+1=0", domain="reals"))
    assert real.plain in {"EmptySet", "{}"}
    complexes = compute(Solve(op="solve", equation="x^{2}+1=0", domain="complexes"))
    assert "I" in complexes.plain


def test_solve_accepts_a_bare_expression_as_equal_to_zero() -> None:
    result = compute(Solve(op="solve", equation="x-3"))
    assert result.plain == "{3}"


def test_solve_returning_a_condition_set_is_unsupported() -> None:
    with pytest.raises(UnsupportedProblem) as excinfo:
        compute(Solve(op="solve", equation=r"\cos(x)=x"))
    assert excinfo.value.details["reason"] == "condition_set"


# ---------------------------------------------------------------------------------------------
# rewrites
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (Simplify(op="simplify", expr=r"\sin^{2}(x)+\cos^{2}(x)"), "1"),
        (Expand(op="expand", expr="(x+1)^{2}"), "x**2 + 2*x + 1"),
        (Factor(op="factor", expr="x^{2}-1"), "(x - 1)*(x + 1)"),
    ],
)
def test_rewrites(command: Simplify | Expand | Factor, expected: str) -> None:
    assert compute(command).plain == expected


# ---------------------------------------------------------------------------------------------
# series, evaluate, plot
# ---------------------------------------------------------------------------------------------
def test_series_keeps_the_landau_term() -> None:
    result = compute(Series(op="series", expr="e^{x}", point="0", order=4))
    assert result.plain.startswith("1 + x + x**2/2 + x**3/6")
    assert "O(x**4)" in result.plain


def test_series_around_a_non_zero_point() -> None:
    result = compute(Series(op="series", expr=r"\ln(x)", var="x", point="1", order=3))
    assert "x - 1" in result.plain


def test_evaluate_substitutes_and_stays_exact() -> None:
    result = compute(Evaluate(op="evaluate", expr="x^{2}+1", subs={"x": "3"}))
    assert result.kind == "number"
    assert result.plain == "10"


def test_evaluate_with_a_symbolic_substitution() -> None:
    result = compute(Evaluate(op="evaluate", expr=r"\sin(x)", subs={"x": r"\frac{\pi}{2}"}))
    assert result.plain == "1"


def test_evaluate_leaving_a_free_symbol_is_an_expression() -> None:
    result = compute(Evaluate(op="evaluate", expr="x+y", subs={"x": "1"}))
    assert result.kind == "expr"
    assert result.plain == "y + 1"


def test_plot_result_carries_png_bytes_and_the_expression() -> None:
    result = compute(Plot(op="plot", expr="x^{2}", xmin=-2.0, xmax=2.0))
    assert result.kind == "plot"
    assert result.plot_png is not None
    assert result.plot_png[:8] == b"\x89PNG\r\n\x1a\n"
    assert result.plain == "x**2"


def test_plot_of_an_expression_with_a_second_symbol_is_unsupported() -> None:
    with pytest.raises(UnsupportedProblem) as excinfo:
        compute(Plot(op="plot", expr="x+y"))
    assert excinfo.value.details["reason"] == "free_symbols_in_plot"


# ---------------------------------------------------------------------------------------------
# error policy
# ---------------------------------------------------------------------------------------------
def test_unsupported_problem_returns_contract_turkish_message() -> None:
    """İP-4 risk plan: the exact sentence, and nothing about SymPy internals."""
    with pytest.raises(UnsupportedProblem) as excinfo:
        compute(Solve(op="solve", equation=r"\cos(x)=x"))
    error = excinfo.value
    assert error.message_tr == "Bu problem tipi şu an için desteklenmemektedir."
    assert error.message_tr == messages_tr.MSG_UNSUPPORTED_PROBLEM
    assert error.code is ErrorCode.UNSUPPORTED_PROBLEM
    problem = error.to_problem()
    assert problem["status"] == 422
    assert problem["detail_tr"] == messages_tr.MSG_UNSUPPORTED_PROBLEM
    assert problem["title"] == "Desteklenmeyen problem tipi"


def test_a_parse_error_is_not_flattened_into_unsupported() -> None:
    with pytest.raises(ParseError):
        compute(Diff(op="diff", expr="x^{"))


def test_sympy_guard_wraps_foreign_exceptions() -> None:
    with pytest.raises(UnsupportedProblem) as excinfo, sympy_guard("diff", reason="boom"):
        raise ValueError("sympy exploded")
    assert excinfo.value.details["error"] == "ValueError"
    assert excinfo.value.details["reason"] == "boom"


def test_sympy_guard_wraps_recursion_errors() -> None:
    with pytest.raises(UnsupportedProblem) as excinfo, sympy_guard("simplify"):
        raise RecursionError
    assert excinfo.value.details["reason"] == "recursion_limit"


def test_sympy_guard_lets_our_own_errors_through() -> None:
    with pytest.raises(UnsupportedProblem, match="desteklenmemektedir"), sympy_guard("diff"):
        raise UnsupportedProblem("already_ours")


def test_reject_unevaluated_refuses_a_non_sympy_value() -> None:
    with pytest.raises(UnsupportedProblem):
        reject_unevaluated("not sympy", "diff", "bad_type")


def test_classify_covers_every_result_kind() -> None:
    x = sympy.Symbol("x")
    assert classify(sympy.Integer(3)) == "number"
    assert classify(x + 1) == "expr"
    assert classify(sympy.FiniteSet(1, 2)) == "set"
    assert classify(sympy.true) == "bool"
    assert classify(sympy.Matrix([[1, 2]])) == "expr"


def test_build_context_drops_the_real_assumption_for_the_complexes() -> None:
    assert build_context("x").var.is_real is True
    assert build_context("x", real=False).var.is_real is None
