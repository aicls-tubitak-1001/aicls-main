"""Property-based checks on the CAS operations (hypothesis).

The headline property is the Fundamental Theorem of Calculus as the engine can observe it:
**integrating the derivative of a polynomial returns the polynomial up to a constant.** It ties
three subsystems together in one assertion -- the LaTeX printer must emit something the parser
accepts, ``diff`` and ``integrate`` must agree, and ``symbolic_equivalent`` must recognise the
result -- which is exactly the kind of coupling a hand-written example set misses.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
import pytest
import sympy

from aicls.core.errors import AiclsError
from aicls.mathengine import (
    Diff,
    Expand,
    Factor,
    Integrate,
    MathEngine,
    Simplify,
    canonical_key,
    symbolic_equivalent,
)

pytestmark = pytest.mark.property

X = sympy.Symbol("x", real=True)

#: Integer coefficients kept small so SymPy stays fast and the LaTeX stays readable.
COEFFICIENTS = st.integers(min_value=-9, max_value=9)

#: Polynomial degree; degree 0 (a constant) is included on purpose -- its derivative is 0 and
#: integrating that back gives 0, which is the polynomial up to a constant.
DEGREES = st.integers(min_value=0, max_value=5)


@st.composite
def polynomials(draw: st.DrawFn) -> sympy.Expr:
    """A polynomial in ``x`` with small integer coefficients."""
    degree = draw(DEGREES)
    coefficients = draw(st.lists(COEFFICIENTS, min_size=degree + 1, max_size=degree + 1))
    return sympy.Add(*(c * X**power for power, c in enumerate(coefficients)))


def _latex(expr: sympy.Expr) -> str:
    from aicls.latex2sympy import sympy_to_latex

    return sympy_to_latex(expr)


@settings(max_examples=60, deadline=None)
@given(polynomials())
def test_integrating_the_derivative_returns_the_polynomial_up_to_a_constant(
    poly: sympy.Expr,
) -> None:
    engine = MathEngine()
    latex = _latex(poly)
    derivative = engine.execute(Diff(op="diff", expr=latex))
    antiderivative = engine.execute(Integrate(op="integrate", expr=derivative.latex))
    recovered = sympy.sympify(antiderivative.srepr)
    difference = sympy.simplify(recovered - poly)
    assert difference.free_symbols == set(), f"{recovered} vs {poly} differ by {difference}"


@settings(max_examples=60, deadline=None)
@given(polynomials())
def test_differentiating_the_integral_returns_the_polynomial(poly: sympy.Expr) -> None:
    engine = MathEngine()
    antiderivative = engine.execute(Integrate(op="integrate", expr=_latex(poly)))
    derivative = engine.execute(Diff(op="diff", expr=antiderivative.latex))
    assert symbolic_equivalent(sympy.sympify(derivative.srepr), poly) == "equal"


@settings(max_examples=60, deadline=None)
@given(polynomials())
def test_expand_and_factor_preserve_the_function(poly: sympy.Expr) -> None:
    engine = MathEngine()
    latex = _latex(poly)
    expanded = engine.execute(Expand(op="expand", expr=latex))
    assert symbolic_equivalent(sympy.sympify(expanded.srepr), poly) == "equal"
    try:
        factored = engine.execute(Factor(op="factor", expr=latex))
    except AiclsError:  # pragma: no cover - factoring a constant is still a valid answer
        return
    assert symbolic_equivalent(sympy.sympify(factored.srepr), poly) == "equal"


@settings(max_examples=60, deadline=None)
@given(polynomials())
def test_results_re_enter_the_grammar_they_came_from(poly: sympy.Expr) -> None:
    """Every ``Result.latex`` must be re-parsable: the UI lets the user send it back as input."""
    from aicls.latex2sympy import latex_to_sympy

    engine = MathEngine()
    result = engine.execute(Simplify(op="simplify", expr=_latex(poly)))
    reparsed = latex_to_sympy(result.latex)
    assert symbolic_equivalent(reparsed, poly) == "equal"


@settings(max_examples=60, deadline=None)
@given(polynomials())
def test_the_cache_never_changes_an_answer(poly: sympy.Expr) -> None:
    engine = MathEngine()
    command = Diff(op="diff", expr=_latex(poly))
    first = engine.execute(command)
    second = engine.execute(command)
    assert second.cache_hit is True
    assert (second.latex, second.plain, second.srepr) == (first.latex, first.plain, first.srepr)
    assert canonical_key(command) in engine.cache
