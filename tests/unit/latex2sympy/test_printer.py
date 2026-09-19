"""sympy -> KaTeX-safe LaTeX printer (docs/mimari/latex-ayristirici.md 3.6)."""

from __future__ import annotations

import pytest
import sympy
from sympy.core.relational import Relational

from aicls.core.latex_safety import find_disallowed_commands
from aicls.latex2sympy import PrintStyle, latex_to_sympy, parse_latex, sympy_to_latex

x = sympy.Symbol("x", real=True)
y = sympy.Symbol("y", real=True)
n = sympy.Symbol("n", integer=True)


def test_sympy_one_third_prints_frac_1_3() -> None:
    """Proposal A.2.2: ``latex(Rational(1, 3))`` must render as ``\\frac{1}{3}``."""
    assert sympy_to_latex(sympy.Rational(1, 3)) == r"\frac{1}{3}"
    assert sympy_to_latex(sympy.Integer(2) / 6) == r"\frac{1}{3}"


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        (sympy.Integer(2), "2"),
        (x, "x"),
        (2 * x, r"2 \cdot x"),
        (x**2, "x^{2}"),
        (sympy.sqrt(x), r"\sqrt{x}"),
        (sympy.pi, r"\pi"),
        (sympy.oo, r"\infty"),
        (sympy.exp(x), "e^{x}"),
        (sympy.Eq(x, 2), "x = 2"),
    ],
)
def test_sympy_to_latex_basic_forms(expr: sympy.Basic, expected: str) -> None:
    assert sympy_to_latex(expr) == expected


def test_sympy_to_latex_uses_ln_and_full_inverse_trig() -> None:
    assert sympy_to_latex(sympy.log(x)).startswith(r"\ln")
    assert sympy_to_latex(sympy.asin(x)).startswith(r"\arcsin")


def test_sympy_to_latex_is_katex_safe() -> None:
    exprs = [
        sympy.Sum(n**2, (n, 1, 10)),
        sympy.Product(n, (n, 1, 10)),
        sympy.Limit(sympy.sin(x) / x, x, 0, "+"),
        sympy.Integral(x**2, (x, 0, 1)),
        sympy.Derivative(x**2, x),
        sympy.Matrix([[1, 2], [3, 4]]),
        sympy.Interval(0, 1),
        sympy.FiniteSet(1, 2),
    ]
    for expr in exprs:
        rendered = sympy_to_latex(expr)
        assert find_disallowed_commands(rendered) == [], rendered
        assert r"\operatorname*" not in rendered
        assert r"\substack" not in rendered


def test_print_style_raw_keeps_sympy_output() -> None:
    raw = sympy_to_latex(sympy.Rational(1, 3), style=PrintStyle.RAW)
    assert raw == sympy.latex(sympy.Rational(1, 3))


def test_sympy_to_latex_round_trips_through_the_grammar() -> None:
    exprs = [
        x**2 + 2 * x + 1,
        sympy.sin(x) * sympy.cos(x),
        sympy.Rational(1, 3),
        sympy.sqrt(x**2 + 1),
        sympy.exp(-x),
        sympy.log(x + 1),
        sympy.Integral(x**2, (x, 0, 1)),
        sympy.Derivative(sympy.sin(x), x),
        sympy.Limit(sympy.sin(x) / x, x, 0),
        sympy.Sum(n**2, (n, 1, 10)),
        sympy.Abs(x),
        sympy.Eq(x**2 - 4, 0),
        sympy.Interval(0, 1),
        x / (y + 1),
        sympy.pi * x**2,
        sympy.Rational(-3, 4) * x,
        sympy.cos(x) ** 2,
        sympy.atan(x),
    ]
    for expr in exprs:
        rendered = sympy_to_latex(expr)
        back = latex_to_sympy(rendered)
        if back == expr:
            continue
        assert not isinstance(back, Relational | sympy.Set), rendered
        assert sympy.simplify(back - expr) == 0, rendered


def test_sympy_to_latex_of_relations_and_sets() -> None:
    assert latex_to_sympy(sympy_to_latex(sympy.Eq(x, 2))) == sympy.Eq(x, 2)
    assert parse_latex(sympy_to_latex(sympy.Interval(0, 1))) is not None


def test_sympy_to_latex_rejects_non_basic() -> None:
    from aicls.core.errors import InputValidationError

    with pytest.raises(InputValidationError):
        sympy_to_latex("not an expression")  # type: ignore[arg-type]
