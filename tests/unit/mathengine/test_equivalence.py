"""Tri-state symbolic equivalence: the metric ``aicls.eval`` scores CAS correctness with."""

from __future__ import annotations

import pytest
import sympy

from aicls.mathengine import symbolic_equivalent
from aicls.mathengine.equivalence import MAX_PROBE_SYMBOLS, PROBE_POINTS

X = sympy.Symbol("x", real=True)
Y = sympy.Symbol("y", real=True)


def test_symbolic_equivalent_sin2_plus_cos2_vs_1() -> None:
    """The canonical identity: different trees, same function."""
    assert symbolic_equivalent(sympy.sin(X) ** 2 + sympy.cos(X) ** 2, sympy.Integer(1)) == "equal"


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (2 * X, X + X, "equal"),
        (sympy.Rational(1, 2), sympy.Float("0.5"), "equal"),
        (sympy.expand((X + 1) ** 2), (X + 1) ** 2, "equal"),
        (X**2, X**3, "different"),
        (2 * X, 2 * X + 1, "different"),
        (sympy.Integer(1), sympy.Integer(2), "different"),
        (sympy.log(X), sympy.log(X) + sympy.Integer(0), "equal"),
    ],
)
def test_tri_state_verdicts(a: sympy.Basic, b: sympy.Basic, expected: str) -> None:
    assert symbolic_equivalent(a, b) == expected


def test_latex_strings_are_accepted_on_either_side() -> None:
    assert symbolic_equivalent("2x", 2 * X) == "equal"
    assert symbolic_equivalent(r"\frac{1}{2}", "0.5") == "equal"


def test_unparsable_when_a_side_is_not_mathematics() -> None:
    assert symbolic_equivalent("x^{", "x") == "unparsable"
    assert symbolic_equivalent(None, X) == "unparsable"
    assert symbolic_equivalent(X, object()) == "unparsable"


def test_python_numbers_are_accepted() -> None:
    assert symbolic_equivalent(1, sympy.Integer(1)) == "equal"
    assert symbolic_equivalent(0.5, sympy.Rational(1, 2)) == "equal"


def test_sets_compare_element_wise_regardless_of_order() -> None:
    assert symbolic_equivalent(sympy.FiniteSet(-2, 2), sympy.FiniteSet(2, -2)) == "equal"
    assert symbolic_equivalent(sympy.FiniteSet(-2, 2), sympy.FiniteSet(2, 3)) == "different"
    assert symbolic_equivalent(sympy.FiniteSet(1), sympy.FiniteSet(1, 2)) == "different"
    assert symbolic_equivalent(sympy.S.Reals, sympy.S.Reals) == "equal"
    assert symbolic_equivalent(sympy.S.Reals, sympy.EmptySet) == "different"


def test_sets_compare_members_up_to_equivalence() -> None:
    assert symbolic_equivalent(sympy.FiniteSet(sympy.sqrt(4)), sympy.FiniteSet(2)) == "equal"


def test_booleans_and_relations() -> None:
    assert symbolic_equivalent(sympy.true, sympy.true) == "equal"
    assert symbolic_equivalent(sympy.true, sympy.false) == "different"
    assert symbolic_equivalent(sympy.Eq(X, 1), sympy.Eq(X, 1)) == "equal"
    assert symbolic_equivalent(sympy.Eq(X, 1), sympy.Eq(X, 2)) == "different"
    assert symbolic_equivalent(sympy.Eq(X, 1), sympy.Lt(X, 1)) == "different"
    assert symbolic_equivalent(sympy.Eq(X, 1), X) == "different"


def test_matrices_compare_structurally() -> None:
    assert symbolic_equivalent(sympy.Matrix([[1, 2]]), sympy.Matrix([[1, 2]])) == "equal"
    assert symbolic_equivalent(sympy.Matrix([[1, 2]]), sympy.Matrix([[1, 3]])) == "different"


def test_numeric_probes_settle_what_simplify_cannot() -> None:
    """A pair ``simplify`` does not reduce to zero, but every probe agrees on."""
    a = sympy.atan(X) + sympy.atan(1 / X)
    b = sympy.pi / 2
    assert symbolic_equivalent(a, b, numeric_probes=3) in {"equal", "different"}


def test_probe_grid_is_fixed_and_deterministic() -> None:
    assert len(PROBE_POINTS) >= MAX_PROBE_SYMBOLS
    first = symbolic_equivalent(sympy.sin(X) * sympy.cos(X), sympy.sin(2 * X) / 2)
    second = symbolic_equivalent(sympy.sin(X) * sympy.cos(X), sympy.sin(2 * X) / 2)
    assert first == second == "equal"


def test_many_free_symbols_short_circuit() -> None:
    symbols = sympy.symbols("a b c d e f", real=True)
    assert symbolic_equivalent(sum(symbols), sum(symbols)) == "equal"
    assert symbolic_equivalent(sum(symbols), sum(symbols) + 1) == "different"


def test_two_variable_expressions_are_probed() -> None:
    assert symbolic_equivalent(X * Y, Y * X) == "equal"
    assert symbolic_equivalent(X + Y, X - Y) == "different"


def test_constant_that_cannot_be_evaluated_is_unparsable() -> None:
    unevaluated = sympy.Integral(sympy.Function("f")(X), (X, 0, 1))
    assert symbolic_equivalent(unevaluated, sympy.Integer(0)) == "unparsable"


def test_truncated_series_compare_by_order_and_polynomial_part() -> None:
    """``O(x^n) - O(x^n)`` is not zero, so series need their own rule."""
    a = sympy.series(sympy.exp(X), X, 0, 4)
    b = sympy.series(sympy.exp(X), X, 0, 4)
    assert symbolic_equivalent(a, b) == "equal"
    assert symbolic_equivalent(a, sympy.series(sympy.exp(X), X, 0, 5)) == "different"
    assert symbolic_equivalent(a, sympy.series(sympy.sin(X), X, 0, 4)) == "different"
    assert symbolic_equivalent(a, a.removeO()) == "different"
    assert symbolic_equivalent(a.removeO(), a) == "different"


def test_probe_that_never_evaluates_is_unparsable() -> None:
    """Two undefined functions: nothing to simplify and no probe produces a number."""
    f, g = sympy.Function("f"), sympy.Function("g")
    assert symbolic_equivalent(f(X), g(X)) == "unparsable"


def test_probes_settle_an_identity_simplify_does_not_know() -> None:
    """``erf(x) + erfc(x) == 1`` survives ``simplify`` unreduced; the probe grid closes it."""
    assert sympy.simplify(sympy.erf(X) + sympy.erfc(X) - 1) != 0
    assert symbolic_equivalent(sympy.erf(X) + sympy.erfc(X), sympy.Integer(1)) == "equal"


def test_a_pole_at_a_probe_point_is_skipped_not_counted() -> None:
    """The first probe point is 5/16; a pole there must not decide the verdict either way."""
    pole = X / (X - sympy.Rational(5, 16))
    assert symbolic_equivalent(pole, sympy.Integer(0)) == "different"


def test_relations_whose_sides_are_unparsable_abstain() -> None:
    f, g = sympy.Function("f"), sympy.Function("g")
    assert symbolic_equivalent(sympy.Eq(f(X), 1), sympy.Eq(g(X), 1)) == "unparsable"


def test_a_simplify_failure_is_an_abstention(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> sympy.Basic:
        raise RuntimeError("simplify exploded")

    monkeypatch.setattr("sympy.simplify", boom)
    assert symbolic_equivalent(X, X + 1) == "unparsable"
