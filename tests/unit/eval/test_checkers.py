"""Answer comparison strategies, and the two traps that would misreport CAS accuracy.

Both traps were found while running this harness over the committed curriculum set, and both
would have produced a *wrong number in a progress report* rather than a crash: a correct answer
scored as wrong.
"""

from __future__ import annotations

import pytest
import sympy

from aicls.core.errors import InputValidationError
from aicls.eval.checkers import (
    check_antiderivative,
    compare,
    compare_latex,
    compare_numeric,
    rebuild_srepr,
)
from aicls.eval.metrics import symbolic_equivalence


# --------------------------------------------------------------------------------- srepr input
def test_rebuild_srepr_round_trips_a_sympy_object() -> None:
    """The dataset stores ``srepr`` because it is exact; rebuilding must give the same object."""
    value = sympy.Integer(2) * sympy.Symbol("x", real=True)
    assert rebuild_srepr(sympy.srepr(value)) == value


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "__import__('os').system('echo')",
        "lambda: 1",
        "Symbol('x'); import os",
        "eval('1+1')",
    ],
)
def test_rebuild_srepr_rejects_anything_that_is_not_an_srepr(text: str) -> None:
    """``srepr`` text is evaluated, so it is validated first (CODING_STANDARDS: no eval)."""
    with pytest.raises(InputValidationError):
        rebuild_srepr(text)


def test_rebuild_srepr_rejects_text_sympy_cannot_rebuild() -> None:
    """A well-formed but meaningless srepr is a data error, not a crash."""
    with pytest.raises(InputValidationError):
        rebuild_srepr("Add(Integer(1)")


# ------------------------------------------------------------------------------------ compare
def test_compare_symbolic_ignores_symbol_assumptions() -> None:
    """``Symbol('x')`` and ``Symbol('x', real=True)`` are different *symbols* to SymPy.

    The engine builds its answers with ``real=True``; a hand-written expected answer usually
    does not. Without this rule ``2x`` and ``2x`` would be scored as a wrong answer, which is
    the worst failure mode a metric has.
    """
    engine_answer = sympy.Integer(2) * sympy.Symbol("x", real=True)
    authored = sympy.Integer(2) * sympy.Symbol("x")
    assert engine_answer != authored  # SymPy itself says they are different objects
    assert compare("symbolic", engine_answer, authored, {}) == (True, "")


def test_compare_still_rejects_a_genuinely_different_answer() -> None:
    """Dropping assumptions must not make the comparison permissive."""
    ok, reason = compare(
        "symbolic", sympy.Integer(2) * sympy.Symbol("x", real=True), sympy.Symbol("x"), {}
    )
    assert ok is False
    assert "different" in reason


def test_compare_up_to_constant_accepts_a_shifted_antiderivative() -> None:
    """An antiderivative is only defined up to a constant (İP-4 indefinite integrals)."""
    x = sympy.Symbol("x", real=True)
    engine = x**2 / 2
    authored = x**2 / 2 + 7
    assert compare("symbolic", engine, authored, {"up_to_constant": True, "var": "x"}) == (
        True,
        "",
    )
    ok, reason = compare("symbolic", x**3, authored, {"up_to_constant": True, "var": "x"})
    assert ok is False
    assert "constant" in reason


def test_compare_bool_and_exact_latex() -> None:
    """``bool`` and ``exact_latex`` are literal comparisons by design."""
    assert compare("bool", sympy.true, sympy.false, {}) == (False, "bool True != False")
    assert compare("bool", sympy.true, sympy.true, {}) == (True, "")
    assert compare_latex("2 x", "2 x") == (True, "")
    assert compare_latex("2 x", "x 2")[0] is False


def test_compare_numeric_samples_the_domain() -> None:
    """The ``numeric`` checker compares on a deterministic grid, not by simplification."""
    x = sympy.Symbol("x", real=True)
    assert compare_numeric(sympy.sin(x) ** 2, 1 - sympy.cos(x) ** 2, {"var": "x"}) == (True, "")
    ok, reason = compare_numeric(x, x + 1, {"var": "x"})
    assert ok is False
    assert "numeric gap" in reason


def test_compare_numeric_on_constants_and_on_an_abstaining_probe() -> None:
    """A constant answer is compared directly; ``nan``/``oo`` makes the checker abstain."""
    assert compare_numeric(sympy.Rational(1, 2), sympy.Float(0.5), {}) == (True, "")
    ok, reason = compare_numeric(sympy.oo, sympy.oo, {})
    assert ok is False
    assert "could not evaluate" in reason


def test_check_antiderivative_verifies_against_the_integrand() -> None:
    """The stronger check: the answer must differentiate back to what was integrated."""
    x = sympy.Symbol("x", real=True)
    assert check_antiderivative(x**2 / 2, "x", {"var": "x"}) == ""
    assert "not the integrand" in check_antiderivative(x**3, "x", {"var": "x"})
    assert "did not parse" in check_antiderivative(x, r"\binom{n}{k}", {"var": "x"})


# ------------------------------------------------------------ the identity rule of the metric
def test_symbolic_equivalence_never_calls_two_identical_answers_different() -> None:
    """``simplify(a - a)`` is ``nan`` for an infinite ``a`` — the metric must not fall for it.

    ``\\lim_{x\\to0^{+}}\\ln x=-\\infty`` compared with itself came back as ``"different"`` from
    the CAS before this rule, i.e. a perfect recognition scored as a wrong answer.
    """
    infinite = r"\lim_{x\to0^{+}}\ln x=-\infty"
    assert symbolic_equivalence(infinite, infinite) == "equal"
    assert symbolic_equivalence(r"-\infty", r"-\infty") == "equal"
    # a normalisation difference is still the same answer
    assert symbolic_equivalence("x^2", "x^{2}") == "equal"
    # ... and two different infinite answers are still judged by the CAS
    assert symbolic_equivalence(r"\infty", r"-\infty") in {"different", "unparsable"}


def test_symbolic_equivalence_identity_rule_also_covers_out_of_grammar_answers() -> None:
    """An expression the grammar cannot read is still equal to itself."""
    assert symbolic_equivalence(r"\binom{n}{k}", r"\binom{n}{k}") == "equal"
    assert symbolic_equivalence(r"\binom{n}{k}", r"\binom{n}{m}") == "unparsable"


# ------------------------------------------------------------------------ abstention branches
def test_compare_numeric_reports_an_evaluation_failure_instead_of_a_verdict() -> None:
    """Two values SymPy cannot subtract is a data error, not "different"."""
    ok, reason = compare_numeric("a", "b", {})
    assert ok is False
    assert "numeric evaluation failed" in reason


def test_compare_numeric_skips_probes_that_hit_a_pole() -> None:
    """A pole inside the sampling domain must not decide the comparison."""
    x = sympy.Symbol("x", real=True)
    # 1/(x+3) has a pole at the first grid point (-3); the remaining probes still agree
    assert compare_numeric(1 / (x + 3), 1 / (x + 3), {"var": "x"}) == (True, "")


def test_compare_numeric_abstains_when_no_probe_can_be_evaluated() -> None:
    """Every probe infinite means the checker has no evidence at all."""
    x = sympy.Symbol("x", real=True)
    ok, reason = compare_numeric(sympy.zoo * x, sympy.Integer(0), {"var": "x"})
    assert ok is False
    assert "could not evaluate any probe" in reason
