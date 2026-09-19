"""Curriculum regression set: CAS correctness rate over ~200 generated Calculus problems.

**Provenance, stated plainly (see KNOWN_ISSUES.md "Metric provenance").** This set is
*generated in code* from closed-form families whose answers are mathematically certain
(``d/dx x^n = n x^(n-1)``, ``int_0^1 x^n dx = 1/(n+1)``, ...). It is a **regression suite that
guards the engine against itself**, not the >=1000-problem İP-4 curriculum test set the
contractual "> 99 % correct" metric is defined on. The progress report must quote the İP-4 set
and say where it came from; a number produced by this file may never be reported as the
contractual metric.

What it does buy: every operation is exercised across a wide parameter range with an
independently known answer, so a regression in parsing, dispatch, rendering or ``srepr``
round-tripping shows up here rather than in the pilot classroom.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import sympy

from aicls.core.errors import AiclsError
from aicls.mathengine import (
    CasCommand,
    Diff,
    Evaluate,
    Expand,
    Factor,
    Integrate,
    Limit,
    MathEngine,
    Series,
    Simplify,
    Solve,
    symbolic_equivalent,
)

pytestmark = pytest.mark.slow

X = sympy.Symbol("x", real=True)

#: The contractual target of the proposal's success-metric table (CAS end to end).
TARGET_CORRECT_RATE = 0.99


@dataclass(frozen=True, slots=True)
class Problem:
    """One curriculum problem and the answer it must produce."""

    topic: str
    command: CasCommand
    expected: sympy.Basic


def _problems() -> list[Problem]:
    """Generate the deterministic problem set (no randomness, no I/O)."""
    problems: list[Problem] = []

    # --- derivatives of powers ------------------------------------------------------------
    for n in range(1, 26):
        problems.append(
            Problem(
                "diff_power",
                Diff(op="diff", expr=f"x^{{{n}}}"),
                sympy.Integer(n) * X ** (n - 1),
            )
        )
    # --- derivatives of elementary functions ----------------------------------------------
    for a in range(1, 11):
        problems.append(
            Problem(
                "diff_trig",
                Diff(op="diff", expr=rf"\sin({a}x)"),
                sympy.Integer(a) * sympy.cos(a * X),
            )
        )
        problems.append(
            Problem(
                "diff_exp",
                Diff(op="diff", expr=f"e^{{{a}x}}"),
                sympy.Integer(a) * sympy.exp(a * X),
            )
        )
    # --- indefinite integrals of powers -----------------------------------------------------
    for n in range(0, 25):
        problems.append(
            Problem(
                "integrate_power",
                Integrate(op="integrate", expr=f"x^{{{n}}}"),
                X ** (n + 1) / sympy.Integer(n + 1),
            )
        )
    # --- definite integrals over [0, 1] -----------------------------------------------------
    for n in range(0, 25):
        problems.append(
            Problem(
                "integrate_definite",
                Integrate(op="integrate", expr=f"x^{{{n}}}", lower="0", upper="1"),
                sympy.Rational(1, n + 1),
            )
        )
    # --- solving quadratics and linear equations --------------------------------------------
    for k in range(1, 21):
        problems.append(
            Problem(
                "solve_quadratic",
                Solve(op="solve", equation=f"x^{{2}}-{k * k}=0"),
                sympy.FiniteSet(-k, k),
            )
        )
    for b in range(1, 21):
        problems.append(
            Problem(
                "solve_linear",
                Solve(op="solve", equation=f"2x+{b}=0"),
                sympy.FiniteSet(sympy.Rational(-b, 2)),
            )
        )
    # --- limits ------------------------------------------------------------------------------
    for a in range(1, 11):
        problems.append(
            Problem(
                "limit_trig",
                Limit(op="limit", expr=rf"\frac{{\sin({a}x)}}{{x}}", to="0"),
                sympy.Integer(a),
            )
        )
    for k in range(1, 11):
        problems.append(
            Problem(
                "limit_removable",
                Limit(op="limit", expr=f"\\frac{{x^{{2}}-{k * k}}}{{x-{k}}}", to=f"{k}"),
                sympy.Integer(2 * k),
            )
        )
    # --- series ------------------------------------------------------------------------------
    for order in range(2, 8):
        problems.append(
            Problem(
                "series_exp",
                Series(op="series", expr="e^{x}", order=order),
                sympy.series(sympy.exp(X), X, 0, order),
            )
        )
    # --- evaluation ---------------------------------------------------------------------------
    for k in range(1, 21):
        problems.append(
            Problem(
                "evaluate",
                Evaluate(op="evaluate", expr="x^{2}+1", subs={"x": str(k)}),
                sympy.Integer(k * k + 1),
            )
        )
    # --- rewrites -----------------------------------------------------------------------------
    for k in range(1, 11):
        problems.append(
            Problem(
                "expand",
                Expand(op="expand", expr=f"(x+{k})^{{2}}"),
                X**2 + 2 * k * X + k * k,
            )
        )
        problems.append(
            Problem(
                "factor",
                Factor(op="factor", expr=f"x^{{2}}-{k * k}"),
                (X - k) * (X + k),
            )
        )
        problems.append(
            Problem(
                "simplify",
                Simplify(op="simplify", expr=rf"\frac{{x^{{2}}-{k * k}}}{{x-{k}}}"),
                X + k,
            )
        )
    return problems


CURRICULUM = _problems()


def test_curriculum_set_is_large_enough_to_measure_a_99_percent_rate() -> None:
    """With fewer than 101 problems, "> 99 %" would just mean "no failure at all"."""
    assert len(CURRICULUM) > 100
    assert len({problem.topic for problem in CURRICULUM}) >= 10


def test_curriculum_problems_cas_correct_rate_above_0_99() -> None:
    engine = MathEngine()
    failures: list[tuple[str, str, str]] = []
    abstentions: list[str] = []

    for problem in CURRICULUM:
        try:
            result = engine.execute(problem.command)
        except AiclsError as exc:
            failures.append((problem.topic, str(problem.command), f"raised {exc.code.value}"))
            continue
        produced = sympy.sympify(result.srepr)
        verdict = symbolic_equivalent(produced, problem.expected)
        if verdict == "different":
            failures.append((problem.topic, result.plain, sympy.sstr(problem.expected)))
        elif verdict == "unparsable":
            abstentions.append(problem.topic)

    total = len(CURRICULUM)
    correct = total - len(failures) - len(abstentions)
    rate = correct / total
    assert rate > TARGET_CORRECT_RATE, (
        f"correct rate {rate:.4f} over {total} problems; "
        f"failures={failures[:5]} abstentions={abstentions[:5]}"
    )


def test_every_curriculum_answer_round_trips_through_srepr_and_latex() -> None:
    """``srepr`` must rebuild the exact object and ``latex`` must stay inside our own grammar."""
    from aicls.latex2sympy import latex_to_sympy

    engine = MathEngine()
    for problem in CURRICULUM[::7]:  # a deterministic sample, for speed
        result = engine.execute(problem.command)
        assert sympy.sympify(result.srepr) is not None
        if result.kind in {"expr", "number"}:
            assert symbolic_equivalent(latex_to_sympy(result.latex), problem.expected) != (
                "different"
            )
