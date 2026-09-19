"""Tri-state symbolic equivalence: ``equal`` | ``different`` | ``unparsable``.

``aicls.eval`` needs to decide whether a student's (or a model's) answer matches the reference
answer over a >=1000-problem curriculum set, and the CAS correctness metric is defined on that
decision (docs/adr/0015). Plain ``a == b`` is far too strict (``\\frac{1}{2}`` vs ``0.5``) and
``simplify(a - b) == 0`` is neither total nor fast, so the decision is deliberately three
valued:

* ``equal`` -- symbolic simplification proves it, or every numeric probe agrees within ``tol``;
* ``different`` -- simplification or a probe finds a genuine gap;
* ``unparsable`` -- one side is not a SymPy object (or a string that parses), or no probe could
  be evaluated. This is the honest "the metric abstains" outcome; ``aicls.eval`` reports these
  separately instead of scoring them as failures.

Probes are deterministic (a fixed rational grid, not ``random``) so a metric never moves
between runs -- the seed rule of ``docs/gelistirme/test-stratejisi.md`` applies to metrics too.
"""

from __future__ import annotations

import cmath
from typing import Any, Literal, TypeAlias

import sympy
from sympy.core.relational import Relational
from sympy.logic.boolalg import BooleanAtom

from aicls.core.errors import AiclsError
from aicls.core.logging import get_logger

#: Result of :func:`symbolic_equivalent`.
Equivalence: TypeAlias = Literal["equal", "different", "unparsable"]

#: Deterministic probe grid; chosen away from 0, +-1 and small integers so that poles, branch
#: cuts and trivial identities do not dominate the sample.
PROBE_POINTS: tuple[float, ...] = (0.3125, 1.7, -0.8125, 2.4375, -1.9, 3.6875, -2.5625, 4.15)

#: Largest number of free symbols a numeric probe is attempted for.
MAX_PROBE_SYMBOLS = 4

_logger = get_logger(__name__)


def _coerce(value: Any) -> sympy.Basic | None:
    """Return ``value`` as a SymPy object, parsing a LaTeX string if needed; None if hopeless."""
    if isinstance(value, sympy.Basic | sympy.MatrixBase):
        return value
    if isinstance(value, bool | int | float):
        return sympy.sympify(value)
    if isinstance(value, str):
        from aicls.latex2sympy import latex_to_sympy

        try:
            return latex_to_sympy(value)
        except Exception:
            # A string that does not parse is exactly the "unparsable" verdict, not a failure;
            # both our ParseError and any raw SymPy/Lark error mean the same thing here.
            return None
    return None


def _sets_equal(a: sympy.Basic, b: sympy.Basic) -> Equivalence | None:
    """Compare two SymPy sets element-wise; None when they are not both finite sets."""
    if not (isinstance(a, sympy.FiniteSet) and isinstance(b, sympy.FiniteSet)):
        return None
    if len(a) != len(b):
        return "different"
    remaining = list(b.args)
    for element in a.args:
        match = next(
            (other for other in remaining if symbolic_equivalent(element, other) == "equal"), None
        )
        if match is None:
            return "different"
        remaining.remove(match)
    return "equal"


def _numeric_probe(
    difference: sympy.Basic, symbols: list[sympy.Symbol], probes: int, tol: float
) -> Equivalence:
    """Sample ``difference`` on the deterministic grid; ``unparsable`` when nothing evaluates."""
    evaluated = 0
    for index in range(probes):
        substitutions = {
            symbol: sympy.Rational(PROBE_POINTS[(index + offset) % len(PROBE_POINTS)])
            for offset, symbol in enumerate(symbols)
        }
        try:
            value = complex(sympy.N(difference.subs(substitutions), 20))
        except Exception:
            # SymPy raises many concrete types while evaluating (TypeError, ZeroDivisionError,
            # PolynomialError, ...); a probe that cannot be evaluated simply does not count.
            continue
        if cmath.isnan(value) or cmath.isinf(value):  # a pole or a NaN: this probe abstains
            continue
        evaluated += 1
        if abs(value) > tol:
            return "different"
    if evaluated == 0:
        return "unparsable"
    return "equal"


def symbolic_equivalent(
    a: Any,
    b: Any,
    *,
    numeric_probes: int = 5,
    tol: float = 1e-6,
) -> Equivalence:
    """Decide whether ``a`` and ``b`` denote the same mathematical object.

    Args:
        a: A ``sympy.Basic`` (or a LaTeX string, or a Python number).
        b: The other side, same accepted types.
        numeric_probes: How many points of :data:`PROBE_POINTS` to sample when symbolic
            simplification is inconclusive.
        tol: Absolute tolerance for a numeric probe.

    Returns:
        ``"equal"``, ``"different"`` or ``"unparsable"``.

    Example:
        >>> import sympy
        >>> x = sympy.Symbol("x")
        >>> symbolic_equivalent(sympy.sin(x) ** 2 + sympy.cos(x) ** 2, sympy.Integer(1))
        'equal'
    """
    left = _coerce(a)
    right = _coerce(b)
    if left is None or right is None:
        return "unparsable"

    set_verdict = _sets_equal(left, right)
    if set_verdict is not None:
        return set_verdict
    if isinstance(left, sympy.Set) or isinstance(right, sympy.Set):
        return "equal" if left == right else "different"
    if isinstance(left, sympy.MatrixBase) or isinstance(right, sympy.MatrixBase):
        return "equal" if left == right else "different"
    if isinstance(left, BooleanAtom) or isinstance(right, BooleanAtom):
        return "equal" if left == right else "different"
    if isinstance(left, Relational) or isinstance(right, Relational):
        return _relations_equal(left, right, numeric_probes=numeric_probes, tol=tol)

    order_verdict = _series_equal(left, right, numeric_probes=numeric_probes, tol=tol)
    if order_verdict is not None:
        return order_verdict

    try:
        difference = sympy.simplify(left - right)
    except AiclsError:
        raise
    except Exception as exc:
        # simplify() can fail outright on exotic trees; that is an abstention, not an error.
        _logger.debug("mathengine.equivalence.simplify_failed", error=type(exc).__name__)
        return "unparsable"

    if difference == 0 or difference.is_zero is True:
        return "equal"

    symbols = sorted(difference.free_symbols, key=str)
    if not symbols:
        try:
            value = complex(sympy.N(difference, 20))
        except Exception:
            # A constant tree SymPy cannot turn into a number (an unevaluated integral, say).
            return "unparsable"
        return "equal" if abs(value) <= tol else "different"
    if len(symbols) > MAX_PROBE_SYMBOLS:
        return "different" if difference.is_zero is False else "unparsable"
    return _numeric_probe(difference, list(symbols), max(1, numeric_probes), tol)


def _order_term(expr: sympy.Basic) -> sympy.Basic | None:
    """Return the Landau ``O(...)`` term of ``expr``, or None when it carries none."""
    if not isinstance(expr, sympy.Expr):
        return None
    try:
        return expr.getO()
    except AttributeError:  # pragma: no cover - every Expr defines getO()
        return None


def _series_equal(
    left: sympy.Basic, right: sympy.Basic, *, numeric_probes: int, tol: float
) -> Equivalence | None:
    """Compare two truncated series; None when neither side carries a Landau term.

    ``O(x^6) - O(x^6)`` is ``O(x^6)``, not zero, so subtracting two series never simplifies to
    zero and the numeric probes cannot evaluate an ``O`` term either -- a series comparison
    would abstain forever. Two truncated series are equal when they are truncated at the same
    order *and* their polynomial parts agree; a series is never equal to a bare polynomial,
    because the truncation is part of the statement.
    """
    left_order = _order_term(left)
    right_order = _order_term(right)
    if left_order is None and right_order is None:
        return None
    if left_order is None or right_order is None:
        return "different"
    if left_order != right_order:
        return "different"
    return symbolic_equivalent(
        left.removeO(), right.removeO(), numeric_probes=numeric_probes, tol=tol
    )


def _relations_equal(
    left: sympy.Basic, right: sympy.Basic, *, numeric_probes: int, tol: float
) -> Equivalence:
    """Two relations are equal when the same operator relates equivalent sides."""
    if not (isinstance(left, Relational) and isinstance(right, Relational)):
        return "different"
    if left.rel_op != right.rel_op:
        return "different"
    lhs = symbolic_equivalent(left.lhs, right.lhs, numeric_probes=numeric_probes, tol=tol)
    rhs = symbolic_equivalent(left.rhs, right.rhs, numeric_probes=numeric_probes, tol=tol)
    if lhs == "equal" and rhs == "equal":
        return "equal"
    if "unparsable" in (lhs, rhs):
        return "unparsable"
    return "different"


__all__ = ["MAX_PROBE_SYMBOLS", "PROBE_POINTS", "Equivalence", "symbolic_equivalent"]
