"""``op="integrate"`` -- indefinite and definite integrals, infinite bounds included.

``\\int_{0}^{1} x^{2}\\,dx = 1/3`` is the proposal's second worked example. Bounds are ordinary
LaTeX fields, so ``\\infty`` and ``-\\infty`` arrive as ``sympy.oo`` through the normal parser
path and improper integrals need no special case.

An indefinite integral is returned *without* the constant of integration: SymPy's antiderivative
is one representative of the family, and adding a symbolic ``C`` would make the answer
un-differentiable back to the input in ``aicls.eval``'s round-trip check. The user interface
appends "+ C" when it displays an indefinite integral (docs/mimari/api.md §3.4).
"""

from __future__ import annotations

import sympy

from aicls.mathengine.models import Integrate, Result
from aicls.mathengine.ops.common import (
    build_context,
    make_result,
    measured,
    parse,
    reject_unevaluated,
    sympy_guard,
)

#: Node whose survival in the answer means SymPy could not find an antiderivative.
_FAILURE_NODES = (sympy.Integral,)


def run(cmd: Integrate) -> Result:
    """Integrate ``cmd.expr`` with respect to ``cmd.var``, definitely when bounds are given.

    Args:
        cmd: The typed command; ``lower`` and ``upper`` are either both set or both None
            (enforced by the model).

    Returns:
        A :class:`~aicls.mathengine.models.Result` of kind ``number`` for a definite integral
        that collapses to a constant, otherwise ``expr``.

    Raises:
        aicls.mathengine.errors.UnsupportedProblem: If the integral stays unevaluated -- the
            İP-4 "bu problem tipi desteklenmemektedir" path.
    """
    context = build_context(cmd.var)
    expr = parse(cmd.expr, context)
    with measured() as elapsed, sympy_guard(cmd.op, reason="integrate_failed"):
        if cmd.lower is None or cmd.upper is None:
            raw = sympy.integrate(expr, context.var)
        else:
            lower = parse(cmd.lower, context)
            upper = parse(cmd.upper, context)
            raw = sympy.integrate(expr, (context.var, lower, upper))
        value = reject_unevaluated(raw, cmd.op, "unevaluated_integral", _FAILURE_NODES)
    return make_result(value, cas_ms=elapsed[0])


__all__ = ["run"]
