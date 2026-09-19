"""``op="diff"`` -- the proposal's first worked example, ``diff(x**2, x) == 2*x``."""

from __future__ import annotations

import sympy

from aicls.mathengine.models import Diff, Result
from aicls.mathengine.ops.common import build_context, make_result, measured, parse, sympy_guard
from aicls.mathengine.steps import diff_steps

#: Only a first derivative is narrated; higher orders would repeat the same story n times.
NARRATED_ORDER = 1


def run(cmd: Diff) -> Result:
    """Differentiate ``cmd.expr`` ``cmd.order`` times with respect to ``cmd.var``.

    Args:
        cmd: The typed command.

    Returns:
        A :class:`~aicls.mathengine.models.Result` of kind ``expr`` (or ``number`` when the
        derivative is constant), with product/chain-rule steps when the first derivative of a
        product or a composition was asked for.

    Raises:
        aicls.mathengine.errors.UnsupportedProblem: If SymPy cannot differentiate the input.
    """
    context = build_context(cmd.var)
    expr = parse(cmd.expr, context)
    with measured() as elapsed, sympy_guard(cmd.op, reason="diff_failed"):
        value = sympy.diff(expr, context.var, cmd.order)
        steps = diff_steps(expr, context.var, value) if cmd.order == NARRATED_ORDER else []
    return make_result(value, steps=steps, cas_ms=elapsed[0])


__all__ = ["NARRATED_ORDER", "run"]
