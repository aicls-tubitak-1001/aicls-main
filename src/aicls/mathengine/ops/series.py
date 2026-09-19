"""``op="series"`` -- Taylor/Laurent expansion around a point.

The Landau term is kept: ``e^{x}`` to order 6 is ``1 + x + x^2/2 + ... + O(x^6)``, and the
``O`` is part of the mathematics a Calculus student is being taught, not noise. Callers that
want the polynomial alone can ask SymPy for ``.removeO()`` on the ``srepr``.
"""

from __future__ import annotations

import sympy

from aicls.mathengine.models import Result, Series
from aicls.mathengine.ops.common import (
    build_context,
    make_result,
    measured,
    parse,
    reject_unevaluated,
    sympy_guard,
)

#: Nodes whose survival means SymPy could not expand the series.
_FAILURE_NODES = (sympy.Integral, sympy.Limit, sympy.Sum, sympy.Product)


def run(cmd: Series) -> Result:
    """Expand ``cmd.expr`` around ``cmd.point`` to ``cmd.order`` terms.

    Args:
        cmd: The typed command.

    Returns:
        A :class:`~aicls.mathengine.models.Result` of kind ``expr``.

    Raises:
        aicls.mathengine.errors.UnsupportedProblem: If SymPy cannot expand the expression.
    """
    context = build_context(cmd.var)
    expr = parse(cmd.expr, context)
    point = parse(cmd.point, context)
    with measured() as elapsed, sympy_guard(cmd.op, reason="series_failed"):
        raw = sympy.series(expr, context.var, point, cmd.order)
        value = reject_unevaluated(raw, cmd.op, "unevaluated_series", _FAILURE_NODES)
    return make_result(value, kind="expr", cas_ms=elapsed[0])


__all__ = ["run"]
