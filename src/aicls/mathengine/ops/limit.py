"""``op="limit"`` -- the proposal's ``lim_{x->0} sin(x)/x == 1`` example.

``direction`` maps one to one onto SymPy's ``dir`` argument: ``"+"`` right-hand, ``"-"``
left-hand, ``"+-"`` two-sided (the default, and the one a Calculus course means by "the
limit"). A two-sided limit whose one-sided limits differ does not exist; SymPy signals that
with a ``ValueError``, and the engine reports it as an unsupported problem rather than
inventing a value -- the platform's "no hallucination" rule applies to limits too.
"""

from __future__ import annotations

import sympy

from aicls.mathengine.errors import UnsupportedProblem
from aicls.mathengine.models import Limit, Result
from aicls.mathengine.ops.common import (
    build_context,
    make_result,
    measured,
    parse,
    reject_unevaluated,
    sympy_guard,
)

#: Node whose survival in the answer means SymPy could not evaluate the limit.
_FAILURE_NODES = (sympy.Limit,)

#: Substring SymPy uses when a two-sided limit does not exist.
_NO_LIMIT_MARKER = "does not exist"


def run(cmd: Limit) -> Result:
    """Evaluate ``lim_{var -> to} expr`` in the requested direction.

    Args:
        cmd: The typed command.

    Returns:
        A :class:`~aicls.mathengine.models.Result`; kind ``number`` for a finite value,
        ``expr`` when the limit is symbolic or infinite.

    Raises:
        aicls.mathengine.errors.UnsupportedProblem: If the limit does not exist or SymPy
            cannot evaluate it.
    """
    context = build_context(cmd.var)
    expr = parse(cmd.expr, context)
    target = parse(cmd.to, context)
    with measured() as elapsed, sympy_guard(cmd.op, reason="limit_failed"):
        try:
            raw = sympy.limit(expr, context.var, target, cmd.direction)
        except ValueError as exc:
            if _NO_LIMIT_MARKER in str(exc):
                raise UnsupportedProblem(
                    "limit_does_not_exist",
                    details={"op": cmd.op, "var": cmd.var, "to": cmd.to},
                ) from exc
            raise
        value = reject_unevaluated(raw, cmd.op, "unevaluated_limit", _FAILURE_NODES)
    return make_result(value, cas_ms=elapsed[0])


__all__ = ["run"]
