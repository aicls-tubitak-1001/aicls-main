"""``op="solve"`` -- the proposal's ``x^{2}-4=0 -> {-2, 2}`` example.

``sympy.solveset`` is used rather than ``sympy.solve`` for two reasons that matter to the
platform: it always returns a *set* (so ``Result.kind`` is honest -- ``{-2, 2}`` is a set, not
a list) and it takes an explicit domain, which is how ``domain="reals"`` versus
``"complexes"`` is implemented. Over the complexes the variable is declared without the
``real=True`` assumption, otherwise ``x^{2}+1=0`` would come back empty.

When ``solveset`` cannot characterise the solutions it returns a ``ConditionSet`` -- the set of
``x`` such that the equation holds, which restates the question instead of answering it. That
is exactly the İP-4 "desteklenmiyor" case.
"""

from __future__ import annotations

import sympy

from aicls.mathengine.errors import UnsupportedProblem
from aicls.mathengine.models import Result, Solve, SolveDomain
from aicls.mathengine.ops.common import build_context, make_result, measured, parse, sympy_guard

#: ``Solve.domain`` -> the SymPy set solutions are looked for in.
DOMAINS: dict[SolveDomain, sympy.Set] = {
    "reals": sympy.S.Reals,
    "complexes": sympy.S.Complexes,
}


def run(cmd: Solve) -> Result:
    """Solve ``cmd.equation`` for ``cmd.var`` over the requested domain.

    Args:
        cmd: The typed command. A bare expression is read as ``expression = 0``.

    Returns:
        A :class:`~aicls.mathengine.models.Result` of kind ``set``.

    Raises:
        aicls.mathengine.errors.UnsupportedProblem: If SymPy returns a ``ConditionSet`` (it
            could not characterise the solutions) or fails outright.
    """
    context = build_context(cmd.var, real=cmd.domain == "reals")
    equation = parse(cmd.equation, context)
    with measured() as elapsed, sympy_guard(cmd.op, reason="solve_failed"):
        value = sympy.solveset(equation, context.var, domain=DOMAINS[cmd.domain])
        if isinstance(value, sympy.ConditionSet) or value.has(sympy.ConditionSet):
            raise UnsupportedProblem(
                "condition_set", details={"op": cmd.op, "value": str(value)[:200]}
            )
    return make_result(value, kind="set", cas_ms=elapsed[0])


__all__ = ["DOMAINS", "run"]
