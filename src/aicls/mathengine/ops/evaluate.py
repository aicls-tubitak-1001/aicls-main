"""``op="evaluate"`` -- substitute values into an expression and evaluate it.

Every substitution *value* is itself LaTeX and goes through the same parser as the expression,
so ``f(\\pi/2)`` and ``x = \\sqrt{2}`` work without a numeric detour. The answer stays exact
(``\\frac{1}{3}``, not ``0.333...``); ``sympy.simplify`` is applied only to close the obvious
gaps (``\\sqrt{4}`` -> ``2``), and a decimal is produced only when the user wrote one.
"""

from __future__ import annotations

from typing import Any

import sympy

from aicls.mathengine.models import Evaluate, Result
from aicls.mathengine.ops.common import build_context, make_result, measured, parse, sympy_guard

#: The symbol table's anchor variable; substitution keys add themselves to the same table.
DEFAULT_VAR = "x"


def run(cmd: Evaluate) -> Result:
    """Substitute ``cmd.subs`` into ``cmd.expr`` and evaluate.

    Args:
        cmd: The typed command; ``subs`` maps variable names to LaTeX values.

    Returns:
        A :class:`~aicls.mathengine.models.Result` of kind ``number`` when the substitution
        closes every free symbol, otherwise ``expr``.

    Raises:
        aicls.mathengine.errors.UnsupportedProblem: If SymPy fails while substituting or
            simplifying.
    """
    context = build_context(DEFAULT_VAR)
    expr = parse(cmd.expr, context)
    mapping: dict[Any, Any] = {
        context.table.symbol(name): parse(value, context) for name, value in cmd.subs.items()
    }
    with measured() as elapsed, sympy_guard(cmd.op, reason="evaluate_failed"):
        value = expr.subs(mapping)
        value = sympy.simplify(value)
    return make_result(value, cas_ms=elapsed[0])


__all__ = ["DEFAULT_VAR", "run"]
