"""``op="simplify" | "expand" | "factor"`` -- the three form-preserving rewrites.

They share one implementation because they differ only in which SymPy function is applied and
in nothing else: same parsing, same guard, same rendering. ``simplify`` is the expensive one
(it is the reason ADR-0016 exists at all); the sandbox timeout is what bounds it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import sympy

from aicls.mathengine.models import Expand, Factor, Result, Simplify
from aicls.mathengine.ops.common import build_context, make_result, measured, parse, sympy_guard

#: ``op`` -> the SymPy rewrite it stands for.
REWRITES: dict[str, Callable[[Any], Any]] = {
    "simplify": sympy.simplify,
    "expand": sympy.expand,
    "factor": sympy.factor,
}

#: The variable a rewrite is parsed against; rewrites are variable-free but the parser needs a
#: symbol table, and ``x`` keeps the assumptions consistent with every other command.
DEFAULT_VAR = "x"


def run(cmd: Simplify | Expand | Factor) -> Result:
    """Apply ``simplify``/``expand``/``factor`` to ``cmd.expr``.

    Args:
        cmd: The typed command.

    Returns:
        A :class:`~aicls.mathengine.models.Result` of kind ``expr`` (or ``number`` when the
        rewrite collapses to a constant).

    Raises:
        aicls.mathengine.errors.UnsupportedProblem: If the rewrite fails inside SymPy.
    """
    context = build_context(DEFAULT_VAR)
    expr = parse(cmd.expr, context)
    rewrite = REWRITES[cmd.op]
    with measured() as elapsed, sympy_guard(cmd.op, reason=f"{cmd.op}_failed"):
        value = rewrite(expr)
    return make_result(value, cas_ms=elapsed[0])


__all__ = ["DEFAULT_VAR", "REWRITES", "run"]
