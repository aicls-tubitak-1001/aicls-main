"""The ten CAS operations and the dispatcher that runs one.

:func:`compute` is the *pure* half of the engine: no cache, no timeout, no processes. It takes
a validated :data:`~aicls.mathengine.models.CasCommand` and returns a
:class:`~aicls.mathengine.models.Result`, raising only
:class:`~aicls.core.errors.AiclsError` subclasses. That is what makes it safe to call from
three very different places -- :class:`~aicls.mathengine.executor.InProcessExecutor`, a spawned
sandbox worker, and ``aicls.eval``'s batch runner -- and what makes the whole operation layer
testable without any concurrency.
"""

from __future__ import annotations

from aicls.core.logging import get_logger
from aicls.mathengine.models import (
    CasCommand,
    Diff,
    Evaluate,
    Expand,
    Factor,
    Integrate,
    Limit,
    Plot,
    Result,
    Series,
    Simplify,
    Solve,
    check_input_size,
)
from aicls.mathengine.ops import differentiate, evaluate, integrate, limit, plot, rewrite, series
from aicls.mathengine.ops import solve as solve_op

#: The ``op`` values :func:`compute` can execute, in the order they appear in the union.
SUPPORTED_OPS: tuple[str, ...] = (
    "diff",
    "integrate",
    "limit",
    "solve",
    "simplify",
    "expand",
    "factor",
    "series",
    "evaluate",
    "plot",
)

_logger = get_logger(__name__)


def compute(cmd: CasCommand) -> Result:
    """Execute one CAS command with SymPy and return its result.

    Args:
        cmd: A validated command of the union.

    Returns:
        The :class:`~aicls.mathengine.models.Result`. ``timings["cas"]`` holds the milliseconds
        the SymPy call took; the engine adds the surrounding stages.

    Raises:
        aicls.core.errors.InputValidationError: If the command's total LaTeX payload exceeds
            :data:`~aicls.mathengine.models.MAX_INPUT_CHARS`.
        aicls.latex2sympy.ParseError: If a LaTeX field is not in the grammar.
        aicls.mathengine.errors.UnsupportedProblem: If SymPy cannot answer the problem.
    """
    check_input_size(cmd)
    _logger.debug("mathengine.compute", op=cmd.op)
    match cmd:
        case Diff():
            return differentiate.run(cmd)
        case Integrate():
            return integrate.run(cmd)
        case Limit():
            return limit.run(cmd)
        case Solve():
            return solve_op.run(cmd)
        case Simplify() | Expand() | Factor():
            return rewrite.run(cmd)
        case Series():
            return series.run(cmd)
        case Evaluate():
            return evaluate.run(cmd)
        case Plot():
            return plot.run(cmd)


__all__ = ["SUPPORTED_OPS", "compute"]
