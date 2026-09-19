"""``op="plot"`` -- render an expression over an interval as PNG bytes.

The result's ``latex``/``plain``/``srepr`` still describe the *expression* (the API needs them
for ``operations.result_latex``), while ``plot_png`` carries the picture and ``kind`` is
``"plot"``. The API stores those bytes as an asset and returns ``plot_url``
(docs/mimari/api.md §3.4); nothing outside this package ever renders a figure.
"""

from __future__ import annotations

from aicls.mathengine.models import Plot, Result
from aicls.mathengine.ops.common import build_context, make_result, measured, parse, sympy_guard
from aicls.mathengine.plotting import DEFAULT_SAMPLES, render_plot


def run(cmd: Plot) -> Result:
    """Plot ``cmd.expr`` over ``[cmd.xmin, cmd.xmax]``.

    Args:
        cmd: The typed command.

    Returns:
        A :class:`~aicls.mathengine.models.Result` of kind ``plot`` whose ``plot_png`` holds
        the deterministic PNG bytes.

    Raises:
        aicls.mathengine.errors.UnsupportedProblem: If the expression has no finite real value
            on the interval, or depends on a symbol other than ``cmd.var``.
    """
    context = build_context(cmd.var)
    expr = parse(cmd.expr, context)
    with measured() as elapsed, sympy_guard(cmd.op, reason="plot_failed"):
        png = render_plot(expr, context.var, cmd.xmin, cmd.xmax, fmt="png", samples=DEFAULT_SAMPLES)
    return make_result(expr, kind="plot", plot_png=png, cas_ms=elapsed[0])


__all__ = ["run"]
