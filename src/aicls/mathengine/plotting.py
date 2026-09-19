"""Deterministic Matplotlib plots rendered to bytes (ADR-0016).

Three properties matter and each is engineered for, not hoped for:

* **Headless.** The Agg backend is selected explicitly and the figure is built through the
  object API (``Figure`` + ``FigureCanvasAgg``), never ``pyplot``. ``pyplot`` keeps global
  figure state, which would leak between requests and is not safe inside a sandbox worker.
* **Deterministic.** Every rc parameter that can vary between machines or user rc files is
  pinned inside a :func:`matplotlib.rc_context`, the sampling grid is a fixed ``linspace`` and
  the PNG ``Software`` tag (which carries the Matplotlib version) is stripped. The same
  expression therefore hashes to the same bytes -- the property
  ``test_plot_renders_deterministic_png_hash`` locks down.
* **Bounded.** 6.4x4.0 inches at 100 dpi keeps a curve under the ~300 kB the ADR budgets for
  moving ``Result.plot_png`` back through the sandbox pipe.

Matplotlib is imported lazily by :func:`load_backend`, not at module import: ``aicls.nlu``
imports the command models from this package on every request path, and paying ~200 ms of
Matplotlib import for a derivative would eat a tenth of the 2.5 s budget for nothing. Sandbox
workers call :func:`load_backend` in their warm-up instead, so a *plot* request still finds the
backend ready.

Turkish labels come from :func:`aicls.mathengine.messages.tr` (see that module for why the
strings are not yet in ``aicls.core.messages_tr``).
"""

from __future__ import annotations

from functools import lru_cache
import io
from typing import Any, Literal, TypeAlias

import numpy as np
import sympy

from aicls.core.logging import get_logger
from aicls.mathengine.errors import UnsupportedProblem
from aicls.mathengine.messages import tr


@lru_cache(maxsize=1)
def load_backend() -> tuple[Any, Any, Any]:
    """Import Matplotlib, force the Agg backend and return ``(matplotlib, Figure, Canvas)``.

    Cached, so the cost is paid once per process. Sandbox workers call this during warm-up
    (:func:`aicls.mathengine.sandbox.worker_loop`) rather than on the first plot request.

    Returns:
        The ``matplotlib`` module, the ``Figure`` class and the ``FigureCanvasAgg`` class.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    # Imported after use("Agg"): the backend must be chosen before a canvas class is created.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    return matplotlib, Figure, FigureCanvasAgg


#: Output formats :func:`render_plot` can produce.
PlotFormat: TypeAlias = Literal["png", "svg"]

#: Figure geometry, fixed so that byte-for-byte comparison is meaningful.
FIGURE_SIZE_INCHES: tuple[float, float] = (6.4, 4.0)
FIGURE_DPI = 100

#: Number of samples when the caller does not say.
DEFAULT_SAMPLES = 400

#: Bounds on the sample count (a plot is a picture, not a numerical experiment).
MIN_SAMPLES = 16
MAX_SAMPLES = 4000

#: Fraction of the value range trimmed away when an asymptote blows the y-axis up.
Y_PERCENTILE = 2.0

#: rc parameters pinned for reproducibility; anything not listed here is Matplotlib's default.
DETERMINISTIC_RC: dict[str, Any] = {
    "figure.dpi": FIGURE_DPI,
    "savefig.dpi": FIGURE_DPI,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10.0,
    "axes.grid": True,
    "axes.unicode_minus": False,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
    "lines.linewidth": 1.6,
    "path.simplify": False,
    "svg.hashsalt": "aicls",
    "svg.fonttype": "path",
    "figure.autolayout": True,
}

#: PNG/SVG metadata keys removed so the output does not embed a version string or a timestamp.
_METADATA: dict[PlotFormat, dict[str, None]] = {
    "png": {"Software": None},
    "svg": {"Date": None},
}

_logger = get_logger(__name__)


def _sample(expr: sympy.Basic, var: sympy.Symbol, xs: Any) -> Any:
    """Evaluate ``expr`` on ``xs``, returning a float array with non-real points as NaN."""
    try:
        func = sympy.lambdify(var, expr, "numpy")
    except Exception as exc:
        raise UnsupportedProblem("lambdify_failed", details={"error": type(exc).__name__}) from exc
    with np.errstate(all="ignore"):
        try:
            raw = func(xs)
        except Exception:
            # A function NumPy cannot vectorise (piecewise, special branch): fall back to a
            # point-by-point evaluation, which is slower but always available.
            raw = np.array([_scalar(func, value) for value in xs], dtype=complex)
        values = np.asarray(raw, dtype=complex) * np.ones_like(xs, dtype=complex)
        ys = np.where(np.abs(values.imag) < 1e-9, values.real, np.nan)
    return np.asarray(ys, dtype=float)


def _scalar(func: Any, value: float) -> complex:
    with np.errstate(all="ignore"):
        try:
            return complex(func(value))
        except Exception:
            # An undefined point (a pole, log of a negative number) becomes a gap in the curve.
            return complex("nan")


def _y_limits(ys: Any) -> tuple[float, float] | None:
    """Robust y-range: trim the outer percentiles so one asymptote cannot flatten the curve."""
    finite = ys[np.isfinite(ys)]
    if finite.size == 0:
        return None
    low = float(np.percentile(finite, Y_PERCENTILE))
    high = float(np.percentile(finite, 100.0 - Y_PERCENTILE))
    if not np.isfinite(low) or not np.isfinite(high) or high - low <= 0.0:
        low, high = float(finite.min()), float(finite.max())
    if high - low <= 0.0:
        centre = low
        low, high = centre - 1.0, centre + 1.0
    margin = (high - low) * 0.05
    return low - margin, high + margin


def render_plot(
    expr: sympy.Basic,
    var: sympy.Symbol,
    xmin: float,
    xmax: float,
    *,
    fmt: PlotFormat = "png",
    samples: int = DEFAULT_SAMPLES,
) -> bytes:
    """Render ``expr`` over ``[xmin, xmax]`` and return the image bytes.

    Args:
        expr: The expression to draw; it must depend on at most ``var``.
        var: The free variable placed on the x-axis.
        xmin: Left edge of the interval.
        xmax: Right edge of the interval (must be greater than ``xmin``).
        fmt: ``"png"`` (default, what the API stores as an asset) or ``"svg"``.
        samples: Number of points on the grid, between :data:`MIN_SAMPLES` and
            :data:`MAX_SAMPLES`.

    Returns:
        The encoded image. A PNG starts with the 8-byte PNG signature.

    Raises:
        UnsupportedProblem: If the range is degenerate, the expression has free symbols other
            than ``var``, it cannot be turned into a numeric function, or it has no finite real
            value anywhere on the grid (a plot of nothing is not a plot).
    """
    if not (xmax > xmin) or not all(map(np.isfinite, (xmin, xmax))):
        raise UnsupportedProblem("empty_plot_range", details={"xmin": xmin, "xmax": xmax})
    extra = {str(symbol) for symbol in expr.free_symbols} - {str(var)}
    if extra:
        raise UnsupportedProblem("free_symbols_in_plot", details={"symbols": sorted(extra)})
    count = max(MIN_SAMPLES, min(MAX_SAMPLES, samples))

    xs = np.linspace(float(xmin), float(xmax), count)
    ys = _sample(expr, var, xs)
    limits = _y_limits(ys)
    if limits is None:
        raise UnsupportedProblem(
            "no_finite_values", details={"xmin": xmin, "xmax": xmax, "samples": count}
        )

    title = tr("PLOT_TITLE", expr=sympy.sstr(expr))
    matplotlib, figure_class, canvas_class = load_backend()
    with matplotlib.rc_context(rc=DETERMINISTIC_RC):
        figure = figure_class(figsize=FIGURE_SIZE_INCHES, dpi=FIGURE_DPI)
        canvas_class(figure)
        axes = figure.add_subplot(111)
        axes.plot(xs, ys)
        axes.set_xlim(float(xmin), float(xmax))
        axes.set_ylim(*limits)
        axes.set_xlabel(tr("PLOT_X_LABEL", var=var.name))
        axes.set_ylabel(tr("PLOT_Y_LABEL", var=var.name))
        axes.set_title(title)
        axes.axhline(0.0, color="0.4", linewidth=0.8)
        axes.axvline(0.0, color="0.4", linewidth=0.8)
        buffer = io.BytesIO()
        figure.savefig(buffer, format=fmt, metadata=_METADATA[fmt])
    payload = buffer.getvalue()
    _logger.debug("mathengine.plot.rendered", fmt=fmt, samples=count, bytes=len(payload))
    return payload


__all__ = [
    "DEFAULT_SAMPLES",
    "DETERMINISTIC_RC",
    "FIGURE_DPI",
    "FIGURE_SIZE_INCHES",
    "MAX_SAMPLES",
    "MIN_SAMPLES",
    "PlotFormat",
    "load_backend",
    "render_plot",
]
