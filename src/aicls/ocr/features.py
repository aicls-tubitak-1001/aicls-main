"""Online-path features: arc-length resampling, size normalisation, deltas and Bezier fits.

The online recogniser (ADR-0013, ``model.family: seq2seq_stroke``) never sees raw pixels: a
sample becomes a ``(T, F)`` float32 matrix where every row is one resampled trace point.

Feature layout (:data:`STROKE_FEATURE_NAMES`, ``F`` = :data:`STROKE_FEATURE_DIM`):

===================  ==========================================================================
``x``, ``y``         position, translated to the ink centre and divided by the larger bbox side
``dx``, ``dy``       offset from the previous point (0 at the first point of the sample)
``dt``               time since the previous point, in seconds (0 at the first point)
``pen_down``         1 on the first point of a stroke
``pen_up``           1 on the last point of a stroke
``bezier_c1x/c1y``   first cubic-Bezier control point of the stroke, relative to its start point
``bezier_c2x/c2y``   second control point, same frame — constant within a stroke
``curvature``        signed turn angle at the point, in radians / pi
===================  ==========================================================================

Normalisation is translation- and scale-invariant, so the same handwriting drawn on a phone and
on a tablet produces the same features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aicls.ocr.types import Ink, OcrInputError

if TYPE_CHECKING:
    import numpy.typing as npt

#: Ordered feature names of one row of :func:`normalize_strokes`.
STROKE_FEATURE_NAMES: tuple[str, ...] = (
    "x",
    "y",
    "dx",
    "dy",
    "dt",
    "pen_down",
    "pen_up",
    "bezier_c1x",
    "bezier_c1y",
    "bezier_c2x",
    "bezier_c2y",
    "curvature",
)
#: Number of features per trace point.
STROKE_FEATURE_DIM = len(STROKE_FEATURE_NAMES)

#: Default arc-length spacing (source pixels) used when resampling traces.
DEFAULT_RESAMPLE_PX = 2.0


def resample_polyline(points: npt.NDArray[np.float32], spacing: float) -> npt.NDArray[np.float32]:
    """Resample a polyline at a constant arc-length ``spacing``.

    Args:
        points: ``(N, 2)`` coordinates in source units.
        spacing: Distance between consecutive output points; must be positive.

    Returns:
        ``(M, 2)`` float32 points starting at ``points[0]`` and ending at ``points[-1]``. A
        polyline of zero length collapses to its single first point.

    Raises:
        OcrInputError: If ``spacing`` is not positive or the array is not ``(N, 2)``.
    """
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise OcrInputError(details={"reason": "polyline_shape", "shape": list(array.shape)})
    if not spacing > 0:
        raise OcrInputError(details={"reason": "resample_spacing", "spacing": float(spacing)})
    if array.shape[0] < 2:
        return array.astype(np.float32, copy=True)

    steps = np.linalg.norm(np.diff(array, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(cumulative[-1])
    if total <= 0.0:
        return array[:1].astype(np.float32, copy=True)

    count = max(int(np.floor(total / spacing)), 1)
    targets = np.linspace(0.0, total, count + 1)
    out = np.empty((targets.size, 2), dtype=np.float32)
    out[:, 0] = np.interp(targets, cumulative, array[:, 0])
    out[:, 1] = np.interp(targets, cumulative, array[:, 1])
    out[0] = array[0]
    out[-1] = array[-1]
    return out


def fit_cubic_bezier(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Least-squares fit of one cubic Bezier curve to a stroke.

    Endpoints are pinned to the first and last point; the two interior control points are solved
    with chord-length parameterisation (the classic Schneider fit without subdivision).

    Args:
        points: ``(N, 2)`` stroke coordinates.

    Returns:
        ``(4, 2)`` float32 control points ``P0, P1, P2, P3``.

    Raises:
        OcrInputError: If the array is not ``(N, 2)`` with at least one point.
    """
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0:
        raise OcrInputError(details={"reason": "bezier_shape", "shape": list(array.shape)})
    start, end = array[0], array[-1]
    if array.shape[0] < 4:
        control1 = start + (end - start) / 3.0
        control2 = start + 2.0 * (end - start) / 3.0
        return np.stack([start, control1, control2, end]).astype(np.float32)

    steps = np.linalg.norm(np.diff(array, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    total = cumulative[-1]
    u = cumulative / total if total > 0 else np.linspace(0.0, 1.0, array.shape[0])

    b0 = (1 - u) ** 3
    b1 = 3 * u * (1 - u) ** 2
    b2 = 3 * u**2 * (1 - u)
    b3 = u**3
    residual = array - (b0[:, None] * start + b3[:, None] * end)
    design = np.stack([b1, b2], axis=1)
    normal = design.T @ design
    try:
        solution = np.linalg.solve(normal, design.T @ residual)
    except np.linalg.LinAlgError:  # pragma: no cover - singular only for degenerate strokes
        solution = np.stack([start + (end - start) / 3.0, start + 2.0 * (end - start) / 3.0])
    return np.stack([start, solution[0], solution[1], end]).astype(np.float32)


def _curvature(xy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Signed turn angle at every point, normalised to ``[-1, 1]`` (radians / pi)."""
    n = xy.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if n < 3:
        return out
    previous = xy[1:-1] - xy[:-2]
    following = xy[2:] - xy[1:-1]
    cross = previous[:, 0] * following[:, 1] - previous[:, 1] * following[:, 0]
    dot = (previous * following).sum(axis=1)
    out[1:-1] = np.arctan2(cross, dot) / np.pi
    return out


def normalize_strokes(
    ink: Ink, *, resample_px: float = DEFAULT_RESAMPLE_PX
) -> npt.NDArray[np.float32]:
    """Turn an :class:`~aicls.ocr.types.Ink` into the ``(T, F)`` online feature matrix.

    Args:
        ink: The handwriting sample.
        resample_px: Arc-length spacing (in source pixels) used to resample every stroke; the
            spacing is scaled with the ink so the result is size invariant.

    Returns:
        ``(T, STROKE_FEATURE_DIM)`` float32 features; ``(0, F)`` for an empty ink.

    Raises:
        OcrInputError: If ``resample_px`` is not positive.
    """
    if not resample_px > 0:
        raise OcrInputError(details={"reason": "resample_spacing", "spacing": float(resample_px)})
    if ink.is_empty:
        return np.zeros((0, STROKE_FEATURE_DIM), dtype=np.float32)

    x_min, y_min, x_max, y_max = ink.bbox
    scale = max(x_max - x_min, y_max - y_min)
    if scale <= 0.0:
        scale = 1.0
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    rows: list[npt.NDArray[np.float64]] = []
    for stroke in ink.strokes:
        xy = np.asarray(stroke.xy, dtype=np.float32)
        times = np.asarray(stroke.t, dtype=np.float64)
        resampled = resample_polyline(xy, resample_px)
        # map the resampled arc positions back onto the original timeline
        if xy.shape[0] > 1 and resampled.shape[0] > 1:
            source = np.linspace(0.0, 1.0, xy.shape[0])
            target = np.linspace(0.0, 1.0, resampled.shape[0])
            resampled_t = np.interp(target, source, times)
        else:
            resampled_t = np.full(resampled.shape[0], float(times[0]))

        normalised = np.empty((resampled.shape[0], 2), dtype=np.float64)
        normalised[:, 0] = (resampled[:, 0] - center_x) / scale
        normalised[:, 1] = (resampled[:, 1] - center_y) / scale

        deltas = np.zeros_like(normalised)
        deltas[1:] = normalised[1:] - normalised[:-1]
        dt = np.zeros(resampled.shape[0], dtype=np.float64)
        dt[1:] = (resampled_t[1:] - resampled_t[:-1]) / 1000.0

        pen_down = np.zeros(resampled.shape[0], dtype=np.float64)
        pen_up = np.zeros(resampled.shape[0], dtype=np.float64)
        pen_down[0] = 1.0
        pen_up[-1] = 1.0

        control = fit_cubic_bezier(normalised.astype(np.float32)).astype(np.float64)
        c1 = np.repeat((control[1] - control[0])[None, :], resampled.shape[0], axis=0)
        c2 = np.repeat((control[2] - control[0])[None, :], resampled.shape[0], axis=0)

        block = np.column_stack(
            [
                normalised,
                deltas,
                dt,
                pen_down,
                pen_up,
                c1,
                c2,
                _curvature(normalised),
            ]
        )
        rows.append(block)

    features = np.concatenate(rows, axis=0).astype(np.float32)
    features[0, STROKE_FEATURE_NAMES.index("dx")] = 0.0
    features[0, STROKE_FEATURE_NAMES.index("dy")] = 0.0
    return np.nan_to_num(features, copy=False)


__all__ = [
    "DEFAULT_RESAMPLE_PX",
    "STROKE_FEATURE_DIM",
    "STROKE_FEATURE_NAMES",
    "fit_cubic_bezier",
    "normalize_strokes",
    "resample_polyline",
]
