"""Deterministic ink -> raster rendering (the offline path of ADR-0013).

``render_ink`` is the bridge between the two input families: a browser trace can be trained and
evaluated with the same image models as an uploaded PNG. The mapping is fully determined by the
ink's bounding box, so the same strokes always produce the same pixels regardless of where on the
canvas they were written or how large the writer's handwriting is (verified by
``test_render_ink_shape_and_range_0_1`` and the translation/scale test).

Convention: paper is ``1.0`` (white), ink is ``0.0`` (black) — the same polarity as a scanned
page, so :func:`aicls.ocr.preprocess.preprocess_image` sees comparable statistics for both
families.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

from aicls.ocr.types import Image, Ink, OcrInputError

if TYPE_CHECKING:
    import numpy.typing as npt

#: Default raster size ``(H, W)`` used by the offline model family.
DEFAULT_RENDER_SIZE: tuple[int, int] = (96, 384)
#: Fraction of the shorter side kept empty around the drawing.
MARGIN_RATIO = 0.08
#: Supersampling factor used before the final box downscale (deterministic anti-aliasing).
SUPERSAMPLE = 2


def _check_size(size: tuple[int, int]) -> tuple[int, int]:
    height, width = (int(size[0]), int(size[1])) if len(size) == 2 else (0, 0)
    if height <= 0 or width <= 0:
        raise OcrInputError(details={"reason": "render_size", "size": list(size)})
    return height, width


def render_ink(
    ink: Ink,
    size: tuple[int, int] = DEFAULT_RENDER_SIZE,
    *,
    stroke_width: int = 2,
) -> npt.NDArray[np.float32]:
    """Rasterise an ink sample into an ``(H, W)`` float32 array in ``[0, 1]``.

    Args:
        ink: The handwriting sample; an empty ink renders a blank page.
        size: Target ``(height, width)`` in pixels.
        stroke_width: Pen width in output pixels (>= 1).

    Returns:
        ``(H, W)`` float32 array, background ``1.0``, ink ``0.0``.

    Raises:
        OcrInputError: If ``size`` or ``stroke_width`` is not positive.
    """
    height, width = _check_size(size)
    if int(stroke_width) < 1:
        raise OcrInputError(details={"reason": "stroke_width", "stroke_width": int(stroke_width)})

    canvas = PILImage.new("L", (width * SUPERSAMPLE, height * SUPERSAMPLE), color=255)
    if not ink.is_empty:
        draw = ImageDraw.Draw(canvas)
        pen = max(1, int(stroke_width) * SUPERSAMPLE)
        for stroke in ink.strokes:
            polyline = [
                (float(x), float(y))
                for x, y in _project(stroke.xy, ink, height * SUPERSAMPLE, width * SUPERSAMPLE)
            ]
            if len(polyline) == 1:
                x, y = polyline[0]
                radius = pen / 2.0
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=0)
            else:
                draw.line(polyline, fill=0, width=pen, joint="curve")
    if SUPERSAMPLE > 1:
        canvas = canvas.resize((width, height), resample=PILImage.Resampling.BOX)
    return (np.asarray(canvas, dtype=np.float32) / 255.0).astype(np.float32)


def render_ink_to_image(
    ink: Ink,
    size: tuple[int, int] = DEFAULT_RENDER_SIZE,
    *,
    stroke_width: int = 2,
) -> Image:
    """Same as :func:`render_ink` but wrapped in an uint8 :class:`~aicls.ocr.types.Image`."""
    raster = render_ink(ink, size, stroke_width=stroke_width)
    return Image(np.round(raster * 255.0).astype(np.uint8))


def _project(
    xy: npt.NDArray[np.float32], ink: Ink, height: int, width: int
) -> npt.NDArray[np.float64]:
    """Map ink coordinates into pixel coordinates, preserving the aspect ratio."""
    x_min, y_min, x_max, y_max = ink.bbox
    span_x = max(x_max - x_min, 1e-6)
    span_y = max(y_max - y_min, 1e-6)
    margin = MARGIN_RATIO * min(height, width)
    usable_h = max(height - 2.0 * margin, 1.0)
    usable_w = max(width - 2.0 * margin, 1.0)
    scale = min(usable_w / span_x, usable_h / span_y)
    offset_x = (width - span_x * scale) / 2.0
    offset_y = (height - span_y * scale) / 2.0
    points = np.asarray(xy, dtype=np.float64)
    out = np.empty_like(points)
    out[:, 0] = (points[:, 0] - x_min) * scale + offset_x
    out[:, 1] = (points[:, 1] - y_min) * scale + offset_y
    return out


__all__ = ["DEFAULT_RENDER_SIZE", "MARGIN_RATIO", "render_ink", "render_ink_to_image"]
