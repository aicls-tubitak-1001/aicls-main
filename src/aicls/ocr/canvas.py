"""Browser Canvas JSON <-> :class:`~aicls.ocr.types.Ink` conversion (ADR-0013, ADR-0014).

The web client buffers pointer events and ships them as JSON, both over ``POST /ocr/recognize``
and over the ``/ws/v1/ink`` socket. Three shapes are accepted, because the front-end evolves
faster than this contract:

``{"strokes": [{"points": [{"x": 1, "y": 2, "t": 1700000000000}, ...]}]}``
``{"strokes": [{"x": [...], "y": [...], "t": [...]}]}``
``{"strokes": [[[x, y, t], ...]]}``

The drawing-surface size travels either as a nested ``{"canvas": {"w": 800, "h": 300}}`` object
(what the web client and ``docs/mimari/api.md`` §5 use) or as flat ``width``/``height`` keys; both
land in ``meta["canvas_width"]`` / ``meta["canvas_height"]``.

Timestamps are the reason this module exists: browsers send absolute epoch milliseconds, which
float32 cannot represent. The smallest stamp of the payload is stored once in ``meta["t0_ms"]``
and every point keeps an exact relative offset, so ``t0_ms + t`` reproduces the original stamp
(``test_ink_from_canvas_json_preserves_timestamps``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aicls.ocr.types import MAX_STROKES_PER_INK, Ink, OcrInputError, Stroke

#: Payload keys copied into :attr:`Ink.meta` when present.
CANVAS_META_KEYS: tuple[str, ...] = ("width", "height", "device", "version", "session_id")

#: Keys of the nested ``canvas`` object (``docs/mimari/api.md`` §5) mapped to ``meta`` entries.
CANVAS_SIZE_KEYS: dict[str, str] = {"w": "canvas_width", "h": "canvas_height"}


def _canvas_size_meta(obj: dict[str, Any]) -> dict[str, float]:
    """Read the drawing-surface size from the nested ``canvas`` object, if the payload has one.

    The web client serialises the surface as ``{"canvas": {"w": 800, "h": 300}}`` (api.md §5 and
    ``web/src/lib/inkml.ts``), while older fixtures use flat ``width``/``height`` keys. Both are
    accepted; unusable values are ignored rather than rejected, because the size is provenance
    metadata and never affects recognition.
    """
    canvas = obj.get("canvas")
    if not isinstance(canvas, dict):
        return {}
    meta: dict[str, float] = {}
    for key, meta_key in CANVAS_SIZE_KEYS.items():
        value = canvas.get(key, canvas.get(meta_key.removeprefix("canvas_")))
        if isinstance(value, int | float) and not isinstance(value, bool):
            meta[meta_key] = float(value)
    return meta


def _as_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise OcrInputError(
            details={"reason": "point_not_numeric", "field": field, "value": repr(value)[:64]}
        ) from exc
    if not np.isfinite(number):
        raise OcrInputError(details={"reason": "point_not_finite", "field": field})
    return number


def _points_from_stroke(raw: Any, index: int) -> list[tuple[float, float, float | None]]:
    """Normalise one stroke entry into ``(x, y, t | None)`` triples."""
    if isinstance(raw, dict):
        if "points" in raw:
            points = raw["points"]
            if not isinstance(points, list):
                raise OcrInputError(details={"reason": "stroke_points_type", "stroke": index})
            return [_point_from_item(item, index) for item in points]
        if "x" in raw and "y" in raw:
            xs, ys = raw["x"], raw["y"]
            ts = raw.get("t")
            if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) != len(ys):
                raise OcrInputError(details={"reason": "stroke_arrays", "stroke": index})
            if ts is not None and (not isinstance(ts, list) or len(ts) != len(xs)):
                raise OcrInputError(details={"reason": "stroke_arrays_t", "stroke": index})
            return [
                (
                    _as_float(xs[i], field="x"),
                    _as_float(ys[i], field="y"),
                    None if ts is None else _as_float(ts[i], field="t"),
                )
                for i in range(len(xs))
            ]
        raise OcrInputError(details={"reason": "stroke_shape", "stroke": index})
    if isinstance(raw, list):
        return [_point_from_item(item, index) for item in raw]
    raise OcrInputError(details={"reason": "stroke_type", "stroke": index})


def _point_from_item(item: Any, index: int) -> tuple[float, float, float | None]:
    if isinstance(item, dict):
        if "x" not in item or "y" not in item:
            raise OcrInputError(details={"reason": "point_missing_xy", "stroke": index})
        stamp = item.get("t", item.get("time"))
        return (
            _as_float(item["x"], field="x"),
            _as_float(item["y"], field="y"),
            None if stamp is None else _as_float(stamp, field="t"),
        )
    if isinstance(item, list | tuple) and len(item) >= 2:
        return (
            _as_float(item[0], field="x"),
            _as_float(item[1], field="y"),
            _as_float(item[2], field="t") if len(item) > 2 else None,
        )
    raise OcrInputError(details={"reason": "point_shape", "stroke": index})


def ink_from_canvas_json(obj: dict[str, Any]) -> Ink:
    """Convert a Canvas JSON payload into an :class:`~aicls.ocr.types.Ink`.

    Args:
        obj: The decoded JSON object; ``strokes`` is required, ``label``, ``writer_id``,
            ``width``, ``height``, ``device`` and ``session_id`` are optional.

    Returns:
        The ink with relative timestamps and ``meta["t0_ms"]`` holding the absolute origin.

    Raises:
        OcrInputError: If the payload is not an object, ``strokes`` is missing or malformed, a
            coordinate is not a finite number, or the stroke/point budgets are exceeded.
    """
    if not isinstance(obj, dict):
        raise OcrInputError(details={"reason": "payload_type", "type": type(obj).__name__})
    raw_strokes = obj.get("strokes")
    if not isinstance(raw_strokes, list):
        raise OcrInputError(details={"reason": "strokes_missing"})
    if len(raw_strokes) > MAX_STROKES_PER_INK:
        raise OcrInputError(
            details={
                "reason": "too_many_strokes",
                "strokes": len(raw_strokes),
                "limit": MAX_STROKES_PER_INK,
            }
        )

    parsed = [_points_from_stroke(raw, index) for index, raw in enumerate(raw_strokes)]
    stamps = [t for stroke in parsed for _, _, t in stroke if t is not None]
    t0 = min(stamps) if stamps else 0.0

    strokes: list[Stroke] = []
    for stroke_points in parsed:
        if not stroke_points:
            continue
        rows = [
            (x, y, (float(order) if t is None else t - t0))
            for order, (x, y, t) in enumerate(stroke_points)
        ]
        strokes.append(Stroke(np.array(rows, dtype=np.float32)))

    meta: dict[str, Any] = {"t0_ms": float(t0)}
    meta.update(_canvas_size_meta(obj))
    for key in CANVAS_META_KEYS:
        if key in obj and obj[key] is not None:
            value = obj[key]
            meta[f"canvas_{key}" if key in ("width", "height") else key] = (
                float(value) if isinstance(value, int | float) else str(value)
            )
    label = obj.get("label")
    writer_id = obj.get("writer_id", obj.get("writerId"))
    return Ink(
        strokes=strokes,
        label=str(label) if label else None,
        writer_id=str(writer_id) if writer_id else None,
        meta=meta,
    )


def ink_to_canvas_json(ink: Ink) -> dict[str, Any]:
    """Serialise an ink back into the Canvas JSON payload shape (absolute timestamps).

    The surface size is emitted twice when known: as the nested ``canvas`` object the web client
    sends and as the flat ``width``/``height`` keys older fixtures read.
    """
    t0 = ink.t0_ms
    width = ink.meta.get("canvas_width")
    height = ink.meta.get("canvas_height")
    payload: dict[str, Any] = {
        "version": 1,
        "label": ink.label,
        "writer_id": ink.writer_id,
        "width": width,
        "height": height,
        "strokes": [
            {
                "points": [
                    {"x": float(x), "y": float(y), "t": float(t) + t0}
                    for x, y, t in stroke.points.tolist()
                ]
            }
            for stroke in ink.strokes
        ],
    }
    if width is not None and height is not None:
        payload["canvas"] = {"w": float(width), "h": float(height)}
    return payload


__all__ = [
    "CANVAS_META_KEYS",
    "CANVAS_SIZE_KEYS",
    "ink_from_canvas_json",
    "ink_to_canvas_json",
]
