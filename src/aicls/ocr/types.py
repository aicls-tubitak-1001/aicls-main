"""Handwriting data types shared by every OCR path (ADR-0013).

Two input families live behind one :class:`Recognizer` interface: online traces
(:class:`Ink`, a list of :class:`Stroke`) and offline rasters (:class:`Image`). Both are plain,
validated records: no torch, no I/O. Sizes are capped here (docs/mimari/api.md §1 upload limits)
so an oversized payload is rejected before any model touches it.

Coordinate convention: ``x`` grows to the right, ``y`` grows downwards (browser Canvas), ``t`` is
milliseconds relative to :attr:`Ink.t0_ms` (the absolute epoch offset kept in ``meta``), so
float32 keeps stamps exact for the whole session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from aicls.core.errors import AiclsError, ErrorCode

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

#: Maximum number of trace points accepted in one ink payload (docs/mimari/api.md §1).
MAX_POINTS_PER_INK = 20_000
#: Maximum number of strokes accepted in one ink payload.
MAX_STROKES_PER_INK = 2_000
#: Maximum side length (pixels) of an uploaded raster.
MAX_IMAGE_SIDE = 4096
#: Maximum encoded size (bytes) of an uploaded raster (docs/mimari/api.md §1).
MAX_IMAGE_BYTES = 2 * 1024 * 1024


class OcrInputError(AiclsError):
    """Handwriting input is invalid or unsupported (``OCR_INPUT_INVALID``, HTTP 400).

    Args:
        message_tr: Optional Turkish override; defaults to the registered
            ``OCR_INPUT_INVALID`` message.
        details: Structured diagnostics (``reason``, ``shape``, ``limit``, ...) that the API
            layer copies into the problem-details document.
    """

    default_code = ErrorCode.OCR_INPUT_INVALID

    def __init__(
        self,
        message_tr: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(self.default_code, message_tr, details)


@dataclass(frozen=True, slots=True)
class Stroke:
    """One pen-down..pen-up trace: an ``(N, 3)`` float32 array of ``x, y, t``.

    The array is copied and made read-only, so a :class:`Stroke` is safe to share between the
    dataset pipeline, the renderer and the recogniser.
    """

    points: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        try:
            array = np.array(self.points, dtype=np.float32, copy=True)
        except (TypeError, ValueError) as exc:
            raise OcrInputError(
                details={"reason": "points_not_numeric", "error": str(exc)}
            ) from exc
        if array.ndim != 2 or array.shape[1] != 3:
            raise OcrInputError(
                details={"reason": "points_shape", "shape": list(array.shape), "expected": "(N,3)"}
            )
        if array.shape[0] == 0:
            raise OcrInputError(details={"reason": "empty_stroke"})
        if array.shape[0] > MAX_POINTS_PER_INK:
            raise OcrInputError(
                details={
                    "reason": "too_many_points",
                    "points": int(array.shape[0]),
                    "limit": MAX_POINTS_PER_INK,
                }
            )
        if not bool(np.isfinite(array).all()):
            raise OcrInputError(details={"reason": "non_finite_points"})
        array.setflags(write=False)
        object.__setattr__(self, "points", array)

    @classmethod
    def from_xyt(
        cls,
        xs: Sequence[float] | npt.NDArray[Any],
        ys: Sequence[float] | npt.NDArray[Any],
        ts: Sequence[float] | npt.NDArray[Any] | None = None,
    ) -> Stroke:
        """Build a stroke from parallel coordinate sequences.

        Args:
            xs: X coordinates.
            ys: Y coordinates (same length as ``xs``).
            ts: Timestamps in milliseconds; ``0, 1, 2, ...`` when omitted.

        Raises:
            OcrInputError: If the sequences have different lengths or are not numeric.
        """
        x_arr = np.asarray(xs, dtype=np.float64).reshape(-1)
        y_arr = np.asarray(ys, dtype=np.float64).reshape(-1)
        if x_arr.size != y_arr.size:
            raise OcrInputError(
                details={"reason": "xy_length_mismatch", "x": int(x_arr.size), "y": int(y_arr.size)}
            )
        if ts is None:
            t_arr = np.arange(x_arr.size, dtype=np.float64)
        else:
            t_arr = np.asarray(ts, dtype=np.float64).reshape(-1)
            if t_arr.size != x_arr.size:
                raise OcrInputError(
                    details={"reason": "t_length_mismatch", "t": int(t_arr.size)},
                )
        return cls(np.stack([x_arr, y_arr, t_arr], axis=1).astype(np.float32))

    def __len__(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_points(self) -> int:
        """Number of sampled points."""
        return int(self.points.shape[0])

    @property
    def xy(self) -> npt.NDArray[np.float32]:
        """``(N, 2)`` view of the coordinates."""
        return self.points[:, :2]

    @property
    def t(self) -> npt.NDArray[np.float32]:
        """``(N,)`` view of the timestamps (ms, relative to ``Ink.t0_ms``)."""
        return self.points[:, 2]

    @property
    def duration_ms(self) -> float:
        """Time between the first and the last sample."""
        return float(self.points[-1, 2] - self.points[0, 2])

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Bounding box as ``(x_min, y_min, x_max, y_max)``."""
        xy = self.xy
        return (
            float(xy[:, 0].min()),
            float(xy[:, 1].min()),
            float(xy[:, 0].max()),
            float(xy[:, 1].max()),
        )


@dataclass(frozen=True, slots=True)
class Ink:
    """An online handwriting sample: ordered strokes plus optional label and provenance.

    Attributes:
        strokes: Pen traces in writing order.
        label: Ground-truth (or corrected) LaTeX when known — datasets and the stub recogniser
            use it; recognisers never read it during inference.
        writer_id: Anonymous writer/session identifier used for speaker/writer-disjoint splits.
        meta: Free-form provenance (``t0_ms``, ``canvas_width``, ``source``, ...).
    """

    strokes: list[Stroke]
    label: str | None = None
    writer_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            raw = list(self.strokes)
        except TypeError as exc:
            raise OcrInputError(details={"reason": "strokes_not_iterable"}) from exc
        if len(raw) > MAX_STROKES_PER_INK:
            raise OcrInputError(
                details={
                    "reason": "too_many_strokes",
                    "strokes": len(raw),
                    "limit": MAX_STROKES_PER_INK,
                }
            )
        strokes = [s if isinstance(s, Stroke) else Stroke(s) for s in raw]
        total = sum(s.n_points for s in strokes)
        if total > MAX_POINTS_PER_INK:
            raise OcrInputError(
                details={"reason": "too_many_points", "points": total, "limit": MAX_POINTS_PER_INK}
            )
        object.__setattr__(self, "strokes", strokes)
        object.__setattr__(self, "meta", dict(self.meta))

    @property
    def n_strokes(self) -> int:
        """Number of strokes."""
        return len(self.strokes)

    @property
    def n_points(self) -> int:
        """Total number of points across all strokes."""
        return sum(s.n_points for s in self.strokes)

    @property
    def is_empty(self) -> bool:
        """True when the sample carries no point at all."""
        return self.n_points == 0

    @property
    def t0_ms(self) -> float:
        """Absolute epoch offset (ms) of the relative timestamps; ``0.0`` when unknown."""
        value = self.meta.get("t0_ms", 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @property
    def duration_ms(self) -> float:
        """Time between the first and the last sample of the whole ink."""
        if self.is_empty:
            return 0.0
        starts = [float(s.points[0, 2]) for s in self.strokes]
        ends = [float(s.points[-1, 2]) for s in self.strokes]
        return max(ends) - min(starts)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Bounding box over all strokes; all zeros for an empty ink."""
        if self.is_empty:
            return (0.0, 0.0, 0.0, 0.0)
        boxes = np.array([s.bbox for s in self.strokes], dtype=np.float64)
        return (
            float(boxes[:, 0].min()),
            float(boxes[:, 1].min()),
            float(boxes[:, 2].max()),
            float(boxes[:, 3].max()),
        )

    def all_points(self) -> npt.NDArray[np.float32]:
        """Concatenate every stroke into a single ``(N, 3)`` array."""
        if not self.strokes:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate([s.points for s in self.strokes], axis=0)


@dataclass(frozen=True, slots=True)
class Image:
    """An offline handwriting sample: ``(H, W)`` grayscale (or ``(H, W, C)``) uint8 pixels."""

    array: npt.NDArray[np.uint8]

    def __post_init__(self) -> None:
        try:
            array = np.array(self.array, dtype=np.uint8, copy=True)
        except (TypeError, ValueError) as exc:
            raise OcrInputError(details={"reason": "image_not_numeric", "error": str(exc)}) from exc
        if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in (1, 3, 4)):
            raise OcrInputError(
                details={"reason": "image_shape", "shape": list(array.shape)},
            )
        if max(array.shape[0], array.shape[1]) > MAX_IMAGE_SIDE:
            raise OcrInputError(
                details={
                    "reason": "image_too_large",
                    "shape": list(array.shape),
                    "limit": MAX_IMAGE_SIDE,
                }
            )
        array.setflags(write=False)
        object.__setattr__(self, "array", array)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        max_bytes: int = MAX_IMAGE_BYTES,
        grayscale: bool = True,
    ) -> Image:
        """Decode PNG/JPEG bytes with Pillow.

        Args:
            data: Encoded image bytes (PNG or JPEG).
            max_bytes: Reject payloads larger than this (docs/mimari/api.md §1).
            grayscale: Convert to a single ``L`` channel (the model input convention).

        Raises:
            OcrInputError: If the payload is too large or cannot be decoded as an image.
        """
        if len(data) > max_bytes:
            raise OcrInputError(
                details={"reason": "too_large", "size": len(data), "limit": max_bytes}
            )
        import io

        from PIL import Image as PILImage
        from PIL import UnidentifiedImageError

        try:
            with PILImage.open(io.BytesIO(data)) as handle:
                handle.load()
                converted = handle.convert("L") if grayscale else handle.convert("RGB")
                return cls(np.asarray(converted, dtype=np.uint8))
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise OcrInputError(
                details={"reason": "image_decode_failed", "error": str(exc)}
            ) from exc

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape."""
        return tuple(int(d) for d in self.array.shape)

    @property
    def height(self) -> int:
        """Pixel rows."""
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        """Pixel columns."""
        return int(self.array.shape[1])

    def to_float01(self) -> npt.NDArray[np.float32]:
        """Return the pixels as float32 scaled to ``[0, 1]``."""
        return (self.array.astype(np.float32) / 255.0).astype(np.float32)


@dataclass(frozen=True, slots=True)
class Recognition:
    """A recogniser's answer for one handwriting sample.

    Attributes:
        latex: Best hypothesis, already postprocessed and sanitised.
        tokens: Tokenisation of :attr:`latex` (the model's output units).
        confidence: Normalised score in ``[0, 1]``.
        alternatives: ``(latex, score)`` pairs sorted by descending score, best first.
        elapsed_ms: Wall-clock recognition time.
        model_version: Registry version string (``"stub-0.1.0"``, ``"ocr-0.3.0"``, ...).
    """

    latex: str
    tokens: list[str]
    confidence: float
    alternatives: list[tuple[str, float]]
    elapsed_ms: float
    model_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", list(self.tokens))
        object.__setattr__(
            self, "alternatives", [(str(text), float(score)) for text, score in self.alternatives]
        )
        object.__setattr__(self, "confidence", float(min(1.0, max(0.0, self.confidence))))
        object.__setattr__(self, "elapsed_ms", float(max(0.0, self.elapsed_ms)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict (``RecognizeOut``, docs/mimari/api.md §3)."""
        return {
            "latex": self.latex,
            "tokens": list(self.tokens),
            "confidence": self.confidence,
            "alternatives": [[text, score] for text, score in self.alternatives],
            "elapsed_ms": self.elapsed_ms,
            "model_version": self.model_version,
        }


@runtime_checkable
class Recognizer(Protocol):
    """Handwriting -> LaTeX interface implemented by the stub and the torch recogniser."""

    def recognize(self, inp: Ink | Image, *, beam: int = 4) -> Recognition:
        """Recognise one sample.

        Args:
            inp: An online :class:`Ink` or an offline :class:`Image`.
            beam: Beam width; ``1`` means greedy decoding.

        Returns:
            The best hypothesis with its alternatives.
        """
        ...

    def warmup(self) -> None:
        """Run whatever the first call would pay for (weight load, kernel autotune)."""
        ...


__all__ = [
    "MAX_IMAGE_BYTES",
    "MAX_IMAGE_SIDE",
    "MAX_POINTS_PER_INK",
    "MAX_STROKES_PER_INK",
    "Image",
    "Ink",
    "OcrInputError",
    "Recognition",
    "Recognizer",
    "Stroke",
]
