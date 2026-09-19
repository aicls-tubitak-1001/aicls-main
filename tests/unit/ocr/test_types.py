"""Tests for the handwriting data types (`aicls.ocr.types`)."""

from __future__ import annotations

import numpy as np
import pytest

from aicls.core.errors import AiclsError, ErrorCode
from aicls.ocr import Image, Ink, OcrInputError, Recognition, Recognizer, Stroke


def _stroke(n: int = 5, *, x0: float = 0.0) -> Stroke:
    xs = np.linspace(x0, x0 + 10.0, n)
    ys = np.linspace(0.0, 5.0, n)
    ts = np.arange(n, dtype=np.float64) * 8.0
    return Stroke.from_xyt(xs, ys, ts)


class TestStroke:
    def test_from_xyt_builds_float32_n_by_3(self) -> None:
        s = _stroke(4)
        assert s.points.shape == (4, 3)
        assert s.points.dtype == np.float32
        assert len(s) == 4
        assert s.n_points == 4

    def test_points_are_read_only_and_copied(self) -> None:
        raw = np.zeros((3, 3), dtype=np.float32)
        s = Stroke(raw)
        raw[0, 0] = 99.0
        assert s.points[0, 0] == 0.0
        with pytest.raises(ValueError):
            s.points[0, 0] = 1.0

    def test_bbox_and_duration(self) -> None:
        s = _stroke(5)
        x0, y0, x1, y1 = s.bbox
        assert (x0, y0) == (0.0, 0.0)
        assert (x1, y1) == (10.0, 5.0)
        assert s.duration_ms == pytest.approx(32.0)

    @pytest.mark.parametrize(
        "bad",
        [
            np.zeros((3, 2), dtype=np.float32),
            np.zeros((3,), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.array([[np.nan, 0.0, 0.0]], dtype=np.float32),
        ],
    )
    def test_invalid_point_arrays_rejected(self, bad: np.ndarray) -> None:
        with pytest.raises(OcrInputError) as exc:
            Stroke(bad)
        assert exc.value.code is ErrorCode.OCR_INPUT_INVALID
        assert isinstance(exc.value, AiclsError)

    def test_too_many_points_rejected(self) -> None:
        with pytest.raises(OcrInputError):
            Stroke(np.zeros((20_001, 3), dtype=np.float32))

    def test_from_xyt_length_mismatch_rejected(self) -> None:
        with pytest.raises(OcrInputError):
            Stroke.from_xyt([0, 1], [0])
        with pytest.raises(OcrInputError):
            Stroke.from_xyt([0, 1], [0, 1], [0])

    def test_from_xyt_defaults_timestamps_to_the_index(self) -> None:
        stroke = Stroke.from_xyt([0, 1, 2], [3, 4, 5])
        assert stroke.t.tolist() == [0.0, 1.0, 2.0]

    def test_resampled_and_xy_views(self) -> None:
        s = _stroke(5)
        assert s.xy.shape == (5, 2)
        assert s.t.shape == (5,)


class TestInk:
    def test_defaults_and_meta_copy(self) -> None:
        meta = {"source": "canvas"}
        ink = Ink(strokes=[_stroke()], meta=meta)
        meta["source"] = "changed"
        assert ink.meta == {"source": "canvas"}
        assert ink.label is None
        assert ink.writer_id is None

    def test_geometry_helpers(self) -> None:
        ink = Ink(strokes=[_stroke(3), _stroke(3, x0=20.0)])
        assert ink.n_strokes == 2
        assert ink.n_points == 6
        assert ink.bbox == (0.0, 0.0, 30.0, 5.0)
        assert ink.duration_ms == pytest.approx(16.0)
        assert ink.is_empty is False

    def test_empty_ink_is_allowed(self) -> None:
        ink = Ink(strokes=[])
        assert ink.is_empty is True
        assert ink.n_points == 0
        assert ink.bbox == (0.0, 0.0, 0.0, 0.0)

    def test_arrays_accepted_and_wrapped(self) -> None:
        ink = Ink(strokes=[np.zeros((2, 3), dtype=np.float32)])  # type: ignore[list-item]
        assert isinstance(ink.strokes[0], Stroke)

    def test_point_budget_enforced_across_strokes(self) -> None:
        big = np.zeros((10_000, 3), dtype=np.float32)
        with pytest.raises(OcrInputError):
            Ink(strokes=[Stroke(big), Stroke(big), Stroke(big)])

    def test_t0_ms_from_meta(self) -> None:
        ink = Ink(strokes=[_stroke()], meta={"t0_ms": 1700.0})
        assert ink.t0_ms == 1700.0
        assert Ink(strokes=[_stroke()]).t0_ms == 0.0

    def test_t0_ms_falls_back_when_metadata_is_not_numeric(self) -> None:
        assert Ink(strokes=[_stroke()], meta={"t0_ms": "gecersiz"}).t0_ms == 0.0

    def test_too_many_strokes_rejected(self) -> None:
        single = np.zeros((1, 3), dtype=np.float32)
        with pytest.raises(OcrInputError):
            Ink(strokes=[Stroke(single) for _ in range(2_001)])

    def test_non_iterable_strokes_rejected(self) -> None:
        with pytest.raises(OcrInputError):
            Ink(strokes=42)  # type: ignore[arg-type]

    def test_all_points_concatenates(self) -> None:
        ink = Ink(strokes=[_stroke(3), _stroke(2)])
        assert ink.all_points().shape == (5, 3)
        assert Ink(strokes=[]).all_points().shape == (0, 3)

    def test_empty_ink_duration_is_zero(self) -> None:
        assert Ink(strokes=[]).duration_ms == 0.0


class TestImage:
    def test_grayscale_array_accepted(self) -> None:
        img = Image(np.zeros((4, 6), dtype=np.uint8))
        assert img.shape == (4, 6)
        assert img.height == 4
        assert img.width == 6
        assert img.to_float01().max() == 0.0

    def test_rgb_array_accepted(self) -> None:
        img = Image(np.zeros((4, 6, 3), dtype=np.uint8))
        assert img.shape == (4, 6, 3)
        assert img.width == 6

    def test_bad_arrays_rejected(self) -> None:
        with pytest.raises(OcrInputError):
            Image(np.zeros((4,), dtype=np.uint8))
        with pytest.raises(OcrInputError):
            Image(np.zeros((4, 4, 7), dtype=np.uint8))

    def test_from_bytes_reads_png(self) -> None:
        from PIL import Image as PILImage

        buf = _png_bytes(PILImage.new("L", (8, 5), color=200))
        img = Image.from_bytes(buf)
        assert img.shape == (5, 8)
        assert int(img.array[0, 0]) == 200

    def test_from_bytes_can_keep_colour(self) -> None:
        from PIL import Image as PILImage

        buf = _png_bytes(PILImage.new("RGB", (6, 4), color=(10, 20, 30)))
        img = Image.from_bytes(buf, grayscale=False)
        assert img.shape == (4, 6, 3)

    def test_oversized_raster_rejected(self) -> None:
        with pytest.raises(OcrInputError):
            Image(np.zeros((4097, 4), dtype=np.uint8))

    def test_non_numeric_array_rejected(self) -> None:
        with pytest.raises(OcrInputError):
            Image("not an array")  # type: ignore[arg-type]

    def test_from_bytes_rejects_non_image(self) -> None:
        with pytest.raises(OcrInputError):
            Image.from_bytes(b"not an image at all")

    def test_from_bytes_rejects_oversize_payload(self) -> None:
        with pytest.raises(OcrInputError):
            Image.from_bytes(b"\x89PNG" + b"0" * 100, max_bytes=16)


def _png_bytes(pil_image: object) -> bytes:
    import io

    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")  # type: ignore[attr-defined]
    return buf.getvalue()


class TestRecognition:
    def test_to_dict_is_json_friendly(self) -> None:
        rec = Recognition(
            latex="x^{2}",
            tokens=["x", "^", "{", "2", "}"],
            confidence=0.5,
            alternatives=[("x^{3}", 0.2)],
            elapsed_ms=1.5,
            model_version="stub-0.1.0",
        )
        payload = rec.to_dict()
        assert payload["latex"] == "x^{2}"
        assert payload["alternatives"] == [["x^{3}", 0.2]]
        assert payload["model_version"] == "stub-0.1.0"

    def test_confidence_is_clamped(self) -> None:
        rec = Recognition(
            latex="1",
            tokens=["1"],
            confidence=5.0,
            alternatives=[],
            elapsed_ms=0.0,
            model_version="v",
        )
        assert rec.confidence == 1.0

    def test_stub_satisfies_recognizer_protocol(self) -> None:
        from aicls.ocr import StubRecognizer

        assert isinstance(StubRecognizer(), Recognizer)
