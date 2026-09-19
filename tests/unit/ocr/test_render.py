"""Deterministic ink rasterisation tests (ADR-0013 offline path)."""

from __future__ import annotations

import numpy as np
import pytest

from aicls.ocr import Ink, OcrInputError, Stroke, render_ink

BACKGROUND = 1.0


def sample_ink() -> Ink:
    return Ink(
        strokes=[
            Stroke.from_xyt([10, 40, 70], [10, 60, 10], [0, 8, 16]),
            Stroke.from_xyt([20, 60], [40, 40], [30, 38]),
        ]
    )


class TestRenderInk:
    def test_render_ink_shape_and_range_0_1(self) -> None:
        arr = render_ink(sample_ink())
        assert arr.shape == (96, 384)
        assert arr.dtype == np.float32
        assert float(arr.min()) >= 0.0
        assert float(arr.max()) <= 1.0
        assert float(arr.min()) < 0.5, "no ink was drawn"
        assert float(arr.max()) == pytest.approx(BACKGROUND, abs=1e-6)

    def test_custom_size_is_respected(self) -> None:
        arr = render_ink(sample_ink(), size=(32, 64))
        assert arr.shape == (32, 64)

    def test_rendering_is_deterministic(self) -> None:
        first = render_ink(sample_ink())
        second = render_ink(sample_ink())
        assert np.array_equal(first, second)

    def test_translation_and_scale_invariance(self) -> None:
        base = render_ink(sample_ink())
        moved = Ink(
            strokes=[
                Stroke.from_xyt(
                    (stroke.points[:, 0] * 3.0 + 500.0).tolist(),
                    (stroke.points[:, 1] * 3.0 + 250.0).tolist(),
                    stroke.points[:, 2].tolist(),
                )
                for stroke in sample_ink().strokes
            ]
        )
        assert np.array_equal(base, render_ink(moved))

    def test_thicker_strokes_cover_more_pixels(self) -> None:
        thin = render_ink(sample_ink(), stroke_width=1)
        thick = render_ink(sample_ink(), stroke_width=5)
        assert (thick < 0.5).sum() > (thin < 0.5).sum()

    def test_empty_ink_renders_blank_canvas(self) -> None:
        arr = render_ink(Ink(strokes=[]), size=(16, 32))
        assert arr.shape == (16, 32)
        assert np.allclose(arr, BACKGROUND)

    def test_single_point_ink_renders_a_dot(self) -> None:
        arr = render_ink(Ink(strokes=[Stroke.from_xyt([5.0], [5.0], [0.0])]), size=(16, 16))
        assert float(arr.min()) < 0.9

    def test_aspect_ratio_is_preserved(self) -> None:
        tall = Ink(strokes=[Stroke.from_xyt([0, 1], [0, 100], [0, 1])])
        arr = render_ink(tall, size=(64, 256), stroke_width=1)
        inked_columns = np.flatnonzero((arr < 0.5).any(axis=0))
        assert inked_columns.size <= 6, "a vertical stroke must not be stretched horizontally"

    @pytest.mark.parametrize("size", [(0, 10), (10, 0), (-1, 4)])
    def test_invalid_size_rejected(self, size: tuple[int, int]) -> None:
        with pytest.raises(OcrInputError):
            render_ink(sample_ink(), size=size)

    def test_invalid_stroke_width_rejected(self) -> None:
        with pytest.raises(OcrInputError):
            render_ink(sample_ink(), stroke_width=0)

    def test_render_to_image_returns_uint8(self) -> None:
        from aicls.ocr import Image, render_ink_to_image

        img = render_ink_to_image(sample_ink(), size=(32, 64))
        assert isinstance(img, Image)
        assert img.array.dtype == np.uint8
        assert img.shape == (32, 64)
