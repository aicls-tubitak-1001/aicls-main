"""Ablatable, ordered preprocessing pipeline tests (proposal A.2.1, ADR-0013)."""

from __future__ import annotations

import numpy as np
import pytest

from aicls.ocr import OcrInputError, PreprocessConfig, preprocess_image
from aicls.ocr.preprocess import PREPROCESS_STEPS


def checkerboard(h: int = 24, w: int = 40) -> np.ndarray:
    rng = np.random.default_rng(1234)
    img = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    img[8:16, 10:30] = 0
    return img


class TestPreprocessConfig:
    def test_default_config_is_a_noop_pipeline(self) -> None:
        cfg = PreprocessConfig()
        assert tuple(cfg.steps) == ()
        assert cfg.size == (96, 384)

    def test_steps_are_stored_in_the_given_order(self) -> None:
        cfg = PreprocessConfig(steps=["normalize01", "grayscale"])
        assert list(cfg.steps) == ["normalize01", "grayscale"]

    def test_unknown_step_is_rejected(self) -> None:
        with pytest.raises(OcrInputError) as exc:
            PreprocessConfig(steps=["sharpen_everything"])
        assert "sharpen_everything" in str(exc.value.details)

    def test_invalid_size_is_rejected(self) -> None:
        with pytest.raises(OcrInputError):
            PreprocessConfig(size=(0, 10))

    def test_config_is_hashable_and_reportable(self) -> None:
        cfg = PreprocessConfig(steps=["grayscale"])
        assert cfg.as_dict() == {
            "steps": ["grayscale"],
            "size": [96, 384],
            "blur_sigma": cfg.blur_sigma,
            "sharpen_amount": cfg.sharpen_amount,
            "pad_value": cfg.pad_value,
        }
        assert hash(cfg) == hash(PreprocessConfig(steps=["grayscale"]))


class TestPipeline:
    def test_preprocess_noop_when_all_flags_off(self) -> None:
        img = checkerboard()
        out = preprocess_image(img, PreprocessConfig(steps=[]))
        assert out.dtype == img.dtype
        assert out.shape == img.shape
        assert np.array_equal(out, img)

    def test_preprocess_steps_are_ablatable_and_ordered(self) -> None:
        img = checkerboard().astype(np.float32) / 255.0
        blur_then_sharpen = preprocess_image(
            img, PreprocessConfig(steps=["gaussian_blur", "laplacian_sharpen"])
        )
        sharpen_then_blur = preprocess_image(
            img, PreprocessConfig(steps=["laplacian_sharpen", "gaussian_blur"])
        )
        only_blur = preprocess_image(img, PreprocessConfig(steps=["gaussian_blur"]))
        only_sharpen = preprocess_image(img, PreprocessConfig(steps=["laplacian_sharpen"]))

        # ordered: swapping two steps changes the result
        assert not np.allclose(blur_then_sharpen, sharpen_then_blur)
        # ablatable: removing a step changes the result, every step is observable
        assert not np.allclose(blur_then_sharpen, only_blur)
        assert not np.allclose(blur_then_sharpen, only_sharpen)
        assert not np.allclose(only_blur, img)
        assert not np.allclose(only_sharpen, img)

    def test_full_pipeline_produces_model_input(self) -> None:
        rgb = np.zeros((30, 50, 3), dtype=np.uint8)
        rgb[5:20, 5:40] = 200
        cfg = PreprocessConfig(
            steps=["gaussian_blur", "laplacian_sharpen", "grayscale", "normalize01", "resize_pad"],
            size=(48, 96),
        )
        out = preprocess_image(rgb, cfg)
        assert out.shape == (48, 96)
        assert out.dtype == np.float32
        assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0

    def test_every_declared_step_runs(self) -> None:
        img = checkerboard()
        for step in PREPROCESS_STEPS:
            out = preprocess_image(img, PreprocessConfig(steps=[step], size=(16, 20)))
            assert out.size > 0

    def test_grayscale_reduces_channels(self) -> None:
        rgb = np.dstack([np.full((4, 4), 255, np.uint8), np.zeros((4, 4), np.uint8)] * 2)[:, :, :3]
        out = preprocess_image(rgb, PreprocessConfig(steps=["grayscale"]))
        assert out.shape == (4, 4)

    def test_grayscale_on_gray_input_is_identity(self) -> None:
        img = checkerboard()
        out = preprocess_image(img, PreprocessConfig(steps=["grayscale"]))
        assert np.array_equal(out, img)

    def test_normalize01_scales_uint8_to_unit_range(self) -> None:
        img = np.array([[0, 128, 255]], dtype=np.uint8)
        out = preprocess_image(img, PreprocessConfig(steps=["normalize01"]))
        assert out.dtype == np.float32
        assert out[0].tolist() == pytest.approx([0.0, 128 / 255, 1.0])

    def test_normalize01_is_idempotent_for_unit_floats(self) -> None:
        img = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
        out = preprocess_image(img, PreprocessConfig(steps=["normalize01"]))
        assert np.allclose(out, img)

    def test_resize_pad_keeps_aspect_and_pads(self) -> None:
        img = np.zeros((10, 100), dtype=np.float32)
        out = preprocess_image(img, PreprocessConfig(steps=["resize_pad"], size=(96, 384)))
        assert out.shape == (96, 384)
        # a 1:10 source padded into a 1:4 canvas leaves background rows
        assert float(out.max()) == pytest.approx(1.0)

    def test_resize_pad_is_deterministic(self) -> None:
        img = checkerboard()
        cfg = PreprocessConfig(steps=["resize_pad"], size=(24, 24))
        assert np.array_equal(preprocess_image(img, cfg), preprocess_image(img, cfg))

    def test_gaussian_blur_smooths_an_impulse(self) -> None:
        img = np.zeros((9, 9), dtype=np.float32)
        img[4, 4] = 1.0
        out = preprocess_image(img, PreprocessConfig(steps=["gaussian_blur"]))
        assert out[4, 4] < 1.0
        assert out[4, 3] > 0.0
        assert float(out.sum()) == pytest.approx(1.0, abs=1e-3)

    def test_laplacian_sharpen_increases_local_contrast(self) -> None:
        img = np.zeros((9, 9), dtype=np.float32)
        img[:, 4:] = 0.5
        out = preprocess_image(img, PreprocessConfig(steps=["laplacian_sharpen"]))
        assert out[4, 4] >= img[4, 4]
        assert out[4, 3] <= img[4, 3]

    @pytest.mark.parametrize(
        "bad",
        [np.zeros((4,), dtype=np.uint8), np.zeros((2, 2, 2, 2), dtype=np.uint8)],
    )
    def test_invalid_image_rank_rejected(self, bad: np.ndarray) -> None:
        with pytest.raises(OcrInputError):
            preprocess_image(bad, PreprocessConfig(steps=["grayscale"]))

    def test_accepts_ocr_image_wrapper(self) -> None:
        from aicls.ocr import Image

        out = preprocess_image(Image(checkerboard()), PreprocessConfig(steps=["normalize01"]))
        assert out.dtype == np.float32
