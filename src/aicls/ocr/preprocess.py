"""Ablatable image preprocessing (proposal A.2.1, ADR-0013).

The proposal names four preprocessing operations (Gaussian blur, Laplacian sharpening, grayscale
conversion, ``[0, 1]`` normalisation); the model input additionally needs a fixed canvas
(``resize_pad``). None of them is hard-wired: :class:`PreprocessConfig` carries an **ordered**
list of steps, so every combination is a configuration key that the ablation table in the İP-3
report can switch on and off (``training/configs/ocr/ablation_preproc.yaml``). An empty list is a
strict no-op — the array comes back unchanged, dtype included.

All steps are pure numpy, deterministic and dtype-preserving where that makes sense:
integer input stays integer until ``normalize01`` turns it into float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from aicls.ocr.types import Image, OcrInputError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

PreprocessStep = Literal[
    "gaussian_blur", "laplacian_sharpen", "grayscale", "normalize01", "resize_pad"
]

#: The five supported steps, in the canonical order used by the ablation table.
PREPROCESS_STEPS: tuple[str, ...] = (
    "gaussian_blur",
    "laplacian_sharpen",
    "grayscale",
    "normalize01",
    "resize_pad",
)

#: Luma weights (ITU-R BT.601) used by the grayscale step.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)
_LAPLACIAN = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """Ordered, ablatable preprocessing pipeline.

    Attributes:
        steps: Steps to apply, in order; an empty tuple is a no-op.
        size: Target ``(height, width)`` of ``resize_pad``.
        blur_sigma: Gaussian sigma in pixels.
        sharpen_amount: Weight of the Laplacian response added back to the image.
        pad_value: Padding value of ``resize_pad``, in the *normalised* range (paper = 1.0).
    """

    steps: Sequence[PreprocessStep] = ()
    size: tuple[int, int] = (96, 384)
    blur_sigma: float = 1.0
    sharpen_amount: float = 1.0
    pad_value: float = 1.0

    def __post_init__(self) -> None:
        steps = tuple(self.steps)
        unknown = [step for step in steps if step not in PREPROCESS_STEPS]
        if unknown:
            raise OcrInputError(
                details={"reason": "unknown_step", "steps": unknown, "allowed": PREPROCESS_STEPS}
            )
        size = tuple(int(v) for v in self.size)
        if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
            raise OcrInputError(details={"reason": "preprocess_size", "size": list(self.size)})
        if self.blur_sigma <= 0:
            raise OcrInputError(details={"reason": "blur_sigma", "value": float(self.blur_sigma)})
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "size", (size[0], size[1]))

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (goes into ``config.yaml`` and the run manifest)."""
        return {
            "steps": list(self.steps),
            "size": [self.size[0], self.size[1]],
            "blur_sigma": self.blur_sigma,
            "sharpen_amount": self.sharpen_amount,
            "pad_value": self.pad_value,
        }


def gaussian_kernel1d(sigma: float) -> npt.NDArray[np.float64]:
    """Normalised 1-D Gaussian kernel with radius ``ceil(3 * sigma)``."""
    radius = max(int(np.ceil(3.0 * sigma)), 1)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _convolve2d(
    plane: npt.NDArray[np.float64], kernel: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """2-D convolution with reflected borders (small kernels, no scipy dependency)."""
    kh, kw = kernel.shape
    padded = np.pad(plane, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kh, kw))
    return np.einsum("ijkl,kl->ij", windows, kernel)


def _per_plane(array: npt.NDArray[np.float64], func: Any) -> npt.NDArray[np.float64]:
    """Apply ``func`` to every channel of a 2-D or 3-D array."""
    if array.ndim == 2:
        return func(array)
    return np.stack([func(array[:, :, c]) for c in range(array.shape[2])], axis=2)


def _as_float(array: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    return array.astype(np.float64, copy=False)


def _restore_dtype(
    result: npt.NDArray[np.float64], reference: npt.NDArray[Any]
) -> npt.NDArray[Any]:
    """Clip and cast ``result`` back to the dtype of ``reference``."""
    if np.issubdtype(reference.dtype, np.integer):
        info = np.iinfo(reference.dtype)
        return np.clip(np.round(result), info.min, info.max).astype(reference.dtype)
    low = min(0.0, float(np.min(reference))) if reference.size else 0.0
    high = max(1.0, float(np.max(reference))) if reference.size else 1.0
    return np.clip(result, low, high).astype(reference.dtype, copy=False)


def _gaussian_blur(array: npt.NDArray[Any], cfg: PreprocessConfig) -> npt.NDArray[Any]:
    kernel1d = gaussian_kernel1d(cfg.blur_sigma)
    kernel = np.outer(kernel1d, kernel1d).astype(np.float64)
    blurred = _per_plane(_as_float(array), lambda plane: _convolve2d(plane, kernel))
    return _restore_dtype(blurred, array)


def _laplacian_sharpen(array: npt.NDArray[Any], cfg: PreprocessConfig) -> npt.NDArray[Any]:
    source = _as_float(array)
    response = _per_plane(source, lambda plane: _convolve2d(plane, _LAPLACIAN))
    return _restore_dtype(source + cfg.sharpen_amount * response, array)


def _grayscale(array: npt.NDArray[Any], cfg: PreprocessConfig) -> npt.NDArray[Any]:
    del cfg
    if array.ndim == 2:
        return array
    if array.shape[2] == 1:
        return array[:, :, 0]
    weighted = _as_float(array[:, :, :3]) @ _LUMA
    return _restore_dtype(weighted, array)


def _normalize01(array: npt.NDArray[Any], cfg: PreprocessConfig) -> npt.NDArray[Any]:
    del cfg
    values = array.astype(np.float32, copy=True)
    if np.issubdtype(array.dtype, np.integer) or (values.size and float(values.max()) > 1.0):
        values /= 255.0
    return np.clip(values, 0.0, 1.0, out=values)


def _resize_pad(array: npt.NDArray[Any], cfg: PreprocessConfig) -> npt.NDArray[Any]:
    from PIL import Image as PILImage

    target_h, target_w = cfg.size
    source = array if array.ndim == 2 else _grayscale(array, cfg)
    float_source = source.astype(np.float32, copy=False)
    scale_to_255 = float(float_source.max()) <= 1.0 if float_source.size else True
    pixels = float_source * 255.0 if scale_to_255 else float_source
    src_h, src_w = pixels.shape
    ratio = min(target_h / src_h, target_w / src_w)
    new_h = max(1, min(target_h, round(src_h * ratio)))
    new_w = max(1, min(target_w, round(src_w * ratio)))
    resized = PILImage.fromarray(np.clip(pixels, 0.0, 255.0).astype(np.uint8)).resize(
        (new_w, new_h), resample=PILImage.Resampling.BILINEAR
    )
    resized_array = np.asarray(resized, dtype=np.float32) / 255.0
    if not scale_to_255:
        resized_array = resized_array * 255.0

    canvas = np.full((target_h, target_w), float(cfg.pad_value), dtype=np.float32)
    if not scale_to_255:
        canvas *= 255.0
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized_array
    return canvas


_STEP_FUNCTIONS: dict[str, Any] = {
    "gaussian_blur": _gaussian_blur,
    "laplacian_sharpen": _laplacian_sharpen,
    "grayscale": _grayscale,
    "normalize01": _normalize01,
    "resize_pad": _resize_pad,
}


def preprocess_image(img: npt.NDArray[Any] | Image, cfg: PreprocessConfig) -> npt.NDArray[Any]:
    """Run the configured steps, in order, over one image.

    Args:
        img: ``(H, W)`` or ``(H, W, C)`` pixels (numpy array or :class:`~aicls.ocr.types.Image`).
        cfg: The pipeline; ``cfg.steps == ()`` returns the input unchanged.

    Returns:
        The processed array. With ``normalize01``/``resize_pad`` the result is float32 in
        ``[0, 1]``; otherwise the input dtype is preserved.

    Raises:
        OcrInputError: If the array is not 2-D or 3-D.
    """
    array = img.array if isinstance(img, Image) else np.asarray(img)
    if array.ndim not in (2, 3):
        raise OcrInputError(details={"reason": "image_rank", "shape": list(array.shape)})
    if not cfg.steps:
        return array
    for step in cfg.steps:
        array = _STEP_FUNCTIONS[step](array, cfg)
    return array


__all__ = [
    "PREPROCESS_STEPS",
    "PreprocessConfig",
    "PreprocessStep",
    "gaussian_kernel1d",
    "preprocess_image",
]
