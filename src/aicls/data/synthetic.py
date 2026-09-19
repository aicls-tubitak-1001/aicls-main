"""Deterministic synthetic data: Bezier ink, mathtext images, sine+noise audio, InkML corpora.

Until the ``DATASET_DOWNLOAD`` gate opens (ADR-0012) this module is the *only* source of training
and smoke data in the repository, so everything here is a pure function of its seed: the same seed
gives the same strokes, the same pixels and the same samples on every machine
(``test_synthetic_generators_reproducible_with_seed``).

* :func:`synth_ink` — cubic-Bezier scribbles whose stroke count and shape are derived from the
  LaTeX label, so different expressions look different and the same expression always looks the
  same. Coordinates follow the browser Canvas convention of ADR-0013 (y grows downwards, ``t`` in
  milliseconds).
* :func:`synth_image` — matplotlib **mathtext** (no TeX installation, no network) rendered to a
  grayscale raster; unparsable input falls back to rasterising a :func:`synth_ink` sample so the
  function never fails on a caller's expression.
* :func:`synth_audio` — an utterance-shaped waveform from :mod:`aicls.asr.synth` with optional
  additive noise at a requested SNR.
* :func:`synth_inkml_dataset` — a directory of InkML files that :func:`aicls.data.prepare_shards`
  and every OCR test can consume like a miniature MathWriting.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import zlib

import numpy as np

from aicls.asr.interfaces import TARGET_SAMPLE_RATE, Audio
from aicls.asr.synth import make_speech_like
from aicls.core.logging import get_logger
from aicls.data.errors import invalid_data
from aicls.data.shards import normalize_label
from aicls.ocr import CURRICULUM_LATEX, Ink, PreprocessConfig, Stroke
from aicls.ocr import preprocess_image as _preprocess_image
from aicls.ocr import render_ink as _render_ink
from aicls.ocr.inkml import write_inkml

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy.typing as npt

_log = get_logger(__name__)

#: Points sampled along one Bezier stroke.
POINTS_PER_STROKE = 24

#: Milliseconds between two consecutive samples of a stroke.
SAMPLE_INTERVAL_MS = 12.0

#: Milliseconds of pen-up time between two strokes.
PEN_UP_MS = 90.0

#: Canvas the synthetic strokes are drawn on (pixels).
CANVAS: tuple[float, float] = (600.0, 200.0)

#: Number of distinct synthetic writers.
N_WRITERS = 8


def _as_rng(rng: np.random.Generator | int) -> np.random.Generator:
    """Accept a generator or a plain seed."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(int(rng))


def synth_ink(rng: np.random.Generator | int, latex: str) -> Ink:
    """Draw a deterministic Bezier "handwriting" sample for ``latex``.

    Args:
        rng: A ``numpy`` generator or an integer seed.
        latex: The label the sample is supposed to represent; drives the stroke count so two
            different expressions never produce the same drawing.

    Returns:
        An :class:`~aicls.ocr.types.Ink` with ``label`` set and a synthetic ``writer_id``.

    Raises:
        InputValidationError: If ``latex`` is empty.
    """
    if not latex or not latex.strip():
        raise invalid_data("empty_latex")
    generator = _as_rng(rng)
    # a stable (process-independent) hash of the label, so the same seed still draws two
    # different expressions differently
    signature = zlib.crc32(latex.encode("utf-8"))
    n_strokes = 2 + (signature % 6)
    slant = ((signature >> 8) % 100) / 500.0
    width, height = CANVAS
    strokes: list[Stroke] = []
    time_ms = 0.0
    for index in range(n_strokes):
        # each stroke lives in its own horizontal band, like glyphs on a line
        left = width * index / n_strokes
        right = width * (index + 1) / n_strokes
        control = generator.uniform(0.0, 1.0, size=(4, 2))
        control[:, 0] = left + control[:, 0] * (right - left)
        control[:, 1] = height * (0.15 + 0.7 * control[:, 1] * (1.0 - slant))
        points = _cubic_bezier(control, POINTS_PER_STROKE)
        times = time_ms + np.arange(POINTS_PER_STROKE, dtype=np.float64) * SAMPLE_INTERVAL_MS
        time_ms = float(times[-1]) + PEN_UP_MS
        strokes.append(Stroke(np.column_stack([points, times]).astype(np.float32)))
    writer = int(generator.integers(0, N_WRITERS))
    return Ink(
        strokes=strokes,
        label=latex,
        writer_id=f"W{writer:03d}",
        meta={"source": "synthetic", "canvas_width": width, "canvas_height": height, "t0_ms": 0.0},
    )


def synth_image(
    rng: np.random.Generator | int,
    latex: str,
    size: tuple[int, int] = (96, 384),
    *,
    noise: float = 0.01,
) -> npt.NDArray[np.float32]:
    """Render ``latex`` with matplotlib mathtext into a ``(H, W)`` float32 raster in ``[0, 1]``.

    Args:
        rng: Generator or seed (drives the additive noise).
        latex: Expression to typeset; ``$`` delimiters are added when missing.
        size: Target ``(height, width)``.
        noise: Standard deviation of the additive Gaussian noise.

    Returns:
        Grayscale pixels, background 1.0, ink 0.0.

    Raises:
        InputValidationError: If ``size`` is not positive.
    """
    height, width = (int(size[0]), int(size[1])) if len(size) == 2 else (0, 0)
    if height <= 0 or width <= 0:
        raise invalid_data("image_size_not_positive", size=list(size))
    generator = _as_rng(rng)
    raster = _mathtext_raster(latex, (height, width))
    if raster is None:
        raster = _render_ink(synth_ink(generator, latex or "x"), (height, width))
    if noise > 0.0:
        raster = raster + generator.normal(0.0, noise, size=raster.shape).astype(np.float32)
    return np.clip(raster, 0.0, 1.0).astype(np.float32)


def synth_audio(
    rng: np.random.Generator | int, seconds: float, *, snr_db: float | None = None
) -> Audio:
    """Build an utterance-shaped 16 kHz waveform.

    Args:
        rng: Generator or seed.
        seconds: Duration in seconds (> 0).
        snr_db: When given, additive white noise is mixed in at this signal-to-noise ratio.

    Returns:
        The waveform.

    Raises:
        InputValidationError: If ``seconds`` is not positive.
    """
    if seconds <= 0.0:
        raise invalid_data("seconds_not_positive", seconds=float(seconds))
    generator = _as_rng(rng)
    seed = int(generator.integers(0, 2**31 - 1))
    audio = make_speech_like(duration_s=float(seconds), sr=TARGET_SAMPLE_RATE, seed=seed)
    if snr_db is None:
        return audio
    samples = np.asarray(audio.samples, dtype=np.float64)
    signal_power = float(np.mean(np.square(samples))) or 1e-12
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise = generator.normal(0.0, float(np.sqrt(noise_power)), size=samples.size)
    return Audio(np.clip(samples + noise, -1.0, 1.0).astype(np.float32), audio.sr)


def synth_inkml_dataset(
    out: Path,
    n: int,
    seed: int = 1234,
    *,
    latexes: tuple[str, ...] | None = None,
    prefix: str = "sample",
) -> Path:
    """Write ``n`` synthetic InkML files into ``out`` and return the directory.

    Args:
        out: Target directory (created, including parents).
        n: Number of files (> 0).
        seed: Base seed; file ``i`` uses ``seed + i``.
        latexes: Label pool; defaults to :data:`aicls.ocr.CURRICULUM_LATEX`.
        prefix: File-name prefix (``sample-00000.inkml``).

    Returns:
        The directory that was written.

    Raises:
        InputValidationError: If ``n`` is not positive.
    """
    if int(n) <= 0:
        raise invalid_data("n_not_positive", n=int(n))
    pool = tuple(latexes) if latexes else tuple(CURRICULUM_LATEX)
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(int(n)):
        ink = synth_ink(np.random.default_rng(int(seed) + index), pool[index % len(pool)])
        path = directory / f"{prefix}-{index:05d}.inkml"
        path.write_bytes(write_inkml(ink))
    _log.info("data.synthetic.inkml_written", out=str(directory), n=int(n), seed=int(seed))
    return directory


def synth_image_dataset(
    out: Path,
    n: int,
    seed: int = 1234,
    *,
    size: tuple[int, int] = (96, 384),
) -> Path:
    """Write ``n`` synthetic rasters as ``.npy`` files plus a ``labels.jsonl`` sidecar.

    Args:
        out: Target directory.
        n: Number of images (> 0).
        seed: Base seed.
        size: Raster ``(height, width)``.

    Returns:
        The directory that was written.

    Raises:
        InputValidationError: If ``n`` is not positive.
    """
    if int(n) <= 0:
        raise invalid_data("n_not_positive", n=int(n))
    import json

    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(int(n)):
        latex = CURRICULUM_LATEX[index % len(CURRICULUM_LATEX)]
        image = synth_image(np.random.default_rng(int(seed) + index), latex, size)
        name = f"image-{index:05d}.npy"
        np.save(directory / name, image)
        rows.append(
            {"id": name.removesuffix(".npy"), "file": name, "label": normalize_label(latex)}
        )
    with (directory / "labels.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return directory


# ------------------------------------------------------------------------------------ internals
def _cubic_bezier(control: npt.NDArray[np.float64], n_points: int) -> npt.NDArray[np.float64]:
    """Sample a cubic Bezier curve defined by four control points."""
    t = np.linspace(0.0, 1.0, n_points)[:, None]
    p0, p1, p2, p3 = (control[i][None, :] for i in range(4))
    return (
        ((1.0 - t) ** 3) * p0
        + 3.0 * ((1.0 - t) ** 2) * t * p1
        + 3.0 * (1.0 - t) * (t**2) * p2
        + (t**3) * p3
    )


def _mathtext_raster(latex: str, size: tuple[int, int]) -> npt.NDArray[np.float32] | None:
    """Typeset ``latex`` with matplotlib mathtext; ``None`` when it cannot be parsed."""
    import io

    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        from matplotlib import mathtext
        from PIL import Image as PILImage
    except ImportError:  # pragma: no cover - matplotlib is a runtime dependency
        return None

    text = latex if latex.startswith("$") else f"${latex}$"
    buffer = io.BytesIO()
    try:
        mathtext.math_to_image(text, buffer, dpi=120, format="png")
    except Exception as exc:
        # mathtext raises several unrelated exception types (ValueError, pyparsing errors,
        # KeyError for unknown symbols); every one of them means "this string is not mathtext",
        # which is a fallback condition, not a failure. It is logged, never swallowed silently.
        _log.info("data.synthetic.mathtext_failed", latex=latex[:80], error=str(exc)[:200])
        return None
    buffer.seek(0)
    with PILImage.open(buffer) as handle:
        handle.load()
        pixels = np.asarray(handle.convert("L"), dtype=np.uint8)
    cfg = PreprocessConfig(steps=("normalize01", "resize_pad"), size=size, pad_value=1.0)
    return np.asarray(_preprocess_image(pixels, cfg), dtype=np.float32)


__all__ = [
    "CANVAS",
    "N_WRITERS",
    "POINTS_PER_STROKE",
    "synth_audio",
    "synth_image",
    "synth_image_dataset",
    "synth_ink",
    "synth_inkml_dataset",
]
