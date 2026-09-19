"""Synthetic generators: Bezier ink, mathtext images, sine+noise audio, InkML datasets.

These are the only "data" the repository can produce before the ``DATASET_DOWNLOAD`` gate opens
(ADR-0012), so their reproducibility under a seed is a contract, not a convenience.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aicls.asr.interfaces import Audio
from aicls.core.errors import InputValidationError
from aicls.data import synth_audio, synth_image, synth_ink, synth_inkml_dataset
from aicls.ocr import Ink, read_inkml

LATEX = r"\int x^{2}\,dx=\frac{x^{3}}{3}+C"


def test_synthetic_generators_reproducible_with_seed(tmp_path: Path) -> None:
    ink_a = synth_ink(np.random.default_rng(1234), LATEX)
    ink_b = synth_ink(np.random.default_rng(1234), LATEX)
    ink_other = synth_ink(np.random.default_rng(99), LATEX)

    assert isinstance(ink_a, Ink)
    assert ink_a.label == LATEX
    assert ink_a.n_strokes == ink_b.n_strokes and ink_a.n_strokes > 0
    for first, second in zip(ink_a.strokes, ink_b.strokes, strict=True):
        np.testing.assert_array_equal(first.points, second.points)
    assert ink_other.strokes[0].points.shape != ink_a.strokes[0].points.shape or not np.array_equal(
        ink_other.strokes[0].points, ink_a.strokes[0].points
    )

    image_a = synth_image(np.random.default_rng(1234), LATEX, (32, 96))
    image_b = synth_image(np.random.default_rng(1234), LATEX, (32, 96))
    np.testing.assert_array_equal(image_a, image_b)
    assert image_a.shape == (32, 96)
    assert image_a.dtype == np.float32
    assert float(image_a.min()) >= 0.0 and float(image_a.max()) <= 1.0
    assert float(image_a.min()) < 0.5  # something was actually drawn

    audio_a = synth_audio(np.random.default_rng(1234), 0.5)
    audio_b = synth_audio(np.random.default_rng(1234), 0.5)
    assert isinstance(audio_a, Audio)
    np.testing.assert_array_equal(audio_a.samples, audio_b.samples)
    assert audio_a.sr == 16000
    assert audio_a.duration_s == pytest.approx(0.5, abs=1e-3)

    first_dir = synth_inkml_dataset(tmp_path / "one", 5, 1234)
    second_dir = synth_inkml_dataset(tmp_path / "two", 5, 1234)
    names = sorted(p.name for p in first_dir.glob("*.inkml"))
    assert names == sorted(p.name for p in second_dir.glob("*.inkml"))
    for name in names:
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()


def test_synth_ink_accepts_an_int_seed_and_varies_with_the_expression() -> None:
    same = synth_ink(1234, LATEX)
    again = synth_ink(1234, LATEX)
    other = synth_ink(1234, r"\lim_{x\to0}\frac{\sin x}{x}=1")
    np.testing.assert_array_equal(same.all_points(), again.all_points())
    assert other.n_strokes != same.n_strokes or not np.array_equal(
        other.all_points(), same.all_points()
    )
    assert same.writer_id is not None
    assert same.meta["source"] == "synthetic"


def test_synth_audio_honours_the_requested_snr() -> None:
    rng = np.random.default_rng(1234)
    clean = synth_audio(rng, 1.0)
    noisy = synth_audio(np.random.default_rng(1234), 1.0, snr_db=5.0)
    quiet = synth_audio(np.random.default_rng(1234), 1.0, snr_db=30.0)

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))

    assert rms(noisy.samples - clean.samples) > rms(quiet.samples - clean.samples)
    assert noisy.duration_s == pytest.approx(1.0, abs=1e-3)


def test_synth_image_falls_back_to_ink_rendering_for_unparsable_latex() -> None:
    image = synth_image(np.random.default_rng(1234), r"\thisisnotmathtext{{{", (32, 96))
    assert image.shape == (32, 96)
    assert image.dtype == np.float32


def test_synth_inkml_dataset_files_are_readable_by_the_ocr_parser(tmp_path: Path) -> None:
    out = synth_inkml_dataset(tmp_path / "inkml", 4, 1234)
    files = sorted(out.glob("*.inkml"))

    assert len(files) == 4
    labels = set()
    for path in files:
        ink = read_inkml(path)
        assert ink.n_strokes > 0
        assert ink.label
        assert ink.writer_id
        labels.add(ink.label)
    assert len(labels) > 1
    assert sum(p.stat().st_size for p in files) < 2 * 1024 * 1024


def test_synthetic_generators_validate_their_arguments() -> None:
    with pytest.raises(InputValidationError):
        synth_ink(1234, "")
    with pytest.raises(InputValidationError):
        synth_image(1234, LATEX, (0, 10))
    with pytest.raises(InputValidationError):
        synth_audio(1234, 0.0)
    with pytest.raises(InputValidationError):
        synth_inkml_dataset(Path("."), 0, 1234)
