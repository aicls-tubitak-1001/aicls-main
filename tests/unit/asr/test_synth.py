"""Synthetic waveform generators — the only audio fixtures the repository ships."""

from __future__ import annotations

import numpy as np
import pytest

from aicls.asr import make_noise, make_speech_like, make_tone
from aicls.asr.errors import AsrInputError
from aicls.asr.qc import estimate_snr_db, run_qc, silence_ratio


def test_generators_are_deterministic_for_a_seed() -> None:
    assert np.array_equal(make_noise(seed=7).samples, make_noise(seed=7).samples)
    assert not np.array_equal(make_noise(seed=7).samples, make_noise(seed=8).samples)
    assert np.array_equal(
        make_speech_like(duration_s=0.5).samples, make_speech_like(duration_s=0.5).samples
    )


def test_make_tone_shape_amplitude_and_optional_noise() -> None:
    tone = make_tone(freq=440.0, duration_s=0.5, sr=48000, amplitude=0.4)
    assert tone.sr == 48000
    assert tone.n_samples == 24000
    assert float(np.abs(tone.samples).max()) == pytest.approx(0.4, abs=0.01)
    noisy = make_tone(duration_s=0.5, amplitude=0.4, noise_amplitude=0.05)
    assert float(np.abs(noisy.samples).max()) > 0.4


def test_make_speech_like_passes_quality_control() -> None:
    """The canonical "good recording" fixture the collection tab is tested against."""
    report = run_qc(make_speech_like(duration_s=2.0))
    assert report.passed is True
    assert estimate_snr_db(make_speech_like(duration_s=2.0)) > 20.0
    assert silence_ratio(make_speech_like(duration_s=2.0)) < 0.6


def test_make_noise_amplitude_is_the_rms() -> None:
    noise = make_noise(duration_s=1.0, amplitude=0.05)
    assert float(np.sqrt(np.mean(noise.samples.astype(np.float64) ** 2))) == pytest.approx(
        0.05, rel=0.1
    )


@pytest.mark.parametrize("kwargs", [{"duration_s": 0.0}, {"sr": 0}])
def test_generators_reject_impossible_arguments(kwargs: dict[str, float]) -> None:
    with pytest.raises(AsrInputError) as excinfo:
        make_tone(**kwargs)
    assert excinfo.value.details["reason"] == "synth_arguments"


def test_make_speech_like_rejects_impossible_bursts() -> None:
    with pytest.raises(AsrInputError) as excinfo:
        make_speech_like(burst_s=0.0)
    assert excinfo.value.details["reason"] == "synth_burst_arguments"
