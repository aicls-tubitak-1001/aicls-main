"""Numpy log-Mel front end: shape, framing, filter bank and argument validation."""

from __future__ import annotations

import numpy as np
import pytest

from aicls.asr.errors import AsrInputError
from aicls.asr.features import (
    DEFAULT_HOP,
    DEFAULT_N_FFT,
    frame_rms,
    hz_to_mel,
    log_mel,
    mel_filterbank,
    mel_to_hz,
    power_spectrogram,
)
from aicls.asr.interfaces import Audio
from aicls.asr.synth import make_speech_like, make_tone


def test_log_mel_shape_80_mels() -> None:
    """The contractual shape: 80 mel bands by roughly ``len(samples) / hop`` frames."""
    audio = make_speech_like(duration_s=2.0)
    mel = log_mel(audio)

    assert mel.shape[0] == 80
    assert mel.dtype == np.float32
    expected_frames = audio.n_samples // DEFAULT_HOP
    assert abs(mel.shape[1] - expected_frames) <= 2
    assert np.isfinite(mel).all()


@pytest.mark.parametrize("n_mels", [40, 64, 80, 128])
def test_log_mel_honours_n_mels(n_mels: int) -> None:
    mel = log_mel(make_tone(duration_s=0.5), n_mels=n_mels)
    assert mel.shape[0] == n_mels


def test_log_mel_frame_count_follows_the_hop() -> None:
    audio = make_tone(duration_s=1.0)
    dense = log_mel(audio, hop=80)
    sparse = log_mel(audio, hop=160)
    assert dense.shape[1] > sparse.shape[1]
    assert abs(dense.shape[1] - 2 * sparse.shape[1]) <= 4


def test_log_mel_normalisation_bounds_the_dynamic_range() -> None:
    """Whisper's compression: 8 decades below the peak, mapped to about [-1, 1]."""
    mel = log_mel(make_speech_like(duration_s=1.0))
    assert mel.max() <= 1.5
    assert mel.min() >= -1.5
    raw = log_mel(make_speech_like(duration_s=1.0), normalize=False)
    assert raw.min() < mel.min()


def test_log_mel_of_silence_is_constant() -> None:
    silence = Audio(np.zeros(16000, dtype=np.float32), 16000)
    mel = log_mel(silence)
    assert np.isfinite(mel).all()
    assert float(mel.std()) == pytest.approx(0.0, abs=1e-6)


def test_log_mel_puts_a_tone_in_the_expected_band() -> None:
    """A 1 kHz tone must dominate a mid band, not the lowest or highest one."""
    mel = log_mel(make_tone(freq=1000.0, duration_s=0.5, amplitude=0.5), normalize=False)
    peak_band = int(np.argmax(mel.mean(axis=1)))
    assert 20 < peak_band < 60


def test_log_mel_handles_audio_shorter_than_one_window() -> None:
    tiny = Audio(np.ones(50, dtype=np.float32) * 0.1, 16000)
    mel = log_mel(tiny)
    assert mel.shape[0] == 80
    assert mel.shape[1] >= 1


@pytest.mark.parametrize(
    "kwargs",
    [{"n_mels": 0}, {"n_fft": 1}, {"hop": 0}],
)
def test_log_mel_rejects_impossible_arguments(kwargs: dict[str, int]) -> None:
    with pytest.raises(AsrInputError):
        log_mel(make_tone(duration_s=0.2), **kwargs)


def test_mel_filterbank_shape_and_normalisation() -> None:
    bank = mel_filterbank(16000, DEFAULT_N_FFT, 80)
    assert bank.shape == (80, DEFAULT_N_FFT // 2 + 1)
    assert (bank >= 0).all()
    assert bank.sum() > 0.0
    # Cached: the same arguments return the identical object.
    assert mel_filterbank(16000, DEFAULT_N_FFT, 80) is bank


def test_mel_filterbank_rejects_an_impossible_frequency_range() -> None:
    with pytest.raises(AsrInputError) as excinfo:
        mel_filterbank(16000, DEFAULT_N_FFT, 80, 9000.0, 8000.0)
    assert excinfo.value.details["reason"] == "mel_frequency_range"


def test_mel_scale_roundtrip() -> None:
    hz = np.asarray([0.0, 100.0, 1000.0, 4000.0, 8000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(hz)), hz, atol=1e-6)


def test_power_spectrogram_shape_and_rejections() -> None:
    audio = make_tone(duration_s=0.5)
    power = power_spectrogram(audio, n_fft=DEFAULT_N_FFT, hop=DEFAULT_HOP)
    assert power.shape[0] == DEFAULT_N_FFT // 2 + 1
    assert (power >= 0).all()
    assert power_spectrogram(audio, center=False).shape[1] < power.shape[1]
    with pytest.raises(AsrInputError):
        power_spectrogram(audio, n_fft=0)


def test_frame_rms_tracks_energy_and_handles_short_input() -> None:
    loud = np.full(1600, 0.5, dtype=np.float32)
    assert float(frame_rms(loud, frame_len=320, hop=160).mean()) == pytest.approx(0.5, abs=1e-3)
    assert frame_rms(np.zeros(0, dtype=np.float32), frame_len=320, hop=160).size == 1
    assert frame_rms(np.full(10, 0.2, np.float32), frame_len=320, hop=160).size == 1
    with pytest.raises(AsrInputError):
        frame_rms(loud, frame_len=0, hop=160)
