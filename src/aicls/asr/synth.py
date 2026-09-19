"""Deterministic synthetic waveforms — the only "audio fixtures" this repository ships.

No real recording is committed (KVKK / consent), and no dataset may be downloaded (ADR-0012), so
every audio test, every smoke run and every documentation figure builds its input here from a
seed. Two shapes cover the needs:

* :func:`make_tone` — a steady sine plus a little noise: right for header, resampling and
  spectrogram tests, and *deliberately* fails the SNR check (see :mod:`aicls.asr.qc`).
* :func:`make_speech_like` — bursts of a two-harmonic tone separated by near-silent gaps, an
  utterance-shaped envelope that passes quality control and exercises the SNR and silence
  estimators the way a real read prompt does.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from aicls.asr.errors import invalid_audio
from aicls.asr.types import TARGET_SAMPLE_RATE, Audio

#: Seed used whenever a generator is not passed explicitly (repo-wide convention).
DEFAULT_SEED = 1234


def _time_axis(duration_s: float, sr: int) -> npt.NDArray[np.float64]:
    if duration_s <= 0.0 or sr <= 0:
        raise invalid_audio("synth_arguments", duration_s=duration_s, sr=sr)
    return np.arange(round(duration_s * sr), dtype=np.float64) / float(sr)


def make_noise(
    *,
    duration_s: float = 1.0,
    sr: int = TARGET_SAMPLE_RATE,
    amplitude: float = 0.02,
    seed: int = DEFAULT_SEED,
) -> Audio:
    """Gaussian white noise of the given RMS ``amplitude`` (a one-entry ``noise_bank``)."""
    n = _time_axis(duration_s, sr).size
    rng = np.random.default_rng(seed)
    return Audio((rng.standard_normal(n) * amplitude).astype(np.float32), sr)


def make_tone(
    *,
    freq: float = 220.0,
    duration_s: float = 1.0,
    sr: int = TARGET_SAMPLE_RATE,
    amplitude: float = 0.3,
    noise_amplitude: float = 0.0,
    seed: int = DEFAULT_SEED,
) -> Audio:
    """A steady sine wave, optionally with additive Gaussian noise.

    Args:
        freq: Tone frequency in Hz.
        duration_s: Length in seconds.
        sr: Sample rate in Hz.
        amplitude: Peak amplitude of the sine in ``[0, 1]``.
        noise_amplitude: RMS of the additive noise (0 for a pure tone).
        seed: Seed for the noise generator.

    Returns:
        The waveform.
    """
    t = _time_axis(duration_s, sr)
    signal = amplitude * np.sin(2.0 * np.pi * freq * t)
    if noise_amplitude > 0.0:
        signal = signal + np.random.default_rng(seed).standard_normal(t.size) * noise_amplitude
    return Audio(signal.astype(np.float32), sr)


def make_speech_like(
    *,
    duration_s: float = 2.0,
    sr: int = TARGET_SAMPLE_RATE,
    burst_s: float = 0.25,
    gap_s: float = 0.15,
    freq: float = 180.0,
    amplitude: float = 0.35,
    noise_amplitude: float = 0.004,
    seed: int = DEFAULT_SEED,
) -> Audio:
    """An utterance-shaped signal: voiced bursts separated by quiet gaps.

    Each burst is a fundamental plus its second harmonic under a raised-cosine envelope, so the
    frame-energy distribution has the two clear modes that :func:`aicls.asr.qc.estimate_snr_db`
    and :func:`aicls.asr.qc.silence_ratio` are designed for. With the default arguments the
    result passes :func:`aicls.asr.qc.run_qc`.

    Args:
        duration_s: Total length in seconds.
        sr: Sample rate in Hz.
        burst_s: Length of one voiced burst.
        gap_s: Length of the silence between bursts.
        freq: Fundamental frequency in Hz.
        amplitude: Peak amplitude of a burst.
        noise_amplitude: RMS of the background noise present throughout.
        seed: Seed for the background noise.

    Returns:
        The waveform.

    Raises:
        AsrInputError: If ``burst_s`` or ``gap_s`` is not positive.
    """
    if burst_s <= 0.0 or gap_s <= 0.0:
        raise invalid_audio("synth_burst_arguments", burst_s=burst_s, gap_s=gap_s)
    t = _time_axis(duration_s, sr)
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(t.size) * noise_amplitude

    burst_n = max(1, round(burst_s * sr))
    gap_n = max(1, round(gap_s * sr))
    # Tukey-style envelope: flat body with short fades, so a burst reads as one loud region
    # rather than a slow swell (which would inflate the measured silence ratio).
    fade_n = max(1, burst_n // 10)
    envelope = np.ones(burst_n, dtype=np.float64)
    fade = np.hanning(2 * fade_n)[:fade_n]
    envelope[:fade_n] = fade
    envelope[burst_n - fade_n :] = fade[::-1]
    tone_t = np.arange(burst_n, dtype=np.float64) / float(sr)
    burst = (
        amplitude
        * envelope
        * (np.sin(2.0 * np.pi * freq * tone_t) + 0.5 * np.sin(2.0 * np.pi * 2.0 * freq * tone_t))
        / 1.5
    )
    start = 0
    while start < t.size:
        stop = min(start + burst_n, t.size)
        signal[start:stop] += burst[: stop - start]
        start = stop + gap_n
    return Audio(signal.astype(np.float32), sr)


__all__ = ["DEFAULT_SEED", "make_noise", "make_speech_like", "make_tone"]
