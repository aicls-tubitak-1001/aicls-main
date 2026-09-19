"""Log-Mel spectrogram in pure numpy (no ``torchaudio``, no ``librosa``).

Two consumers, one implementation:

* **quality control and augmentation** (:mod:`aicls.asr.qc`, :mod:`aicls.asr.augment`) need a
  cheap, dependency-free spectrogram that behaves identically on every machine;
* **the Whisper inference path** uses Hugging Face's ``WhisperFeatureExtractor`` instead, because
  the model was trained against exactly those filter banks — :func:`log_mel` is deliberately
  Whisper-shaped (80 mels, 400-sample window, 160-sample hop, the same log compression and
  ``max - 8`` floor) so plots and SpecAugment behave the same in both paths (ADR-0017).

The mel scale is HTK (``2595 * log10(1 + f / 700)``) with Slaney area normalisation, matching the
filter bank ``librosa.filters.mel`` produces for the same arguments closely enough for QC use.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Literal

import numpy as np
import numpy.typing as npt

from aicls.asr.errors import invalid_audio
from aicls.asr.types import Audio

#: Number of mel bands Whisper expects.
DEFAULT_N_MELS = 80

#: Analysis window length in samples (25 ms at 16 kHz).
DEFAULT_N_FFT = 400

#: Analysis hop in samples (10 ms at 16 kHz).
DEFAULT_HOP = 160

#: Floor applied before the logarithm, so silence maps to a finite value.
_LOG_FLOOR = 1e-10

#: Dynamic range kept above the loudest bin, as in Whisper's ``log_mel_spectrogram``.
_DYNAMIC_RANGE_DECADES = 8.0


def hz_to_mel(hz: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64]:
    """Convert Hz to the HTK mel scale."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64]:
    """Convert HTK mel values back to Hz."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


@lru_cache(maxsize=16)
def mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> npt.NDArray[np.float32]:
    """Build a triangular mel filter bank.

    Args:
        sr: Sample rate of the signal the FFT was taken from.
        n_fft: FFT size (the bank has ``n_fft // 2 + 1`` columns).
        n_mels: Number of mel bands.
        fmin: Lowest edge frequency in Hz.
        fmax: Highest edge frequency in Hz; defaults to the Nyquist frequency.

    Returns:
        Array of shape ``(n_mels, n_fft // 2 + 1)``, area-normalised (Slaney style).

    Raises:
        AsrInputError: If the arguments cannot describe a filter bank.
    """
    if sr <= 0 or n_fft < 2 or n_mels < 1:
        raise invalid_audio("mel_filterbank_arguments", sr=sr, n_fft=n_fft, n_mels=n_mels)
    top = float(sr) / 2.0 if fmax is None else float(fmax)
    if not 0.0 <= fmin < top:
        raise invalid_audio("mel_frequency_range", fmin=fmin, fmax=top)

    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_bins, dtype=np.float64)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(top), n_mels + 2))

    # Triangle m spans edges[m] .. edges[m + 2] with its peak at edges[m + 1].
    diff = np.diff(edges)
    ramps = edges[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / np.maximum(diff[:-1, None], np.finfo(np.float64).tiny)
    upper = ramps[2:] / np.maximum(diff[1:, None], np.finfo(np.float64).tiny)
    bank = np.maximum(0.0, np.minimum(lower, upper))

    enorm = 2.0 / np.maximum(edges[2 : n_mels + 2] - edges[:n_mels], np.finfo(np.float64).tiny)
    return (bank * enorm[:, None]).astype(np.float32)


def _frames(
    samples: npt.NDArray[np.float32], n_fft: int, hop: int, *, center: bool
) -> npt.NDArray[np.float32]:
    """Slice ``samples`` into overlapping analysis frames of ``n_fft`` samples."""
    signal = samples
    if center:
        pad = n_fft // 2
        mode: Literal["reflect", "constant"] = "reflect" if signal.size > pad else "constant"
        signal = np.pad(signal, pad, mode=mode)
    if signal.size < n_fft:
        signal = np.pad(signal, (0, n_fft - signal.size))
    n_frames = 1 + (signal.size - n_fft) // hop
    strided: npt.NDArray[np.float32] = np.lib.stride_tricks.sliding_window_view(signal, n_fft)[
        :: int(hop)
    ][:n_frames]
    return strided


def power_spectrogram(
    audio: Audio,
    *,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
    center: bool = True,
) -> npt.NDArray[np.float32]:
    """Short-time power spectrum, shape ``(n_fft // 2 + 1, n_frames)``.

    Uses a periodic Hann window and, by default, centred frames with reflect padding, so frame
    ``t`` is centred on sample ``t * hop``.
    """
    if n_fft < 2 or hop < 1:
        raise invalid_audio("stft_arguments", n_fft=n_fft, hop=hop)
    window = np.hanning(n_fft + 1)[:n_fft].astype(np.float32)
    frames = _frames(audio.samples, n_fft, hop, center=center)
    spectrum = np.fft.rfft(frames * window, n=n_fft, axis=-1)
    power: npt.NDArray[np.float32] = (spectrum.real**2 + spectrum.imag**2).astype(np.float32)
    return power.T


def log_mel(
    audio: Audio,
    *,
    n_mels: int = DEFAULT_N_MELS,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP,
    normalize: bool = True,
) -> npt.NDArray[np.float32]:
    """Log-Mel spectrogram of ``audio``.

    Args:
        audio: Waveform (any sample rate; the filter bank follows ``audio.sr``).
        n_mels: Number of mel bands (80 for Whisper).
        n_fft: Analysis window length in samples.
        hop: Analysis hop in samples.
        normalize: Apply Whisper's compression — clamp to 8 decades below the maximum, then map
            to roughly ``[-1, 1]`` with ``(x + 4) / 4``. Turn it off to get raw ``log10`` power.

    Returns:
        Array of shape ``(n_mels, n_frames)``, ``float32``, with ``n_frames`` about
        ``len(samples) / hop``.

    Raises:
        AsrInputError: On non-positive ``n_mels``/``hop`` or an ``n_fft`` below 2.
    """
    power = power_spectrogram(audio, n_fft=n_fft, hop=hop)
    bank = mel_filterbank(audio.sr, n_fft, n_mels)
    mel = bank @ power
    log_spec = np.log10(np.maximum(mel, _LOG_FLOOR))
    if normalize:
        log_spec = np.maximum(log_spec, log_spec.max() - _DYNAMIC_RANGE_DECADES)
        log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float32)


def frame_rms(
    samples: npt.NDArray[np.float32], *, frame_len: int, hop: int
) -> npt.NDArray[np.float32]:
    """Root-mean-square energy per analysis frame (no padding, no windowing).

    Short signals yield a single frame covering everything, so callers never have to special-case
    an empty result.
    """
    if frame_len < 1 or hop < 1:
        raise invalid_audio("frame_arguments", frame_len=frame_len, hop=hop)
    if samples.size == 0:
        return np.zeros(1, dtype=np.float32)
    if samples.size < frame_len:
        return np.asarray([math.sqrt(float(np.mean(samples.astype(np.float64) ** 2)))], np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(samples, frame_len)[::hop]
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)).astype(np.float32)


__all__ = [
    "DEFAULT_HOP",
    "DEFAULT_N_FFT",
    "DEFAULT_N_MELS",
    "frame_rms",
    "hz_to_mel",
    "log_mel",
    "mel_filterbank",
    "mel_to_hz",
    "power_spectrogram",
]
