"""Audio and spectrogram augmentation for ASR fine-tuning (İP-3).

The proposal names SpecAugment, pitch shift, noise injection and TTS augmentation; speed
perturbation is the standard fourth member of that family. Augmentation is applied **only to the
training split**, and synthetic (TTS) utterances are tagged ``source="tts"`` in the manifest and
never enter the test split (docs/ml/veri-yonetimi.md).

Everything here is numpy, deterministic given a ``numpy.random.Generator``, and free of
``torchaudio``/``librosa``. Waveform-rate changes reuse the windowed-sinc resampler of
:mod:`aicls.asr.audio_io`.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

import numpy as np
import numpy.typing as npt

from aicls.asr.audio_io import resample_by_ratio
from aicls.asr.errors import invalid_audio
from aicls.asr.types import Audio

#: Default seed for every helper called without an explicit generator (repo-wide convention).
DEFAULT_SEED = 1234

#: Largest denominator used when a real-valued rate change is approximated by a fraction.
_MAX_DENOMINATOR = 2000

#: Overlap-add analysis window for time stretching (64 ms at 16 kHz).
_OLA_FRAME = 1024

#: Overlap-add synthesis hop (a quarter of the frame keeps the Hann overlap well conditioned).
_OLA_HOP = 256

#: Pitch shifts beyond this are refused: they stop sounding like speech.
MAX_SEMITONES = 12.0


def _generator(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng(DEFAULT_SEED)


def _fit_length(samples: npt.NDArray[np.float32], length: int) -> npt.NDArray[np.float32]:
    """Trim or zero-pad ``samples`` to exactly ``length`` samples."""
    if samples.size == length:
        return samples
    if samples.size > length:
        return np.ascontiguousarray(samples[:length])
    return np.concatenate((samples, np.zeros(length - samples.size, dtype=np.float32)))


def _ola_time_stretch(samples: npt.NDArray[np.float32], rate: float) -> npt.NDArray[np.float32]:
    """Stretch a signal to ``rate`` times its length without changing its pitch.

    Plain overlap-add with a Hann window and no phase correction: good enough for data
    augmentation (the artefacts it introduces are themselves a useful perturbation) and cheap.
    """
    if samples.size < _OLA_FRAME or abs(rate - 1.0) < 1e-9:
        return samples
    hop_analysis = max(1, round(_OLA_HOP / rate))
    window = np.hanning(_OLA_FRAME + 1)[:_OLA_FRAME].astype(np.float32)
    n_frames = 1 + (samples.size - _OLA_FRAME) // hop_analysis
    out_len = _OLA_FRAME + (n_frames - 1) * _OLA_HOP
    out = np.zeros(out_len, dtype=np.float32)
    weight = np.zeros(out_len, dtype=np.float32)
    for index in range(n_frames):
        src = index * hop_analysis
        dst = index * _OLA_HOP
        out[dst : dst + _OLA_FRAME] += samples[src : src + _OLA_FRAME] * window
        weight[dst : dst + _OLA_FRAME] += window
    # numpy 2.4's stubs type this division as float64 even though every operand is float32;
    # the cast is a no-op at runtime (copy=False) and pins the declared dtype for the checker.
    stretched = out / np.maximum(weight, np.float32(1e-6))
    return stretched.astype(np.float32, copy=False)


def _resample_by(samples: npt.NDArray[np.float32], factor: float) -> npt.NDArray[np.float32]:
    """Read ``samples`` ``factor`` times faster (higher pitch, shorter) at the same sample rate."""
    ratio = Fraction(1.0 / float(factor)).limit_denominator(_MAX_DENOMINATOR)
    return resample_by_ratio(samples, ratio)


def spec_augment(
    mel: npt.NDArray[np.float32],
    *,
    time_masks: int = 2,
    freq_masks: int = 2,
    rng: np.random.Generator | None = None,
    time_mask_max: int = 30,
    freq_mask_max: int = 13,
    fill: float | None = None,
) -> npt.NDArray[np.float32]:
    """Apply SpecAugment time and frequency masks to a log-Mel spectrogram.

    Args:
        mel: Spectrogram of shape ``(n_mels, n_frames)``, the layout
            :func:`aicls.asr.features.log_mel` returns. Never modified in place.
        time_masks: Number of time masks to draw.
        freq_masks: Number of frequency masks to draw.
        rng: Generator for reproducibility; a seed-1234 generator is used when omitted.
        time_mask_max: Largest time-mask width in frames.
        freq_mask_max: Largest frequency-mask height in mel bands.
        fill: Value written into the masks; the mean of ``mel`` when omitted, which is the usual
            choice for a normalised log-Mel input.

    Returns:
        A masked copy of ``mel``.

    Raises:
        AsrInputError: If ``mel`` is not 2-D or the mask counts are negative.
    """
    if mel.ndim != 2:
        raise invalid_audio("spec_augment_rank", ndim=mel.ndim)
    if time_masks < 0 or freq_masks < 0:
        raise invalid_audio("spec_augment_counts", time_masks=time_masks, freq_masks=freq_masks)
    generator = _generator(rng)
    out = np.array(mel, dtype=np.float32, copy=True)
    n_mels, n_frames = out.shape
    value = float(np.mean(out)) if fill is None else float(fill)

    for _ in range(freq_masks):
        width = int(generator.integers(1, max(1, min(freq_mask_max, n_mels)) + 1))
        if width >= n_mels:
            width = max(1, n_mels - 1)
        start = int(generator.integers(0, n_mels - width + 1))
        out[start : start + width, :] = value
    for _ in range(time_masks):
        width = int(generator.integers(1, max(1, min(time_mask_max, n_frames)) + 1))
        if width >= n_frames:
            width = max(1, n_frames - 1)
        start = int(generator.integers(0, n_frames - width + 1))
        out[:, start : start + width] = value
    return out


def pitch_shift(
    audio: Audio,
    semitones: float,
    rng: np.random.Generator | None = None,
) -> Audio:
    """Shift the pitch by ``semitones`` while keeping the duration bit-for-bit identical.

    Implemented as overlap-add time stretching by ``2 ** (semitones / 12)`` followed by a
    windowed-sinc resample of the same factor, then trimmed/padded to the original length. The
    proposal's range is +/-2 semitones.

    Args:
        audio: Input waveform.
        semitones: Positive raises the pitch, negative lowers it; 0 returns the input unchanged.
        rng: Unused today; accepted so the call site matches the other augmenters and a future
            random-range variant stays source compatible.

    Returns:
        Audio with the same sample rate, the same number of samples and a shifted pitch.

    Raises:
        AsrInputError: If ``|semitones|`` exceeds :data:`MAX_SEMITONES`.
    """
    del rng  # accepted for signature symmetry; the shift itself is deterministic
    if abs(float(semitones)) > MAX_SEMITONES:
        raise invalid_audio("pitch_shift_out_of_range", semitones=semitones, max=MAX_SEMITONES)
    if semitones == 0 or audio.n_samples == 0:
        return audio
    factor = 2.0 ** (float(semitones) / 12.0)
    stretched = _ola_time_stretch(audio.samples, factor)
    shifted = _resample_by(stretched, factor)
    return Audio(_fit_length(shifted, audio.n_samples), audio.sr)


def speed_perturb(audio: Audio, factor: float) -> Audio:
    """Play the recording ``factor`` times faster (Kaldi-style speed perturbation, 0.9-1.1).

    Both duration and pitch change, which is the point: it models speaking-rate variation. The
    sample rate is unchanged, so the returned audio has about ``n / factor`` samples.

    Raises:
        AsrInputError: If ``factor`` is not in ``(0.25, 4.0)``.
    """
    speed = float(factor)
    if not 0.25 < speed < 4.0:
        raise invalid_audio("speed_factor_out_of_range", factor=speed)
    if speed == 1.0 or audio.n_samples == 0:
        return audio
    return Audio(_resample_by(audio.samples, speed), audio.sr)


def _noise_samples(
    noise_bank: Sequence[Audio | npt.NDArray[np.float32]] | None,
    length: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float32]:
    """Pick one noise clip and tile/crop it to ``length`` samples (white noise when no bank)."""
    if not noise_bank:
        return rng.standard_normal(length).astype(np.float32)
    choice = noise_bank[int(rng.integers(0, len(noise_bank)))]
    clip = choice.samples if isinstance(choice, Audio) else np.asarray(choice, dtype=np.float32)
    clip = np.asarray(clip, dtype=np.float32).reshape(-1)
    if clip.size == 0:
        return rng.standard_normal(length).astype(np.float32)
    if clip.size < length:
        clip = np.tile(clip, int(np.ceil(length / clip.size)))
    offset = int(rng.integers(0, clip.size - length + 1)) if clip.size > length else 0
    return np.ascontiguousarray(clip[offset : offset + length])


def add_noise(
    audio: Audio,
    snr_db: float,
    noise_bank: Sequence[Audio | npt.NDArray[np.float32]] | None = None,
    rng: np.random.Generator | None = None,
    *,
    clip: bool = False,
) -> Audio:
    """Mix noise into ``audio`` at a target signal-to-noise ratio.

    Args:
        audio: Clean waveform.
        snr_db: Target SNR in dB (the proposal's range is 10-25 dB).
        noise_bank: Recorded noise clips (``Audio`` or raw arrays); Gaussian white noise is used
            when the bank is empty or omitted. Clips shorter than the signal are tiled, longer
            clips are cropped at a random offset.
        rng: Generator for clip and offset selection; seed 1234 when omitted.
        clip: Clamp the mix to ``[-1, 1]``. Off by default so the realised SNR is exactly the
            requested one; the collection pipeline turns it on before writing PCM16.

    Returns:
        The noisy waveform at the same sample rate and length.
    """
    if audio.n_samples == 0:
        return audio
    generator = _generator(rng)
    signal = audio.samples.astype(np.float64)
    signal_power = float(np.mean(signal**2))
    if signal_power <= 0.0:
        return audio
    noise = _noise_samples(noise_bank, audio.n_samples, generator).astype(np.float64)
    noise_power = float(np.mean(noise**2))
    if noise_power <= 0.0:
        return audio
    target_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    mixed = signal + noise * np.sqrt(target_power / noise_power)
    if clip:
        mixed = np.clip(mixed, -1.0, 1.0)
    return Audio(mixed.astype(np.float32), audio.sr)


def measured_snr_db(clean: Audio, noisy: Audio) -> float:
    """Signal-to-noise ratio in dB between a clean signal and its noisy version.

    Inverse of :func:`add_noise`, used by tests and by the augmentation report in the model card.

    Raises:
        AsrInputError: If the two waveforms differ in length or sample rate.
    """
    if clean.sr != noisy.sr or clean.n_samples != noisy.n_samples:
        raise invalid_audio("snr_shape_mismatch", clean=clean.n_samples, noisy=noisy.n_samples)
    signal = clean.samples.astype(np.float64)
    residual = noisy.samples.astype(np.float64) - signal
    noise_power = float(np.mean(residual**2))
    signal_power = float(np.mean(signal**2))
    if noise_power <= 0.0 or signal_power <= 0.0:
        return float("inf") if signal_power > 0.0 else 0.0
    return 10.0 * float(np.log10(signal_power / noise_power))


__all__ = [
    "DEFAULT_SEED",
    "MAX_SEMITONES",
    "add_noise",
    "measured_snr_db",
    "pitch_shift",
    "spec_augment",
    "speed_perturb",
]
