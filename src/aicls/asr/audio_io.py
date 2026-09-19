"""WAV (PCM16) reading, writing and resampling with the standard library plus numpy.

ADR-0017: the browser encodes 16 kHz mono PCM16 WAV itself, the server accepts **only** RIFF/WAVE
PCM16 and has no media dependency at all — no ffmpeg, no ``torchaudio``, no ``soundfile``, no
``librosa``. 44.1 kHz and 48 kHz uploads (older browsers whose ``OfflineAudioContext`` refuses to
resample) are converted here with a windowed-sinc polyphase resampler written in numpy.

Accepted input (anything else is an ``ASR_INPUT_INVALID``):

* RIFF/WAVE container, verified on the raw bytes before ``wave`` is even opened, so a PNG renamed
  to ``.wav`` is refused rather than mis-parsed;
* 16-bit PCM (``sampwidth == 2``);
* 1 or 2 channels (stereo is downmixed to mono by averaging);
* 16 000, 44 100 or 48 000 Hz;
* at most 30 s and at most 10 MB.
"""

from __future__ import annotations

import contextlib
from fractions import Fraction
import io
import math
import wave

import numpy as np
import numpy.typing as npt

from aicls.asr.errors import invalid_audio
from aicls.asr.types import TARGET_SAMPLE_RATE, Audio

#: Hard upload ceiling (ADR-0017); 30 s of 16 kHz PCM16 is ~960 kB, so this is generous.
MAX_WAV_BYTES = 10 * 1024 * 1024

#: Hard duration ceiling (ADR-0017). ``max_seconds`` may lower it but never raise it.
MAX_SECONDS = 30.0

#: Sample rates the browser recorder can produce.
ACCEPTED_SAMPLE_RATES: tuple[int, ...] = (16000, 44100, 48000)

#: PCM16 sample width in bytes.
_PCM16_WIDTH = 2

#: Smallest byte count that can hold a canonical RIFF/WAVE header.
_MIN_WAV_BYTES = 44

#: Full-scale value of a signed 16-bit sample.
_INT16_SCALE = 32768.0

#: Number of sinc zero crossings kept on each side of the resampling kernel.
_RESAMPLE_ZEROS = 16

#: Output samples resampled per numpy block (bounds peak memory for 30 s uploads).
_RESAMPLE_BLOCK = 8192


def _blackman(arg: npt.NDArray[np.float64], half_width: float) -> npt.NDArray[np.float64]:
    """Blackman window evaluated at ``arg`` (in input samples), zero outside ``+/-half_width``."""
    scaled = np.pi * arg / half_width
    window = 0.42 + 0.5 * np.cos(scaled) + 0.08 * np.cos(2.0 * scaled)
    return np.where(np.abs(arg) < half_width, window, 0.0)


def _sinc_kernels(up: int, half: int, cutoff: float) -> npt.NDArray[np.float32]:
    """Precompute one windowed-sinc kernel per output phase.

    Args:
        up: Number of distinct phases (numerator of the rational resampling ratio).
        half: Kernel half-width in input samples.
        cutoff: Low-pass cutoff relative to the input Nyquist frequency (``<= 1``).

    Returns:
        Array of shape ``(up, 2 * half + 1)``; row ``p`` is the kernel for fractional offset
        ``p / up``, normalised so that its taps sum to 1 (unity DC gain).
    """
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    phases = np.arange(up, dtype=np.float64)[:, None] / float(up)
    args = phases - offsets[None, :]
    kernels = cutoff * np.sinc(cutoff * args) * _blackman(args, float(half) + 1.0)
    norms = kernels.sum(axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (kernels / norms).astype(np.float32)


def _resample_frac(samples: npt.NDArray[np.float32], ratio: Fraction) -> npt.NDArray[np.float32]:
    """Resample so that the output rate is ``ratio`` times the input rate.

    Uses a polyphase windowed-sinc filter: output sample ``m`` reads the input at position
    ``m / ratio``, low-passed at ``min(1, ratio)`` of the input Nyquist frequency to avoid
    aliasing when downsampling. Pure numpy, deterministic, no external DSP dependency.

    Args:
        samples: Contiguous 1-D float32 input.
        ratio: Output rate divided by input rate, as an exact fraction.

    Returns:
        The resampled signal, of length ``floor(len(samples) * ratio)``.
    """
    up, down = ratio.numerator, ratio.denominator
    if up == down or samples.size == 0:
        return samples
    n_out = int(samples.size * up // down)
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)

    cutoff = min(1.0, up / down)
    half = math.ceil(_RESAMPLE_ZEROS / cutoff)
    kernels = _sinc_kernels(up, half, cutoff)
    offsets = np.arange(-half, half + 1, dtype=np.int64)
    padded = np.concatenate(
        (
            np.zeros(half, dtype=np.float32),
            samples,
            np.zeros(half + 1, dtype=np.float32),
        )
    )

    out = np.empty(n_out, dtype=np.float32)
    for start in range(0, n_out, _RESAMPLE_BLOCK):
        stop = min(start + _RESAMPLE_BLOCK, n_out)
        positions = np.arange(start, stop, dtype=np.int64) * down
        centres = positions // up
        phases = positions % up
        # +half shifts into the zero-padded array; index i of `offsets` is input index c + d.
        window_idx = (centres + half)[:, None] + offsets[None, :]
        out[start:stop] = (kernels[phases] * padded[window_idx]).sum(axis=1)
    return out


def resample_by_ratio(samples: npt.NDArray[np.float32], ratio: Fraction) -> npt.NDArray[np.float32]:
    """Resample by an arbitrary rational ``ratio`` of output rate to input rate.

    :mod:`aicls.asr.augment` uses this directly for pitch shifting and speed perturbation, where
    the rate change is a real number approximated by a fraction rather than a pair of Hz values.

    Raises:
        AsrInputError: If ``ratio`` is not strictly positive.
    """
    if ratio <= 0:
        raise invalid_audio("resample_ratio_not_positive", ratio=str(ratio))
    return _resample_frac(samples, ratio)


def resample(samples: npt.NDArray[np.float32], sr_in: int, sr_out: int) -> npt.NDArray[np.float32]:
    """Resample raw samples from ``sr_in`` to ``sr_out`` (windowed-sinc, numpy only)."""
    if sr_in <= 0 or sr_out <= 0:
        raise invalid_audio("sample_rate_not_positive", sr_in=sr_in, sr_out=sr_out)
    return _resample_frac(samples, Fraction(int(sr_out), int(sr_in)))


def resample_to_16k(audio: Audio) -> Audio:
    """Return ``audio`` at 16 kHz (the rate every model and quality check assumes).

    Audio that is already at 16 kHz is returned unchanged (same object).
    """
    if audio.sr == TARGET_SAMPLE_RATE:
        return audio
    return Audio(resample(audio.samples, audio.sr, TARGET_SAMPLE_RATE), TARGET_SAMPLE_RATE)


def _check_container(data: bytes) -> None:
    """Reject anything that is not a RIFF/WAVE container before ``wave`` sees it."""
    if len(data) == 0:
        raise invalid_audio("empty_upload")
    if len(data) > MAX_WAV_BYTES:
        raise invalid_audio("file_too_large", size=len(data), max_size=MAX_WAV_BYTES)
    if len(data) < _MIN_WAV_BYTES:
        raise invalid_audio("file_too_short", size=len(data), min_size=_MIN_WAV_BYTES)
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise invalid_audio("not_riff_wave", magic=data[:12].hex())


def read_wav_pcm16(
    data: bytes | bytearray | memoryview, *, max_seconds: float = MAX_SECONDS
) -> Audio:
    """Decode PCM16 WAV bytes into a mono 16 kHz :class:`Audio`.

    Args:
        data: The raw uploaded bytes (``bytes``, ``bytearray`` or ``memoryview``).
        max_seconds: Duration ceiling; clamped to :data:`MAX_SECONDS` so a caller can only make
            the limit stricter, never looser.

    Returns:
        Mono float32 audio at 16 kHz, resampled with :func:`resample_to_16k` when needed.

    Raises:
        AsrInputError: For a non-WAV container, a non-PCM16 encoding, more than two channels, an
            unsupported sample rate, an empty stream, or audio that is too long or too large.
    """
    # Typed as `object` on purpose: the caller is an HTTP upload handler, so the runtime guard
    # has to survive whatever the framework hands over, not only what the annotation promises.
    raw: object = data
    if not isinstance(raw, bytes | bytearray | memoryview):
        raise invalid_audio("not_bytes", type=type(raw).__name__)
    payload = bytes(raw)
    _check_container(payload)

    limit = min(float(max_seconds), MAX_SECONDS)
    if limit <= 0.0:
        raise invalid_audio("max_seconds_not_positive", max_seconds=max_seconds)

    try:
        with contextlib.closing(wave.open(io.BytesIO(payload), "rb")) as handle:
            n_channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frame_rate = handle.getframerate()
            n_frames = handle.getnframes()
            if sample_width != _PCM16_WIDTH:
                raise invalid_audio("not_pcm16", sample_width_bytes=sample_width)
            if n_channels not in (1, 2):
                raise invalid_audio("channel_count_unsupported", channels=n_channels)
            if frame_rate not in ACCEPTED_SAMPLE_RATES:
                raise invalid_audio(
                    "sample_rate_unsupported",
                    sample_rate=frame_rate,
                    accepted=list(ACCEPTED_SAMPLE_RATES),
                )
            if n_frames <= 0:
                raise invalid_audio("no_audio_frames")
            duration_s = n_frames / float(frame_rate)
            if duration_s > limit:
                raise invalid_audio(
                    "duration_too_long", duration_s=round(duration_s, 3), max_seconds=limit
                )
            raw = handle.readframes(n_frames)
    except (wave.Error, EOFError) as exc:
        raise invalid_audio("wave_parse_failed", error=str(exc)) from exc

    frame_bytes = n_channels * _PCM16_WIDTH
    usable = (len(raw) // frame_bytes) * frame_bytes
    if usable == 0:
        raise invalid_audio("no_audio_frames")
    interleaved = np.frombuffer(raw[:usable], dtype="<i2").astype(np.float32) / _INT16_SCALE
    if n_channels == 2:
        interleaved = interleaved.reshape(-1, 2).mean(axis=1)

    return resample_to_16k(Audio(interleaved, frame_rate))


def write_wav_pcm16(audio: Audio, *, channels: int = 1) -> bytes:
    """Encode :class:`Audio` as PCM16 WAV bytes (the inverse of :func:`read_wav_pcm16`).

    Used by the synthetic fixtures, by ``aicls.collect`` when it stores an accepted recording, and
    by tests that need a byte stream the server would accept.

    Args:
        audio: Mono waveform; samples outside ``[-1, 1]`` are clipped.
        channels: 1 for mono, 2 to duplicate the mono signal into both channels.

    Returns:
        A complete RIFF/WAVE byte string.

    Raises:
        AsrInputError: If ``channels`` is not 1 or 2.
    """
    if channels not in (1, 2):
        raise invalid_audio("channel_count_unsupported", channels=channels)
    clipped = np.clip(audio.samples, -1.0, 1.0)
    pcm = np.round(clipped * (_INT16_SCALE - 1.0)).astype("<i2")
    if channels == 2:
        pcm = np.repeat(pcm[:, None], 2, axis=1).reshape(-1)
    buffer = io.BytesIO()
    with contextlib.closing(wave.open(buffer, "wb")) as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(_PCM16_WIDTH)
        handle.setframerate(audio.sr)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


__all__ = [
    "ACCEPTED_SAMPLE_RATES",
    "MAX_SECONDS",
    "MAX_WAV_BYTES",
    "read_wav_pcm16",
    "resample",
    "resample_by_ratio",
    "resample_to_16k",
    "write_wav_pcm16",
]
