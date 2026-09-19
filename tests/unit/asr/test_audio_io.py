"""WAV container validation, PCM16 decoding and windowed-sinc resampling (ADR-0017)."""

from __future__ import annotations

import io
import struct
import wave

import numpy as np
import pytest

from aicls.asr.audio_io import (
    ACCEPTED_SAMPLE_RATES,
    MAX_SECONDS,
    MAX_WAV_BYTES,
    read_wav_pcm16,
    resample,
    resample_by_ratio,
    resample_to_16k,
    write_wav_pcm16,
)
from aicls.asr.errors import AsrInputError
from aicls.asr.interfaces import Audio
from aicls.asr.synth import make_speech_like, make_tone
from aicls.core.errors import ErrorCode

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_bytes() -> bytes:
    """A minimal but genuine PNG byte stream, long enough to reach the magic-number check."""
    ihdr = struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0)
    chunk = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + struct.pack(">I", 0)
    return PNG_SIGNATURE + chunk + b"\x00" * 64


def raw_wav(samples: np.ndarray, sr: int, *, sample_width: int = 2, channels: int = 1) -> bytes:
    """Encode a WAV with an arbitrary sample width, for the negative tests."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sr)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


def test_read_wav_rejects_png_bytes_with_wav_extension() -> None:
    """A PNG renamed to .wav must be refused by the container check, not mis-parsed."""
    with pytest.raises(AsrInputError) as excinfo:
        read_wav_pcm16(png_bytes())
    error = excinfo.value
    assert error.code is ErrorCode.ASR_INPUT_INVALID
    assert error.http_status == 400
    assert error.details["reason"] == "not_riff_wave"
    assert error.message_tr == "Ses girdisi geçersiz veya desteklenmeyen biçimde."


def test_read_wav_accepts_44100_and_resamples_to_16k() -> None:
    """44.1 kHz uploads are accepted and converted to the 16 kHz working rate."""
    original = make_tone(freq=440.0, duration_s=1.0, sr=44100, amplitude=0.5)
    audio = read_wav_pcm16(write_wav_pcm16(original))

    assert audio.sr == 16000
    assert audio.n_samples == 16000
    assert audio.duration_s == pytest.approx(1.0, abs=1e-3)
    assert audio.samples.dtype == np.float32
    # The 440 Hz tone survives: peak amplitude preserved, waveform matches a native 16 kHz tone.
    reference = make_tone(freq=440.0, duration_s=1.0, sr=16000, amplitude=0.5)
    assert np.abs(audio.samples[400:-400] - reference.samples[400:-400]).max() < 1e-3


@pytest.mark.parametrize("sr", ACCEPTED_SAMPLE_RATES)
def test_read_wav_accepts_every_supported_rate(sr: int) -> None:
    audio = read_wav_pcm16(write_wav_pcm16(make_tone(duration_s=0.5, sr=sr)))
    assert audio.sr == 16000
    assert audio.n_samples == 8000


def test_read_wav_downmixes_stereo_to_mono() -> None:
    original = make_tone(duration_s=0.5, sr=16000, amplitude=0.4)
    stereo = write_wav_pcm16(original, channels=2)
    audio = read_wav_pcm16(stereo)
    assert audio.n_samples == original.n_samples
    assert np.abs(audio.samples - original.samples).max() < 1e-3


def test_read_wav_roundtrip_is_pcm16_accurate() -> None:
    original = make_speech_like(duration_s=1.0)
    audio = read_wav_pcm16(write_wav_pcm16(original))
    assert np.abs(audio.samples - original.samples).max() < 2.0 / 32768.0


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (b"", "empty_upload"),
        (b"RIFF" + b"\x00" * 8, "file_too_short"),
        (b"RIFX" + b"\x00" * 4 + b"WAVE" + b"\x00" * 40, "not_riff_wave"),
    ],
)
def test_read_wav_rejects_broken_containers(payload: bytes, reason: str) -> None:
    with pytest.raises(AsrInputError) as excinfo:
        read_wav_pcm16(payload)
    assert excinfo.value.details["reason"] == reason


def test_read_wav_rejects_oversized_upload() -> None:
    with pytest.raises(AsrInputError) as excinfo:
        read_wav_pcm16(b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * MAX_WAV_BYTES)
    assert excinfo.value.details["reason"] == "file_too_large"


def test_read_wav_rejects_non_pcm16_sample_width() -> None:
    eight_bit = np.full(8000, 128, dtype=np.uint8)
    with pytest.raises(AsrInputError) as excinfo:
        read_wav_pcm16(raw_wav(eight_bit, 16000, sample_width=1))
    assert excinfo.value.details["reason"] == "not_pcm16"


def test_read_wav_rejects_unsupported_sample_rate() -> None:
    pcm = np.zeros(8000, dtype="<i2")
    with pytest.raises(AsrInputError) as excinfo:
        read_wav_pcm16(raw_wav(pcm, 22050))
    assert excinfo.value.details["reason"] == "sample_rate_unsupported"
    assert excinfo.value.details["sample_rate"] == 22050


def test_read_wav_rejects_more_than_two_channels() -> None:
    pcm = np.zeros(8000 * 3, dtype="<i2")
    with pytest.raises(AsrInputError) as excinfo:
        read_wav_pcm16(raw_wav(pcm, 16000, channels=3))
    assert excinfo.value.details["reason"] == "channel_count_unsupported"


def test_read_wav_rejects_empty_frame_count() -> None:
    with pytest.raises(AsrInputError) as excinfo:
        read_wav_pcm16(raw_wav(np.zeros(0, dtype="<i2"), 16000))
    assert excinfo.value.details["reason"] == "no_audio_frames"


def test_read_wav_rejects_audio_longer_than_thirty_seconds() -> None:
    """The 30 s ceiling of ADR-0017 holds even when the caller asks for more."""
    long_audio = make_tone(duration_s=MAX_SECONDS + 1.0, sr=16000, amplitude=0.1)
    payload = write_wav_pcm16(long_audio)
    with pytest.raises(AsrInputError) as excinfo:
        read_wav_pcm16(payload, max_seconds=120.0)
    assert excinfo.value.details["reason"] == "duration_too_long"
    assert excinfo.value.details["max_seconds"] == MAX_SECONDS


def test_read_wav_honours_a_stricter_max_seconds() -> None:
    payload = write_wav_pcm16(make_tone(duration_s=5.0))
    with pytest.raises(AsrInputError) as excinfo:
        read_wav_pcm16(payload, max_seconds=2.0)
    assert excinfo.value.details["reason"] == "duration_too_long"
    assert read_wav_pcm16(payload, max_seconds=10.0).duration_s == pytest.approx(5.0)


def test_read_wav_rejects_non_positive_max_seconds() -> None:
    payload = write_wav_pcm16(make_tone(duration_s=1.0))
    with pytest.raises(AsrInputError) as excinfo:
        read_wav_pcm16(payload, max_seconds=0.0)
    assert excinfo.value.details["reason"] == "max_seconds_not_positive"


def test_read_wav_accepts_bytearray_and_memoryview() -> None:
    payload = write_wav_pcm16(make_tone(duration_s=0.5))
    assert read_wav_pcm16(bytearray(payload)).n_samples == 8000
    assert read_wav_pcm16(memoryview(payload)).n_samples == 8000


def test_read_wav_rejects_a_non_bytes_argument() -> None:
    with pytest.raises(AsrInputError) as excinfo:
        read_wav_pcm16("not bytes")  # type: ignore[arg-type]  # deliberate misuse
    assert excinfo.value.details["reason"] == "not_bytes"


def test_read_wav_reports_a_truncated_data_chunk() -> None:
    payload = bytearray(write_wav_pcm16(make_tone(duration_s=0.5)))
    del payload[44:]
    with pytest.raises(AsrInputError) as excinfo:
        read_wav_pcm16(bytes(payload))
    assert excinfo.value.details["reason"] in {"wave_parse_failed", "no_audio_frames"}


def test_resample_to_16k_returns_the_same_object_at_16k() -> None:
    audio = make_tone(duration_s=0.25, sr=16000)
    assert resample_to_16k(audio) is audio


def test_resample_preserves_a_constant_signal() -> None:
    """Unity DC gain: the kernels are normalised, so a constant stays constant."""
    constant = np.full(4410, 0.25, dtype=np.float32)
    out = resample(constant, 44100, 16000)
    assert out.size == 1600
    assert np.abs(out[50:-50] - 0.25).max() < 1e-4


def test_resample_upsamples_as_well_as_downsamples() -> None:
    audio = make_tone(freq=200.0, duration_s=0.25, sr=16000, amplitude=0.5)
    up = resample(audio.samples, 16000, 48000)
    assert up.size == 12000
    assert abs(float(np.abs(up).max()) - 0.5) < 5e-3


def test_resample_handles_empty_and_degenerate_inputs() -> None:
    empty = np.zeros(0, dtype=np.float32)
    assert resample(empty, 44100, 16000).size == 0
    assert resample(np.zeros(1, dtype=np.float32), 48000, 16000).size == 0
    with pytest.raises(AsrInputError):
        resample(empty, 0, 16000)


def test_resample_by_ratio_rejects_a_non_positive_ratio() -> None:
    from fractions import Fraction

    with pytest.raises(AsrInputError) as excinfo:
        resample_by_ratio(np.zeros(4, dtype=np.float32), Fraction(0, 1))
    assert excinfo.value.details["reason"] == "resample_ratio_not_positive"


def test_write_wav_pcm16_clips_and_validates_channels() -> None:
    loud = Audio(np.asarray([2.0, -2.0, 0.0], dtype=np.float32), 16000)
    decoded = np.frombuffer(write_wav_pcm16(loud)[44:], dtype="<i2")
    assert decoded.max() == 32767
    assert decoded.min() == -32767
    with pytest.raises(AsrInputError) as excinfo:
        write_wav_pcm16(loud, channels=3)
    assert excinfo.value.details["reason"] == "channel_count_unsupported"
