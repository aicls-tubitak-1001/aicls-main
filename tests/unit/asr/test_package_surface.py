"""The ``aicls.asr`` package surface: re-exports, lazy torch wrappers and small edge branches."""

from __future__ import annotations

import struct
from types import SimpleNamespace

import numpy as np
import pytest

from aicls import asr
from aicls.asr._backend import mean_probability
from aicls.asr.augment import spec_augment
from aicls.asr.errors import AsrInputError
from aicls.asr.features import log_mel
from aicls.asr.interfaces import Audio, resolve_device


def test_public_names_are_all_importable() -> None:
    for name in asr.__all__:
        assert hasattr(asr, name), name
    assert asr.__dir__() == sorted(set(asr.__all__))


def test_torch_wrappers_are_exported_lazily() -> None:
    """``from aicls.asr import WhisperTranscriber`` works, but only pulls torch on demand."""
    from aicls.asr.wav2vec2_transcriber import Wav2Vec2Transcriber
    from aicls.asr.whisper_transcriber import WhisperTranscriber

    assert asr.WhisperTranscriber is WhisperTranscriber
    assert asr.Wav2Vec2Transcriber is Wav2Vec2Transcriber
    with pytest.raises(AttributeError):
        _ = asr.NoSuchTranscriber  # type: ignore[attr-defined]  # deliberate miss


def test_read_wav_reports_a_wave_parse_failure() -> None:
    """A RIFF/WAVE header with no fmt or data chunk is a container error, not a crash."""
    payload = b"RIFF" + struct.pack("<I", 36) + b"WAVE" + b"JUNK" + b"\x00" * 40
    with pytest.raises(AsrInputError) as excinfo:
        asr.read_wav_pcm16(payload)
    assert excinfo.value.details["reason"] == "wave_parse_failed"


def test_log_mel_pads_with_zeros_when_reflection_is_impossible() -> None:
    """Audio shorter than half a window cannot be reflect-padded; zero padding takes over."""
    mel = log_mel(Audio(np.full(4, 0.2, dtype=np.float32), 16000))
    assert mel.shape[0] == 80
    assert np.isfinite(mel).all()


def test_power_spectrogram_pads_uncentred_audio_up_to_one_window() -> None:
    from aicls.asr.features import power_spectrogram

    power = power_spectrogram(Audio(np.full(4, 0.2, dtype=np.float32), 16000), center=False)
    assert power.shape[1] == 1


@pytest.mark.parametrize("shape", [(1, 6), (2, 1), (1, 1)])
def test_spec_augment_clamps_masks_that_would_erase_everything(shape: tuple[int, int]) -> None:
    """A mask may never cover a whole axis, however small the spectrogram is."""
    mel = np.arange(shape[0] * shape[1], dtype=np.float32).reshape(shape)
    out = spec_augment(
        mel,
        time_masks=1,
        freq_masks=1,
        time_mask_max=99,
        freq_mask_max=99,
        rng=np.random.default_rng(0),
    )
    assert out.shape == mel.shape
    assert np.isfinite(out).all()


def test_resolve_device_reports_cpu_when_torch_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ImportError("no torch here")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]  # passthrough

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert resolve_device("cuda") == "cpu"


def test_mean_probability_without_scores_is_full_confidence() -> None:
    torch = SimpleNamespace(softmax=lambda tensor, dim: tensor)
    assert mean_probability(torch, None, [[0]]) == 1.0
    assert mean_probability(torch, (), [[0]]) == 1.0


def test_whisper_token_budget_falls_back_without_a_position_limit() -> None:
    from aicls.asr.whisper_transcriber import WhisperTranscriber

    transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
    transcriber.max_new_tokens = 64
    transcriber._model = SimpleNamespace(config=SimpleNamespace())
    assert transcriber._token_budget() == 64
    transcriber._model = SimpleNamespace(config=SimpleNamespace(max_target_positions=32))
    assert transcriber._token_budget() == 31
