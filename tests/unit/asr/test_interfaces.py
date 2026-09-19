"""Value types, protocols and the :func:`get_transcriber` factory."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from aicls.asr.errors import AsrInputError
from aicls.asr.interfaces import (
    TARGET_SAMPLE_RATE,
    TRANSCRIBER_NAMES,
    Audio,
    Segment,
    Transcriber,
    Transcript,
    concat,
    get_transcriber,
    resolve_device,
)
from aicls.asr.stub import StubTranscriber
from aicls.asr.synth import make_tone
from aicls.core.errors import ErrorCode, InputValidationError


def test_audio_normalises_dtype_shape_and_duration() -> None:
    audio = Audio([[0.1, 0.2], [0.3, 0.4]], 16000)
    assert audio.samples.dtype == np.float32
    assert audio.samples.ndim == 1
    assert audio.n_samples == 4
    assert audio.duration_s == pytest.approx(4 / 16000)
    assert "n_samples=4" in repr(audio)


def test_audio_recomputes_a_duration_passed_by_the_caller() -> None:
    """``duration_s`` is derived, so it can never contradict the samples."""
    audio = Audio(np.zeros(8000, dtype=np.float32), 16000, 99.0)
    assert audio.duration_s == pytest.approx(0.5)


def test_audio_rejects_a_non_positive_sample_rate() -> None:
    with pytest.raises(AsrInputError) as excinfo:
        Audio(np.zeros(10, dtype=np.float32), 0)
    assert excinfo.value.details["reason"] == "sample_rate_not_positive"
    assert excinfo.value.code is ErrorCode.ASR_INPUT_INVALID


def test_audio_rejects_non_numeric_samples() -> None:
    with pytest.raises(AsrInputError) as excinfo:
        Audio(["not", "audio"], 16000)
    assert excinfo.value.details["reason"] == "samples_not_numeric"


def test_audio_with_samples_keeps_the_sample_rate() -> None:
    audio = make_tone(duration_s=0.1, sr=44100)
    other = audio.with_samples(np.zeros(441, dtype=np.float32))
    assert other.sr == 44100
    assert other.n_samples == 441


def test_concat_joins_matching_audio_and_refuses_the_rest() -> None:
    first = make_tone(duration_s=0.1)
    second = make_tone(freq=330.0, duration_s=0.1)
    joined = concat([first, second])
    assert joined.n_samples == first.n_samples + second.n_samples
    with pytest.raises(AsrInputError) as excinfo:
        concat([])
    assert excinfo.value.details["reason"] == "empty_audio_sequence"
    with pytest.raises(AsrInputError) as excinfo:
        concat([first, make_tone(duration_s=0.1, sr=48000)])
    assert excinfo.value.details["reason"] == "sample_rate_mismatch"


def test_transcript_defaults_are_api_shaped() -> None:
    transcript = Transcript(text="x karenin türevini al")
    assert transcript.confidence == 1.0
    assert transcript.segments == []
    assert transcript.elapsed_ms == 0.0
    assert transcript.model_version == "unknown"
    segment = Segment(0.0, 1.5, "x karenin türevini al")
    assert segment.confidence == 1.0


def test_stub_satisfies_the_transcriber_protocol() -> None:
    assert isinstance(StubTranscriber(), Transcriber)
    assert not isinstance(object(), Transcriber)


def test_get_transcriber_returns_the_stub_by_default() -> None:
    transcriber = get_transcriber()
    assert isinstance(transcriber, StubTranscriber)
    assert isinstance(get_transcriber("  STUB  "), StubTranscriber)


def test_get_transcriber_does_not_import_torch() -> None:
    """The default path must stay usable on a host with no ML stack (ADR-0002 layering)."""
    before = "torch" in sys.modules
    get_transcriber("stub").warmup()
    assert ("torch" in sys.modules) is before


def test_get_transcriber_rejects_an_unknown_backend() -> None:
    with pytest.raises(InputValidationError) as excinfo:
        get_transcriber("deepspeech")
    assert excinfo.value.details["reason"] == "unknown_transcriber"
    assert excinfo.value.details["known"] == list(TRANSCRIBER_NAMES)


@pytest.mark.parametrize("name", ["whisper", "wav2vec2"])
def test_get_transcriber_requires_a_local_model_dir(name: str) -> None:
    with pytest.raises(InputValidationError) as excinfo:
        get_transcriber(name)
    assert excinfo.value.details["reason"] == "model_dir_required"
    assert excinfo.value.details["backend"] == name


@pytest.mark.parametrize("name", ["whisper", "wav2vec2"])
def test_get_transcriber_reports_a_missing_model_dir(name: str, tmp_path: Path) -> None:
    from aicls.core.errors import NotFoundError

    with pytest.raises(NotFoundError) as excinfo:
        get_transcriber(name, model_dir=tmp_path / "absent")
    assert excinfo.value.details["reason"] == "model_dir_missing"


def test_resolve_device_falls_back_to_cpu_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("") == "cpu"
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    assert resolve_device("cuda") == "cpu"
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    )
    assert resolve_device("cuda:0") == "cuda:0"


def test_target_sample_rate_is_the_adr_value() -> None:
    assert TARGET_SAMPLE_RATE == 16000
