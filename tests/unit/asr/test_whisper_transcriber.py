"""Whisper inference edge — local checkpoints only, no download, no hub id (ADR-0012/0017)."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pytest

from aicls.asr.errors import AsrModelError
from aicls.asr.synth import make_speech_like
from aicls.asr.types import Audio
from aicls.asr.whisper_transcriber import (
    DEFAULT_COMPRESSION_RATIO_THRESHOLD,
    GUARD_LOW_LOGPROB,
    GUARD_REPETITIVE,
    MAX_UTTERANCE_SECONDS,
    WhisperTranscriber,
    compression_ratio,
    guard_reason,
)
from aicls.core.errors import NotFoundError


def test_whisper_ft_builds_from_tiny_random_config_without_download(
    tiny_whisper_dir: Path,
) -> None:
    """A tiny randomly initialised Whisper is saved locally, loaded and used end to end.

    This is the offline proof of the whole Whisper path: no weights are fetched, the checkpoint
    comes from ``save_pretrained`` in the fixture, and ``from_pretrained`` sees a filesystem path
    with ``local_files_only=True``.
    """
    transcriber = WhisperTranscriber.load(tiny_whisper_dir, "cpu")

    assert transcriber.device == "cpu"
    assert transcriber.language == "tr"
    assert transcriber.task == "transcribe"
    assert transcriber.model_version == tiny_whisper_dir.name

    transcript = transcriber.transcribe(make_speech_like(duration_s=1.0))

    assert isinstance(transcript.text, str)
    assert 0.0 <= transcript.confidence <= 1.0
    assert transcript.elapsed_ms > 0.0
    assert transcript.model_version == tiny_whisper_dir.name
    assert len(transcript.segments) == 1


def test_whisper_warmup_runs_one_silent_inference(tiny_whisper_dir: Path) -> None:
    transcriber = WhisperTranscriber.load(tiny_whisper_dir)
    assert transcriber.warmup() is None


def test_whisper_resamples_a_44100_hz_utterance(tiny_whisper_dir: Path) -> None:
    transcriber = WhisperTranscriber.load(tiny_whisper_dir)
    transcript = transcriber.transcribe(make_speech_like(duration_s=1.0, sr=44100))
    assert isinstance(transcript.text, str)
    assert transcript.segments[0].end_s == pytest.approx(1.0, abs=0.01)


def test_whisper_load_reports_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        WhisperTranscriber.load(tmp_path / "asr-0.2.0")
    assert excinfo.value.details["reason"] == "model_dir_missing"
    assert excinfo.value.http_status == 404


def test_whisper_load_reports_a_directory_without_a_config(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(AsrModelError) as excinfo:
        WhisperTranscriber.load(tmp_path / "empty")
    assert excinfo.value.details["reason"] == "model_config_missing"


def test_whisper_load_reports_a_directory_without_weights(tmp_path: Path) -> None:
    directory = tmp_path / "half-imported"
    directory.mkdir()
    (directory / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AsrModelError) as excinfo:
        WhisperTranscriber.load(directory)
    assert excinfo.value.details["reason"] == "model_weights_missing"


def test_whisper_load_wraps_a_corrupt_checkpoint(tmp_path: Path) -> None:
    directory = tmp_path / "corrupt"
    directory.mkdir()
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"not a checkpoint")
    with pytest.raises(AsrModelError) as excinfo:
        WhisperTranscriber.load(directory)
    assert excinfo.value.details["reason"] == "whisper_load_failed"


def test_whisper_forced_language_kwargs_only_when_the_checkpoint_knows_them(
    tiny_whisper_dir: Path,
) -> None:
    """The tiny model has no language tokens; a real fine-tune does, and then they are forced."""
    transcriber = WhisperTranscriber.load(tiny_whisper_dir)
    assert transcriber._forced_kwargs() == {}
    transcriber._model.generation_config.lang_to_id = {"<|tr|>": 1}
    assert transcriber._forced_kwargs() == {"language": "tr", "task": "transcribe"}
    transcriber._model.generation_config.lang_to_id = None


def test_whisper_wraps_a_generation_failure(tiny_whisper_dir: Path) -> None:
    transcriber = WhisperTranscriber.load(tiny_whisper_dir)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("kernel exploded")

    transcriber._model.generate = boom
    with pytest.raises(AsrModelError) as excinfo:
        transcriber.transcribe(make_speech_like(duration_s=0.5))
    assert excinfo.value.details["reason"] == "whisper_generate_failed"


def test_module_never_passes_a_hub_id_to_from_pretrained() -> None:
    """``tools/check_gates.py`` greps for this; keep the source honest here too."""
    source = Path(__file__).resolve().parents[3] / "src" / "aicls" / "asr"
    for name in ("whisper_transcriber.py", "wav2vec2_transcriber.py"):
        text = (source / name).read_text(encoding="utf-8")
        assert not re.search(r"from_pretrained\(\s*[\"'][\w.\-]+/", text)
        assert "local_files_only=True" in text


# --------------------------------------------------------------------------------------------
# what comes out of the model is checked before it is handed on
#
# Whisper is a language model sitting upstream of a pipeline whose whole claim is that answers
# never come from one (proposal A.2.2). The claim survives only if its output is filtered rather
# than trusted: a withheld transcript costs the speaker a repeat, an unchecked one puts an
# invented command in front of a student.
# --------------------------------------------------------------------------------------------
def test_compression_ratio_separates_prose_from_a_decoding_loop() -> None:
    """The detector has to tell ordinary Turkish from a stuck decoder, not just measure text."""
    prose = "x karenin türevini al ve sonucu grafiğe dök"
    loop = "evet " * 60

    assert compression_ratio(prose) < DEFAULT_COMPRESSION_RATIO_THRESHOLD
    assert compression_ratio(loop) > DEFAULT_COMPRESSION_RATIO_THRESHOLD
    assert compression_ratio("") == 0.0, "an empty transcript must not be rejected as repetitive"


def test_a_repetitive_transcript_is_withheld() -> None:
    """A decoding loop is the failure this guard is for."""
    assert guard_reason("tamam " * 80, avg_logprob=-0.1) == GUARD_REPETITIVE


def test_a_transcript_the_decoder_was_unsure_of_is_withheld() -> None:
    """A number that describes the decoder's uncertainty is not a transcription."""
    assert guard_reason("türevini al", avg_logprob=-3.0) == GUARD_LOW_LOGPROB


def test_ordinary_confident_turkish_passes_both_guards() -> None:
    """The half that matters as much: a guard that fires on real speech is worse than none."""
    assert guard_reason("sinüs x'in integralini hesapla", avg_logprob=-0.2) is None


def test_an_empty_transcript_is_not_a_guard_failure() -> None:
    """Silence in, nothing out. That is the correct answer, not a rejected one."""
    assert guard_reason("", avg_logprob=float("-inf")) is None


def test_a_withheld_transcript_reports_no_text_and_no_confidence(
    tiny_whisper_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: the caller must be told nothing was understood, not given something invented.

    Confidence 0.0 matters as much as the empty text: the voice path asks for confirmation below
    0.8 (docs/mimari/api.md §3.6), so a withheld transcript can never be acted on silently.
    """
    from aicls.asr import whisper_transcriber as module

    transcriber = WhisperTranscriber.load(tiny_whisper_dir)
    monkeypatch.setattr(module, "guard_reason", lambda *a, **k: module.GUARD_REPETITIVE)

    result = transcriber.transcribe(Audio(np.zeros(8000, np.float32), 16_000))

    assert result.text == ""
    assert result.confidence == 0.0


def test_an_utterance_longer_than_the_receptive_field_is_refused(tiny_whisper_dir: Path) -> None:
    """Whisper sees 30 seconds. The feature extractor would keep the first 30 and say nothing.

    Silent cropping is how a data-loss bug survives for months: the caller gets a plausible
    transcript of part of what was said and nothing records that the rest was dropped.
    """
    transcriber = WhisperTranscriber.load(tiny_whisper_dir)
    too_long = Audio(np.zeros(int(16_000 * (MAX_UTTERANCE_SECONDS + 1)), np.float32), 16_000)

    with pytest.raises(AsrModelError) as excinfo:
        transcriber.transcribe(too_long)

    assert excinfo.value.details["reason"] == "whisper_utterance_too_long"
