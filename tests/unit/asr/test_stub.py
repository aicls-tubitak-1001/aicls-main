"""The deterministic stub transcriber that every pipeline test runs against."""

from __future__ import annotations

import numpy as np
import pytest

from aicls.asr.audio_io import read_wav_pcm16, write_wav_pcm16
from aicls.asr.errors import AsrInputError
from aicls.asr.interfaces import Audio
from aicls.asr.stub import STUB_MODEL_VERSION, STUB_SENTENCES, StubTranscriber, fingerprint
from aicls.asr.synth import make_speech_like, make_tone


def test_stub_transcriber_deterministic() -> None:
    """Same audio, same sentence — across calls, across instances, forever."""
    audio = make_speech_like(duration_s=2.0)

    first = StubTranscriber().transcribe(audio)
    second = StubTranscriber().transcribe(audio)

    assert first.text == second.text
    assert first.confidence == second.confidence
    assert first.text in STUB_SENTENCES
    assert first.model_version == STUB_MODEL_VERSION
    assert 0.8 <= first.confidence <= 1.0
    assert first.elapsed_ms >= 0.0


def test_stub_transcriber_distinguishes_different_audio() -> None:
    transcriber = StubTranscriber()
    texts = {
        transcriber.transcribe(make_tone(freq=freq, duration_s=1.0)).text
        for freq in (110.0, 220.0, 330.0, 440.0, 550.0, 660.0)
    }
    assert len(texts) > 1


def test_stub_transcript_has_one_segment_spanning_the_utterance() -> None:
    audio = make_speech_like(duration_s=1.5)
    transcript = StubTranscriber().transcribe(audio)
    assert len(transcript.segments) == 1
    segment = transcript.segments[0]
    assert segment.start_s == 0.0
    assert segment.end_s == pytest.approx(audio.duration_s)
    assert segment.text == transcript.text


def test_stub_transcriber_accepts_an_injected_sentence_pool() -> None:
    pool = ["türev tanımını getir", "integral tanımını getir"]
    transcriber = StubTranscriber(pool, model_version="asr-stub-test")
    assert transcriber.sentences == tuple(pool)
    transcript = transcriber.transcribe(make_speech_like(duration_s=1.0))
    assert transcript.text in pool
    assert transcript.model_version == "asr-stub-test"


def test_stub_transcriber_refuses_an_empty_pool() -> None:
    with pytest.raises(AsrInputError) as excinfo:
        StubTranscriber([])
    assert excinfo.value.details["reason"] == "stub_sentences_empty"


def test_stub_transcriber_returns_an_empty_transcript_for_empty_audio() -> None:
    transcript = StubTranscriber().transcribe(Audio(np.zeros(0, dtype=np.float32), 16000))
    assert transcript.text == ""
    assert transcript.confidence == 0.0
    assert transcript.segments == []


def test_stub_warmup_is_a_no_op() -> None:
    assert StubTranscriber().warmup() is None


def test_fingerprint_survives_a_pcm16_wav_roundtrip() -> None:
    """Quantising to PCM16 first means the stored WAV and the decoded audio agree."""
    audio = make_speech_like(duration_s=1.0)
    decoded = read_wav_pcm16(write_wav_pcm16(audio))
    assert fingerprint(audio) == fingerprint(decoded)
    assert fingerprint(audio).startswith("sha256:")


def test_fingerprint_depends_on_the_sample_rate() -> None:
    samples = make_tone(duration_s=0.5, sr=16000).samples
    assert fingerprint(Audio(samples, 16000)) != fingerprint(Audio(samples, 48000))


def test_stub_sentences_are_turkish_calculus_commands() -> None:
    assert len(STUB_SENTENCES) >= 8
    assert len(set(STUB_SENTENCES)) == len(STUB_SENTENCES)
    assert any("türev" in sentence for sentence in STUB_SENTENCES)
    assert any("integral" in sentence for sentence in STUB_SENTENCES)
