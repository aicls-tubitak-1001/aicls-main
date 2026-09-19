"""Deterministic stub transcriber — the default backend everywhere weights are absent.

The stub is what makes the whole platform testable before a single model is fine-tuned: the API,
the WebSocket audio stream, the NLU layer and the end-to-end pipeline tests all run against it
(``AICLS_MODELS_ENABLED=false``). It never imports torch.

"Deterministic" means the *content* of the waveform selects the sentence: the samples are
quantised to PCM16, hashed with SHA-256 together with the sample rate, and the digest indexes the
sentence table. The same audio therefore always yields the same Turkish sentence, on every
machine and in every process, while different audio yields different sentences — which is exactly
what a fixture-driven pipeline test needs.
"""

from __future__ import annotations

from collections.abc import Sequence
import time

import numpy as np

from aicls.asr.errors import invalid_audio
from aicls.asr.types import Audio, Segment, Transcript
from aicls.core.hashing import sha256_bytes

#: Version string reported in :attr:`aicls.asr.interfaces.Transcript.model_version`.
STUB_MODEL_VERSION = "asr-stub-1"

# Seed sentences from the reading-scenario library (İP-3; the full library lives in
# ``datasets/nlu/reading_scenarios.yaml`` and is loaded by ``aicls.collect``). These are fixture
# data for a fake model, not user-interface text, so they do not belong in
# ``aicls.core.messages_tr``; pass ``sentences=`` to drive the stub from the real library.
STUB_SENTENCES: tuple[str, ...] = (
    "x karenin türevini al",
    "sinüs x'in integralini hesapla",
    "x kare artı üç x eksi iki fonksiyonunun kökünü bul",
    "limit x sıfıra giderken sinüs x bölü x",
    "türev tanımını getir",
    "iki x üzeri üç fonksiyonunun grafiğini çiz",
    "x kare eksi dört ifadesini çarpanlarına ayır",
    "e üzeri x fonksiyonunun ikinci türevini al",
    "sıfırdan bire kadar x kare integralini hesapla",
    "kosinüs x fonksiyonunun türevini bul",
    "logaritma x fonksiyonunun tanım kümesini söyle",
    "üç x artı beş eşittir on denklemini çöz",
)

#: Hex digits of the digest used to pick the sentence.
_SENTENCE_DIGITS = 8

#: Hex digits of the digest used to derive the reported confidence.
_CONFIDENCE_DIGITS = 4

#: Confidence band the stub reports: high enough to exercise the accept path, never exactly 1.0.
_MIN_CONFIDENCE = 0.82
_MAX_CONFIDENCE = 0.99


def fingerprint(audio: Audio) -> str:
    """Stable content hash of a waveform (``"sha256:<hex>"``).

    The samples are quantised to PCM16 first, so audio that survived a WAV round trip hashes
    identically to the audio that was written.
    """
    pcm = np.round(np.clip(audio.samples, -1.0, 1.0) * 32767.0).astype("<i2")
    return "sha256:" + sha256_bytes(pcm.tobytes() + str(audio.sr).encode("ascii"))


class StubTranscriber:
    """A :class:`~aicls.asr.interfaces.Transcriber` that needs no weights and no torch.

    Args:
        sentences: Sentence pool to choose from; :data:`STUB_SENTENCES` when omitted.
        model_version: Value reported in the transcript.

    Raises:
        AsrInputError: If ``sentences`` is empty.
    """

    def __init__(
        self,
        sentences: Sequence[str] | None = None,
        *,
        model_version: str = STUB_MODEL_VERSION,
    ) -> None:
        pool = tuple(sentences) if sentences is not None else STUB_SENTENCES
        if not pool:
            raise invalid_audio("stub_sentences_empty")
        self._sentences = pool
        self.model_version = model_version

    @property
    def sentences(self) -> tuple[str, ...]:
        """The sentence pool this instance selects from."""
        return self._sentences

    def warmup(self) -> None:
        """No-op: the stub has nothing to load."""

    def transcribe(self, audio: Audio) -> Transcript:
        """Map the waveform's fingerprint to a fixed sentence.

        Args:
            audio: Any waveform; empty audio yields an empty transcript with confidence 0.

        Returns:
            The :class:`~aicls.asr.interfaces.Transcript`, with one segment spanning the whole
            recording.
        """
        started = time.perf_counter()
        if audio.n_samples == 0:
            return Transcript(
                text="",
                confidence=0.0,
                segments=[],
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                model_version=self.model_version,
            )
        digest = fingerprint(audio).removeprefix("sha256:")
        index = int(digest[:_SENTENCE_DIGITS], 16) % len(self._sentences)
        text = self._sentences[index]
        span = _MAX_CONFIDENCE - _MIN_CONFIDENCE
        steps = 16**_CONFIDENCE_DIGITS
        confidence = _MIN_CONFIDENCE + span * (
            int(digest[_SENTENCE_DIGITS : _SENTENCE_DIGITS + _CONFIDENCE_DIGITS], 16) / steps
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return Transcript(
            text=text,
            confidence=round(confidence, 4),
            segments=[Segment(0.0, audio.duration_s, text, round(confidence, 4))],
            elapsed_ms=elapsed_ms,
            model_version=self.model_version,
        )


__all__ = ["STUB_MODEL_VERSION", "STUB_SENTENCES", "StubTranscriber", "fingerprint"]
