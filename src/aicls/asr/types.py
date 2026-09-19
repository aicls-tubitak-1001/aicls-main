"""Torch-free speech-edge value types: :class:`Audio`, :class:`Segment`, :class:`Transcript`.

This module is the **pure** half of the speech edge. It holds the value types and protocols that
every caller needs and nothing that could ever reach ``torch`` or ``transformers`` — not even
behind a lazy import. The backend factory (:func:`aicls.asr.interfaces.get_transcriber`) and the
device resolver (:func:`aicls.asr.interfaces.resolve_device`) stay in
:mod:`aicls.asr.interfaces`, which re-exports everything defined here for backwards compatibility.

The split exists because of the ADR-0002 layering contract ``no-torch-in-core``: deterministic
packages such as :mod:`aicls.collect` consume :mod:`aicls.asr.audio_io` and :mod:`aicls.asr.qc`,
and a static import path from those modules to ``torch`` — even one that never executes — breaks
the contract. Pure modules therefore import from here, never from
:mod:`aicls.asr.interfaces`.

``Audio`` always holds **mono float32 samples in [-1, 1]**; the constructor normalises whatever
array-like it is given, so callers never have to think about dtype or shape. ``duration_s`` is
derived from ``len(samples) / sr`` and cannot drift out of sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from aicls.asr.errors import invalid_audio

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

#: Sample rate every model and every quality check works at (ADR-0017).
TARGET_SAMPLE_RATE = 16000

#: Backend names accepted by :func:`aicls.asr.interfaces.get_transcriber`.
TRANSCRIBER_NAMES: tuple[str, ...] = ("stub", "whisper", "wav2vec2")


@dataclass(frozen=True, slots=True, eq=False)
class Audio:
    """A mono waveform with its sample rate.

    Args:
        samples: Any array-like of samples; converted to a contiguous 1-D ``float32`` array.
        sr: Sample rate in Hz (must be positive).
        duration_s: Accepted for call compatibility but always **recomputed** from
            ``len(samples) / sr``; passing a value never overrides the derived duration.

    Raises:
        AsrInputError: If ``sr`` is not positive or the samples are not numeric.
    """

    samples: npt.NDArray[np.float32] = field(repr=False)
    sr: int
    duration_s: float = 0.0

    def __post_init__(self) -> None:
        try:
            array = np.asarray(self.samples, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise invalid_audio("samples_not_numeric", error=str(exc)) from exc
        if array.ndim > 1:
            array = array.reshape(-1)
        sr = int(self.sr)
        if sr <= 0:
            raise invalid_audio("sample_rate_not_positive", sample_rate=sr)
        object.__setattr__(self, "samples", np.ascontiguousarray(array))
        object.__setattr__(self, "sr", sr)
        object.__setattr__(self, "duration_s", float(array.size) / float(sr))

    @property
    def n_samples(self) -> int:
        """Number of samples in the waveform."""
        return int(self.samples.size)

    def with_samples(self, samples: npt.NDArray[np.float32]) -> Audio:
        """Return a copy carrying ``samples`` at the same sample rate."""
        return Audio(samples, self.sr)

    def __repr__(self) -> str:
        return f"Audio(n_samples={self.n_samples}, sr={self.sr}, duration_s={self.duration_s:.3f})"


@dataclass(frozen=True, slots=True)
class Segment:
    """One timed piece of a transcript (Whisper returns several for long audio)."""

    start_s: float
    end_s: float
    text: str
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class Transcript:
    """The result of :meth:`Transcriber.transcribe`.

    Attributes:
        text: Recognised Turkish text (already stripped).
        confidence: Mean token/frame probability in ``[0, 1]``; the web UI always asks for
            confirmation on the voice path when it drops below 0.8 (docs/mimari/api.md §3.6).
        segments: Timed segments; a single whole-utterance segment for the stub and CTC paths.
        elapsed_ms: Wall-clock inference time, for ``Server-Timing`` and the latency budget.
        model_version: Registry version string (``"asr-0.2.0"``, ``"asr-stub-1"``, ...).
    """

    text: str
    confidence: float = 1.0
    segments: list[Segment] = field(default_factory=list)
    elapsed_ms: float = 0.0
    model_version: str = "unknown"


@runtime_checkable
class Transcriber(Protocol):
    """Speech-to-text backend contract (stub, Whisper, Wav2Vec2)."""

    def transcribe(self, audio: Audio) -> Transcript:
        """Transcribe one utterance.

        Args:
            audio: Mono waveform; implementations resample to 16 kHz themselves.

        Returns:
            The :class:`Transcript`.
        """
        ...

    def warmup(self) -> None:
        """Run one throwaway inference so the first real request is not slow."""
        ...


@runtime_checkable
class TtsAugmenter(Protocol):
    """Text-to-speech augmentation contract.

    No engine is chosen yet: the engine decision is a separate ADR and its use is gated, because
    synthetic speech must be tagged ``source="tts"`` in the manifest and **never** enters the test
    split (docs/ml/veri-yonetimi.md §77).
    """

    def synthesize(self, text: str, voice: str) -> Audio:
        """Render ``text`` with ``voice`` as a waveform."""
        ...


def concat(audios: Sequence[Audio]) -> Audio:
    """Concatenate waveforms that share a sample rate.

    Raises:
        AsrInputError: If the sequence is empty or the sample rates differ.
    """
    if not audios:
        raise invalid_audio("empty_audio_sequence")
    sr = audios[0].sr
    if any(a.sr != sr for a in audios):
        raise invalid_audio("sample_rate_mismatch", sample_rates=sorted({a.sr for a in audios}))
    return Audio(np.concatenate([a.samples for a in audios]), sr)


__all__ = [
    "TARGET_SAMPLE_RATE",
    "TRANSCRIBER_NAMES",
    "Audio",
    "Segment",
    "Transcriber",
    "Transcript",
    "TtsAugmenter",
    "concat",
]
