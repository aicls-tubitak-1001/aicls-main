"""Wav2Vec2 CTC transcriber — the İP-3 risk plan's alternative to Whisper, kept warm in code.

The proposal names Wav2Vec/DeepSpeech alongside Whisper and the İP-3 risk plan mentions an
ensemble. A CTC model has a different error profile from a sequence-to-sequence model (it never
hallucinates fluent-but-wrong text), which makes it a genuine fallback rather than a duplicate.

Same two rules as :mod:`aicls.asr.whisper_transcriber`: local checkpoint directory only (never a
hub id, ``local_files_only=True``, ADR-0012) and no ``torch``/``transformers`` import at module
scope. Decoding is greedy CTC collapse, which the Hugging Face tokenizer does for us.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from aicls.asr._backend import (
    import_torch,
    import_transformers,
    require_model_dir,
    select_device,
)
from aicls.asr.audio_io import resample_to_16k
from aicls.asr.errors import model_not_ready
from aicls.asr.types import TARGET_SAMPLE_RATE, Audio, Segment, Transcript
from aicls.core.logging import get_logger

_log = get_logger("aicls.asr")

#: Length of the silent warm-up utterance, in seconds.
_WARMUP_SECONDS = 0.5


class Wav2Vec2Transcriber:
    """Wav2Vec2 CTC transcriber (Hugging Face ``transformers``).

    Args:
        model: A ``Wav2Vec2ForCTC`` (or anything returning ``logits``).
        processor: A ``Wav2Vec2Processor`` (feature extractor + CTC tokenizer).
        device: Device the model already lives on.
        model_version: Registry version string reported in the transcript.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        device: str = "cpu",
        model_version: str = "wav2vec2-local",
    ) -> None:
        self._model = model
        self._processor = processor
        self._torch = import_torch()
        self.device = device
        self.model_version = model_version

    @classmethod
    def load(cls, model_dir: Path | str, device: str = "cpu") -> Wav2Vec2Transcriber:
        """Load a fine-tuned Wav2Vec2 CTC checkpoint from a local directory.

        Args:
            model_dir: Directory holding ``config.json``, the weights, ``vocab.json`` and the
                feature-extractor config. Never a hub id.
            device: ``"cpu"`` or ``"cuda"``; falls back to CPU when CUDA is unavailable.

        Returns:
            A ready transcriber (call :meth:`warmup` before serving traffic).

        Raises:
            NotFoundError: If ``model_dir`` does not exist.
            AsrModelError: If the directory is not a checkpoint, or torch/transformers are
                missing, or the checkpoint cannot be instantiated.
        """
        path = require_model_dir(model_dir)
        transformers = import_transformers()
        import_torch()
        resolved_device = select_device(device)
        local_dir = str(path)
        try:
            processor = transformers.Wav2Vec2Processor.from_pretrained(
                local_dir, local_files_only=True
            )
            model = transformers.Wav2Vec2ForCTC.from_pretrained(local_dir, local_files_only=True)
        except (OSError, ValueError, KeyError) as exc:
            raise model_not_ready("wav2vec2_load_failed", path=local_dir, error=str(exc)) from exc
        model.to(resolved_device)
        model.eval()
        _log.info("asr_wav2vec2_loaded", path=local_dir, device=resolved_device)
        return cls(model, processor, device=resolved_device, model_version=path.name)

    def warmup(self) -> None:
        """Transcribe half a second of silence so the first real request is not slow."""
        import numpy as np

        silence = Audio(
            np.zeros(int(TARGET_SAMPLE_RATE * _WARMUP_SECONDS), np.float32), TARGET_SAMPLE_RATE
        )
        self.transcribe(silence)

    def transcribe(self, audio: Audio) -> Transcript:
        """Transcribe one utterance with greedy CTC decoding.

        Args:
            audio: Waveform at any accepted sample rate; resampled to 16 kHz first.

        Returns:
            The transcript; ``confidence`` is the mean of the per-frame maximum softmax
            probabilities, the standard cheap CTC confidence.

        Raises:
            AsrModelError: If the forward pass fails inside ``transformers``.
        """
        started = time.perf_counter()
        torch = self._torch
        prepared = resample_to_16k(audio)
        inputs = self._processor(
            prepared.samples, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        )
        values = inputs["input_values"].to(self.device)
        try:
            with torch.no_grad():
                logits = self._model(values).logits
        except (RuntimeError, ValueError) as exc:
            raise model_not_ready("wav2vec2_forward_failed", error=str(exc)) from exc
        probabilities = torch.softmax(logits.float(), dim=-1)
        predicted = torch.argmax(probabilities, dim=-1)
        peaks = probabilities.max(dim=-1).values
        confidence = float(peaks.mean()) if peaks.numel() else 1.0
        text = self._processor.batch_decode(predicted)[0].strip()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return Transcript(
            text=text,
            confidence=round(confidence, 4),
            segments=[Segment(0.0, prepared.duration_s, text, round(confidence, 4))],
            elapsed_ms=elapsed_ms,
            model_version=self.model_version,
        )


__all__ = ["Wav2Vec2Transcriber"]
