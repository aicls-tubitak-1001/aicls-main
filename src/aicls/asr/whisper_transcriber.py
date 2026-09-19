"""Whisper inference behind a lazy import, from a **local** checkpoint directory only.

The proposal names Whisper as the ASR base model (A.2.3) and the target is WER < 10 % on a
speaker-disjoint Turkish test set. Fine-tuning lives in ``aicls.ml.whisper_ft``; this module is
only the inference edge the API talks to.

Two rules shape the code:

* **No download, ever.** ``WhisperTranscriber.load`` takes a directory under
  ``AICLS_MODELS_ROOT`` and passes it to ``from_pretrained`` with ``local_files_only=True``. A
  Hugging Face hub id is never accepted; ``tools/check_gates.py`` greps for that (ADR-0012).
* **No import at module scope.** ``transformers`` and ``torch`` are imported inside
  :meth:`WhisperTranscriber.load`, so importing :mod:`aicls.asr` costs milliseconds and the
  layering contract (``no-torch-in-core``) holds.

Decoding is greedy and forced to Turkish transcription (``language="tr"``, ``task="transcribe"``)
whenever the checkpoint's generation config knows those tokens; a tiny randomly initialised test
model does not, and is then decoded without the forced ids.

**What comes out is checked before it is handed on.** The project's guarantee is that answers
never come from a language model (proposal A.2.2); Whisper is one, and it sits upstream of the
deterministic path, so the guarantee holds only if its output is filtered rather than trusted.
:func:`guard_reason` withholds a transcript that is degenerate repetition or that the decoder was
not confident about, and the caller then receives empty text with confidence ``0.0`` -- the system
says it did not understand instead of inventing a command.

**One failure mode is not covered here, and this checkpoint cannot cover it.** Handed two seconds
of digital silence, the real ``whisper-large-v3-turbo`` returns "Altyazı M.K." with confidence
**0.9668**; handed quiet noise it returns "Evet." Neither guard above sees anything wrong, and
correctly so: the text is fluent, unrepetitive Turkish and the decoder is confident about it.

The guard designed for this is Whisper's ``no_speech`` probability, and it was measured on the RTX
5090 on 2026-09-09 before being relied on. It does not exist on this checkpoint. The token id
resolves correctly (``<|nospeech|>`` = 50363, which is also what
``generation_config.no_timestamps_token_id - 1`` gives, the way ``transformers`` locates it), but
on pure silence its probability at that position is **0.00000000** -- a hard zero, not a small
number. The position is dominated by ``<|transcribe|>`` (0.537) rather than a language token, so
``large-v3-turbo`` does not emit the no-speech signal at all.

So the guard for silence cannot be built from the model's output; it has to come from the audio.
:mod:`aicls.asr.qc` already measures exactly the right things (``silence_ratio``, ``snr_db``) for
contributed recordings, and wiring it into the live path is a decision about the API contract
rather than a change to this module -- its thresholds are tuned for read-aloud uploads, not for
short spoken commands. Recorded here so the next person starts from the measurement rather than
from the plan it invalidates (docs/ml/ses-ve-llm-degerlendirmesi.md §4.4, item 3).
"""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

from aicls.asr._backend import (
    import_torch,
    import_transformers,
    mean_probability,
    require_model_dir,
    select_device,
)
from aicls.asr.audio_io import resample_to_16k
from aicls.asr.errors import model_not_ready
from aicls.asr.types import TARGET_SAMPLE_RATE, Audio, Segment, Transcript
from aicls.core.logging import get_logger

_log = get_logger("aicls.asr")

#: Upper bound on generated tokens (a spoken Calculus command is far shorter).
DEFAULT_MAX_NEW_TOKENS = 128

#: Length of the silent warm-up utterance, in seconds.
_WARMUP_SECONDS = 0.5

#: Whisper's own receptive field. One call sees 30 seconds and no more.
MAX_UTTERANCE_SECONDS = 30.0

#: Mean token log-probability below which a transcript is not trusted.
#:
#: Whisper's published default. A decoder that is guessing spreads its probability mass, and the
#: number it produces then describes the decoder's uncertainty rather than the speaker.
DEFAULT_LOGPROB_THRESHOLD = -1.0

#: gzip compression ratio above which a transcript is degenerate repetition.
#:
#: Whisper's published default. A decoding loop ("evet evet evet evet ...") compresses far better
#: than Turkish prose, which is what makes the ratio a usable detector for it.
DEFAULT_COMPRESSION_RATIO_THRESHOLD = 2.4

#: Reason codes :meth:`WhisperTranscriber.transcribe` reports when it withholds a transcript.
GUARD_LOW_LOGPROB = "low_logprob"
GUARD_REPETITIVE = "repetitive"


def compression_ratio(text: str) -> float:
    """Ratio of a transcript's size to its gzip-compressed size (Whisper's definition).

    Prose sits near 1.5-2.0; a decoding loop climbs far above that because the repeated span
    compresses almost completely away.

    Args:
        text: The decoded transcript.

    Returns:
        The ratio; ``0.0`` for an empty string, which no threshold should reject.
    """
    import zlib

    raw = text.encode("utf-8")
    if not raw:
        return 0.0
    return len(raw) / len(zlib.compress(raw))


def guard_reason(
    text: str,
    *,
    avg_logprob: float,
    logprob_threshold: float = DEFAULT_LOGPROB_THRESHOLD,
    compression_ratio_threshold: float = DEFAULT_COMPRESSION_RATIO_THRESHOLD,
) -> str | None:
    """Why this transcript must not be handed on, or ``None`` when it may be.

    The project's central claim is that answers never come from a language model (proposal
    A.2.2). Whisper is a language model, and it sits upstream of the deterministic path, so the
    claim survives only if what it produces is *checked* before anything acts on it. A withheld
    transcript costs the speaker a repeat; an unchecked one puts an invented command in front of
    a student.

    Args:
        text: The decoded transcript.
        avg_logprob: Mean log-probability of the emitted tokens.
        logprob_threshold: Reject below this.
        compression_ratio_threshold: Reject above this.

    Returns:
        :data:`GUARD_LOW_LOGPROB`, :data:`GUARD_REPETITIVE`, or ``None``.
    """
    if not text:
        return None
    if compression_ratio(text) > compression_ratio_threshold:
        return GUARD_REPETITIVE
    if avg_logprob < logprob_threshold:
        return GUARD_LOW_LOGPROB
    return None


class WhisperTranscriber:
    """Whisper sequence-to-sequence transcriber (Hugging Face ``transformers``).

    Build it with :meth:`load`; the constructor takes already-built objects so tests and
    ``aicls.ml`` can inject a tiny model without touching the filesystem.

    Args:
        model: A ``WhisperForConditionalGeneration`` (or anything with the same ``generate``).
        processor: A ``WhisperProcessor`` (feature extractor + tokenizer).
        device: Device the model already lives on.
        language: Decoder language token; Turkish for this project.
        task: ``"transcribe"`` (never ``"translate"``).
        model_version: Registry version string reported in the transcript.
        max_new_tokens: Decoding cap.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        device: str = "cpu",
        language: str = "tr",
        task: str = "transcribe",
        model_version: str = "whisper-local",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> None:
        self._model = model
        self._processor = processor
        self._torch = import_torch()
        self.device = device
        self.language = language
        self.task = task
        self.model_version = model_version
        self.max_new_tokens = max_new_tokens

    @classmethod
    def load(
        cls,
        model_dir: Path | str,
        device: str = "cpu",
        *,
        language: str = "tr",
        task: str = "transcribe",
    ) -> WhisperTranscriber:
        """Load a fine-tuned Whisper checkpoint from a local directory.

        Args:
            model_dir: Directory holding ``config.json``, the weights and the processor files
                (written by ``save_pretrained``). Never a hub id.
            device: ``"cpu"`` or ``"cuda"``; falls back to CPU when CUDA is unavailable.
            language: Decoder language token.
            task: Decoder task token.

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
            processor = transformers.WhisperProcessor.from_pretrained(
                local_dir, local_files_only=True
            )
            torch = import_torch()
            # fp16 on CUDA only. On CPU half precision is emulated and slower than fp32, which
            # is the configuration the container's own tests run in.
            kwargs: dict[str, Any] = {"local_files_only": True}
            if resolved_device.startswith("cuda"):
                kwargs["dtype"] = torch.float16
                kwargs["attn_implementation"] = "sdpa"
            model = transformers.WhisperForConditionalGeneration.from_pretrained(
                local_dir, **kwargs
            )
        except (OSError, ValueError, KeyError) as exc:
            raise model_not_ready("whisper_load_failed", path=local_dir, error=str(exc)) from exc
        model.to(resolved_device)
        model.eval()
        _log.info("asr_whisper_loaded", path=local_dir, device=resolved_device)
        return cls(
            model,
            processor,
            device=resolved_device,
            language=language,
            task=task,
            model_version=path.name,
        )

    def _model_dtype(self) -> Any:
        """The dtype the weights actually live in, so inputs can be made to match.

        Read from a parameter rather than remembered from ``load``: the constructor also accepts
        a model somebody else built, and a stub in a test has no dtype of ours to recall.
        """
        try:
            return next(self._model.parameters()).dtype
        except (StopIteration, AttributeError):  # pragma: no cover - a parameterless test double
            return self._torch.float32

    def _token_budget(self) -> int:
        """Decoding cap, lowered to what the checkpoint's decoder positions actually allow.

        A production Whisper has 448 target positions and the cap never binds; a tiny test model
        has far fewer, and asking for more is a hard error inside ``generate``.
        """
        positions = getattr(getattr(self._model, "config", None), "max_target_positions", None)
        if isinstance(positions, int) and positions > 1:
            return min(self.max_new_tokens, positions - 1)
        return self.max_new_tokens

    def _forced_kwargs(self) -> dict[str, str]:
        """Language/task kwargs, but only when the checkpoint's tokenizer knows those tokens."""
        generation_config = getattr(self._model, "generation_config", None)
        if generation_config is None or not getattr(generation_config, "lang_to_id", None):
            return {}
        return {"language": self.language, "task": self.task}

    def warmup(self) -> None:
        """Transcribe half a second of silence so the first real request is not slow."""
        import numpy as np

        samples = np.zeros(int(TARGET_SAMPLE_RATE * _WARMUP_SECONDS), np.float32)
        self.transcribe(Audio(samples, TARGET_SAMPLE_RATE))

    def transcribe(self, audio: Audio) -> Transcript:
        """Transcribe one utterance with greedy decoding.

        Args:
            audio: Waveform at any accepted sample rate; resampled to 16 kHz first.

        Returns:
            The transcript; ``confidence`` is the mean probability of the emitted tokens.

        Raises:
            AsrModelError: If generation fails inside ``transformers``.
        """
        started = time.perf_counter()
        torch = self._torch
        prepared = resample_to_16k(audio)
        if prepared.duration_s > MAX_UTTERANCE_SECONDS:
            # The feature extractor would silently keep the first 30 seconds. Silent cropping is
            # how a data-loss bug survives for months: the caller gets a plausible transcript of
            # a fraction of what was said, and nothing anywhere records that the rest was
            # dropped. The upper layers already bound utterance length, so this cannot fire on a
            # legitimate request.
            raise model_not_ready(
                "whisper_utterance_too_long",
                duration_s=round(prepared.duration_s, 3),
                limit_s=MAX_UTTERANCE_SECONDS,
            )
        features = self._processor.feature_extractor(
            prepared.samples, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        )
        # The features must follow the weights. Loading the checkpoint in fp16 without casting
        # here fails inside the first convolution with "Input type (float) and bias type
        # (c10::Half) should be the same" - caught on the RTX 5090 with the real
        # whisper-large-v3-turbo, which is the only place a dtype mismatch can show up.
        input_features = features["input_features"].to(self.device, dtype=self._model_dtype())
        try:
            with torch.no_grad():
                output = self._model.generate(
                    input_features,
                    max_new_tokens=self._token_budget(),
                    num_beams=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **self._forced_kwargs(),
                )
        except (RuntimeError, ValueError) as exc:
            raise model_not_ready("whisper_generate_failed", error=str(exc)) from exc
        sequences = output.sequences
        decoded = self._processor.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        text = decoded[0].strip()
        confidence = mean_probability(torch, getattr(output, "scores", None), sequences)
        avg_logprob = math.log(confidence) if confidence > 0.0 else float("-inf")
        withheld = guard_reason(text, avg_logprob=avg_logprob)
        if withheld is not None:
            _log.warning(
                "asr_whisper_transcript_withheld",
                reason=withheld,
                chars=len(text),
                compression_ratio=round(compression_ratio(text), 3),
                avg_logprob=round(avg_logprob, 4),
                model_version=self.model_version,
            )
            text = ""
            confidence = 0.0
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return Transcript(
            text=text,
            confidence=round(confidence, 4),
            segments=[Segment(0.0, prepared.duration_s, text, round(confidence, 4))],
            elapsed_ms=elapsed_ms,
            model_version=self.model_version,
        )


__all__ = [
    "DEFAULT_COMPRESSION_RATIO_THRESHOLD",
    "DEFAULT_LOGPROB_THRESHOLD",
    "DEFAULT_MAX_NEW_TOKENS",
    "GUARD_LOW_LOGPROB",
    "GUARD_REPETITIVE",
    "MAX_UTTERANCE_SECONDS",
    "WhisperTranscriber",
    "compression_ratio",
    "guard_reason",
]
