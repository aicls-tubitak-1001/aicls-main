"""Speech-edge public surface: the value types plus the backend factory.

The value types themselves (:class:`Audio`, :class:`Segment`, :class:`Transcript`, the
``Transcriber``/``TtsAugmenter`` protocols, :func:`concat`, :data:`TARGET_SAMPLE_RATE`,
:data:`TRANSCRIBER_NAMES`) live in the torch-free :mod:`aicls.asr.types` and are re-exported here
unchanged, so ``from aicls.asr.interfaces import Audio`` keeps working.

What is defined *here* is the part that may touch the heavy stack: :func:`get_transcriber` and
:func:`resolve_device`. Both keep ``torch``/``transformers`` behind a lazy import inside the
function body (ADR-0002 layering, ADR-0017 audio format), but even that lazy edge is enough to
break the ``no-torch-in-core`` import contract for deterministic consumers — which is why the
pure modules (:mod:`aicls.asr.audio_io`, :mod:`aicls.asr.qc`, :mod:`aicls.asr.features`,
:mod:`aicls.asr.augment`, :mod:`aicls.asr.synth`, :mod:`aicls.asr.stub`) import from
:mod:`aicls.asr.types` and never from this module.

The API layer (``aicls.api.model_registry``) only ever sees these types plus
:func:`get_transcriber`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aicls.asr.types import (
    TARGET_SAMPLE_RATE,
    TRANSCRIBER_NAMES,
    Audio,
    Segment,
    Transcriber,
    Transcript,
    TtsAugmenter,
    concat,
)
from aicls.core.errors import InputValidationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path


def get_transcriber(
    name: str = "stub",
    *,
    model_dir: Path | None = None,
    device: str = "cpu",
) -> Transcriber:
    """Build a :class:`Transcriber` by backend name.

    Args:
        name: One of :data:`TRANSCRIBER_NAMES`. ``"stub"`` needs no weights and no torch.
        model_dir: Local checkpoint directory under ``AICLS_MODELS_ROOT``. Required for
            ``"whisper"`` and ``"wav2vec2"``; a Hugging Face hub id is never accepted (ADR-0012).
        device: ``"cpu"`` or ``"cuda"``/``"cuda:0"``; falls back to CPU when CUDA is absent.

    Returns:
        A ready-to-use transcriber (not warmed up).

    Raises:
        InputValidationError: On an unknown backend name, or a missing ``model_dir`` for a
            backend that needs weights.
        AsrModelError: If the backend cannot be loaded (see :mod:`aicls.asr.errors`).
    """
    key = name.strip().lower()
    if key == "stub":
        from aicls.asr.stub import StubTranscriber

        return StubTranscriber()
    if key in {"whisper", "wav2vec2"}:
        if model_dir is None:
            raise InputValidationError(
                details={"reason": "model_dir_required", "backend": key},
            )
        if key == "whisper":
            from aicls.asr.whisper_transcriber import WhisperTranscriber

            return WhisperTranscriber.load(model_dir, device)
        from aicls.asr.wav2vec2_transcriber import Wav2Vec2Transcriber

        return Wav2Vec2Transcriber.load(model_dir, device)
    raise InputValidationError(
        details={"reason": "unknown_transcriber", "name": name, "known": list(TRANSCRIBER_NAMES)},
    )


def resolve_device(device: str) -> str:
    """Return a device string that ``torch`` can actually use.

    ``"cuda"`` degrades to ``"cpu"`` when torch is missing or reports no CUDA device, so the
    CPU-only development host never crashes on a configuration meant for the workstation.
    """
    wanted = device.strip().lower() or "cpu"
    if not wanted.startswith("cuda"):
        return wanted
    try:
        import torch
    except ImportError:
        return "cpu"
    return wanted if torch.cuda.is_available() else "cpu"


__all__ = [
    "TARGET_SAMPLE_RATE",
    "TRANSCRIBER_NAMES",
    "Audio",
    "Segment",
    "Transcriber",
    "Transcript",
    "TtsAugmenter",
    "concat",
    "get_transcriber",
    "resolve_device",
]
