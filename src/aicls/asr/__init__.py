"""Speech edge (İP-3): WAV in, Turkish text out — plus quality control and augmentation.

``aicls.asr`` is the whole voice side of the platform except training:

* :mod:`~aicls.asr.types` — the torch-free value types: :class:`Audio`, :class:`Segment`,
  :class:`Transcript` and the ``Transcriber``/``TtsAugmenter`` protocols. Deterministic
  consumers (``aicls.collect``) and every pure module below import from *here*;
* :mod:`~aicls.asr.interfaces` — re-exports those types and adds the parts that may reach the
  heavy stack: :func:`get_transcriber` and :func:`resolve_device`;
* :mod:`~aicls.asr.audio_io` — PCM16 WAV reading/writing and the numpy windowed-sinc resampler
  (stdlib ``wave`` only; no ffmpeg, no ``torchaudio``, no ``soundfile`` — ADR-0017);
* :mod:`~aicls.asr.features` — log-Mel spectrograms in numpy;
* :mod:`~aicls.asr.qc` — :func:`run_qc` for the "Model Gelişimine Katkıda Bulun" tab;
* :mod:`~aicls.asr.augment` — SpecAugment, pitch shift, noise injection, speed perturbation;
* :mod:`~aicls.asr.synth` — deterministic synthetic waveforms (no recording is ever committed);
* :mod:`~aicls.asr.stub`, :mod:`~aicls.asr.whisper_transcriber`,
  :mod:`~aicls.asr.wav2vec2_transcriber` — the three backends.

Importing this package never imports ``torch`` or ``transformers``: the two model wrappers are
re-exported lazily through :func:`__getattr__`, so ``from aicls.asr import WhisperTranscriber``
still works but costs nothing until it is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
from aicls.asr.augment import (
    add_noise,
    measured_snr_db,
    pitch_shift,
    spec_augment,
    speed_perturb,
)
from aicls.asr.errors import AsrInputError, AsrModelError
from aicls.asr.features import DEFAULT_HOP, DEFAULT_N_FFT, DEFAULT_N_MELS, log_mel, mel_filterbank
from aicls.asr.interfaces import (
    TARGET_SAMPLE_RATE,
    TRANSCRIBER_NAMES,
    Audio,
    Segment,
    Transcriber,
    Transcript,
    TtsAugmenter,
    get_transcriber,
)
from aicls.asr.qc import (
    QcCheck,
    QcReport,
    QcResult,
    QcThresholds,
    clipping_ratio,
    dc_offset,
    estimate_snr_db,
    run_qc,
    silence_ratio,
)
from aicls.asr.stub import STUB_SENTENCES, StubTranscriber
from aicls.asr.synth import make_noise, make_speech_like, make_tone

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aicls.asr.wav2vec2_transcriber import Wav2Vec2Transcriber
    from aicls.asr.whisper_transcriber import WhisperTranscriber

_LAZY = {
    "WhisperTranscriber": "aicls.asr.whisper_transcriber",
    "Wav2Vec2Transcriber": "aicls.asr.wav2vec2_transcriber",
}


def __getattr__(name: str) -> Any:
    """Import the torch-backed transcribers only when they are actually referenced."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(set(__all__))


__all__ = [
    "ACCEPTED_SAMPLE_RATES",
    "DEFAULT_HOP",
    "DEFAULT_N_FFT",
    "DEFAULT_N_MELS",
    "MAX_SECONDS",
    "MAX_WAV_BYTES",
    "STUB_SENTENCES",
    "TARGET_SAMPLE_RATE",
    "TRANSCRIBER_NAMES",
    "AsrInputError",
    "AsrModelError",
    "Audio",
    "QcCheck",
    "QcReport",
    "QcResult",
    "QcThresholds",
    "Segment",
    "StubTranscriber",
    "Transcriber",
    "Transcript",
    "TtsAugmenter",
    "Wav2Vec2Transcriber",
    "WhisperTranscriber",
    "add_noise",
    "clipping_ratio",
    "dc_offset",
    "estimate_snr_db",
    "get_transcriber",
    "log_mel",
    "make_noise",
    "make_speech_like",
    "make_tone",
    "measured_snr_db",
    "mel_filterbank",
    "pitch_shift",
    "read_wav_pcm16",
    "resample",
    "resample_by_ratio",
    "resample_to_16k",
    "run_qc",
    "silence_ratio",
    "spec_augment",
    "speed_perturb",
    "write_wav_pcm16",
]
