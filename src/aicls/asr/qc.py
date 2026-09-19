"""Automatic quality control for contributed recordings (İP-3, "Model Gelişimine Katkıda Bulun").

``POST /contribute/recordings`` runs :func:`run_qc` synchronously (budget < 300 ms) and stores the
report in ``recordings.qc_json``; a rejected upload is never written to disk
(docs/mimari/api.md §3.8, docs/mimari/genel-bakis.md §2.5). Five measurements, all in numpy:

===================  ====================================================================
``duration_min``     the utterance is long enough to carry a command
``duration_max``     the utterance is not a runaway recording
``clipping_ratio``   fraction of samples at digital full scale (input gain too high)
``snr_db``           loud-frame energy over quiet-frame energy (noisy room, far microphone)
``silence_ratio``    fraction of quiet frames (participant did not speak, or dead air)
``dc_offset``        mean sample value (a broken or badly biased microphone)
===================  ====================================================================

The SNR estimate needs pauses to see a noise floor: it compares the 90th and 10th percentiles of
the frame RMS, so a recording that is *uniformly* loud (a pure tone, a continuous hum) reads as
low SNR by design. Real read-aloud prompts always contain leading, trailing and inter-word
silence, which is exactly what the estimator uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np
import numpy.typing as npt

from aicls.asr.features import frame_rms
from aicls.asr.types import Audio
from aicls.core.messages_tr import (
    MSG_QC_CLIPPING_RATIO,
    MSG_QC_DC_OFFSET,
    MSG_QC_DURATION_MAX,
    MSG_QC_DURATION_MIN,
    MSG_QC_SILENCE_RATIO,
    MSG_QC_SNR_DB,
)

#: Analysis frame for the energy-based measures (20 ms at 16 kHz).
_FRAME_MS = 20.0

#: Analysis hop for the energy-based measures (10 ms at 16 kHz).
_HOP_MS = 10.0

#: A sample at or above this magnitude counts as clipped.
_CLIP_LEVEL = 0.99

#: A frame quieter than this fraction of the loud-frame level counts as silence.
_SILENCE_RELATIVE = 0.1

#: Absolute silence floor (~-60 dBFS) so an all-quiet recording is not "all speech".
_SILENCE_ABSOLUTE = 1e-3

#: Percentiles used for the noise floor and the speech level of the SNR estimate.
_NOISE_PERCENTILE = 10.0
_SIGNAL_PERCENTILE = 90.0

#: RMS below which a noise floor is treated as digital silence.
_NOISE_EPSILON = 1e-6

#: Reported SNR when the noise floor is digital silence.
MAX_SNR_DB = 100.0

# ---------------------------------------------------------------------------------------------
# Turkish rejection texts.
#
# House rule: Turkish user-facing strings are *defined* in ``aicls.core.messages_tr`` and nowhere
# else. This table only binds each ``QcCheck.name`` to its ``MSG_QC_*`` template, so a translation
# change happens in one file and this module keeps its stable English keys.
# ---------------------------------------------------------------------------------------------
QC_REASONS_TR: dict[str, str] = {
    "duration_min": MSG_QC_DURATION_MIN,
    "duration_max": MSG_QC_DURATION_MAX,
    "clipping_ratio": MSG_QC_CLIPPING_RATIO,
    "snr_db": MSG_QC_SNR_DB,
    "silence_ratio": MSG_QC_SILENCE_RATIO,
    "dc_offset": MSG_QC_DC_OFFSET,
}


@dataclass(frozen=True, slots=True)
class QcThresholds:
    """Acceptance thresholds for a contributed recording.

    Defaults follow docs/mimari/moduller.md; the collection tab may tighten them per campaign.
    """

    min_duration_s: float = 0.8
    max_duration_s: float = 20.0
    max_clipping_ratio: float = 0.001
    min_snr_db: float = 15.0
    max_silence_ratio: float = 0.6
    max_dc_offset: float = 0.02


@dataclass(frozen=True, slots=True)
class QcCheck:
    """One measurement and its verdict.

    Attributes:
        name: Stable English key (also the key into :data:`QC_REASONS_TR`).
        value: The measured value.
        threshold: The bound it was compared against.
        passed: Whether the measurement satisfies the bound.
    """

    name: str
    value: float
    threshold: float
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form for ``recordings.qc_json`` and the API response."""
        return {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class QcReport:
    """Outcome of :func:`run_qc`.

    Attributes:
        passed: True only when every check passed.
        checks: Every measurement, in a stable order, whether it passed or not.
        reasons_tr: One Turkish sentence per failed check, in the same order.
    """

    passed: bool
    checks: list[QcCheck] = field(default_factory=list)
    reasons_tr: list[str] = field(default_factory=list)

    @property
    def failed_checks(self) -> list[QcCheck]:
        """The checks that did not pass."""
        return [check for check in self.checks if not check.passed]

    def value(self, name: str) -> float:
        """Measured value of check ``name``.

        Raises:
            KeyError: If no check by that name was run.
        """
        for check in self.checks:
            if check.name == name:
                return check.value
        raise KeyError(name)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable form matching ``RecordingOut.qc`` (docs/mimari/api.md §3.8)."""
        return {
            "passed": self.passed,
            "checks": [check.as_dict() for check in self.checks],
            "reasons_tr": list(self.reasons_tr),
        }


#: Alias kept because early drafts of the module spec called the report ``QcResult``.
QcResult = QcReport


def clipping_ratio(audio: Audio) -> float:
    """Fraction of samples at or beyond :data:`_CLIP_LEVEL` of full scale."""
    if audio.n_samples == 0:
        return 0.0
    return float(np.count_nonzero(np.abs(audio.samples) >= _CLIP_LEVEL)) / float(audio.n_samples)


def dc_offset(audio: Audio) -> float:
    """Absolute mean sample value (a healthy microphone sits near 0)."""
    if audio.n_samples == 0:
        return 0.0
    return float(abs(np.mean(audio.samples.astype(np.float64))))


def _rms_frames(audio: Audio) -> npt.NDArray[np.float32]:
    frame_len = max(1, round(audio.sr * _FRAME_MS / 1000.0))
    hop = max(1, round(audio.sr * _HOP_MS / 1000.0))
    return frame_rms(audio.samples, frame_len=frame_len, hop=hop)


def estimate_snr_db(audio: Audio) -> float:
    """Estimate the signal-to-noise ratio in dB from the frame-energy distribution.

    The loud-frame level is the 90th percentile of the frame RMS and the noise floor the 10th;
    their ratio in dB is the estimate. A digitally silent noise floor reports :data:`MAX_SNR_DB`.
    """
    if audio.n_samples == 0:
        return 0.0
    rms = _rms_frames(audio).astype(np.float64)
    noise = float(np.percentile(rms, _NOISE_PERCENTILE))
    signal = float(np.percentile(rms, _SIGNAL_PERCENTILE))
    if signal <= _NOISE_EPSILON:
        return 0.0
    if noise <= _NOISE_EPSILON:
        return MAX_SNR_DB
    return min(MAX_SNR_DB, 20.0 * math.log10(signal / noise))


def silence_ratio(audio: Audio) -> float:
    """Fraction of frames quieter than 10 % of the loud-frame level (or ~-60 dBFS)."""
    if audio.n_samples == 0:
        return 1.0
    rms = _rms_frames(audio).astype(np.float64)
    loud = float(np.percentile(rms, 95.0))
    threshold = max(_SILENCE_ABSOLUTE, _SILENCE_RELATIVE * loud)
    return float(np.count_nonzero(rms < threshold)) / float(rms.size)


def _reason_tr(check: QcCheck) -> str:
    template = QC_REASONS_TR[check.name]
    return template.format(
        value=check.value,
        threshold=check.threshold,
        value_pct=check.value * 100.0,
        threshold_pct=check.threshold * 100.0,
    )


def run_qc(audio: Audio, thresholds: QcThresholds | None = None) -> QcReport:
    """Run every acceptance check on one recording.

    Args:
        audio: The decoded recording (normally the output of
            :func:`aicls.asr.audio_io.read_wav_pcm16`, i.e. mono 16 kHz).
        thresholds: Bounds to apply; :class:`QcThresholds` defaults when omitted.

    Returns:
        A :class:`QcReport` whose ``reasons_tr`` holds one Turkish sentence per failed check and
        is empty when the recording is accepted.
    """
    limits = thresholds or QcThresholds()
    duration = audio.duration_s
    checks = [
        QcCheck("duration_min", duration, limits.min_duration_s, duration >= limits.min_duration_s),
        QcCheck("duration_max", duration, limits.max_duration_s, duration <= limits.max_duration_s),
    ]
    clipping = clipping_ratio(audio)
    checks.append(
        QcCheck(
            "clipping_ratio",
            clipping,
            limits.max_clipping_ratio,
            clipping <= limits.max_clipping_ratio,
        )
    )
    snr = estimate_snr_db(audio)
    checks.append(QcCheck("snr_db", snr, limits.min_snr_db, snr >= limits.min_snr_db))
    silence = silence_ratio(audio)
    checks.append(
        QcCheck(
            "silence_ratio", silence, limits.max_silence_ratio, silence <= limits.max_silence_ratio
        )
    )
    offset = dc_offset(audio)
    checks.append(
        QcCheck("dc_offset", offset, limits.max_dc_offset, offset <= limits.max_dc_offset)
    )

    reasons = [_reason_tr(check) for check in checks if not check.passed]
    return QcReport(passed=not reasons, checks=checks, reasons_tr=reasons)


__all__ = [
    "MAX_SNR_DB",
    "QC_REASONS_TR",
    "QcCheck",
    "QcReport",
    "QcResult",
    "QcThresholds",
    "clipping_ratio",
    "dc_offset",
    "estimate_snr_db",
    "run_qc",
    "silence_ratio",
]
