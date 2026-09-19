"""Quality control for the "Model Gelişimine Katkıda Bulun" tab (İP-3, docs/mimari/api.md §3.8)."""

from __future__ import annotations

import numpy as np
import pytest

from aicls.asr.interfaces import Audio
from aicls.asr.qc import (
    MAX_SNR_DB,
    QC_REASONS_TR,
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
from aicls.asr.synth import make_speech_like, make_tone

CHECK_NAMES = (
    "duration_min",
    "duration_max",
    "clipping_ratio",
    "snr_db",
    "silence_ratio",
    "dc_offset",
)


def clipped_audio(*, duration_s: float = 1.5, amplitude: float = 1.4) -> Audio:
    """A recording driven past full scale, then clipped by the converter."""
    hot = make_tone(freq=300.0, duration_s=duration_s, amplitude=amplitude)
    return Audio(np.clip(hot.samples, -1.0, 1.0), hot.sr)


def failed_names(report: QcReport) -> set[str]:
    return {check.name for check in report.failed_checks}


def test_qc_accepts_a_clean_speech_like_recording() -> None:
    report = run_qc(make_speech_like(duration_s=2.0))
    assert report.passed is True
    assert report.reasons_tr == []
    assert [check.name for check in report.checks] == list(CHECK_NAMES)


def test_audio_qc_flags_clipping_above_threshold() -> None:
    """Input gain too high: a large share of samples sits at digital full scale."""
    report = run_qc(clipped_audio())

    assert report.passed is False
    assert "clipping_ratio" in failed_names(report)
    assert report.value("clipping_ratio") > QcThresholds().max_clipping_ratio
    assert any("kırpılma" in reason for reason in report.reasons_tr)


def test_audio_qc_flags_low_snr() -> None:
    """A noisy room: the loud frames barely rise above the noise floor."""
    noisy = make_tone(freq=300.0, duration_s=1.5, amplitude=0.2, noise_amplitude=0.15)
    report = run_qc(noisy)

    assert report.passed is False
    assert "snr_db" in failed_names(report)
    assert report.value("snr_db") < QcThresholds().min_snr_db
    assert any("gürültü" in reason for reason in report.reasons_tr)


@pytest.mark.parametrize(
    ("duration_s", "expected"),
    [(0.3, "duration_min"), (25.0, "duration_max")],
)
def test_audio_qc_rejects_duration_out_of_range(duration_s: float, expected: str) -> None:
    report = run_qc(make_speech_like(duration_s=duration_s))

    assert report.passed is False
    assert expected in failed_names(report)
    assert report.value(expected) == pytest.approx(duration_s, abs=1e-3)


def test_audio_qc_flags_silence_ratio() -> None:
    """The participant spoke for a fraction of a second and then let the recorder run."""
    samples = np.zeros(32000, dtype=np.float32)
    speech = make_speech_like(duration_s=0.25)
    samples[: speech.n_samples] = speech.samples

    report = run_qc(Audio(samples, 16000))

    assert report.passed is False
    assert "silence_ratio" in failed_names(report)
    assert report.value("silence_ratio") > QcThresholds().max_silence_ratio
    assert any("sessiz" in reason for reason in report.reasons_tr)


def test_audio_qc_flags_dc_offset() -> None:
    biased = make_speech_like(duration_s=2.0)
    report = run_qc(Audio(biased.samples + 0.05, biased.sr))

    assert report.passed is False
    assert "dc_offset" in failed_names(report)
    assert report.value("dc_offset") == pytest.approx(0.05, abs=1e-3)
    assert any("doğru akım" in reason for reason in report.reasons_tr)


def test_qc_thresholds_can_be_tightened_per_campaign() -> None:
    audio = make_speech_like(duration_s=2.0)
    assert run_qc(audio).passed is True
    strict = QcThresholds(min_duration_s=3.0, min_snr_db=60.0)
    report = run_qc(audio, strict)
    assert report.passed is False
    assert failed_names(report) == {"duration_min", "snr_db"}
    assert len(report.reasons_tr) == 2


def test_qc_report_serialises_for_the_api_and_the_recordings_table() -> None:
    report = run_qc(make_speech_like(duration_s=2.0))
    payload = report.as_dict()

    assert payload["passed"] is True
    assert payload["reasons_tr"] == []
    assert [check["name"] for check in payload["checks"]] == list(CHECK_NAMES)
    assert set(payload["checks"][0]) == {"name", "value", "threshold", "passed"}


def test_qc_report_value_raises_for_an_unknown_check() -> None:
    with pytest.raises(KeyError):
        run_qc(make_speech_like(duration_s=1.0)).value("does_not_exist")


def test_every_check_name_has_a_turkish_reason() -> None:
    assert set(QC_REASONS_TR) == set(CHECK_NAMES)
    for template in QC_REASONS_TR.values():
        assert template.endswith(".") or template.endswith("deneyin.")


def test_qc_result_is_an_alias_of_qc_report() -> None:
    assert QcResult is QcReport


def test_measures_on_empty_audio_are_defined() -> None:
    empty = Audio(np.zeros(0, dtype=np.float32), 16000)
    assert clipping_ratio(empty) == 0.0
    assert dc_offset(empty) == 0.0
    assert estimate_snr_db(empty) == 0.0
    assert silence_ratio(empty) == 1.0
    assert run_qc(empty).passed is False


def test_snr_of_a_signal_over_digital_silence_is_capped() -> None:
    samples = np.zeros(16000, dtype=np.float32)
    samples[4000:8000] = 0.4
    assert estimate_snr_db(Audio(samples, 16000)) == MAX_SNR_DB
    assert estimate_snr_db(Audio(np.zeros(16000, dtype=np.float32), 16000)) == 0.0


def test_qc_check_is_a_plain_serialisable_record() -> None:
    check = QcCheck("snr_db", 21.5, 15.0, True)
    assert check.as_dict() == {
        "name": "snr_db",
        "value": 21.5,
        "threshold": 15.0,
        "passed": True,
    }
