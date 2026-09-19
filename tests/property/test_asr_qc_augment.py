"""Property-based checks on quality control and augmentation.

Quality measures must stay inside their mathematical range for any waveform, augmentation must
respect the shape contracts the training pipeline relies on, and the stub transcriber must be a
pure function of its input.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
import numpy as np
import pytest

from aicls.asr.augment import add_noise, measured_snr_db, pitch_shift, spec_augment, speed_perturb
from aicls.asr.features import log_mel
from aicls.asr.interfaces import Audio
from aicls.asr.qc import MAX_SNR_DB, QcThresholds, clipping_ratio, run_qc, silence_ratio
from aicls.asr.stub import StubTranscriber
from aicls.asr.synth import make_speech_like

pytestmark = pytest.mark.property

SLOW = settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.too_slow])


@st.composite
def waveforms(draw: st.DrawFn, *, min_size: int = 320, max_size: int = 4000) -> Audio:
    """A 16 kHz mono waveform with samples inside [-1, 1].

    The samples are drawn with :func:`hypothesis.extra.numpy.arrays` rather than an
    ``st.lists`` of 4000 floats: the list form draws one float's worth of entropy per sample and
    trips Hypothesis' ``data_too_large`` health check on unlucky seeds (observed 2026-09-08 —
    "9 inputs were generated successfully, 20 exceeded the maximum allowed entropy"), which
    silently reduced this property to a handful of examples before failing the run outright. The
    array strategy fills most of the buffer from one drawn value and varies a few positions, so
    it covers the same domain — any length in range, any sample in [-1, 1] — at a fraction of
    the entropy.
    """
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    samples = draw(
        hnp.arrays(
            dtype=np.float32,
            shape=size,
            elements=st.floats(
                min_value=-1.0,
                max_value=1.0,
                width=32,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )
    return Audio(samples, 16000)


@SLOW
@given(audio=waveforms())
def test_qc_measures_stay_inside_their_range(audio: Audio) -> None:
    report = run_qc(audio)
    assert 0.0 <= report.value("clipping_ratio") <= 1.0
    assert 0.0 <= report.value("silence_ratio") <= 1.0
    assert 0.0 <= report.value("snr_db") <= MAX_SNR_DB
    assert report.value("dc_offset") >= 0.0
    assert report.passed == (report.reasons_tr == [])
    assert len(report.checks) == 6


@SLOW
@given(audio=waveforms())
def test_qc_report_reasons_match_the_failed_checks(audio: Audio) -> None:
    report = run_qc(audio)
    assert len(report.reasons_tr) == len(report.failed_checks)
    assert all(reason.strip() for reason in report.reasons_tr)


@SLOW
@given(level=st.floats(min_value=0.0, max_value=1.0))
def test_clipping_ratio_is_monotone_in_the_signal_level(level: float) -> None:
    audio = Audio(np.full(1600, level, dtype=np.float32), 16000)
    assert clipping_ratio(audio) == (1.0 if level >= 0.99 else 0.0)
    assert silence_ratio(audio) == (1.0 if level < 1e-3 else 0.0)


@SLOW
@given(
    semitones=st.floats(min_value=-4.0, max_value=4.0),
    duration_s=st.floats(min_value=0.3, max_value=1.5),
)
def test_pitch_shift_always_preserves_the_sample_count(semitones: float, duration_s: float) -> None:
    audio = make_speech_like(duration_s=duration_s)
    shifted = pitch_shift(audio, semitones)
    assert shifted.n_samples == audio.n_samples
    assert shifted.sr == audio.sr
    assert np.isfinite(shifted.samples).all()


@SLOW
@given(factor=st.floats(min_value=0.8, max_value=1.25))
def test_speed_perturb_length_follows_the_factor(factor: float) -> None:
    audio = make_speech_like(duration_s=1.0)
    out = speed_perturb(audio, factor)
    assert out.duration_s == pytest.approx(audio.duration_s / factor, rel=0.02)


@SLOW
@given(snr_db=st.floats(min_value=0.0, max_value=30.0))
def test_add_noise_realises_the_requested_snr(snr_db: float) -> None:
    clean = make_speech_like(duration_s=0.5)
    noisy = add_noise(clean, snr_db, rng=np.random.default_rng(1234))
    assert measured_snr_db(clean, noisy) == pytest.approx(snr_db, abs=0.2)


@SLOW
@given(
    time_masks=st.integers(min_value=0, max_value=3),
    freq_masks=st.integers(min_value=0, max_value=3),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_spec_augment_preserves_shape_and_value_range(
    time_masks: int, freq_masks: int, seed: int
) -> None:
    mel = log_mel(make_speech_like(duration_s=1.0))
    out = spec_augment(
        mel, time_masks=time_masks, freq_masks=freq_masks, rng=np.random.default_rng(seed)
    )
    assert out.shape == mel.shape
    assert out.dtype == np.float32
    assert float(out.min()) >= float(mel.min())
    assert float(out.max()) <= float(mel.max())


@SLOW
@given(audio=waveforms(min_size=320, max_size=2000))
def test_stub_transcriber_is_a_pure_function_of_the_audio(audio: Audio) -> None:
    first = StubTranscriber().transcribe(audio)
    second = StubTranscriber().transcribe(audio)
    assert first.text == second.text
    assert first.confidence == second.confidence
    assert first.text in StubTranscriber().sentences


@SLOW
@given(
    min_duration_s=st.floats(min_value=0.1, max_value=1.0),
    max_duration_s=st.floats(min_value=2.0, max_value=30.0),
)
def test_duration_checks_bracket_the_configured_window(
    min_duration_s: float, max_duration_s: float
) -> None:
    thresholds = QcThresholds(min_duration_s=min_duration_s, max_duration_s=max_duration_s)
    audio = make_speech_like(duration_s=1.5)
    report = run_qc(audio, thresholds)
    inside = min_duration_s <= audio.duration_s <= max_duration_s
    failed = {check.name for check in report.failed_checks}
    assert inside == ({"duration_min", "duration_max"}.isdisjoint(failed))
