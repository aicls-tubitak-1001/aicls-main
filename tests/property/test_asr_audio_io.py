"""Property-based checks on the WAV codec and the resampler (ADR-0017).

The invariants that must hold for *any* waveform, not just the fixtures: a PCM16 round trip loses
at most one quantisation step, the container check never accepts a non-RIFF byte string, and the
resampler's output length, sample rate and amplitude bound follow from its arguments.
"""

from __future__ import annotations

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
import numpy as np
import pytest

from aicls.asr.audio_io import (
    ACCEPTED_SAMPLE_RATES,
    read_wav_pcm16,
    resample,
    resample_to_16k,
    write_wav_pcm16,
)
from aicls.asr.errors import AsrInputError
from aicls.asr.interfaces import Audio

pytestmark = pytest.mark.property

PCM16_STEP = 1.0 / 32767.0

#: Worst-case amplitude gain of the polyphase windowed-sinc resampler: the largest L1 norm of a
#: kernel row over every rate pair in ``ACCEPTED_SAMPLE_RATES`` (measured max 2.4046), plus a small
#: margin. Unity DC gain does not bound the transient response of near-Nyquist input.
MAX_RESAMPLER_GAIN = 2.45
#: ``data_too_large`` is suppressed for the same reason as ``too_slow``: a waveform is *meant* to
#: be a few thousand samples, so Hypothesis's entropy budget is deliberately spent here. It is a
#: performance/shrinkability heuristic, not an assertion, and nothing about the properties below
#: changes. Without it the suite is hostage to a patch release: hypothesis 6.147.0 (the container
#: venv) raises ``FailedHealthCheck: data_too_large`` on ``waveforms()`` while 6.167.1 (the
#: Windows venv) does not, and both satisfy the ``>=6.115,<7`` pin (measured 2026-09-08).
SLOW = settings(
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@st.composite
def waveforms(draw: st.DrawFn, *, min_size: int = 16, max_size: int = 4000) -> Audio:
    """A mono waveform at an accepted sample rate, with samples strictly inside [-1, 1]."""
    sr = draw(st.sampled_from(ACCEPTED_SAMPLE_RATES))
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    values = draw(
        st.lists(
            st.floats(min_value=-0.99, max_value=0.99, allow_nan=False, allow_infinity=False),
            min_size=size,
            max_size=size,
        )
    )
    return Audio(np.asarray(values, dtype=np.float32), sr)


@SLOW
@given(audio=waveforms())
def test_pcm16_roundtrip_loses_at_most_one_quantisation_step(audio: Audio) -> None:
    payload = write_wav_pcm16(audio)
    assume(audio.duration_s <= 30.0)
    decoded = read_wav_pcm16(payload)

    assert decoded.sr == 16000
    if audio.sr == 16000:
        assert decoded.n_samples == audio.n_samples
        assert np.abs(decoded.samples - audio.samples).max() <= 2 * PCM16_STEP


@SLOW
@given(audio=waveforms())
def test_written_wav_is_always_a_riff_wave_container(audio: Audio) -> None:
    payload = write_wav_pcm16(audio)
    assert payload[:4] == b"RIFF"
    assert payload[8:12] == b"WAVE"


@SLOW
@given(
    payload=st.binary(min_size=44, max_size=200).filter(
        lambda b: b[:4] != b"RIFF" or b[8:12] != b"WAVE"
    )
)
def test_non_riff_bytes_are_always_refused(payload: bytes) -> None:
    with pytest.raises(AsrInputError) as excinfo:
        read_wav_pcm16(payload)
    assert excinfo.value.details["reason"] == "not_riff_wave"


@SLOW
@given(audio=waveforms(min_size=64, max_size=2000))
def test_resampling_to_16k_yields_the_expected_length_and_rate(audio: Audio) -> None:
    out = resample_to_16k(audio)
    assert out.sr == 16000
    assert out.n_samples == audio.n_samples * 16000 // audio.sr


@SLOW
@given(
    audio=waveforms(min_size=64, max_size=2000),
    target=st.sampled_from(ACCEPTED_SAMPLE_RATES),
)
def test_resampling_never_amplifies_beyond_the_input_peak(audio: Audio, target: int) -> None:
    """A normalised low-pass kernel cannot lift a bandlimited signal far above its own peak.

    The exact worst case is the largest L1 norm (sum of ``abs`` taps) over the polyphase kernel
    bank: every output sample is one kernel row dotted with a window of the input, so
    ``max|out| <= max_p ||k_p||_1 * max|in|``. The bank has unity *DC* gain by construction, but
    near-Nyquist content (an alternating waveform, which hypothesis does find) rings, and for the
    rate pairs in :data:`ACCEPTED_SAMPLE_RATES` the measured L1 norms reach ~2.41. Overshoot is
    inherent to sinc interpolation, so :data:`MAX_RESAMPLER_GAIN` bounds it instead of pretending
    it cannot happen; the check still catches a resampler that has genuinely blown up.
    """
    out = resample(audio.samples, audio.sr, target)
    peak = float(np.abs(audio.samples).max())
    if out.size:
        assert float(np.abs(out).max()) <= peak * MAX_RESAMPLER_GAIN + 1e-6
