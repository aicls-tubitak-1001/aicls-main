"""SpecAugment, pitch shift, noise injection, speed perturbation and the TTS hook (İP-3)."""

from __future__ import annotations

import numpy as np
import pytest

from aicls.asr.augment import (
    MAX_SEMITONES,
    add_noise,
    measured_snr_db,
    pitch_shift,
    spec_augment,
    speed_perturb,
)
from aicls.asr.errors import AsrInputError
from aicls.asr.features import log_mel
from aicls.asr.interfaces import Audio, TtsAugmenter
from aicls.asr.synth import make_noise, make_speech_like, make_tone


def dominant_frequency(audio: Audio) -> float:
    """Frequency of the strongest spectral peak, in Hz."""
    spectrum = np.abs(np.fft.rfft(audio.samples.astype(np.float64)))
    freqs = np.fft.rfftfreq(audio.n_samples, 1.0 / audio.sr)
    return float(freqs[int(np.argmax(spectrum))])


# ---------------------------------------------------------------------------------------------
# SpecAugment
# ---------------------------------------------------------------------------------------------
def test_spec_augment_masks_time_and_freq() -> None:
    """Both mask families are applied, leaving whole constant rows and columns behind."""
    mel = log_mel(make_speech_like(duration_s=2.0))
    rng = np.random.default_rng(1234)

    masked = spec_augment(mel, time_masks=2, freq_masks=2, rng=rng)

    assert masked.shape == mel.shape
    assert not np.array_equal(masked, mel)
    constant_rows = np.isclose(masked, masked[:, :1]).all(axis=1)
    constant_cols = np.isclose(masked, masked[:1, :]).all(axis=0)
    assert constant_rows.any(), "no frequency mask was applied"
    assert constant_cols.any(), "no time mask was applied"
    # The original is untouched.
    assert np.array_equal(mel, log_mel(make_speech_like(duration_s=2.0)))


def test_spec_augment_is_deterministic_for_a_given_seed() -> None:
    mel = log_mel(make_speech_like(duration_s=1.0))
    first = spec_augment(mel, rng=np.random.default_rng(7))
    second = spec_augment(mel, rng=np.random.default_rng(7))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, spec_augment(mel, rng=np.random.default_rng(8)))


def test_spec_augment_without_masks_is_a_copy() -> None:
    mel = log_mel(make_tone(duration_s=0.5))
    out = spec_augment(mel, time_masks=0, freq_masks=0)
    assert np.array_equal(out, mel)
    assert out is not mel


def test_spec_augment_fill_value_is_configurable() -> None:
    mel = log_mel(make_speech_like(duration_s=1.0))
    out = spec_augment(mel, fill=-3.0, rng=np.random.default_rng(1))
    assert float(out.min()) == pytest.approx(-3.0)


def test_spec_augment_survives_a_tiny_spectrogram() -> None:
    tiny = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = spec_augment(tiny, time_masks=1, freq_masks=1, rng=np.random.default_rng(3))
    assert out.shape == tiny.shape


@pytest.mark.parametrize(
    ("mel", "kwargs"),
    [
        (np.zeros(4, dtype=np.float32), {}),
        (np.zeros((2, 2), dtype=np.float32), {"time_masks": -1}),
    ],
)
def test_spec_augment_rejects_bad_arguments(mel: np.ndarray, kwargs: dict[str, int]) -> None:
    with pytest.raises(AsrInputError):
        spec_augment(mel, **kwargs)


# ---------------------------------------------------------------------------------------------
# Pitch shift
# ---------------------------------------------------------------------------------------------
def test_augment_pitch_shift_preserves_duration() -> None:
    """The contractual property: the augmented clip lines up with its transcript."""
    audio = make_speech_like(duration_s=2.0)

    for semitones in (-2.0, -1.0, 1.0, 2.0):
        shifted = pitch_shift(audio, semitones)
        assert shifted.n_samples == audio.n_samples
        assert shifted.sr == audio.sr
        assert shifted.duration_s == pytest.approx(audio.duration_s)


def test_augment_pitch_shift_actually_moves_the_pitch() -> None:
    tone = make_tone(freq=200.0, duration_s=1.0, amplitude=0.5)
    up = dominant_frequency(pitch_shift(tone, 4.0))
    down = dominant_frequency(pitch_shift(tone, -4.0))
    assert up > 220.0
    assert down < 180.0


def test_pitch_shift_of_zero_semitones_is_the_identity() -> None:
    audio = make_speech_like(duration_s=1.0)
    assert pitch_shift(audio, 0.0) is audio
    assert pitch_shift(audio, 1.0, np.random.default_rng(1)).n_samples == audio.n_samples


def test_pitch_shift_handles_clips_shorter_than_one_frame() -> None:
    short = Audio(make_tone(duration_s=0.02, amplitude=0.4).samples, 16000)
    assert pitch_shift(short, 2.0).n_samples == short.n_samples
    assert pitch_shift(Audio(np.zeros(0, dtype=np.float32), 16000), 2.0).n_samples == 0


def test_pitch_shift_refuses_an_absurd_interval() -> None:
    with pytest.raises(AsrInputError) as excinfo:
        pitch_shift(make_tone(duration_s=0.5), MAX_SEMITONES + 1.0)
    assert excinfo.value.details["reason"] == "pitch_shift_out_of_range"


# ---------------------------------------------------------------------------------------------
# Noise injection
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("snr_db", [10.0, 15.0, 20.0, 25.0])
def test_add_noise_hits_the_requested_snr(snr_db: float) -> None:
    """The proposal's 10-25 dB range, realised exactly."""
    clean = make_speech_like(duration_s=1.5)
    noisy = add_noise(clean, snr_db, rng=np.random.default_rng(1234))

    assert noisy.n_samples == clean.n_samples
    assert measured_snr_db(clean, noisy) == pytest.approx(snr_db, abs=0.1)


def test_add_noise_uses_the_noise_bank_when_given() -> None:
    clean = make_speech_like(duration_s=1.0)
    bank = [make_noise(duration_s=0.4, seed=1), make_noise(duration_s=3.0, seed=2)]
    noisy = add_noise(clean, 12.0, bank, np.random.default_rng(5))
    assert measured_snr_db(clean, noisy) == pytest.approx(12.0, abs=0.1)
    # Raw arrays are accepted as well as Audio objects.
    raw_bank = [make_noise(duration_s=1.0, seed=3).samples]
    assert add_noise(clean, 12.0, raw_bank).n_samples == clean.n_samples


def test_add_noise_can_clip_to_the_pcm_range() -> None:
    clean = make_speech_like(duration_s=1.0, amplitude=0.9)
    noisy = add_noise(clean, 0.0, rng=np.random.default_rng(1), clip=True)
    assert float(np.abs(noisy.samples).max()) <= 1.0


def test_add_noise_is_a_no_op_without_signal_or_noise() -> None:
    silence = Audio(np.zeros(1600, dtype=np.float32), 16000)
    assert add_noise(silence, 15.0) is silence
    assert add_noise(Audio(np.zeros(0, dtype=np.float32), 16000), 15.0).n_samples == 0
    clean = make_speech_like(duration_s=0.5)
    silent_bank = [Audio(np.zeros(1600, dtype=np.float32), 16000)]
    assert add_noise(clean, 15.0, silent_bank) is clean
    empty_bank = [Audio(np.zeros(0, dtype=np.float32), 16000)]
    assert add_noise(clean, 15.0, empty_bank).n_samples == clean.n_samples


def test_measured_snr_db_edge_cases() -> None:
    clean = make_speech_like(duration_s=0.5)
    assert measured_snr_db(clean, clean) == float("inf")
    silence = Audio(np.zeros(clean.n_samples, dtype=np.float32), clean.sr)
    assert measured_snr_db(silence, silence) == 0.0
    with pytest.raises(AsrInputError):
        measured_snr_db(clean, Audio(clean.samples[:100], clean.sr))


# ---------------------------------------------------------------------------------------------
# Speed perturbation
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("factor", [0.9, 1.1])
def test_speed_perturb_changes_the_duration(factor: float) -> None:
    audio = make_speech_like(duration_s=2.0)
    out = speed_perturb(audio, factor)
    assert out.sr == audio.sr
    assert out.duration_s == pytest.approx(audio.duration_s / factor, rel=0.01)


def test_speed_perturb_of_one_is_the_identity() -> None:
    audio = make_speech_like(duration_s=0.5)
    assert speed_perturb(audio, 1.0) is audio
    assert speed_perturb(Audio(np.zeros(0, dtype=np.float32), 16000), 1.2).n_samples == 0


@pytest.mark.parametrize("factor", [0.0, 0.2, 5.0])
def test_speed_perturb_refuses_an_extreme_factor(factor: float) -> None:
    with pytest.raises(AsrInputError) as excinfo:
        speed_perturb(make_speech_like(duration_s=0.5), factor)
    assert excinfo.value.details["reason"] == "speed_factor_out_of_range"


# ---------------------------------------------------------------------------------------------
# TTS hook (protocol only — the engine is a separate, gated decision)
# ---------------------------------------------------------------------------------------------
def test_tts_augmenter_protocol_accepts_a_conforming_object() -> None:
    class FakeTts:
        def synthesize(self, text: str, voice: str) -> Audio:
            return make_tone(freq=120.0 + len(text) + len(voice), duration_s=0.5)

    augmenter: TtsAugmenter = FakeTts()
    assert isinstance(augmenter, TtsAugmenter)
    assert augmenter.synthesize("türevini al", "tr-female-1").duration_s == pytest.approx(0.5)
    assert not isinstance(object(), TtsAugmenter)
