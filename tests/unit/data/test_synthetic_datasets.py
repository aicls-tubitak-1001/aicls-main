"""The synthetic datasets the CI smoke configs train on (``data.dataset: synthetic_*``)."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pytest

from aicls.asr.features import log_mel
from aicls.core.errors import InputValidationError
from aicls.data import SyntheticAudioDataset, SyntheticInkDataset, collate_audio, collate_ink
from aicls.data.synthetic import synth_image_dataset
from aicls.ocr import default_tokenizer


def test_synthetic_audio_dataset_is_reproducible_and_collates() -> None:
    a = SyntheticAudioDataset(4, seed=1234, feature_fn=partial(log_mel, n_mels=16), seconds=0.5)
    b = SyntheticAudioDataset(4, seed=1234, feature_fn=partial(log_mel, n_mels=16), seconds=0.5)

    assert len(a) == 4
    np.testing.assert_array_equal(a[0].features, b[0].features)
    assert a[0].features.shape[0] == 16
    assert a[0].text
    assert a[0].sample_rate == 16000
    assert a[-1].sample_id == "synthetic-000003"

    batch = collate_audio(list(a))
    assert batch["features"].shape[0] == 4
    assert len(batch["texts"]) == 4

    raw = SyntheticAudioDataset(2, seed=1, seconds=0.25)
    assert raw[0].features.ndim == 1  # no feature_fn: the waveform itself

    with pytest.raises(InputValidationError):
        SyntheticAudioDataset(0)
    with pytest.raises(IndexError):
        _ = a[4]


def test_synthetic_ink_dataset_iterates_and_collates() -> None:
    ds = SyntheticInkDataset(5, seed=1234, tokenizer=default_tokenizer())
    items = list(ds)

    assert len(items) == 5
    assert all(item.is_synthetic for item in items)
    assert ds[-1].sample_id == "synthetic-000004"
    batch = collate_ink(items)
    assert batch["features"].shape[0] == 5
    assert batch["labels"].shape[0] == 5

    with pytest.raises(InputValidationError):
        SyntheticInkDataset(0)
    with pytest.raises(IndexError):
        _ = ds[5]


def test_synth_image_dataset_writes_images_and_labels(tmp_path: Path) -> None:
    out = synth_image_dataset(tmp_path / "images", 3, 1234, size=(32, 96))
    files = sorted(out.glob("*.npy"))

    assert len(files) == 3
    assert np.load(files[0]).shape == (32, 96)
    lines = (out / "labels.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert '"label"' in lines[0]

    with pytest.raises(InputValidationError):
        synth_image_dataset(tmp_path / "images", 0, 1234)
