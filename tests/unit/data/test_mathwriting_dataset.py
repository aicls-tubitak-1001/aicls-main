"""MathWritingDataset: online/offline items, the synthetic mixing sampler and torch interop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aicls.core.errors import InputValidationError, NotFoundError
from aicls.data import MathWritingDataset, collate_image, collate_ink, prepare_shards
from aicls.data.mathwriting import SyntheticInkDataset
from aicls.ocr import STROKE_FEATURE_DIM, PreprocessConfig, default_tokenizer


@pytest.fixture
def tokenizer():
    """Curriculum tokenizer (process-wide cached)."""
    return default_tokenizer()


def test_mathwriting_dataset_online_yields_features_and_token_ids(
    shards_root: Path, tokenizer
) -> None:
    ds = MathWritingDataset(shards_root, "train", mode="online", tokenizer=tokenizer)

    assert len(ds) == 12
    sample = ds[0]
    assert sample.features.ndim == 2
    assert sample.features.shape[1] == STROKE_FEATURE_DIM
    assert sample.features.dtype == np.float32
    assert sample.label_ids.dtype == np.int64
    assert sample.label_ids.size >= 2  # <s> ... </s>
    assert tokenizer.decode(sample.label_ids.tolist()) == sample.label
    assert sample.is_synthetic is False
    assert sample.sample_id
    features, ids = sample  # tuple-compatible, as documented in the module table
    assert features is sample.features and ids is sample.label_ids


def test_mathwriting_dataset_offline_yields_a_preprocessed_image(
    shards_root: Path, tokenizer
) -> None:
    cfg = PreprocessConfig(steps=("normalize01", "resize_pad"), size=(32, 96))
    ds = MathWritingDataset(
        shards_root, "train", mode="offline", tokenizer=tokenizer, preprocess=cfg
    )

    sample = ds[3]
    assert sample.features.shape == (1, 32, 96)
    assert sample.features.dtype == np.float32
    assert float(sample.features.min()) >= 0.0 and float(sample.features.max()) <= 1.0


def test_mathwriting_dataset_uses_rendered_images_from_the_shard(
    inkml_root: Path, tmp_path: Path, tokenizer
) -> None:
    out = tmp_path / "shards"
    prepare_shards(inkml_root, out, split="train", shard_size=8, render=True, image_size=(32, 96))
    ds = MathWritingDataset(
        out,
        "train",
        mode="offline",
        tokenizer=tokenizer,
        preprocess=PreprocessConfig(steps=("normalize01",), size=(32, 96)),
    )
    assert ds.uses_rendered_images
    assert ds[0].features.shape == (1, 32, 96)


@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 1.0])
def test_synthetic_ratio_sampler_mixes_as_configured(
    mixed_shards_root: Path, tokenizer, ratio: float
) -> None:
    ds = MathWritingDataset(
        mixed_shards_root,
        "train",
        mode="online",
        tokenizer=tokenizer,
        synthetic_ratio=ratio,
    )
    flags = [ds[i].is_synthetic for i in range(len(ds))]
    observed = sum(flags) / len(flags)

    assert observed == pytest.approx(ratio, abs=1.0 / len(ds))
    assert len(ds) >= 12 if ratio < 1.0 else len(ds) == 8
    # the mixture is deterministic: the same index gives the same provenance twice
    assert [ds[i].is_synthetic for i in range(len(ds))] == flags
    if 0.0 < ratio < 1.0:
        real_ids = {ds[i].sample_id for i in range(len(ds)) if not ds[i].is_synthetic}
        assert len(real_ids) == 12  # every real sample is seen exactly once per epoch


def test_mathwriting_dataset_rejects_invalid_arguments(shards_root: Path, tokenizer) -> None:
    with pytest.raises(InputValidationError):
        MathWritingDataset(shards_root, "train", mode="sideways", tokenizer=tokenizer)  # type: ignore[arg-type]
    with pytest.raises(InputValidationError):
        MathWritingDataset(
            shards_root, "train", mode="online", tokenizer=tokenizer, synthetic_ratio=1.5
        )
    with pytest.raises(NotFoundError):
        MathWritingDataset(shards_root, "valid", mode="online", tokenizer=tokenizer)
    with pytest.raises(NotFoundError):
        # synthetic_ratio > 0 needs a prepared `synthetic` split next to the real one
        MathWritingDataset(
            shards_root, "train", mode="online", tokenizer=tokenizer, synthetic_ratio=0.5
        )
    ds = MathWritingDataset(shards_root, "train", mode="online", tokenizer=tokenizer)
    with pytest.raises(IndexError):
        _ = ds[len(ds)]


def test_mathwriting_dataset_truncates_labels_to_max_label_len(
    shards_root: Path, tokenizer
) -> None:
    ds = MathWritingDataset(
        shards_root, "train", mode="online", tokenizer=tokenizer, max_label_len=4
    )
    assert all(ds[i].label_ids.size <= 4 for i in range(len(ds)))


def test_synthetic_ink_dataset_is_reproducible(tokenizer) -> None:
    a = SyntheticInkDataset(6, seed=1234, tokenizer=tokenizer)
    b = SyntheticInkDataset(6, seed=1234, tokenizer=tokenizer)
    assert len(a) == 6
    np.testing.assert_array_equal(a[2].features, b[2].features)
    assert a[2].label == b[2].label
    assert a[2].is_synthetic is True

    offline = SyntheticInkDataset(
        3,
        seed=1234,
        tokenizer=tokenizer,
        mode="offline",
        preprocess=PreprocessConfig(size=(32, 96)),
    )
    assert offline[0].features.shape[0] == 1


def test_dataset_batches_collate_into_padded_arrays(shards_root: Path, tokenizer) -> None:
    ds = MathWritingDataset(shards_root, "train", mode="online", tokenizer=tokenizer)
    batch = collate_ink([ds[0], ds[1], ds[2]])
    assert batch["features"].shape[0] == 3
    assert batch["feature_lengths"].tolist() == [ds[i].features.shape[0] for i in range(3)]

    offline_ds = MathWritingDataset(
        shards_root,
        "train",
        mode="offline",
        tokenizer=tokenizer,
        preprocess=PreprocessConfig(steps=("normalize01", "resize_pad"), size=(32, 96)),
    )
    image_batch = collate_image([offline_ds[0], offline_ds[1]])
    assert image_batch["images"].shape == (2, 1, 32, 96)


@pytest.mark.slow
def test_mathwriting_dataset_works_with_a_torch_dataloader(shards_root: Path, tokenizer) -> None:
    torch = pytest.importorskip("torch")
    from functools import partial

    ds = MathWritingDataset(shards_root, "train", mode="online", tokenizer=tokenizer)
    loader = torch.utils.data.DataLoader(
        ds.to_torch(), batch_size=4, collate_fn=partial(collate_ink, to_torch=True)
    )
    batch = next(iter(loader))
    assert batch["features"].shape[0] == 4
    assert batch["labels"].dtype == torch.int64
