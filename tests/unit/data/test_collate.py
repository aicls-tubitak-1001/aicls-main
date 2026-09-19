"""Collate functions: padding, lengths, masks and the optional torch conversion."""

from __future__ import annotations

import numpy as np
import pytest

from aicls.core.errors import InputValidationError
from aicls.data import collate_audio, collate_image, collate_ink
from aicls.data.collate import PAD_ID, pad_sequences
from aicls.data.mathwriting import InkSample


def make_sample(n_frames: int, n_ids: int, *, dim: int = 3) -> InkSample:
    """A tiny online sample with recognisable values."""
    return InkSample(
        features=np.full((n_frames, dim), float(n_frames), dtype=np.float32),
        label_ids=np.arange(4, 4 + n_ids, dtype=np.int64),
        label="x",
        sample_id=f"s{n_frames}",
        writer_id="w1",
        is_synthetic=False,
    )


def test_pad_sequences_pads_to_the_longest_and_reports_lengths() -> None:
    padded, lengths = pad_sequences(
        [np.ones((2, 3), dtype=np.float32), np.ones((5, 3), dtype=np.float32)]
    )
    assert padded.shape == (2, 5, 3)
    assert lengths.tolist() == [2, 5]
    assert padded.dtype == np.float32
    assert float(padded[0, 2:].sum()) == 0.0

    with pytest.raises(InputValidationError):
        pad_sequences([])


def test_collate_ink_pads_features_and_labels() -> None:
    batch = collate_ink([make_sample(2, 3), make_sample(5, 6)])

    assert batch["features"].shape == (2, 5, 3)
    assert batch["feature_lengths"].tolist() == [2, 5]
    assert batch["labels"].shape == (2, 6)
    assert batch["label_lengths"].tolist() == [3, 6]
    assert batch["labels"][0, 3:].tolist() == [PAD_ID] * 3
    assert batch["feature_mask"].dtype == np.bool_
    assert batch["feature_mask"][0].tolist() == [True, True, False, False, False]
    assert batch["labels_text"] == ["x", "x"]
    assert batch["sample_ids"] == ["s2", "s5"]
    assert batch["is_synthetic"].tolist() == [False, False]


def test_collate_image_stacks_and_pads_to_a_common_canvas() -> None:
    first = make_sample(1, 2)
    second = make_sample(1, 4)
    first = InkSample(
        features=np.zeros((1, 8, 12), dtype=np.float32),
        label_ids=first.label_ids,
        label=first.label,
        sample_id=first.sample_id,
        writer_id=first.writer_id,
        is_synthetic=first.is_synthetic,
    )
    second = InkSample(
        features=np.ones((1, 8, 12), dtype=np.float32),
        label_ids=second.label_ids,
        label=second.label,
        sample_id=second.sample_id,
        writer_id=second.writer_id,
        is_synthetic=True,
    )
    batch = collate_image([first, second])

    assert batch["images"].shape == (2, 1, 8, 12)
    assert batch["labels"].shape == (2, 4)
    assert batch["is_synthetic"].tolist() == [False, True]

    with pytest.raises(InputValidationError):
        collate_image(
            [
                first,
                InkSample(
                    features=np.zeros((1, 4, 4), dtype=np.float32),
                    label_ids=second.label_ids,
                    label="x",
                    sample_id="s",
                    writer_id="w",
                    is_synthetic=False,
                ),
            ]
        )


def test_collate_functions_reject_empty_batches() -> None:
    for fn in (collate_ink, collate_image, collate_audio):
        with pytest.raises(InputValidationError):
            fn([])


def test_collate_accepts_plain_tuples() -> None:
    batch = collate_ink(
        [
            (np.ones((2, 3), dtype=np.float32), np.array([4, 5], dtype=np.int64)),
            (np.ones((3, 3), dtype=np.float32), np.array([6], dtype=np.int64)),
        ]
    )
    assert batch["features"].shape == (2, 3, 3)
    assert batch["label_lengths"].tolist() == [2, 1]


@pytest.mark.slow
def test_collate_can_return_torch_tensors() -> None:
    torch = pytest.importorskip("torch")
    batch = collate_ink([make_sample(2, 3), make_sample(5, 6)], to_torch=True)
    assert isinstance(batch["features"], torch.Tensor)
    assert batch["features"].dtype == torch.float32
    assert batch["labels"].dtype == torch.int64
    assert batch["feature_lengths"].dtype == torch.int64
    assert batch["labels_text"] == ["x", "x"]
