"""Batch collation: padding, lengths and masks — numpy first, torch on request.

The training loop needs padded batches; the evaluation harness and the notebooks want plain
numpy. These functions therefore build numpy batches and convert to torch **only** when
``to_torch=True``, behind a lazy import (ADR-0002 layering). A ``DataLoader`` uses them as
``collate_fn=functools.partial(collate_ink, to_torch=True)``.

Padding value for labels is ``<pad>`` (id 0, :data:`aicls.ocr.tokenizer.PAD_ID`); feature padding
is 0.0 and the accompanying ``*_mask`` marks the valid frames, so a model never has to guess which
part of a row is real.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from aicls.data.errors import invalid_data
from aicls.ocr.tokenizer import PAD_ID

__all__ = [
    "PAD_ID",
    "collate_audio",
    "collate_image",
    "collate_ink",
    "pad_sequences",
]


def pad_sequences(
    arrays: Sequence[np.ndarray],
    *,
    pad_value: float = 0.0,
    dtype: np.dtype[Any] | type | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad a list of ``(T, ...)`` arrays to the longest ``T``.

    Args:
        arrays: Arrays sharing every dimension but the first.
        pad_value: Value written into the padding.
        dtype: Output dtype; defaults to the dtype of the first array.

    Returns:
        ``(batch, lengths)`` where ``batch`` has shape ``(B, T_max, ...)`` and ``lengths`` holds
        the original first-dimension sizes as ``int64``.

    Raises:
        InputValidationError: If the batch is empty or the trailing shapes disagree.
    """
    if not arrays:
        raise invalid_data("empty_batch")
    first = np.asarray(arrays[0])
    trailing = first.shape[1:]
    for index, array in enumerate(arrays):
        if np.asarray(array).shape[1:] != trailing:
            raise invalid_data(
                "inconsistent_shapes",
                index=index,
                expected=list(trailing),
                got=list(np.asarray(array).shape[1:]),
            )
    lengths = np.asarray([np.asarray(a).shape[0] for a in arrays], dtype=np.int64)
    longest = int(lengths.max()) if lengths.size else 0
    out = np.full((len(arrays), longest, *trailing), pad_value, dtype=dtype or first.dtype)
    for index, array in enumerate(arrays):
        values = np.asarray(array)
        out[index, : values.shape[0]] = values
    return out, lengths


def collate_ink(batch: Sequence[Any], *, to_torch: bool = False) -> dict[str, Any]:
    """Collate online ink samples into a padded batch.

    Args:
        batch: :class:`~aicls.data.mathwriting.InkSample` items or ``(features, label_ids)``
            tuples.
        to_torch: Convert the arrays to torch tensors (lazy import).

    Returns:
        ``features (B, T, F)``, ``feature_lengths (B,)``, ``feature_mask (B, T)``,
        ``labels (B, L)``, ``label_lengths (B,)`` plus the ``labels_text``, ``sample_ids``,
        ``writer_ids`` and ``is_synthetic`` metadata.

    Raises:
        InputValidationError: If the batch is empty or the feature shapes disagree.
    """
    items = _items(batch)
    features, feature_lengths = pad_sequences(
        [np.asarray(item.features, dtype=np.float32) for item in items], dtype=np.float32
    )
    labels, label_lengths = pad_sequences(
        [np.asarray(item.label_ids, dtype=np.int64) for item in items],
        pad_value=float(PAD_ID),
        dtype=np.int64,
    )
    result: dict[str, Any] = {
        "features": features,
        "feature_lengths": feature_lengths,
        "feature_mask": _mask(feature_lengths, features.shape[1]),
        "labels": labels,
        "label_lengths": label_lengths,
        "labels_text": [item.label for item in items],
        "sample_ids": [item.sample_id for item in items],
        "writer_ids": [item.writer_id for item in items],
        "is_synthetic": np.asarray([item.is_synthetic for item in items], dtype=np.bool_),
    }
    return _maybe_torch(result, to_torch)


def collate_image(batch: Sequence[Any], *, to_torch: bool = False) -> dict[str, Any]:
    """Collate offline image samples into a stacked batch.

    Args:
        batch: :class:`~aicls.data.mathwriting.InkSample` items whose ``features`` are
            ``(C, H, W)`` rasters, or ``(image, label_ids)`` tuples.
        to_torch: Convert the arrays to torch tensors (lazy import).

    Returns:
        ``images (B, C, H, W)``, ``labels (B, L)``, ``label_lengths (B,)`` plus metadata.

    Raises:
        InputValidationError: If the batch is empty or the images have different shapes
            (the preprocessing pipeline is responsible for a common canvas).
    """
    items = _items(batch)
    images = [np.asarray(item.features, dtype=np.float32) for item in items]
    shape = images[0].shape
    for index, image in enumerate(images):
        if image.shape != shape:
            raise invalid_data(
                "inconsistent_image_shapes",
                index=index,
                expected=list(shape),
                got=list(image.shape),
            )
    labels, label_lengths = pad_sequences(
        [np.asarray(item.label_ids, dtype=np.int64) for item in items],
        pad_value=float(PAD_ID),
        dtype=np.int64,
    )
    result: dict[str, Any] = {
        "images": np.stack(images, axis=0),
        "labels": labels,
        "label_lengths": label_lengths,
        "labels_text": [item.label for item in items],
        "sample_ids": [item.sample_id for item in items],
        "writer_ids": [item.writer_id for item in items],
        "is_synthetic": np.asarray([item.is_synthetic for item in items], dtype=np.bool_),
    }
    return _maybe_torch(result, to_torch)


def collate_audio(batch: Sequence[Any], *, to_torch: bool = False) -> dict[str, Any]:
    """Collate audio samples, padding along the time axis.

    Features are ``(n_mels, T)`` (log-Mel convention) or ``(T,)`` raw waveforms; both are padded
    on their **last** axis, which is time.

    Args:
        batch: :class:`~aicls.data.audio_commands.AudioSample` items or
            ``(features, text, speaker)`` tuples.
        to_torch: Convert the arrays to torch tensors (lazy import).

    Returns:
        ``features (B, n_mels, T)`` (or ``(B, T)``), ``feature_lengths (B,)``,
        ``feature_mask (B, T)``, ``texts``, ``speakers`` and ``sample_ids``.

    Raises:
        InputValidationError: If the batch is empty or the feature ranks/bands disagree.
    """
    items = _audio_items(batch)
    arrays = [np.asarray(item.features, dtype=np.float32) for item in items]
    transposed = [array.T if array.ndim == 2 else array for array in arrays]
    padded, lengths = pad_sequences(transposed, dtype=np.float32)
    features = np.transpose(padded, (0, 2, 1)) if arrays[0].ndim == 2 else padded
    result: dict[str, Any] = {
        "features": np.ascontiguousarray(features),
        "feature_lengths": lengths,
        "feature_mask": _mask(lengths, int(padded.shape[1])),
        "texts": [item.text for item in items],
        "speakers": [item.speaker for item in items],
        "sample_ids": [item.sample_id for item in items],
    }
    return _maybe_torch(result, to_torch)


# ------------------------------------------------------------------------------------ internals
class _Item:
    """Duck-typed view over an ink sample or a plain ``(features, label_ids)`` tuple."""

    __slots__ = ("features", "is_synthetic", "label", "label_ids", "sample_id", "writer_id")

    def __init__(self, source: Any) -> None:
        if hasattr(source, "features") and hasattr(source, "label_ids"):
            self.features = source.features
            self.label_ids = source.label_ids
            self.label = getattr(source, "label", "")
            self.sample_id = getattr(source, "sample_id", "")
            self.writer_id = getattr(source, "writer_id", "")
            self.is_synthetic = bool(getattr(source, "is_synthetic", False))
            return
        try:
            features, label_ids = source[0], source[1]
        except (TypeError, IndexError, KeyError) as exc:
            raise invalid_data("unsupported_batch_item", item_type=type(source).__name__) from exc
        self.features = features
        self.label_ids = label_ids
        self.label = ""
        self.sample_id = ""
        self.writer_id = ""
        self.is_synthetic = False


class _AudioItem:
    """Duck-typed view over an audio sample or a ``(features, text, speaker)`` tuple."""

    __slots__ = ("features", "sample_id", "speaker", "text")

    def __init__(self, source: Any) -> None:
        if hasattr(source, "features") and hasattr(source, "text"):
            self.features = source.features
            self.text = source.text
            self.speaker = getattr(source, "speaker", "")
            self.sample_id = getattr(source, "sample_id", "")
            return
        try:
            self.features, self.text = source[0], source[1]
            self.speaker = source[2] if len(source) > 2 else ""
        except (TypeError, IndexError, KeyError) as exc:
            raise invalid_data("unsupported_batch_item", item_type=type(source).__name__) from exc
        self.sample_id = ""


def _items(batch: Sequence[Any]) -> list[_Item]:
    if not batch:
        raise invalid_data("empty_batch")
    return [_Item(item) for item in batch]


def _audio_items(batch: Sequence[Any]) -> list[_AudioItem]:
    if not batch:
        raise invalid_data("empty_batch")
    return [_AudioItem(item) for item in batch]


def _mask(lengths: np.ndarray, width: int) -> np.ndarray:
    """``(B, width)`` boolean mask, true where the row carries real data."""
    return np.arange(width)[None, :] < np.asarray(lengths, dtype=np.int64)[:, None]


def _maybe_torch(batch: dict[str, Any], to_torch: bool) -> dict[str, Any]:
    """Convert every array of ``batch`` to a torch tensor when asked (lazy import)."""
    if not to_torch:
        return batch
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is installed in the dev environment
        raise ImportError(
            "to_torch=True needs torch; the numpy batches work without it "
            "(see docs/ml/egitim-altyapisi.md §2)"
        ) from exc

    def convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(value))
        return value

    return {key: convert(value) for key, value in batch.items()}
