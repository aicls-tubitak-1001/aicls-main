"""MathWriting datasets: shard-backed and purely synthetic (docs/ml/egitim-altyapisi.md §2).

:class:`MathWritingDataset` is a map-style dataset over the ``npz`` shards produced by
:func:`aicls.data.prepare_shards`. It serves both input families of ADR-0013 from the same shards:

* ``mode="online"`` — ``aicls.ocr.normalize_strokes`` turns the strokes into a ``(T, F)`` feature
  matrix for the stroke encoders;
* ``mode="offline"`` — the sample is rasterised (or the pre-rendered raster is reused) and pushed
  through the configured, ablatable ``aicls.ocr.preprocess_image`` pipeline into ``(1, H, W)``.

``synthetic_ratio`` mixes MathWriting's synthetic subset into the epoch **at sampler level**: the
positions of the synthetic samples are computed from the ratio alone, so an epoch is deterministic
and reproducible, every real sample is still seen exactly once, and ``synthetic_ratio=1.0``
expresses the "two synthetic warm-up epochs" of the training recipe.

torch is never imported here; :meth:`MathWritingDataset.to_torch` wraps the dataset in a
``torch.utils.data.Dataset`` behind a lazy import, and the numpy path works without torch at all.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from aicls.core.logging import get_logger
from aicls.data.errors import invalid_data, missing_path
from aicls.data.shards import ShardReader, ShardSample, load_shard_manifest
from aicls.ocr import Ink, PreprocessConfig, normalize_strokes, preprocess_image, render_ink

_log = get_logger(__name__)

Mode = Literal["online", "offline"]

#: The two supported item modes.
MODES: tuple[str, ...] = ("online", "offline")

#: Split holding MathWriting's synthetic expressions.
SYNTHETIC_SPLIT = "synthetic"


class Tokenizer(Protocol):
    """The tokenizer contract a dataset needs (``aicls.ocr.LatexTokenizer`` implements it)."""

    def encode(
        self, latex: str, *, add_special: bool = False, max_len: int | None = None
    ) -> list[int]:
        """Encode a LaTeX string into token ids."""
        ...


@dataclass(frozen=True, slots=True)
class InkSample:
    """One handwriting training item.

    Attributes:
        features: ``(T, F)`` online features or ``(1, H, W)`` offline image, float32.
        label_ids: Token ids of the label, ``int64``.
        label: The normalised LaTeX label the ids came from.
        sample_id: Source file stem (shard) or synthetic index.
        writer_id: Writer identifier (``""`` when unknown).
        is_synthetic: Whether the sample comes from the synthetic subset.
    """

    features: np.ndarray
    label_ids: np.ndarray
    label: str
    sample_id: str
    writer_id: str
    is_synthetic: bool

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield ``features, label_ids`` so ``features, ids = sample`` keeps working."""
        yield self.features
        yield self.label_ids

    def __getitem__(self, index: int) -> np.ndarray:
        """``sample[0]`` is the features, ``sample[1]`` the token ids."""
        return (self.features, self.label_ids)[index]


class _BaseInkDataset:
    """Shared item building for the shard-backed and synthetic ink datasets."""

    def __init__(
        self,
        *,
        mode: Mode,
        tokenizer: Tokenizer | None,
        preprocess: PreprocessConfig | None,
        max_label_len: int | None,
        add_special: bool,
        stroke_width: int,
    ) -> None:
        if mode not in MODES:
            raise invalid_data("unknown_mode", mode=str(mode), allowed=list(MODES))
        if max_label_len is not None and max_label_len <= 0:
            raise invalid_data("max_label_len_not_positive", max_label_len=max_label_len)
        self._mode: Mode = mode
        self._tokenizer = tokenizer
        self._preprocess = preprocess
        self._max_label_len = max_label_len
        self._add_special = add_special
        self._stroke_width = stroke_width

    @property
    def mode(self) -> Mode:
        """``"online"`` or ``"offline"``."""
        return self._mode

    @property
    def image_size(self) -> tuple[int, int]:
        """Raster size used by the offline path."""
        from aicls.ocr import DEFAULT_RENDER_SIZE

        return self._preprocess.size if self._preprocess is not None else DEFAULT_RENDER_SIZE

    def _encode(self, label: str) -> np.ndarray:
        if self._tokenizer is None:
            return np.zeros((0,), dtype=np.int64)
        ids = self._tokenizer.encode(
            label, add_special=self._add_special, max_len=self._max_label_len
        )
        return np.asarray(ids, dtype=np.int64)

    def _features_from_ink(self, ink: Ink, *, image: np.ndarray | None = None) -> np.ndarray:
        if self._mode == "online":
            return np.ascontiguousarray(normalize_strokes(ink), dtype=np.float32)
        raster = (
            image.astype(np.float32) / 255.0
            if image is not None
            else render_ink(ink, self.image_size, stroke_width=self._stroke_width)
        )
        if self._preprocess is not None:
            raster = preprocess_image(raster, self._preprocess)
        return np.ascontiguousarray(np.asarray(raster, dtype=np.float32)[None, ...])

    def _build(self, sample: ShardSample) -> InkSample:
        return InkSample(
            features=self._features_from_ink(sample.to_ink(), image=sample.image),
            label_ids=self._encode(sample.label),
            label=sample.label,
            sample_id=sample.sample_id,
            writer_id=sample.writer_id,
            is_synthetic=sample.is_synthetic,
        )

    def to_torch(self) -> Any:
        """Wrap this dataset in a ``torch.utils.data.Dataset`` (lazy import).

        Returns:
            A torch map-style dataset delegating to ``__len__``/``__getitem__``.

        Raises:
            ImportError: If torch is not installed (it is an optional, gated dependency).
        """
        return _torch_dataset(self)


class MathWritingDataset(_BaseInkDataset):
    """MathWriting samples read from ``npz`` shards.

    Args:
        root: Shards root (the directory holding ``<split>/shards_manifest.json``).
        split: Split to serve (``train``, ``valid``, ``test``, ``symbols``, ``synthetic``).
        mode: ``"online"`` (stroke features) or ``"offline"`` (rasterised image).
        tokenizer: Label tokenizer; ``None`` yields empty id arrays (feature-only use).
        preprocess: Offline preprocessing pipeline; also fixes the raster size.
        synthetic_ratio: Share of the epoch drawn from the ``synthetic`` split, in ``[0, 1]``.
        max_label_len: Truncate token ids to this length.
        add_special: Wrap the ids in ``<s> ... </s>`` (default: yes, the seq2seq convention).
        stroke_width: Pen width used when a raster has to be produced.

    Raises:
        InputValidationError: On an unknown mode or a ratio outside ``[0, 1]``.
        NotFoundError: If the split (or the synthetic split it needs) has no shard manifest.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        *,
        mode: Mode = "online",
        tokenizer: Tokenizer | None = None,
        preprocess: PreprocessConfig | None = None,
        synthetic_ratio: float = 0.0,
        max_label_len: int | None = None,
        add_special: bool = True,
        stroke_width: int = 2,
    ) -> None:
        super().__init__(
            mode=mode,
            tokenizer=tokenizer,
            preprocess=preprocess,
            max_label_len=max_label_len,
            add_special=add_special,
            stroke_width=stroke_width,
        )
        if not 0.0 <= float(synthetic_ratio) <= 1.0:
            raise invalid_data("synthetic_ratio_out_of_range", synthetic_ratio=synthetic_ratio)

        self._root = Path(root)
        self._split = split
        self._ratio = float(synthetic_ratio)
        self._real = _SplitIndex(self._root, split)
        self._synthetic: _SplitIndex | None = None
        if self._ratio > 0.0:
            self._synthetic = _SplitIndex(self._root, SYNTHETIC_SPLIT)

        if self._ratio >= 1.0 and self._synthetic is not None:
            self._length = len(self._synthetic)
        else:
            self._length = round(len(self._real) / (1.0 - self._ratio))
        _log.info(
            "data.mathwriting.opened",
            root=str(self._root),
            split=split,
            mode=mode,
            n_real=len(self._real),
            n_synthetic=len(self._synthetic) if self._synthetic else 0,
            epoch_length=self._length,
        )

    # ------------------------------------------------------------------ dataset protocol
    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> InkSample:
        """Return item ``index`` of the (possibly mixed) epoch.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(f"dataset index out of range: {index}")
        if self._synthetic is not None and self._is_synthetic_position(index):
            position = floor(index * self._ratio) % len(self._synthetic)
            return self._build(self._synthetic.sample(position))
        real_position = (index - floor(index * self._ratio)) % len(self._real)
        return self._build(self._real.sample(real_position))

    def __iter__(self) -> Iterator[InkSample]:
        for index in range(len(self)):
            yield self[index]

    # ------------------------------------------------------------------ introspection
    @property
    def split(self) -> str:
        """Served split name."""
        return self._split

    @property
    def synthetic_ratio(self) -> float:
        """Configured synthetic share of the epoch."""
        return self._ratio

    @property
    def n_real(self) -> int:
        """Number of real samples behind this dataset."""
        return len(self._real)

    @property
    def n_synthetic(self) -> int:
        """Number of synthetic samples available for mixing."""
        return len(self._synthetic) if self._synthetic is not None else 0

    @property
    def uses_rendered_images(self) -> bool:
        """Whether the offline path reuses rasters stored in the shards."""
        return self._real.has_images

    @property
    def fingerprint(self) -> str:
        """Shard-manifest fingerprint of the served split (goes into the OCR report)."""
        return self._real.fingerprint

    def _is_synthetic_position(self, index: int) -> bool:
        """Deterministic, evenly spread synthetic positions (no RNG, so epochs repeat exactly)."""
        if self._ratio >= 1.0:
            return True
        return floor((index + 1) * self._ratio) > floor(index * self._ratio)


class SyntheticInkDataset(_BaseInkDataset):
    """Deterministic synthetic ink samples — the CPU smoke dataset (no corpus needed).

    Args:
        n: Number of samples.
        seed: Base seed; sample ``i`` uses ``seed + i``.
        tokenizer: Label tokenizer.
        mode: ``"online"`` or ``"offline"``.
        preprocess: Offline preprocessing pipeline.
        latexes: Label pool; defaults to ``aicls.ocr.CURRICULUM_LATEX``.
        max_label_len: Truncate token ids to this length.
        add_special: Wrap the ids in ``<s> ... </s>``.

    Raises:
        InputValidationError: If ``n`` is not positive or the mode is unknown.
    """

    def __init__(
        self,
        n: int,
        *,
        seed: int = 1234,
        tokenizer: Tokenizer | None = None,
        mode: Mode = "online",
        preprocess: PreprocessConfig | None = None,
        latexes: Sequence[str] | None = None,
        max_label_len: int | None = None,
        add_special: bool = True,
    ) -> None:
        super().__init__(
            mode=mode,
            tokenizer=tokenizer,
            preprocess=preprocess,
            max_label_len=max_label_len,
            add_special=add_special,
            stroke_width=2,
        )
        if int(n) <= 0:
            raise invalid_data("n_not_positive", n=int(n))
        from aicls.ocr import CURRICULUM_LATEX

        self._n = int(n)
        self._seed = int(seed)
        self._latexes = tuple(latexes) if latexes else tuple(CURRICULUM_LATEX)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> InkSample:
        """Generate sample ``index`` from ``seed + index``.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        from aicls.data.shards import normalize_label
        from aicls.data.synthetic import synth_ink

        if index < 0:
            index += self._n
        if not 0 <= index < self._n:
            raise IndexError(f"dataset index out of range: {index}")
        latex = self._latexes[index % len(self._latexes)]
        ink = synth_ink(np.random.default_rng(self._seed + index), latex)
        label = normalize_label(latex)
        return InkSample(
            features=self._features_from_ink(ink),
            label_ids=self._encode(label),
            label=label,
            sample_id=f"synthetic-{index:06d}",
            writer_id=ink.writer_id or "",
            is_synthetic=True,
        )

    def __iter__(self) -> Iterator[InkSample]:
        for index in range(len(self)):
            yield self[index]


class _SplitIndex:
    """Shard manifest plus lazily opened readers for one split."""

    def __init__(self, root: Path, split: str) -> None:
        self._root = Path(root)
        self._split = split
        self._manifest = load_shard_manifest(self._root, split)
        self._readers: dict[int, ShardReader] = {}
        self._offsets: list[int] = []
        total = 0
        for entry in self._manifest.shards:
            self._offsets.append(total)
            total += entry.n_samples
        self._length = total
        if total == 0:
            raise missing_path(self._root / split, "shard_split_empty")

    def __len__(self) -> int:
        return self._length

    @property
    def has_images(self) -> bool:
        """Whether the shards carry pre-rendered rasters."""
        return self._manifest.has_images

    @property
    def fingerprint(self) -> str:
        """Shard-manifest fingerprint."""
        return self._manifest.fingerprint

    def sample(self, index: int) -> ShardSample:
        """Return global sample ``index`` of this split."""
        shard_index = max(0, bisect_right(self._offsets, index) - 1)
        local = index - self._offsets[shard_index]
        reader = self._readers.get(shard_index)
        if reader is None:
            entry = self._manifest.shards[shard_index]
            reader = ShardReader(self._root / self._split / entry.name)
            self._readers[shard_index] = reader
        return reader.sample(local)


def _torch_dataset(dataset: Any) -> Any:
    """Build a ``torch.utils.data.Dataset`` wrapper around ``dataset`` (lazy torch import)."""
    try:
        from torch.utils.data import Dataset as TorchDataset
    except ImportError as exc:  # pragma: no cover - torch is installed in the dev environment
        raise ImportError(
            "torch is required for MathWritingDataset.to_torch(); the numpy dataset works "
            "without it (see docs/ml/egitim-altyapisi.md §2)"
        ) from exc

    class _Wrapped(TorchDataset[Any]):
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __len__(self) -> int:
            return len(self._inner)

        def __getitem__(self, index: int) -> Any:
            return self._inner[index]

    return _Wrapped(dataset)


__all__ = [
    "MODES",
    "SYNTHETIC_SPLIT",
    "InkSample",
    "MathWritingDataset",
    "Mode",
    "SyntheticInkDataset",
    "Tokenizer",
]
