"""Model and dataset registries: config names to constructed objects (design document §3).

A training run says ``model.family: seq2seq_image`` and ``data.dataset: mathwriting``; this module
is the only place that turns those two strings into a ``torch.nn.Module`` and a map-style dataset.
Nothing is reimplemented here — the OCR families delegate to :mod:`aicls.ocr.models.seq2seq`, the
speech families to Hugging Face ``transformers``, and the datasets to :mod:`aicls.data`.

Three rules hold for every entry:

* **Lazy heavy imports.** ``import aicls.ml.registry`` must not import torch, torchvision or
  transformers, so an ungated process (the CLI listing configs, a test collecting names) stays
  cheap and torch-free — ``test_importing_the_registry_does_not_import_torch``.
* **Local weights only.** ``whisper``/``wav2vec2`` load from ``model.base_dir``, a directory under
  ``AICLS_MODELS_ROOT``, with ``local_files_only=True``. A hub id would be a download and sits
  behind the ``WEIGHTS_DOWNLOAD`` gate (ADR-0012, ``tools/check_gates.py``). With no ``base_dir``
  the model is built from a small randomly initialised config: that is the CI smoke path of
  ``training/configs/asr/smoke_synthetic.yaml``, and it is a wiring check, never a result.
* **Unknown names fail loudly**, listing what is registered.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from aicls.core import messages_tr
from aicls.core.errors import InputValidationError, NotFoundError
from aicls.core.logging import get_logger
from aicls.ml.config import DEFAULT_MAX_LEN, DEFAULT_SEED, DataCfg, ModelCfg, TrainConfig

_log = get_logger(__name__)

#: Number of samples a synthetic dataset serves when ``data.n_samples`` is not set.
DEFAULT_SYNTHETIC_SAMPLES = 64

#: The only split a training-curriculum knob such as ``data.synthetic_ratio`` may touch.
TRAIN_SPLIT = "train"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Everything a model builder needs, gathered from the config tree.

    ``model.d_model`` and friends live in :class:`~aicls.ml.config.ModelCfg`, but the raster size
    lives in the ``data:`` block, so builders receive this resolved view rather than one of the
    two sub-configs.

    Attributes:
        model: The architecture block.
        image_size: ``(H, W)`` raster size (``data.image_size``).
        max_len: Decoding cap actually used.
    """

    model: ModelCfg
    image_size: tuple[int, int] = (96, 384)
    max_len: int = DEFAULT_MAX_LEN

    @classmethod
    def from_config(cls, cfg: ModelCfg | TrainConfig) -> ModelSpec:
        """Build the spec from a full :class:`TrainConfig` or a bare :class:`ModelCfg`."""
        if isinstance(cfg, TrainConfig):
            return cls(
                model=cfg.model,
                image_size=cfg.data.image_size,
                max_len=cfg.model.max_len or cfg.data.max_label_len or DEFAULT_MAX_LEN,
            )
        return cls(model=cfg, max_len=cfg.max_len or DEFAULT_MAX_LEN)


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Everything a dataset builder needs, gathered from the config tree.

    Attributes:
        data: The data block.
        seed: ``run.seed``; the synthetic generators are seeded from it.
        vocab_path: ``model.vocab_path``; ``None`` falls back to the curriculum tokenizer.
        model_dir: ``model.base_dir``; needed by the Whisper feature extractor.
        n_mels: ``model.num_mel_bins`` for the log-Mel feature function.
    """

    data: DataCfg
    seed: int = DEFAULT_SEED
    vocab_path: Path | None = None
    model_dir: Path | None = None
    n_mels: int = 80

    @classmethod
    def from_config(cls, cfg: DataCfg | TrainConfig) -> DatasetSpec:
        """Build the spec from a full :class:`TrainConfig` or a bare :class:`DataCfg`."""
        if isinstance(cfg, TrainConfig):
            return cls(
                data=cfg.data,
                seed=cfg.run.seed,
                vocab_path=cfg.model.vocab_path,
                model_dir=cfg.model.base_dir,
                n_mels=cfg.model.num_mel_bins,
            )
        return cls(data=cfg)

    def tokenizer(self) -> Any:
        """Return the LaTeX tokenizer: ``model.vocab_path`` when given, else the default one."""
        from aicls.ocr import LatexTokenizer, default_tokenizer

        if self.vocab_path is not None:
            return LatexTokenizer.from_file(self.vocab_path)
        return default_tokenizer()

    def n_samples(self) -> int:
        """Length of a synthetic dataset (``data.n_samples`` or the CI default)."""
        return self.data.n_samples or DEFAULT_SYNTHETIC_SAMPLES


ModelBuilder = Callable[[ModelSpec, int], Any]
DatasetBuilder = Callable[[DatasetSpec, str], Any]

_MODELS: dict[str, ModelBuilder] = {}
_DATASETS: dict[str, DatasetBuilder] = {}

_B = TypeVar("_B", bound=Callable[..., Any])


def _register(table: dict[str, Any], kind: str, name: str) -> Callable[[_B], _B]:
    """Return a decorator registering the decorated callable under ``name``."""

    def decorator(builder: _B) -> _B:
        if name in table:
            raise InputValidationError(
                messages_tr.MSG_VALIDATION_ERROR,
                details={"reason": f"duplicate_{kind}", "name": name},
            )
        table[name] = builder
        return builder

    return decorator


def register_model(name: str) -> Callable[[_B], _B]:
    """Register a model builder under ``name`` (a :data:`aicls.ml.config.MODEL_FAMILIES` entry).

    Args:
        name: The ``model.family`` value this builder answers to.

    Returns:
        A decorator that registers and returns the builder unchanged.

    Raises:
        InputValidationError: If the name is already registered.
    """
    return _register(_MODELS, "model", name)


def register_dataset(name: str) -> Callable[[_B], _B]:
    """Register a dataset builder under ``name`` (a :data:`aicls.ml.config.DATASET_NAMES` entry).

    Args:
        name: The ``data.dataset`` value this builder answers to.

    Returns:
        A decorator that registers and returns the builder unchanged.

    Raises:
        InputValidationError: If the name is already registered.
    """
    return _register(_DATASETS, "dataset", name)


def registered_models() -> tuple[str, ...]:
    """Sorted names of every registered model family."""
    return tuple(sorted(_MODELS))


def registered_datasets() -> tuple[str, ...]:
    """Sorted names of every registered dataset."""
    return tuple(sorted(_DATASETS))


def _lookup(table: dict[str, Any], kind: str, name: str) -> Any:
    """Return ``table[name]`` or raise a :class:`NotFoundError` listing what is registered."""
    builder = table.get(name)
    if builder is None:
        raise NotFoundError(
            details={
                "reason": f"unknown_{kind}",
                "name": name,
                "registered": sorted(table),
            }
        )
    return builder


def get_model_builder(name: str) -> ModelBuilder:
    """Look up a model builder by family name.

    Raises:
        NotFoundError: If nothing is registered under ``name``; ``details["registered"]`` lists
            the available names.
    """
    builder: ModelBuilder = _lookup(_MODELS, "model", name)
    return builder


def get_dataset_builder(name: str) -> DatasetBuilder:
    """Look up a dataset builder by dataset name.

    Raises:
        NotFoundError: If nothing is registered under ``name``; ``details["registered"]`` lists
            the available names.
    """
    builder: DatasetBuilder = _lookup(_DATASETS, "dataset", name)
    return builder


def build_model(cfg: ModelCfg | TrainConfig, vocab_size: int) -> Any:
    """Build the model described by ``cfg``.

    Args:
        cfg: A full :class:`~aicls.ml.config.TrainConfig` (preferred: it also carries the raster
            size) or just its :class:`~aicls.ml.config.ModelCfg` block.
        vocab_size: Output vocabulary size, specials included.

    Returns:
        A ``torch.nn.Module``. For ``whisper``/``wav2vec2`` with no ``model.base_dir`` the weights
        are randomly initialised — a wiring check, not a trained model.

    Raises:
        NotFoundError: On an unknown family, or a ``base_dir`` that does not exist.
        InputValidationError: If ``vocab_size`` is not positive.
        aicls.ocr.torch_utils.TorchUnavailableError: If torch is not installed.
    """
    if vocab_size <= 0:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "vocab_size_not_positive", "vocab_size": int(vocab_size)},
        )
    spec = ModelSpec.from_config(cfg)
    builder = get_model_builder(spec.model.family)
    model = builder(spec, int(vocab_size))
    _log.info(
        "ml.registry.model_built",
        family=spec.model.family,
        encoder=spec.model.encoder,
        vocab_size=int(vocab_size),
        pretrained=spec.model.base_dir is not None,
    )
    return model


def build_dataset(cfg: DataCfg | TrainConfig, split: str) -> Any:
    """Build the dataset described by ``cfg`` for one split.

    Args:
        cfg: A full :class:`~aicls.ml.config.TrainConfig` (preferred: it also carries the seed and
            the vocabulary path) or just its :class:`~aicls.ml.config.DataCfg` block.
        split: ``train``, ``valid``, ``test``, ... The synthetic datasets ignore it: they generate
            their samples from ``run.seed`` and have no splits.

    Returns:
        A map-style dataset from :mod:`aicls.data` (``ds.to_torch()`` wraps it for a DataLoader).

    Raises:
        NotFoundError: On an unknown dataset name or a missing file.
        InputValidationError: If a path the dataset needs is not configured.
    """
    spec = DatasetSpec.from_config(cfg)
    builder = get_dataset_builder(spec.data.dataset)
    dataset = builder(spec, split)
    _log.info(
        "ml.registry.dataset_built",
        dataset=spec.data.dataset,
        split=split,
        mode=spec.data.mode,
        n=len(dataset),
    )
    return dataset


def _require(value: Any, key: str) -> Any:
    """Return ``value`` or raise naming the config key that has to be set."""
    if value is None:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "missing_config_key", "key": key},
        )
    return value


def _require_local_dir(path: Path, key: str) -> Path:
    """Return an existing local model directory.

    Raises:
        NotFoundError: If it does not exist. It is never downloaded: that is the
            ``WEIGHTS_DOWNLOAD`` gate (ADR-0012).
    """
    if not path.is_dir():
        raise NotFoundError(
            messages_tr.msg("FILE_NOT_FOUND", path=str(path)),
            details={
                "reason": "model_dir_missing",
                "key": key,
                "path": str(path),
                "gate": "WEIGHTS_DOWNLOAD",
            },
        )
    return path


def require_transformers() -> Any:
    """Import ``transformers`` lazily, as a typed error rather than a bare ``ImportError``.

    Returns:
        The ``transformers`` module.

    Raises:
        aicls.ocr.torch_utils.TorchUnavailableError: If it is not installed. It ships with the
            ``dev`` extra; nothing here ever downloads it.
    """
    from aicls.ocr.torch_utils import TorchUnavailableError

    try:
        import transformers
    except ImportError as exc:
        raise TorchUnavailableError("transformers", str(exc)) from exc
    return transformers


# ----------------------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------------------
def _build_seq2seq(spec: ModelSpec, vocab_size: int) -> Any:
    """Shared body of the two OCR families (:func:`aicls.ocr.models.seq2seq.build_seq2seq`)."""
    from aicls.ocr.torch_utils import require_torch

    require_torch()  # typed, Turkish error instead of a bare ImportError
    from aicls.ocr.models.seq2seq import build_seq2seq

    ocr_cfg = spec.model.to_ocr_config(image_size=spec.image_size, max_len=spec.max_len)
    return build_seq2seq(ocr_cfg, vocab_size)


@register_model("seq2seq_image")
def build_seq2seq_image(spec: ModelSpec, vocab_size: int) -> Any:
    """Offline OCR family: CNN/torchvision encoder (``weights=None``) plus an attention decoder."""
    return _build_seq2seq(spec, vocab_size)


@register_model("seq2seq_stroke")
def build_seq2seq_stroke(spec: ModelSpec, vocab_size: int) -> Any:
    """Online OCR family: stroke-feature encoder plus the same decoders (ADR-0013)."""
    return _build_seq2seq(spec, vocab_size)


@register_model("whisper")
def build_whisper(spec: ModelSpec, vocab_size: int) -> Any:
    """Whisper sequence-to-sequence ASR model.

    With ``model.base_dir`` set the checkpoint is loaded from that **local** directory; without it
    a small randomly initialised model is built from ``d_model``/``layers``/``heads`` — the CPU
    smoke path, which produces no evidence about WER.
    """
    transformers = require_transformers()

    cfg = spec.model
    if cfg.base_dir is not None:
        local_dir = str(_require_local_dir(Path(cfg.base_dir), "model.base_dir"))
        return transformers.WhisperForConditionalGeneration.from_pretrained(
            local_dir, local_files_only=True
        )
    config = transformers.WhisperConfig(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        encoder_layers=cfg.layers,
        decoder_layers=cfg.layers,
        encoder_attention_heads=cfg.heads,
        decoder_attention_heads=cfg.heads,
        encoder_ffn_dim=cfg.feedforward_dim,
        decoder_ffn_dim=cfg.feedforward_dim,
        dropout=cfg.dropout,
        num_mel_bins=cfg.num_mel_bins,
        max_target_positions=spec.max_len,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    return transformers.WhisperForConditionalGeneration(config)


@register_model("wav2vec2")
def build_wav2vec2(spec: ModelSpec, vocab_size: int) -> Any:
    """Wav2Vec2 CTC model — the İP-3 risk plan's ensemble candidate, kept warm in the registry."""
    transformers = require_transformers()

    cfg = spec.model
    if cfg.base_dir is not None:
        local_dir = str(_require_local_dir(Path(cfg.base_dir), "model.base_dir"))
        return transformers.Wav2Vec2ForCTC.from_pretrained(local_dir, local_files_only=True)
    config = transformers.Wav2Vec2Config(
        vocab_size=vocab_size,
        hidden_size=cfg.d_model,
        num_hidden_layers=cfg.layers,
        num_attention_heads=cfg.heads,
        intermediate_size=cfg.feedforward_dim,
        hidden_dropout=cfg.dropout,
        attention_dropout=cfg.dropout,
        num_feat_extract_layers=2,
        conv_dim=(cfg.d_model, cfg.d_model),
        conv_stride=(5, 2),
        conv_kernel=(10, 3),
    )
    return transformers.Wav2Vec2ForCTC(config)


# ----------------------------------------------------------------------------------------------
# datasets
# ----------------------------------------------------------------------------------------------
def _feature_fn(spec: DatasetSpec) -> Any:
    """Audio feature extractor named by ``data.feature_extractor``.

    ``log_mel`` (the default) is the numpy implementation of :mod:`aicls.asr.features`; ``whisper``
    builds a Hugging Face ``WhisperFeatureExtractor`` from the **local** ``model.base_dir``.
    """
    import numpy as np

    from aicls.asr import log_mel
    from aicls.asr.types import Audio

    if spec.data.feature_extractor == "whisper":
        local_dir = str(
            _require_local_dir(Path(_require(spec.model_dir, "model.base_dir")), "model.base_dir")
        )
        transformers = require_transformers()

        extractor = transformers.WhisperFeatureExtractor.from_pretrained(
            local_dir, local_files_only=True
        )

        def whisper_features(audio: Audio) -> Any:
            batch = extractor(audio.samples, sampling_rate=audio.sr, return_tensors="np")
            return np.asarray(batch["input_features"][0], dtype=np.float32)

        return whisper_features

    def log_mel_features(audio: Audio) -> Any:
        return log_mel(audio, n_mels=spec.n_mels)

    return log_mel_features


@register_dataset("mathwriting")
def build_mathwriting(spec: DatasetSpec, split: str) -> Any:
    """MathWriting shards (``training/scripts/prepare_mathwriting.py`` output).

    The shards themselves are behind the ``DATASET_DOWNLOAD`` gate; this only opens what is on
    disk, and says which key is missing when it is not.

    ``data.synthetic_ratio`` is applied to the **training split only**. It is a curriculum knob
    (design document §1: warm up on rendered synthetic ink, then mix at 0.5), not a property of
    the thing being measured. Mixing synthetic strokes into ``valid`` would make ``val_cer`` —
    the number that selects the best checkpoint — score the model partly on machine-generated
    ink, and mixing them into ``test`` would put them inside the contractual CER itself.
    """
    from aicls.data import MathWritingDataset

    data = spec.data
    root = Path(_require(data.root, "data.root"))
    ratio = data.synthetic_ratio if split == TRAIN_SPLIT else 0.0
    if ratio != data.synthetic_ratio:
        _log.info(
            "ml.registry.synthetic_ratio_not_applied",
            split=split,
            configured=data.synthetic_ratio,
            reason="synthetic_ratio is a training curriculum; evaluation splits stay real",
        )
    return MathWritingDataset(
        root,
        split,
        mode=data.mode,
        tokenizer=spec.tokenizer(),
        preprocess=data.to_preprocess_config(),
        synthetic_ratio=ratio,
        max_label_len=data.max_label_len,
    )


@register_dataset("synthetic_ink")
def build_synthetic_ink(spec: DatasetSpec, split: str) -> Any:
    """Deterministic synthetic handwriting — the gate-free CPU dataset (``split`` is ignored)."""
    from aicls.data import SyntheticInkDataset

    data = spec.data
    return SyntheticInkDataset(
        spec.n_samples(),
        seed=spec.seed,
        tokenizer=spec.tokenizer(),
        mode=data.mode,
        preprocess=data.to_preprocess_config(),
        max_label_len=data.max_label_len,
    )


@register_dataset("audio_commands")
def build_audio_commands(spec: DatasetSpec, split: str) -> Any:
    """The Turkish math-command corpus exported by ``aicls contrib export-manifest``.

    A speaker-disjoint split assignment is mandatory; :class:`~aicls.data.AudioCommandDataset`
    refuses to load without one, because a WER over a seen speaker is not a WER.
    """
    from aicls.data import AudioCommandDataset

    data = spec.data
    manifest = Path(_require(data.manifest, "data.manifest"))
    return AudioCommandDataset(
        manifest,
        split,
        feature_fn=_feature_fn(spec),
        splits=Path(data.splits_path) if data.splits_path is not None else None,
        max_seconds=data.max_seconds,
    )


@register_dataset("synthetic_audio")
def build_synthetic_audio(spec: DatasetSpec, split: str) -> Any:
    """Deterministic synthetic utterances — the gate-free CPU dataset (``split`` is ignored)."""
    from aicls.data import SyntheticAudioDataset

    return SyntheticAudioDataset(
        spec.n_samples(),
        seed=spec.seed,
        feature_fn=_feature_fn(spec),
        seconds=spec.data.sample_seconds,
    )


__all__ = [
    "DEFAULT_SYNTHETIC_SAMPLES",
    "DatasetBuilder",
    "DatasetSpec",
    "ModelBuilder",
    "ModelSpec",
    "build_audio_commands",
    "build_dataset",
    "build_mathwriting",
    "build_model",
    "build_seq2seq_image",
    "build_seq2seq_stroke",
    "build_synthetic_audio",
    "build_synthetic_ink",
    "build_wav2vec2",
    "build_whisper",
    "get_dataset_builder",
    "get_model_builder",
    "register_dataset",
    "register_model",
    "registered_datasets",
    "registered_models",
    "require_transformers",
]
