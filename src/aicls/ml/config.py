"""Typed training configuration: one YAML tree per run (docs/ml/egitim-altyapisi.md §1, ADR-0006).

:class:`TrainConfig` is the single description of a training run — data, model, optimiser, loop,
evaluation, tracking, checkpointing — and every sub-model sets ``extra="forbid"`` so a typo is
caught before the first optimizer step instead of silently training the wrong thing.

Loading (:func:`load_config`) adds the three things a config file needs in practice:

* **``_base_`` inheritance**, resolved relative to the child file and deep-merged (dictionaries
  merge, scalars and lists replace) with a cycle guard, so a task config states only its deltas;
* **dotted CLI overrides** (``"optim.lr=1e-4"``, ``"train.max_steps=2"``) parsed with YAML scalar
  semantics — with the one repair that PyYAML's YAML 1.1 resolver needs, so ``1e-4`` is a float
  and not the string ``"1e-4"``;
* **``${AICLS_DATA_ROOT}`` / ``${AICLS_MODELS_ROOT}`` expansion** through :mod:`aicls.core.paths`,
  so no committed file contains a machine-specific absolute path.

Two rules are worth reading before editing this module:

* ``model.weights`` may only ever be ``None``. Pre-trained torchvision weights are a download
  (ADR-0012); a checkpoint is loaded from a local directory instead.
* :attr:`TrainConfig.is_gate_free` is decided by ``data.dataset`` alone. ``gates.synthetic`` is a
  declaration that can *narrow* nothing and *widen* nothing; a config claiming synthetic data
  while pointing at MathWriting still needs the ``TRAINING`` approval.

The module imports :mod:`aicls.ocr` (for the encoder and preprocessing vocabularies) but never
torch: ``import aicls.ml.config`` stays cheap and safe in an ungated process.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import os
from pathlib import Path
import re
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
import yaml  # type: ignore[import-untyped]  # PyYAML ships no stubs and types-PyYAML is absent

from aicls.core import messages_tr
from aicls.core.errors import InputValidationError, NotFoundError
from aicls.core.hashing import sha256_json
from aicls.core.logging import get_logger
from aicls.core.paths import data_root, models_root, runs_root
from aicls.ocr import PREPROCESS_STEPS, PreprocessConfig
from aicls.ocr.config import IMAGE_ENCODERS, STROKE_ENCODERS, OcrModelConfig
from aicls.ocr.preprocess import PreprocessStep

_log = get_logger(__name__)

#: Default seed used everywhere in the project (docs/gelistirme/test-stratejisi.md).
DEFAULT_SEED = 1234

#: Default decoding length when neither ``model.max_len`` nor ``data.max_label_len`` is set.
DEFAULT_MAX_LEN = 160

#: Model families of the design document §3.
MODEL_FAMILIES: tuple[str, ...] = ("seq2seq_image", "seq2seq_stroke", "whisper", "wav2vec2")

#: Datasets of the design document §2.
DATASET_NAMES: tuple[str, ...] = (
    "mathwriting",
    "audio_commands",
    "synthetic_ink",
    "synthetic_audio",
)

#: Metric names accepted in ``eval.metrics`` (implemented in :mod:`aicls.eval.metrics`).
METRIC_NAMES: tuple[str, ...] = ("cer", "ter", "wer", "exprate", "symeq", "loss")

#: Inheritance key of the design document, plus the spelling used by the committed base file.
BASE_KEY = "_base_"
LEGACY_BASE_KEY = "extends"

#: Fields deliberately excluded from :func:`config_hash` (see its docstring).
HASH_EXCLUDED_KEYS: tuple[str, ...] = ("run.name", "run.notes")

#: Pre-schema layout of ``training/configs/_base.yaml``, migrated on load (see §"Uygulama notları").
LEGACY_TOP_LEVEL_RUN_KEYS: tuple[str, ...] = ("seed", "deterministic")
LEGACY_OPTIM_TO_TRAIN_KEYS: tuple[str, ...] = ("grad_clip", "amp")
LEGACY_TRAIN_RENAMES: dict[str, str] = {"early_stopping_patience": "early_stop_patience"}
LEGACY_MODEL_KINDS: dict[str, str] = {
    "ocr.offline_seq2seq": "seq2seq_image",
    "ocr.online_seq2seq": "seq2seq_stroke",
    "asr.whisper_ft": "whisper",
    "asr.wav2vec2_ft": "wav2vec2",
}
LEGACY_DATASET_NAMES: dict[str, str] = {
    "synthetic_ocr": "synthetic_ink",
    "synthetic_asr": "synthetic_audio",
}

ModelFamily = Literal["seq2seq_image", "seq2seq_stroke", "whisper", "wav2vec2"]
DatasetName = Literal["mathwriting", "audio_commands", "synthetic_ink", "synthetic_audio"]
DataMode = Literal["online", "offline"]
AmpMode = Literal["off", "bf16", "fp16"]

_VARIABLE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
# PyYAML implements the YAML 1.1 resolver, which needs a dot *and* a signed exponent before it
# calls something a float; "1e-4" would come back as a string. The CLI contract says otherwise.
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_PATH_VARIABLES = {
    "AICLS_DATA_ROOT": data_root,
    "AICLS_MODELS_ROOT": models_root,
    "AICLS_RUNS_ROOT": runs_root,
}


class _Cfg(BaseModel):
    """Base of every configuration node: unknown keys are an error, instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RunCfg(_Cfg):
    """Identity and reproducibility of one run.

    Attributes:
        name: Human-readable run name (also the run directory name). Not part of the config hash.
        seed: Seed handed to ``seed_everything``.
        deterministic: Ask torch for deterministic kernels (``warn_only``).
        tags: Free-form tags; they *are* part of the config hash because an ablation tag
            distinguishes two otherwise identical experiments.
        notes: Free text for the run log. Not part of the config hash.
    """

    name: str = ""
    seed: int = Field(default=DEFAULT_SEED, ge=0)
    deterministic: bool = True
    tags: tuple[str, ...] = ()
    notes: str = ""

    @field_validator("name", "notes", mode="before")
    @classmethod
    def _none_is_empty(cls, value: Any) -> Any:
        """Accept ``name: null`` (the committed base file writes it that way)."""
        return "" if value is None else value


class PathsCfg(_Cfg):
    """Filesystem roots; ``None`` means "ask :mod:`aicls.core.paths` at use time"."""

    data_root: Path | None = None
    models_root: Path | None = None
    runs_root: Path | None = None


class SpecAugmentCfg(_Cfg):
    """SpecAugment masking (proposal A.2.3)."""

    time_masks: int = Field(default=0, ge=0)
    freq_masks: int = Field(default=0, ge=0)
    time_mask_width: int = Field(default=10, ge=0)
    freq_mask_width: int = Field(default=8, ge=0)


class AugmentCfg(_Cfg):
    """Augmentation switches for both modalities (all ablatable, all off by default).

    Attributes:
        affine_jitter: Ink affine jitter magnitude.
        stroke_dropout: Probability of dropping a stroke.
        spec_augment: Time/frequency masking for the audio path.
        pitch_semitones: Inclusive pitch-shift range in semitones.
        noise_snr_db: Inclusive SNR range of the injected noise.
        speed: Inclusive speed-perturbation factor range.
        tts_ratio: Share of the epoch drawn from TTS-augmented utterances.
    """

    affine_jitter: float = Field(default=0.0, ge=0.0)
    stroke_dropout: float = Field(default=0.0, ge=0.0, le=1.0)
    spec_augment: SpecAugmentCfg | None = None
    pitch_semitones: tuple[float, float] | None = None
    noise_snr_db: tuple[float, float] | None = None
    speed: tuple[float, float] | None = None
    tts_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class DataCfg(_Cfg):
    """Which data the run reads and how it becomes tensors (design document §2).

    Attributes:
        dataset: Registered dataset name; ``synthetic_*`` needs no approval gate (ADR-0012).
        root: Shard root for :class:`~aicls.data.MathWritingDataset`.
        manifest: JSONL export manifest for :class:`~aicls.data.AudioCommandDataset`.
        splits_path: Speaker-disjoint split file; mandatory for a reportable ASR run.
        split: Split served when the caller does not pass one.
        mode: ``online`` (stroke features) or ``offline`` (rasterised image).
        image_size: ``(H, W)`` of the offline raster.
        preprocess: Ordered, ablatable preprocessing steps (:data:`aicls.ocr.PREPROCESS_STEPS`).
        synthetic_ratio: Share of the epoch drawn from MathWriting's synthetic subset.
        synthetic_warmup_epochs: Epochs trained on synthetic data alone before mixing.
        augment: Augmentation switches.
        max_label_len: Truncate label token ids to this length.
        batch_size: Samples per step (per device).
        num_workers: DataLoader workers; 0 on Windows unless profiled.
        feature_extractor: ``log_mel`` (numpy, default) or ``whisper`` (HF, needs a local model).
        n_samples: Length of the synthetic datasets.
        sample_seconds: Utterance length of :class:`~aicls.data.SyntheticAudioDataset`.
        max_seconds: Longest recording :class:`~aicls.data.AudioCommandDataset` will decode.
    """

    dataset: DatasetName = "synthetic_ink"
    root: Path | None = None
    manifest: Path | None = None
    splits_path: Path | None = None
    split: str = "train"
    mode: DataMode = "online"
    image_size: tuple[int, int] = (96, 384)
    preprocess: tuple[str, ...] = ()
    synthetic_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    synthetic_warmup_epochs: int = Field(default=0, ge=0)
    augment: AugmentCfg = Field(default_factory=AugmentCfg)
    max_label_len: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=8, ge=1)
    num_workers: int = Field(default=0, ge=0)
    feature_extractor: Literal["log_mel", "whisper"] | None = None
    n_samples: int | None = Field(default=None, ge=1)
    sample_seconds: float = Field(default=1.0, gt=0.0)
    max_seconds: float = Field(default=30.0, gt=0.0)

    @field_validator("preprocess")
    @classmethod
    def _known_steps(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = [step for step in value if step not in PREPROCESS_STEPS]
        if unknown:
            raise ValueError(f"preprocess: unknown step(s) {unknown}; allowed: {PREPROCESS_STEPS}")
        return value

    @field_validator("image_size")
    @classmethod
    def _positive_image_size(cls, value: tuple[int, int]) -> tuple[int, int]:
        if value[0] <= 0 or value[1] <= 0:
            raise ValueError(f"image_size must be positive, got {value}")
        return value

    def to_preprocess_config(self) -> PreprocessConfig:
        """Return the :class:`aicls.ocr.PreprocessConfig` this data block describes.

        The result is always built (even with an empty step list) because it also carries the
        raster size the offline path renders at.
        """
        # The strings were validated against PREPROCESS_STEPS above; the Literal is exact.
        steps = cast("Sequence[PreprocessStep]", tuple(self.preprocess))
        return PreprocessConfig(steps=steps, size=self.image_size)


class LoraCfg(_Cfg):
    """LoRA adapter settings for the large ASR checkpoint (design document §1, §8)."""

    enabled: bool = False
    r: int = Field(default=16, ge=1)
    alpha: int = Field(default=32, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    targets: tuple[str, ...] = ("q_proj", "v_proj")


class ModelCfg(_Cfg):
    """Architecture of the model under training (design document §3).

    Attributes:
        family: One of :data:`MODEL_FAMILIES`.
        encoder: Backbone name for the two seq2seq families; ``None`` for the HF families.
            Filled in with the family default (``cnn_small`` / ``stroke_transformer``) when absent.
        decoder: Autoregressive decoder of the seq2seq families.
        d_model: Hidden width.
        layers: Layer count (encoder depth and decoder depth).
        heads: Attention heads; must divide ``d_model``.
        ff_dim: Feed-forward width; ``4 * d_model`` when omitted.
        dropout: Dropout probability.
        max_len: Maximum decoded length.
        vocab_path: ``vocab.json`` of the LaTeX tokenizer.
        base_dir: **Local** directory of a pre-trained HF checkpoint. Never a hub id: a hub id is
            a download and sits behind the ``WEIGHTS_DOWNLOAD`` gate (ADR-0012). ``None`` builds a
            randomly initialised model from the sizes above — that is the CI smoke path.
        pretrained_path: Local checkpoint to warm-start the seq2seq families from.
        weights: Always ``None``. torchvision backbones are built with ``weights=None``; anything
            else would fetch weights over the network.
        freeze_encoder: Freeze the encoder during fine-tuning.
        lora: LoRA adapter settings.
        num_mel_bins: Mel bands the audio models expect.
    """

    family: ModelFamily = "seq2seq_image"
    encoder: str | None = None
    decoder: Literal["transformer", "lstm_attn", "gru_attn"] = "transformer"
    d_model: int = Field(default=256, ge=8, le=2048)
    layers: int = Field(default=6, ge=1, le=24)
    heads: int = Field(default=8, ge=1, le=32)
    ff_dim: int | None = Field(default=None, ge=8, le=8192)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    max_len: int | None = Field(default=None, ge=1, le=1024)
    vocab_path: Path | None = None
    base_dir: Path | None = None
    pretrained_path: Path | None = None
    weights: None = None
    freeze_encoder: bool = False
    lora: LoraCfg = Field(default_factory=LoraCfg)
    num_mel_bins: int = Field(default=80, ge=1)

    @model_validator(mode="before")
    @classmethod
    def _default_encoder(cls, data: Any) -> Any:
        """Fill in the family's default encoder (the design document leaves it implicit)."""
        if not isinstance(data, dict) or data.get("encoder") is not None:
            return data
        defaults = {"seq2seq_image": "cnn_small", "seq2seq_stroke": "stroke_transformer"}
        family = str(data.get("family", "seq2seq_image"))
        if family in defaults:
            return {**data, "encoder": defaults[family]}
        return data

    @model_validator(mode="after")
    def _check_family(self) -> ModelCfg:  # noqa: N804 - "after" validators take self
        if self.d_model % self.heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by heads ({self.heads})")
        allowed: tuple[str, ...] | None = None
        if self.family == "seq2seq_image":
            allowed = IMAGE_ENCODERS
        elif self.family == "seq2seq_stroke":
            allowed = STROKE_ENCODERS
        if allowed is None:
            if self.encoder is not None:
                raise ValueError(f"encoder must be null for family {self.family!r}")
        elif self.encoder not in allowed:
            raise ValueError(
                f"encoder {self.encoder!r} is not available for family {self.family!r}; "
                f"allowed: {list(allowed)}"
            )
        return self

    @property
    def feedforward_dim(self) -> int:
        """Feed-forward width actually used (``4 * d_model`` when ``ff_dim`` is unset)."""
        return self.ff_dim if self.ff_dim is not None else 4 * self.d_model

    @property
    def is_seq2seq(self) -> bool:
        """True for the two OCR families built by :mod:`aicls.ocr.models`."""
        return self.family in ("seq2seq_image", "seq2seq_stroke")

    def to_ocr_config(
        self, *, image_size: tuple[int, int] | None = None, max_len: int | None = None
    ) -> OcrModelConfig:
        """Project this block onto :class:`aicls.ocr.config.OcrModelConfig`.

        Args:
            image_size: Raster size from ``data.image_size`` (the OCR default when omitted).
            max_len: Decoding cap (``model.max_len``, then this argument, then the OCR default).

        Returns:
            The architecture description :func:`aicls.ocr.models.seq2seq.build_seq2seq` takes.

        Raises:
            ValueError: If the family is not one of the two seq2seq families.
        """
        if not self.is_seq2seq:
            raise ValueError(f"family {self.family!r} has no OCR model configuration")
        resolved_len = self.max_len or max_len or DEFAULT_MAX_LEN
        return OcrModelConfig(
            family=self.family,  # type: ignore[arg-type]  # narrowed by is_seq2seq
            encoder=self.encoder,  # type: ignore[arg-type]  # validated against the family lists
            decoder=self.decoder,
            d_model=self.d_model,
            layers=self.layers,
            heads=self.heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            max_len=resolved_len,
            image_size=image_size or (96, 384),
        )


class OptimCfg(_Cfg):
    """Optimiser and learning-rate schedule."""

    name: Literal["adamw", "adam", "sgd"] = "adamw"
    lr: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    betas: tuple[float, float] = (0.9, 0.999)
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0)
    scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    warmup_steps: int = Field(default=0, ge=0)


class LoopCfg(_Cfg):
    """The training loop itself (the ``train:`` block).

    Attributes:
        epochs: Epoch budget; ``None`` means "run until ``max_steps``".
        max_steps: Hard step cap; ``None`` means "run the epoch budget".
        grad_accum: Micro-batches accumulated per optimizer step.
        amp: ``off`` (CPU default), ``bf16`` or ``fp16``. bf16 needs the ``CUDA_INSTALL`` gate.
        grad_clip: Global gradient-norm clip; ``None`` disables clipping.
        eval_every: Validation cadence in steps.
        ckpt_every: Checkpoint cadence in steps.
        log_every: Metric logging cadence in steps.
        early_stop_patience: Validations without improvement before stopping; ``None`` disables.
        allow_dirty: Allow a dirty working tree; such a run is marked ``citable=false`` (ADR-0007).
    """

    epochs: int | None = Field(default=None, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    grad_accum: int = Field(default=1, ge=1)
    amp: AmpMode = "off"
    grad_clip: float | None = Field(default=1.0, gt=0.0)
    eval_every: int = Field(default=1000, ge=1)
    ckpt_every: int = Field(default=1000, ge=1)
    log_every: int = Field(default=10, ge=1)
    early_stop_patience: int | None = Field(default=None, ge=1)
    allow_dirty: bool = False

    @field_validator("amp", mode="before")
    @classmethod
    def _amp_from_bool(cls, value: Any) -> Any:
        """Accept ``amp: false`` / ``amp: true`` (the committed base file writes a bool)."""
        if isinstance(value, bool):
            return "bf16" if value else "off"
        if isinstance(value, str) and value.strip().lower() in ("none", "no", "false"):
            return "off"
        return value


class EvalCfg(_Cfg):
    """Validation decoding and metrics (docs/ml/degerlendirme.md).

    Attributes:
        beam: Beam width for decoding; ``1`` is greedy.
        max_n_val: Cap on decoded validation samples; ``None`` decodes the whole split.
        metrics: Metric names to score the decoded pairs with.
        recalibrate_norm_samples: Training samples used to re-estimate BatchNorm running
            statistics immediately before each validation pass; ``None`` disables it. Needed by
            any model whose encoder carries BatchNorm: measured on the RTX 5090 (2026-09-08,
            ``runs/ocr_r34_tfm_full2`` step 18000), the *same weights* scored a teacher-forced
            loss of 0.32 with batch statistics and 8.52 with the accumulated running statistics,
            and the reported CER moved from 7.34 to 0.152 once the statistics were re-estimated.
            Without this the loop measures, selects and early-stops on numbers that describe the
            statistics rather than the model. The estimate is always taken on the **training**
            split; see :meth:`aicls.ml.trainer.Trainer._recalibrate_norm` for why.
    """

    beam: int = Field(default=1, ge=1, le=16)
    max_n_val: int | None = Field(default=None, ge=1)
    metrics: tuple[str, ...] = ("cer",)
    recalibrate_norm_samples: int | None = Field(default=None, ge=1)

    @field_validator("metrics")
    @classmethod
    def _known_metrics(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = [name for name in value if name not in METRIC_NAMES]
        if unknown:
            raise ValueError(f"metrics: unknown metric(s) {unknown}; allowed: {METRIC_NAMES}")
        return value


class TrackingCfg(_Cfg):
    """Experiment tracking backend (ADR-0007; JSONL is the dependency-free default)."""

    backend: Literal["jsonl", "mlflow"] = "jsonl"
    mlflow_uri: str | None = None
    experiment: str = "default"


class CheckpointCfg(_Cfg):
    """Checkpoint retention policy (design document §5)."""

    keep_last: int = Field(default=2, ge=0)
    keep_best: int = Field(default=1, ge=0)
    metric: str = "val_loss"
    mode: Literal["min", "max"] = "min"


class GatesCfg(_Cfg):
    """Declared approval-gate context.

    ``synthetic`` is documentation only: :attr:`TrainConfig.is_gate_free` is decided by the
    dataset name, so a config cannot talk its way past the ``TRAINING`` gate (ADR-0012).
    """

    synthetic: bool | None = None


class TrainConfig(_Cfg):
    """The complete description of one training run.

    Attributes:
        schema_version: Config schema version (bumped when a key is renamed).
        run: Identity, seed and tags.
        paths: Filesystem roots (usually left empty and taken from the environment).
        data: Dataset selection and batching.
        model: Architecture.
        optim: Optimiser and schedule.
        train: The loop.
        eval: Validation decoding and metrics.
        tracking: Experiment tracker.
        checkpoint: Retention policy.
        gates: Declared approval-gate context.
    """

    schema_version: int = Field(default=1, ge=1)
    run: RunCfg = Field(default_factory=RunCfg)
    paths: PathsCfg = Field(default_factory=PathsCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    model: ModelCfg = Field(default_factory=ModelCfg)
    optim: OptimCfg = Field(default_factory=OptimCfg)
    train: LoopCfg = Field(default_factory=LoopCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)
    tracking: TrackingCfg = Field(default_factory=TrackingCfg)
    checkpoint: CheckpointCfg = Field(default_factory=CheckpointCfg)
    gates: GatesCfg = Field(default_factory=GatesCfg)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_layout(cls, data: Any) -> Any:
        """Move the pre-schema keys of ``training/configs/_base.yaml`` to their canonical place.

        The design document is normative, so ``seed``/``deterministic`` belong to ``run:``,
        ``grad_clip``/``amp`` to ``train:`` and the patience field is ``early_stop_patience``.
        The committed base file predates the schema and writes them elsewhere; rather than break
        it, the old placement is migrated here (and logged), never overriding a value the config
        already states in the canonical place. A key that cannot be migrated is left alone so that
        ``extra="forbid"`` names it.
        """
        if not isinstance(data, dict):
            return data
        out: dict[str, Any] = dict(data)
        for section in ("run", "optim", "train", "model", "data"):
            if isinstance(out.get(section), dict):
                out[section] = dict(out[section])
        moved: list[str] = []

        for key in LEGACY_TOP_LEVEL_RUN_KEYS:
            if key in out:
                _relocate(out, "run", key, out.pop(key))
                moved.append(key)
        optim = out.get("optim")
        if isinstance(optim, dict):
            for key in LEGACY_OPTIM_TO_TRAIN_KEYS:
                if key in optim:
                    _relocate(out, "train", key, optim.pop(key))
                    moved.append(f"optim.{key}")
        train = out.get("train")
        if isinstance(train, dict):
            for old, new in LEGACY_TRAIN_RENAMES.items():
                if old in train:
                    train.setdefault(new, train.pop(old))
                    moved.append(f"train.{old}")
        model = out.get("model")
        if isinstance(model, dict) and "kind" in model:
            kind = model["kind"]
            family = LEGACY_MODEL_KINDS.get(str(kind)) if kind is not None else None
            if kind is None or family is not None:
                model.pop("kind")
                if family is not None:
                    model.setdefault("family", family)
                moved.append("model.kind")
        data_block = out.get("data")
        if isinstance(data_block, dict):
            name = data_block.get("dataset")
            if isinstance(name, str) and name in LEGACY_DATASET_NAMES:
                data_block["dataset"] = LEGACY_DATASET_NAMES[name]
                moved.append(f"data.dataset={name}")

        if moved:
            _log.warning("ml.config.legacy_keys_migrated", keys=moved)
        return out

    @model_validator(mode="after")
    def _warn_on_a_false_synthetic_claim(self) -> TrainConfig:  # noqa: N804 - takes self
        if self.gates.synthetic and not self.data.dataset.startswith("synthetic_"):
            _log.warning(
                "ml.config.gates_synthetic_ignored",
                dataset=self.data.dataset,
                reason="only a synthetic_* dataset is gate-free (ADR-0012)",
            )
        return self

    @property
    def is_gate_free(self) -> bool:
        """True when this run needs no ``TRAINING`` approval, i.e. it uses synthetic data only.

        The dataset name is the only input. ``gates.synthetic`` is a declaration and is ignored
        here on purpose: a wrong declaration must not be able to open a gate (ADR-0012).
        """
        return self.data.dataset.startswith("synthetic_")

    def to_yaml(self) -> str:
        """Render the resolved config as YAML (what a run directory stores as ``config.yaml``)."""
        text = yaml.safe_dump(
            resolved_dict(self), sort_keys=False, allow_unicode=True, default_flow_style=False
        )
        return str(text)


def _relocate(tree: dict[str, Any], section: str, key: str, value: Any) -> None:
    """Put ``value`` at ``tree[section][key]`` unless the section already states it."""
    block = tree.get(section)
    if not isinstance(block, dict):
        block = {}
        tree[section] = block
    block.setdefault(key, value)


# ----------------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------------
def _read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML mapping.

    Raises:
        NotFoundError: If ``path`` does not exist.
        InputValidationError: If the file is not valid YAML or is not a mapping.
    """
    if not path.is_file():
        raise NotFoundError(
            messages_tr.msg("FILE_NOT_FOUND", path=str(path)),
            details={"reason": "config_file_missing", "path": str(path)},
        )
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "config_not_yaml", "path": str(path), "error": str(exc)},
        ) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={
                "reason": "config_not_a_mapping",
                "path": str(path),
                "type": type(loaded).__name__,
            },
        )
    return dict(loaded)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``override`` onto ``base``: dictionaries merge, scalars and lists replace."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _base_paths(value: Any, child: Path) -> list[Path]:
    """Resolve a ``_base_`` entry (a string or a list of strings) against the child's directory."""
    items: Iterable[Any] = [value] if isinstance(value, str) else value
    if isinstance(value, Mapping) or not isinstance(items, Iterable):
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "config_base_type", "path": str(child), "value": repr(value)},
        )
    return [child.parent / str(item) for item in items]


def _load_tree(path: Path, chain: tuple[Path, ...]) -> dict[str, Any]:
    """Read ``path`` and everything it inherits from, deep-merged parents first.

    Raises:
        InputValidationError: If the inheritance chain contains a cycle.
        NotFoundError: If a file in the chain does not exist.
    """
    resolved = path.resolve()
    if resolved in chain:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={
                "reason": "config_base_cycle",
                "path": str(resolved),
                "chain": [str(item) for item in chain],
            },
        )
    raw = _read_yaml(path)
    base_value = raw.pop(BASE_KEY, None)
    legacy_value = raw.pop(LEGACY_BASE_KEY, None)
    if base_value is None:
        base_value = legacy_value
    merged: dict[str, Any] = {}
    if base_value is not None:
        for base_path in _base_paths(base_value, path):
            merged = _deep_merge(merged, _load_tree(base_path, (*chain, resolved)))
    return _deep_merge(merged, raw)


def parse_override(text: str) -> tuple[str, Any]:
    """Parse one ``dotted.key=value`` override.

    The value is read with YAML scalar semantics (``true`` is a bool, ``null`` is ``None``,
    ``[a, b]`` is a list), with one repair: PyYAML's YAML 1.1 resolver calls ``1e-4`` a string,
    and the CLI contract says it is a float.

    Args:
        text: The ``key=value`` string as typed on the command line.

    Returns:
        ``(dotted_key, value)``.

    Raises:
        InputValidationError: If there is no ``=``, the key is empty or malformed, or the value is
            not parsable YAML.
    """
    if "=" not in text:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "override_syntax", "override": text, "expected": "key=value"},
        )
    key, _, raw_value = text.partition("=")
    key = key.strip()
    if not key or any(not part for part in key.split(".")):
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "override_key", "override": text},
        )
    stripped = raw_value.strip()
    try:
        value = yaml.safe_load(stripped) if stripped else None
    except yaml.YAMLError as exc:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "override_value", "override": text, "error": str(exc)},
        ) from exc
    if isinstance(value, str) and _NUMBER_RE.fullmatch(stripped):
        value = float(stripped) if re.search(r"[.eE]", stripped) else int(stripped)
    return key, value


def _apply_override(tree: dict[str, Any], text: str) -> None:
    """Set one dotted key in ``tree`` in place, creating intermediate mappings as needed."""
    key, value = parse_override(text)
    parts = key.split(".")
    node: dict[str, Any] = tree
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _resolve_variable(name: str) -> str:
    """Value of ``${name}``: a project root first, then the process environment.

    Raises:
        InputValidationError: If the name is neither a project root nor an environment variable.
    """
    root = _PATH_VARIABLES.get(name)
    if root is not None:
        return root().as_posix()
    value = os.environ.get(name)
    if value is None:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={
                "reason": "unknown_config_variable",
                "variable": name,
                "known": sorted(_PATH_VARIABLES),
            },
        )
    return value


def expand_variables(value: Any) -> Any:
    """Recursively expand ``${VAR}`` in every string of a config tree.

    Args:
        value: A mapping, sequence or scalar read from YAML.

    Returns:
        The same structure with every ``${VAR}`` replaced.

    Raises:
        InputValidationError: If a referenced variable is unknown.
    """
    if isinstance(value, Mapping):
        return {key: expand_variables(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_variables(item) for item in value]
    if isinstance(value, str):
        return _VARIABLE_RE.sub(lambda match: _resolve_variable(match.group(1)), value)
    return value


def build_config(data: Mapping[str, Any], *, source: Path | None = None) -> TrainConfig:
    """Validate an already-merged mapping into a :class:`TrainConfig`.

    Args:
        data: The resolved configuration tree.
        source: File the tree came from, recorded in the error details.

    Returns:
        The validated configuration.

    Raises:
        InputValidationError: With the offending dotted key(s) in ``details["keys"]``.
    """
    try:
        return TrainConfig.model_validate(dict(data))
    except ValidationError as exc:
        raise _as_validation_error(exc, source) from exc


def _as_validation_error(exc: ValidationError, source: Path | None) -> InputValidationError:
    """Translate a pydantic error into the project's error type, naming every offending key."""
    problems = [
        {
            "key": ".".join(str(part) for part in error["loc"]) or "<root>",
            "type": str(error["type"]),
            "message": str(error["msg"]),
        }
        for error in exc.errors()
    ]
    details: dict[str, Any] = {
        "reason": "config_invalid",
        "keys": [problem["key"] for problem in problems],
        "errors": problems,
    }
    if source is not None:
        details["path"] = str(source)
    _log.warning("ml.config.invalid", keys=details["keys"], path=details.get("path"))
    return InputValidationError(messages_tr.MSG_VALIDATION_ERROR, details=details)


def load_config(path: Path | str, overrides: Sequence[str] = ()) -> TrainConfig:
    """Load a training configuration from YAML.

    The file may inherit from other files with ``_base_`` (relative to itself); the chain is
    deep-merged parents first, then the CLI overrides are applied, then ``${VAR}`` references are
    expanded, then the whole tree is validated.

    Args:
        path: The task configuration file.
        overrides: ``dotted.key=value`` strings, applied in order after inheritance.

    Returns:
        The validated, fully resolved configuration.

    Raises:
        NotFoundError: If the file (or one of its bases) does not exist.
        InputValidationError: On a cycle, a malformed override, an unknown ``${VAR}`` or any
            schema violation — the offending dotted key is named in ``details``.
    """
    file_path = Path(path)
    tree = _load_tree(file_path, ())
    for override in overrides:
        _apply_override(tree, override)
    expanded = expand_variables(tree)
    cfg = build_config(expanded, source=file_path)
    _log.info(
        "ml.config.loaded",
        path=str(file_path),
        run=cfg.run.name,
        dataset=cfg.data.dataset,
        family=cfg.model.family,
        overrides=len(overrides),
        gate_free=cfg.is_gate_free,
    )
    return cfg


def resolved_dict(cfg: TrainConfig) -> dict[str, Any]:
    """JSON-safe dump of the resolved configuration (paths become strings, tuples become lists)."""
    return cfg.model_dump(mode="json")


def config_hash(cfg: TrainConfig) -> str:
    """Stable SHA-256 (64 hex characters) of the resolved configuration.

    The hash is taken over the canonical JSON form (:func:`aicls.core.hashing.sha256_json`), so
    the order in which keys were written in the YAML file cannot change it.

    ``run.name`` and ``run.notes`` are excluded: two runs that differ only by their name or by a
    note are the *same experiment*, and the report needs them to collide so that a re-run can be
    recognised as a repetition. ``run.tags`` is **not** excluded — an ablation tag is part of the
    experiment's identity.

    The dump contains resolved absolute paths, so the same file hashes differently on two machines
    with different ``AICLS_DATA_ROOT`` values; that is intended, because the data location is part
    of what was actually run.

    Args:
        cfg: A validated configuration.

    Returns:
        The hex digest, without a ``sha256:`` prefix.
    """
    payload = resolved_dict(cfg)
    for dotted in HASH_EXCLUDED_KEYS:
        section, _, field = dotted.partition(".")
        block = payload.get(section)
        if isinstance(block, dict):
            block.pop(field, None)
    return sha256_json(payload)


__all__ = [
    "BASE_KEY",
    "DATASET_NAMES",
    "DEFAULT_MAX_LEN",
    "DEFAULT_SEED",
    "HASH_EXCLUDED_KEYS",
    "METRIC_NAMES",
    "MODEL_FAMILIES",
    "AmpMode",
    "AugmentCfg",
    "CheckpointCfg",
    "DataCfg",
    "DataMode",
    "DatasetName",
    "EvalCfg",
    "GatesCfg",
    "LoopCfg",
    "LoraCfg",
    "ModelCfg",
    "ModelFamily",
    "OptimCfg",
    "PathsCfg",
    "RunCfg",
    "SpecAugmentCfg",
    "TrackingCfg",
    "TrainConfig",
    "build_config",
    "config_hash",
    "expand_variables",
    "load_config",
    "parse_override",
    "resolved_dict",
]
