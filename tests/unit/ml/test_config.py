"""Tests for the typed training configuration (docs/ml/egitim-altyapisi.md §1, ADR-0006).

The two shapes that must load are (a) the worked example printed in the design document and
(b) the ``training/configs/_base.yaml`` file already committed to the repository. They are read
here from disk rather than restated, so a later rewrite of ``_base.yaml`` is caught by this test
instead of by a training run.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aicls.core.errors import InputValidationError, NotFoundError
from aicls.core.settings import settings_override
from aicls.ml.config import (
    DATASET_NAMES,
    DEFAULT_SEED,
    MODEL_FAMILIES,
    DataCfg,
    ModelCfg,
    TrainConfig,
    build_config,
    config_hash,
    load_config,
    resolved_dict,
)

# The design document's worked OCR example, verbatim (§1). It inherits from the committed
# _base.yaml, which is copied next to it by the fixture below.
DOC_OCR_YAML = """
_base_: ../_base.yaml
run:   {name: ocr_offline_resnet34_tfm, seed: 1234, deterministic: true, tags: [ocr, offline]}
data:
  dataset: mathwriting
  root: ${AICLS_DATA_ROOT}/mathwriting/shards
  mode: offline
  image_size: [96, 384]
  preprocess: [grayscale, normalize01, resize_pad]
  synthetic_ratio: 0.5
  synthetic_warmup_epochs: 2
  augment: {affine_jitter: 0.05, stroke_dropout: 0.02}
  max_label_len: 160
  batch_size: 64
  num_workers: 6
model:
  family: seq2seq_image
  encoder: resnet34
  decoder: transformer
  d_model: 256
  layers: 6
  heads: 8
  dropout: 0.1
  vocab_path: ${AICLS_DATA_ROOT}/mathwriting/vocab.json
optim: {name: adamw, lr: 3.0e-4, weight_decay: 0.01, betas: [0.9, 0.98], scheduler: cosine, warmup_steps: 2000}
train: {epochs: 30, max_steps: null, grad_accum: 1, amp: bf16, grad_clip: 1.0, eval_every: 2000, ckpt_every: 2000, early_stop_patience: 5, allow_dirty: false}
eval:  {beam: 4, max_n_val: 2000, metrics: [cer, ter, exprate, symeq]}
tracking: {backend: jsonl, mlflow_uri: null, experiment: ocr}
"""

# The Whisper variant described in prose right below the OCR example (§1).
DOC_ASR_YAML = """
run: {name: asr_whisper_small_tr, seed: 1234, tags: [asr]}
data:
  dataset: audio_commands
  manifest: ${AICLS_DATA_ROOT}/manifests/audio_commands.jsonl
  splits_path: ${AICLS_DATA_ROOT}/manifests/audio_commands_splits.json
  feature_extractor: whisper
  batch_size: 32
  augment:
    spec_augment: {time_masks: 2, freq_masks: 2}
    pitch_semitones: [-2, 2]
    noise_snr_db: [10, 25]
    speed: [0.9, 1.1]
    tts_ratio: 0.0
model:
  family: whisper
  base_dir: ${AICLS_MODELS_ROOT}/pretrained/whisper-small
  lora: {enabled: false, r: 16, alpha: 32, targets: [q_proj, v_proj]}
  freeze_encoder: false
train: {max_steps: 4000, amp: bf16}
eval: {metrics: [wer]}
tracking: {backend: jsonl, experiment: asr}
"""

MINIMAL_YAML = """
run: {name: smoke, seed: 7}
data: {dataset: synthetic_ink, batch_size: 4, n_samples: 8}
model: {family: seq2seq_image, encoder: cnn_small, d_model: 32, layers: 2, heads: 2}
train: {max_steps: 2}
"""


#: Fixed, machine-independent roots for the ``${AICLS_DATA_ROOT}`` expansion tests. They have to
#: be **absolute on the platform the tests run on**: ``Path("D:/aicls-data")`` is absolute on
#: Windows but a single *relative* component named "D:" on Linux, where
#: ``Settings._absolute_data_root`` resolves it against the working directory -- so the expected
#: value and the actual value stopped agreeing the first time the suite ran in a container
#: (2026-09-08). What is asserted is unchanged: an expanded path is the root joined with the
#: literal tail from the YAML.
_ABSOLUTE_PREFIX = Path("D:/") if os.name == "nt" else Path("/srv")
DATA_ROOT = _ABSOLUTE_PREFIX / "aicls-data"
MODELS_ROOT = _ABSOLUTE_PREFIX / "aicls-models"


def test_the_fixture_roots_are_absolute_on_this_platform() -> None:
    """The premise the ``${AICLS_DATA_ROOT}`` assertions rest on, asserted rather than assumed."""
    assert DATA_ROOT.is_absolute()
    assert MODELS_ROOT.is_absolute()


def _write(path: Path, text: str) -> Path:
    """Write ``text`` as UTF-8 with LF endings and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


@pytest.fixture
def doc_example(tmp_path: Path, repo_root: Path) -> Path:
    """The design-document OCR config, inheriting the committed ``training/configs/_base.yaml``."""
    committed = (repo_root / "training" / "configs" / "_base.yaml").read_text(encoding="utf-8")
    _write(tmp_path / "_base.yaml", committed)
    return _write(tmp_path / "ocr" / "offline_resnet34_tfm.yaml", DOC_OCR_YAML)


# ------------------------------------------------------------------------------------------
# schema
# ------------------------------------------------------------------------------------------
def test_train_config_defaults_are_a_complete_tree() -> None:
    cfg = TrainConfig()

    assert cfg.run.seed == DEFAULT_SEED
    assert cfg.run.deterministic is True
    assert cfg.data.dataset in DATASET_NAMES
    assert cfg.model.family in MODEL_FAMILIES
    assert cfg.model.encoder == "cnn_small"  # filled in for the image family
    assert cfg.optim.name == "adamw"
    assert cfg.train.amp == "off"  # CPU default; bf16 only after the CUDA_INSTALL gate
    assert cfg.eval.metrics == ("cer",)
    assert cfg.tracking.backend == "jsonl"
    assert cfg.is_gate_free is True  # synthetic_ink needs no TRAINING approval


def test_train_config_rejects_unknown_key(tmp_path: Path) -> None:
    path = _write(tmp_path / "typo.yaml", "data: {batch_sizee: 4}\n")

    with pytest.raises(InputValidationError) as excinfo:
        load_config(path)

    assert "data.batch_sizee" in excinfo.value.details["keys"]
    assert excinfo.value.details["reason"] == "config_invalid"

    # ... and the same holds for a typo introduced by a CLI override.
    good = _write(tmp_path / "good.yaml", MINIMAL_YAML)
    with pytest.raises(InputValidationError) as override_info:
        load_config(good, ["optim.learning_rate=1e-4"])
    assert "optim.learning_rate" in override_info.value.details["keys"]


def test_train_config_override_dotted_path(tmp_path: Path) -> None:
    path = _write(tmp_path / "cfg.yaml", MINIMAL_YAML)

    cfg = load_config(
        path,
        [
            "optim.lr=1e-4",  # YAML 1.1 would call this a string; the loader must not
            "train.max_steps=2",
            "run.deterministic=false",
            "train.early_stop_patience=null",
            "eval.metrics=[cer, wer]",
            "run.name=overridden",
        ],
    )

    assert isinstance(cfg.optim.lr, float)
    assert cfg.optim.lr == pytest.approx(1e-4)
    assert cfg.train.max_steps == 2
    assert cfg.run.deterministic is False
    assert cfg.train.early_stop_patience is None
    assert cfg.eval.metrics == ("cer", "wer")
    assert cfg.run.name == "overridden"


def test_override_without_an_equals_sign_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "cfg.yaml", MINIMAL_YAML)

    with pytest.raises(InputValidationError) as excinfo:
        load_config(path, ["optim.lr"])

    assert excinfo.value.details["reason"] == "override_syntax"


# ------------------------------------------------------------------------------------------
# _base_ inheritance
# ------------------------------------------------------------------------------------------
def test_train_config_base_inheritance(tmp_path: Path) -> None:
    _write(
        tmp_path / "root.yaml",
        "run: {name: root, tags: [a, b]}\n"
        "data: {dataset: synthetic_ink, batch_size: 4, num_workers: 3}\n"
        "optim: {lr: 3.0e-4, warmup_steps: 100}\n",
    )
    _write(
        tmp_path / "mid" / "mid.yaml",
        "_base_: ../root.yaml\ndata: {batch_size: 8}\noptim: {warmup_steps: 5}\n",
    )
    child = _write(
        tmp_path / "mid" / "child.yaml",
        "_base_: mid.yaml\nrun: {name: child, tags: [c]}\ndata: {num_workers: 0}\n",
    )

    cfg = load_config(child)

    assert cfg.run.name == "child"  # scalars replace
    assert cfg.run.tags == ("c",)  # lists replace, they never concatenate
    assert cfg.data.batch_size == 8  # dicts merge across the whole chain
    assert cfg.data.dataset == "synthetic_ink"
    assert cfg.data.num_workers == 0
    assert cfg.optim.warmup_steps == 5
    assert cfg.optim.lr == pytest.approx(3e-4)


def test_load_config_refuses_a_base_cycle(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "_base_: b.yaml\nrun: {name: a}\n")
    _write(tmp_path / "b.yaml", "_base_: a.yaml\nrun: {name: b}\n")

    with pytest.raises(InputValidationError) as excinfo:
        load_config(tmp_path / "a.yaml")

    assert excinfo.value.details["reason"] == "config_base_cycle"


def test_load_config_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError):
        load_config(tmp_path / "absent.yaml")

    _write(tmp_path / "child.yaml", "_base_: nowhere.yaml\n")
    with pytest.raises(NotFoundError):
        load_config(tmp_path / "child.yaml")


def test_a_base_may_be_a_list_merged_left_to_right(tmp_path: Path) -> None:
    _write(tmp_path / "one.yaml", "data: {batch_size: 4, num_workers: 2}\n")
    _write(tmp_path / "two.yaml", "data: {batch_size: 16}\n")
    child = _write(tmp_path / "child.yaml", "_base_: [one.yaml, two.yaml]\nrun: {name: c}\n")

    cfg = load_config(child)

    assert cfg.data.batch_size == 16
    assert cfg.data.num_workers == 2


def test_load_config_reports_a_file_that_is_not_a_yaml_mapping(tmp_path: Path) -> None:
    broken = _write(tmp_path / "broken.yaml", "run: {name: [unclosed\n")
    with pytest.raises(InputValidationError) as yaml_error:
        load_config(broken)
    assert yaml_error.value.details["reason"] == "config_not_yaml"

    listy = _write(tmp_path / "list.yaml", "- one\n- two\n")
    with pytest.raises(InputValidationError) as shape_error:
        load_config(listy)
    assert shape_error.value.details["reason"] == "config_not_a_mapping"

    empty = _write(tmp_path / "empty.yaml", "# nothing but a comment\n")
    assert load_config(empty).run.seed == DEFAULT_SEED


def test_load_config_refuses_a_base_that_is_not_a_path(tmp_path: Path) -> None:
    path = _write(tmp_path / "cfg.yaml", "_base_: 7\n")

    with pytest.raises(InputValidationError) as excinfo:
        load_config(path)

    assert excinfo.value.details["reason"] == "config_base_type"


def test_override_with_an_empty_key_segment_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path / "cfg.yaml", MINIMAL_YAML)

    with pytest.raises(InputValidationError) as excinfo:
        load_config(path, ["optim..lr=1e-4"])

    assert excinfo.value.details["reason"] == "override_key"


def test_legacy_model_kind_is_migrated_and_an_unknown_kind_is_named(tmp_path: Path) -> None:
    migrated = load_config(_write(tmp_path / "old.yaml", "model: {kind: asr.whisper_ft}\n"))
    assert migrated.model.family == "whisper"

    with pytest.raises(InputValidationError) as excinfo:
        load_config(_write(tmp_path / "bad.yaml", "model: {kind: asr.deepspeech}\n"))
    assert "model.kind" in excinfo.value.details["keys"]


# ------------------------------------------------------------------------------------------
# the two shapes that must load
# ------------------------------------------------------------------------------------------
def test_committed_base_yaml_loads(repo_root: Path, tmp_path: Path) -> None:
    """``training/configs/_base.yaml`` uses the pre-schema layout; it must still load."""
    with settings_override(data_root=tmp_path, models_root=tmp_path / "models"):
        cfg = load_config(repo_root / "training" / "configs" / "_base.yaml")

    # Legacy placement is migrated into the canonical (design-document) location.
    assert cfg.run.seed == 1234  # top-level `seed:` -> run.seed
    assert cfg.run.deterministic is True  # top-level `deterministic:` -> run.deterministic
    assert cfg.train.amp == "off"  # optim.amp: false -> train.amp
    assert cfg.train.grad_clip == pytest.approx(1.0)  # optim.grad_clip -> train.grad_clip
    assert cfg.train.early_stop_patience is None  # train.early_stopping_patience -> ...
    assert cfg.data.dataset == "synthetic_ink"  # legacy alias of `synthetic_ocr`
    assert cfg.checkpoint.keep_last == 2
    assert cfg.paths.data_root == tmp_path
    assert cfg.is_gate_free is True


def test_design_document_example_loads_on_top_of_the_committed_base(doc_example: Path) -> None:
    with settings_override(data_root=DATA_ROOT, models_root=MODELS_ROOT):
        cfg = load_config(doc_example)

    assert cfg.run.name == "ocr_offline_resnet34_tfm"
    assert cfg.run.tags == ("ocr", "offline")
    assert cfg.data.dataset == "mathwriting"
    assert cfg.data.root == DATA_ROOT / "mathwriting" / "shards"
    assert cfg.data.mode == "offline"
    assert cfg.data.image_size == (96, 384)
    assert cfg.data.preprocess == ("grayscale", "normalize01", "resize_pad")
    assert cfg.data.synthetic_ratio == pytest.approx(0.5)
    assert cfg.data.synthetic_warmup_epochs == 2
    assert cfg.data.augment.affine_jitter == pytest.approx(0.05)
    assert cfg.data.num_workers == 6
    assert cfg.model.encoder == "resnet34"
    assert cfg.model.vocab_path == DATA_ROOT / "mathwriting" / "vocab.json"
    assert cfg.optim.betas == (0.9, 0.98)
    assert cfg.optim.warmup_steps == 2000
    assert cfg.train.epochs == 30
    assert cfg.train.max_steps is None  # the child's explicit null replaces the base's 2
    assert cfg.train.amp == "bf16"  # the child wins over the base's optim.amp: false
    assert cfg.train.early_stop_patience == 5
    assert cfg.eval.beam == 4
    assert cfg.eval.metrics == ("cer", "ter", "exprate", "symeq")
    assert cfg.tracking.experiment == "ocr"
    # A real MathWriting run is gated even though _base.yaml declares gates.synthetic: true.
    assert cfg.is_gate_free is False


def test_design_document_whisper_example_loads(tmp_path: Path) -> None:
    path = _write(tmp_path / "asr.yaml", DOC_ASR_YAML)

    with settings_override(data_root=DATA_ROOT, models_root=MODELS_ROOT):
        cfg = load_config(path)

    assert cfg.model.family == "whisper"
    assert cfg.model.base_dir == MODELS_ROOT / "pretrained" / "whisper-small"
    assert cfg.model.encoder is None
    assert cfg.model.lora.enabled is False
    assert cfg.model.lora.targets == ("q_proj", "v_proj")
    assert cfg.data.feature_extractor == "whisper"
    assert cfg.data.augment.spec_augment is not None
    assert cfg.data.augment.spec_augment.time_masks == 2
    assert cfg.data.augment.noise_snr_db == (10.0, 25.0)
    assert cfg.eval.metrics == ("wer",)


# ------------------------------------------------------------------------------------------
# variable expansion
# ------------------------------------------------------------------------------------------
def test_config_expands_data_and_models_root(tmp_path: Path) -> None:
    # Block style, like the design document: `${` inside a YAML *flow* mapping is a syntax error.
    path = _write(
        tmp_path / "cfg.yaml",
        "data:\n"
        "  dataset: mathwriting\n"
        "  root: ${AICLS_DATA_ROOT}/mathwriting/shards\n"
        "model:\n"
        "  family: whisper\n"
        "  base_dir: ${AICLS_MODELS_ROOT}/pretrained/whisper-small\n",
    )

    with settings_override(data_root=tmp_path / "data", models_root=tmp_path / "models"):
        cfg = load_config(path)

    assert cfg.data.root == tmp_path / "data" / "mathwriting" / "shards"
    assert cfg.model.base_dir == tmp_path / "models" / "pretrained" / "whisper-small"


def test_config_expands_variables_inside_overrides(tmp_path: Path) -> None:
    path = _write(tmp_path / "cfg.yaml", MINIMAL_YAML)

    with settings_override(data_root=tmp_path / "data"):
        cfg = load_config(path, ["data.root=${AICLS_DATA_ROOT}/shards"])

    assert cfg.data.root == tmp_path / "data" / "shards"


def test_config_refuses_an_unknown_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AICLS_NOT_A_REAL_VARIABLE", raising=False)
    path = _write(tmp_path / "cfg.yaml", "data:\n  root: ${AICLS_NOT_A_REAL_VARIABLE}/x\n")

    with pytest.raises(InputValidationError) as excinfo:
        load_config(path)

    assert excinfo.value.details["variable"] == "AICLS_NOT_A_REAL_VARIABLE"


# ------------------------------------------------------------------------------------------
# hashing
# ------------------------------------------------------------------------------------------
def test_config_hash_is_stable_under_key_reordering() -> None:
    first = build_config(
        {
            "run": {"name": "a", "seed": 1234},
            "data": {"dataset": "synthetic_ink", "batch_size": 4},
            "optim": {"lr": 0.0003, "name": "adamw"},
        }
    )
    second = build_config(
        {
            "optim": {"name": "adamw", "lr": 0.0003},
            "data": {"batch_size": 4, "dataset": "synthetic_ink"},
            "run": {"seed": 1234, "name": "a"},
        }
    )

    assert config_hash(first) == config_hash(second)
    assert len(config_hash(first)) == 64


def test_config_hash_ignores_run_name_and_notes() -> None:
    base = build_config({"run": {"name": "run-a", "notes": "first attempt"}})
    renamed = build_config({"run": {"name": "run-b", "notes": "second attempt"}})
    retagged = build_config({"run": {"name": "run-a", "tags": ["ocr"]}})

    assert config_hash(base) == config_hash(renamed)
    assert config_hash(base) != config_hash(retagged)  # tags DO identify the experiment
    assert resolved_dict(base)["run"]["name"] == "run-a"  # the dump itself keeps everything


def test_config_hash_changes_with_any_substantive_field() -> None:
    base = build_config({"optim": {"lr": 0.0003}})
    other = build_config({"optim": {"lr": 0.0004}})

    assert config_hash(base) != config_hash(other)


def test_to_yaml_round_trips(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path / "cfg.yaml", MINIMAL_YAML))

    reloaded = load_config(_write(tmp_path / "again.yaml", cfg.to_yaml()))

    assert config_hash(reloaded) == config_hash(cfg)


# ------------------------------------------------------------------------------------------
# per-field rules
# ------------------------------------------------------------------------------------------
def test_model_config_rejects_an_encoder_from_the_wrong_family() -> None:
    with pytest.raises(ValueError, match="encoder"):
        ModelCfg(family="seq2seq_image", encoder="stroke_bilstm")

    with pytest.raises(ValueError, match="encoder"):
        ModelCfg(family="whisper", encoder="resnet34")

    stroke = ModelCfg(family="seq2seq_stroke")
    assert stroke.encoder == "stroke_transformer"


def test_model_config_requires_heads_to_divide_d_model() -> None:
    with pytest.raises(ValueError, match="d_model"):
        ModelCfg(d_model=32, heads=5)


def test_model_config_refuses_pretrained_torchvision_weights() -> None:
    """ADR-0012: torchvision backbones are always ``weights=None``."""
    with pytest.raises(ValueError):
        ModelCfg(weights="IMAGENET1K_V1")


def test_data_config_rejects_an_unknown_preprocess_step() -> None:
    with pytest.raises(ValueError, match="preprocess"):
        DataCfg(preprocess=["grayscale", "sharpen_a_lot"])


def test_eval_config_rejects_an_unknown_metric() -> None:
    with pytest.raises(InputValidationError) as excinfo:
        build_config({"eval": {"metrics": ["cer", "bleu"]}})

    assert "eval.metrics" in excinfo.value.details["keys"]


def test_amp_accepts_the_two_spellings() -> None:
    assert build_config({"train": {"amp": False}}).train.amp == "off"
    assert build_config({"train": {"amp": True}}).train.amp == "bf16"
    assert build_config({"train": {"amp": "bf16"}}).train.amp == "bf16"
    with pytest.raises(InputValidationError):
        build_config({"train": {"amp": "int4"}})


def test_gate_freedom_follows_the_dataset_not_the_declaration() -> None:
    """``gates.synthetic`` may never widen the TRAINING gate (ADR-0012)."""
    lying = build_config({"data": {"dataset": "mathwriting"}, "gates": {"synthetic": True}})
    honest = build_config({"data": {"dataset": "synthetic_audio"}, "gates": {"synthetic": False}})

    assert lying.is_gate_free is False
    assert honest.is_gate_free is True


def test_to_preprocess_config_carries_the_image_size() -> None:
    data = DataCfg(mode="offline", image_size=(32, 64), preprocess=["grayscale", "resize_pad"])

    pre = data.to_preprocess_config()

    assert pre.size == (32, 64)
    assert tuple(pre.steps) == ("grayscale", "resize_pad")
