"""Every committed training configuration must load and validate (docs/ml/egitim-altyapisi.md §1).

A typo in a YAML file is otherwise discovered by a six-hour GPU run rather than by CI, so this
module walks ``training/configs/**/*.yaml`` and pushes every file through
:func:`aicls.ml.config.load_config`. It additionally pins the four properties that make the tree
safe to leave lying around while every approval gate (ADR-0012) is closed:

* only the two ``smoke_synthetic.yaml`` files are gate-free (``data.dataset: synthetic_*``);
* every other file carries a ``# GATE:`` header naming the approvals it needs;
* no file ever names a Hugging Face hub id — ``model.base_dir`` is a local directory under
  ``AICLS_MODELS_ROOT`` and ``model.weights`` is always ``None``;
* the learning-rate sweep publishes a machine-readable grid whose every point still validates.

Nothing here trains, downloads or touches the network: ``load_config`` only parses and validates,
and the ``${AICLS_DATA_ROOT}`` paths it expands are never opened.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]  # PyYAML ships no stubs and types-PyYAML is absent

from aicls.core.settings import settings_override
from aicls.ml.config import TrainConfig, config_hash, load_config, resolved_dict

# tests/unit/ml/test_configs_validate.py -> repository root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_ROOT = _REPO_ROOT / "training" / "configs"
_CONFIGS: tuple[str, ...] = tuple(
    sorted(p.relative_to(_CONFIG_ROOT).as_posix() for p in _CONFIG_ROOT.rglob("*.yaml"))
)

#: The tree promised by docs/mimari/genel-bakis.md and training/README.md.
PROMISED: tuple[str, ...] = (
    "_base.yaml",
    "asr/profile.yaml",
    "asr/smoke_synthetic.yaml",
    "asr/whisper_large_v3_turbo_lora.yaml",
    "asr/whisper_small_ft.yaml",
    "ocr/ablation_preproc.yaml",
    "ocr/offline_effnet_b0_tfm.yaml",
    "ocr/offline_resnet34_tfm.yaml",
    "ocr/online_stroke_tfm.yaml",
    "ocr/profile.yaml",
    "ocr/smoke_synthetic.yaml",
    "sweeps/ocr_lr_sweep.yaml",
)

#: Configs that need no approval at all: synthetic data, CPU, two steps.
GATE_FREE: tuple[str, ...] = ("ocr/smoke_synthetic.yaml", "asr/smoke_synthetic.yaml")

#: ``_base.yaml`` is an inheritance fragment, not a runnable task, so it declares no gate.
NOT_A_TASK: tuple[str, ...] = ("_base.yaml",)

GATE_NAMES = frozenset({"DATASET_DOWNLOAD", "WEIGHTS_DOWNLOAD", "CUDA_INSTALL", "TRAINING"})
GATE_HEADER_RE = re.compile(r"^#\s*GATE:\s*(?P<gates>[A-Z_0-9 ,+]+)")
SWEEP_HEADER_RE = re.compile(
    r"^#\s*AICLS-SWEEP:\s*(?P<key>[A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(?P<values>\[.*\])\s*$"
)

#: Fixed, machine-independent roots the resolved-path assertions are written against. They must
#: be **absolute on the platform the tests run on**, and that is the whole point of the pair:
#: ``Path("D:/aicls-data")`` is absolute on Windows but a single *relative* component named
#: "D:" on Linux, where ``Settings`` resolves it against the working directory. The
#: ``is_absolute()`` assertion below then passed for the wrong reason (the resolve made it
#: absolute) while ``MODELS_ROOT in value.parents`` compared an absolute path with a relative
#: one and failed. The guarantee under test -- a weights directory is an absolute local path
#: under ``AICLS_MODELS_ROOT`` -- is unchanged; only the spelling of "absolute" is per-platform.
_ABSOLUTE_PREFIX = Path("D:/") if os.name == "nt" else Path("/srv")
DATA_ROOT = _ABSOLUTE_PREFIX / "aicls-data"
MODELS_ROOT = _ABSOLUTE_PREFIX / "aicls-models"


def _load(rel_path: str, overrides: tuple[str, ...] = ()) -> TrainConfig:
    """Load one committed config under deterministic, machine-independent roots."""
    with settings_override(data_root=DATA_ROOT, models_root=MODELS_ROOT):
        return load_config(_CONFIG_ROOT / rel_path, overrides)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a resolved config dump to ``{"data.batch_size": 64, ...}``."""
    if not isinstance(value, dict):
        return {prefix: value}
    flat: dict[str, Any] = {}
    for key, item in value.items():
        flat.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    return flat


def _differing_keys(left: TrainConfig, right: TrainConfig) -> set[str]:
    """Dotted keys whose resolved values differ between two configs."""
    a, b = _flatten(resolved_dict(left)), _flatten(resolved_dict(right))
    return {key for key in a.keys() | b.keys() if a.get(key) != b.get(key)}


def _header_comment(rel_path: str) -> str:
    """The leading comment block of a config file (everything before the first YAML key)."""
    lines: list[str] = []
    for line in (_CONFIG_ROOT / rel_path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            lines.append(line)
            continue
        break
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# the whole tree
# ----------------------------------------------------------------------------------------------
def test_the_fixture_roots_are_absolute_on_this_platform() -> None:
    """The premise of every path assertion in this module, asserted rather than assumed.

    Without this, a root that is relative on the host under test degrades the assertions below
    into comparisons of two paths that were both silently resolved against the working
    directory -- which is exactly how ``D:/aicls-models`` went unnoticed until the suite first
    ran on Linux (2026-09-08).
    """
    assert DATA_ROOT.is_absolute()
    assert MODELS_ROOT.is_absolute()


def test_every_promised_config_is_committed() -> None:
    """The file list in docs/mimari/genel-bakis.md is the contract; nothing more, nothing less."""
    assert _CONFIGS == PROMISED


@pytest.mark.parametrize("rel_path", _CONFIGS)
def test_smoke_configs_load_and_validate(rel_path: str) -> None:
    """Every committed YAML validates against the schema and hashes reproducibly."""
    cfg = _load(rel_path)

    assert isinstance(cfg, TrainConfig)
    assert cfg.schema_version == 1
    digest = config_hash(cfg)
    assert len(digest) == 64
    assert digest == config_hash(_load(rel_path))  # loading twice is the same experiment


@pytest.mark.parametrize("rel_path", _CONFIGS)
def test_configs_are_utf8_lf_and_tab_free(rel_path: str) -> None:
    """Windows checkouts must not smuggle CRLF or tabs into a YAML file."""
    raw = (_CONFIG_ROOT / rel_path).read_bytes()

    raw.decode("utf-8")  # raises UnicodeDecodeError if the file is not UTF-8
    assert b"\r\n" not in raw
    assert b"\t" not in raw


@pytest.mark.parametrize("rel_path", _CONFIGS)
def test_no_config_names_a_hub_id_or_pretrained_weights(rel_path: str) -> None:
    """Weights come from a local directory only (ADR-0012, WEIGHTS_DOWNLOAD)."""
    cfg = _load(rel_path)

    assert cfg.model.weights is None
    for label, value in (
        ("base_dir", cfg.model.base_dir),
        ("pretrained", cfg.model.pretrained_path),
    ):
        if value is None:
            continue
        assert value.is_absolute(), f"model.{label} must be an absolute local path, got {value}"
        assert MODELS_ROOT in value.parents, f"model.{label} must live under AICLS_MODELS_ROOT"


def test_only_the_smoke_configs_are_gate_free() -> None:
    """``is_gate_free`` follows ``data.dataset`` alone; nothing else may claim to be gate-free."""
    gate_free = {rel for rel in _CONFIGS if _load(rel).is_gate_free}

    assert gate_free == {*GATE_FREE, *NOT_A_TASK}


@pytest.mark.parametrize("rel_path", [r for r in _CONFIGS if r not in (*GATE_FREE, *NOT_A_TASK)])
def test_gated_configs_carry_a_gate_header(rel_path: str) -> None:
    """A real-data config states, in its first lines, which approvals it needs."""
    header = _header_comment(rel_path)

    matches = [GATE_HEADER_RE.match(line) for line in header.splitlines()]
    found = [m for m in matches if m is not None]
    assert found, f"{rel_path} has no '# GATE: ...' header"
    declared = {name.strip() for m in found for name in re.split(r"[,+]", m.group("gates"))}
    declared.discard("")
    assert declared <= GATE_NAMES, f"{rel_path} declares unknown gate(s): {declared - GATE_NAMES}"
    assert "TRAINING" in declared, f"{rel_path} runs on real data and needs the TRAINING gate"


def test_base_config_never_declares_itself_synthetic() -> None:
    """Regression: ``gates.synthetic: true`` in the base leaked into every real-data child."""
    base = _load("_base.yaml")

    assert base.gates.synthetic is None
    assert base.train.amp == "off"  # CPU default; bf16 only after the CUDA_INSTALL gate
    assert base.run.seed == 1234
    assert base.data.num_workers == 0


# ----------------------------------------------------------------------------------------------
# OCR
# ----------------------------------------------------------------------------------------------
def test_offline_resnet34_reproduces_the_design_document() -> None:
    """The worked example printed in docs/ml/egitim-altyapisi.md §1, value for value."""
    cfg = _load("ocr/offline_resnet34_tfm.yaml")

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
    assert cfg.data.augment.stroke_dropout == pytest.approx(0.02)
    assert cfg.data.max_label_len == 160
    assert cfg.data.batch_size == 64
    assert cfg.data.num_workers == 6
    assert cfg.model.family == "seq2seq_image"
    assert cfg.model.encoder == "resnet34"
    assert cfg.model.decoder == "transformer"
    assert (cfg.model.d_model, cfg.model.layers, cfg.model.heads) == (256, 6, 8)
    assert cfg.model.vocab_path == DATA_ROOT / "mathwriting" / "vocab.json"
    assert cfg.optim.betas == (0.9, 0.98)
    assert cfg.optim.lr == pytest.approx(3e-4)
    assert cfg.optim.warmup_steps == 2000
    assert cfg.train.epochs == 30
    assert cfg.train.max_steps is None
    assert cfg.train.amp == "bf16"
    assert cfg.train.eval_every == 2000
    assert cfg.train.ckpt_every == 2000
    # 2026-09-09: patience 5 stopped a healthy run on validation-subset noise; see the config.
    assert cfg.train.early_stop_patience == 25
    assert cfg.train.allow_dirty is False
    assert cfg.eval.beam == 4
    assert cfg.eval.max_n_val == 2000
    assert cfg.eval.metrics == ("cer", "ter", "exprate", "symeq")
    assert cfg.tracking.backend == "jsonl"
    assert cfg.tracking.experiment == "ocr"
    assert cfg.is_gate_free is False


def test_effnet_b0_is_the_resnet34_run_with_one_encoder_swapped() -> None:
    """An encoder ablation must differ in the encoder and in nothing else that touches training."""
    resnet = _load("ocr/offline_resnet34_tfm.yaml")
    effnet = _load("ocr/offline_effnet_b0_tfm.yaml")

    assert effnet.model.encoder == "efficientnet_b0"
    assert _differing_keys(resnet, effnet) == {"run.name", "run.notes", "run.tags", "model.encoder"}


def test_ablation_preproc_adds_the_two_optional_filters() -> None:
    """Proposal A.2.1's Gaussian blur and Laplacian sharpen are config keys, not code branches."""
    resnet = _load("ocr/offline_resnet34_tfm.yaml")
    ablation = _load("ocr/ablation_preproc.yaml")

    assert ablation.data.preprocess == (
        "gaussian_blur",
        "laplacian_sharpen",
        "grayscale",
        "normalize01",
        "resize_pad",
    )
    assert _differing_keys(resnet, ablation) == {
        "run.name",
        "run.notes",
        "run.tags",
        "data.preprocess",
    }


def test_online_stroke_config_is_the_online_path() -> None:
    """ADR-0013's online branch: stroke features, no raster, the §8 batch envelope."""
    cfg = _load("ocr/online_stroke_tfm.yaml")

    assert cfg.model.family == "seq2seq_stroke"
    assert cfg.model.encoder == "stroke_transformer"
    assert cfg.model.decoder == "transformer"
    assert cfg.data.mode == "online"
    assert cfg.data.preprocess == ()  # nothing is rasterised on the online path
    assert cfg.data.batch_size == 128
    assert cfg.data.dataset == "mathwriting"


def test_ocr_smoke_is_tiny_cpu_and_needs_no_approval() -> None:
    """The config CI runs: 64 synthetic traces, two steps, no gate, no download, no CUDA."""
    cfg = _load("ocr/smoke_synthetic.yaml")

    assert cfg.is_gate_free is True
    assert cfg.data.dataset == "synthetic_ink"
    assert cfg.data.n_samples == 64
    assert cfg.data.num_workers == 0  # Windows has no fork
    assert cfg.model.d_model == 32
    assert cfg.model.layers == 2
    assert cfg.train.max_steps == 2
    assert cfg.train.amp == "off"
    assert cfg.model.base_dir is None
    assert cfg.data.root is None and cfg.data.manifest is None  # nothing to download


# ----------------------------------------------------------------------------------------------
# ASR
# ----------------------------------------------------------------------------------------------
def test_whisper_small_is_a_full_finetune_from_a_local_directory() -> None:
    """§8 envelope: batch 32, 30 s windows, bf16, LoRA off, weights from a local directory."""
    cfg = _load("asr/whisper_small_ft.yaml")

    assert cfg.model.family == "whisper"
    assert cfg.model.encoder is None
    assert cfg.model.base_dir == MODELS_ROOT / "pretrained" / "whisper-small"
    assert cfg.model.lora.enabled is False
    assert cfg.model.freeze_encoder is False
    assert cfg.data.dataset == "audio_commands"
    assert cfg.data.feature_extractor == "whisper"
    assert cfg.data.splits_path is not None  # a WER over a seen speaker is not a WER
    assert cfg.data.batch_size == 32
    assert cfg.data.max_seconds == pytest.approx(30.0)
    assert cfg.train.amp == "bf16"
    assert cfg.train.grad_accum == 1
    assert "wer" in cfg.eval.metrics
    assert cfg.data.augment.spec_augment is not None
    assert cfg.data.augment.spec_augment.time_masks == 2
    assert cfg.data.augment.pitch_semitones == (-2.0, 2.0)
    assert cfg.data.augment.noise_snr_db == (10.0, 25.0)
    assert cfg.data.augment.speed == (0.9, 1.1)


def test_whisper_large_turbo_is_lora_only() -> None:
    """§8 envelope: LoRA r=16, batch 16, grad_accum 2 — the reported-model candidate."""
    cfg = _load("asr/whisper_large_v3_turbo_lora.yaml")

    assert cfg.model.base_dir == MODELS_ROOT / "pretrained" / "whisper-large-v3-turbo"
    assert cfg.model.lora.enabled is True
    assert cfg.model.lora.r == 16
    assert cfg.model.lora.alpha == 32
    assert cfg.model.lora.targets == ("q_proj", "v_proj")
    assert cfg.data.batch_size == 16
    assert cfg.train.grad_accum == 2
    assert cfg.train.amp == "bf16"


def test_asr_smoke_is_a_random_tiny_whisper_on_synthetic_audio() -> None:
    """The config CI runs for the ASR path: no checkpoint, no download, two steps."""
    cfg = _load("asr/smoke_synthetic.yaml")

    assert cfg.is_gate_free is True
    assert cfg.data.dataset == "synthetic_audio"
    assert cfg.data.feature_extractor == "log_mel"  # the numpy path; no HF processor needed
    assert cfg.data.n_samples == 64
    assert cfg.data.num_workers == 0
    assert cfg.model.family == "whisper"
    assert cfg.model.base_dir is None  # randomly initialised tiny WhisperConfig
    assert cfg.model.d_model == 32
    assert cfg.model.layers == 2
    assert cfg.model.num_mel_bins == 80
    assert cfg.train.max_steps == 2
    assert cfg.train.amp == "off"


# ----------------------------------------------------------------------------------------------
# the gate-free pair actually runs
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rel_path", GATE_FREE)
def test_gate_free_configs_build_a_model_and_a_dataset_on_cpu(rel_path: str) -> None:
    """The two smoke configs instantiate for real: registry -> torch module + dataset, on CPU.

    Validating a YAML only proves the keys are spelled right; this proves the values are usable
    (encoder name known, d_model divisible by heads, mel bins matching the feature function).

    HONESTY: nothing here is evidence. The weights are randomly initialised and the samples come
    from a synthetic generator, so the only claim being checked is that the wiring holds. No
    optimizer step is taken, no file is downloaded, no approval gate is touched.
    """
    pytest.importorskip("torch")
    from aicls.ml.registry import DatasetSpec, build_dataset, build_model

    cfg = _load(rel_path)
    assert cfg.is_gate_free  # a gated config must never reach this test

    dataset = build_dataset(cfg, cfg.data.split)
    assert len(dataset) == 64
    sample = dataset[0]  # InkSample or AudioSample; both carry .features
    assert sample.features.shape[0] > 0

    tokenizer = DatasetSpec.from_config(cfg).tokenizer()
    model = build_model(cfg, tokenizer.vocab_size)
    assert sum(p.numel() for p in model.parameters()) > 0


# ----------------------------------------------------------------------------------------------
# profiling and sweeps
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rel_path", ["ocr/profile.yaml", "asr/profile.yaml"])
def test_profile_configs_measure_two_hundred_steps(rel_path: str) -> None:
    """§8: the first gated action of every configuration is a 200-step measurement."""
    cfg = _load(rel_path)

    assert cfg.train.max_steps == 200
    assert cfg.train.epochs is None
    assert cfg.train.early_stop_patience is None
    assert cfg.train.eval_every > 200  # a profile run measures throughput, not quality
    assert cfg.train.ckpt_every > 200
    assert cfg.tracking.experiment == "profile"
    assert "profile" in cfg.run.tags


def test_lr_sweep_declares_a_machine_readable_grid() -> None:
    """The sweep grid lives in an ``# AICLS-SWEEP:`` header so CI can validate every point."""
    rel = "sweeps/ocr_lr_sweep.yaml"
    header = _header_comment(rel)

    matches = [SWEEP_HEADER_RE.match(line) for line in header.splitlines()]
    found = [m for m in matches if m is not None]
    assert len(found) == 1, f"{rel} must declare exactly one sweep axis"
    key = found[0].group("key")
    values = yaml.safe_load(found[0].group("values"))

    assert key == "optim.lr"
    assert len(values) >= 3
    assert all(isinstance(v, float) for v in values), values

    digests = set()
    for value in values:
        point = _load(rel, (f"{key}={value}",))
        assert point.optim.lr == pytest.approx(value)
        assert point.model.encoder == "resnet34"  # the sweep is over the resnet34 base
        digests.add(config_hash(point))
    assert len(digests) == len(values)  # every point is a distinct experiment


def test_readme_documents_every_config() -> None:
    """``training/configs/README.md`` is the map; a new config without a line is a lost config."""
    readme = (_CONFIG_ROOT / "README.md").read_text(encoding="utf-8")

    for rel in _CONFIGS:
        assert Path(rel).name in readme, f"{rel} is not mentioned in training/configs/README.md"
