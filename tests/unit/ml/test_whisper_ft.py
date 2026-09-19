"""Tests for the Whisper fine-tuning wrapper (docs/ml/egitim-altyapisi.md §4.2, ADR-0006).

The wrapper does not reimplement Whisper fine-tuning; it wraps Hugging Face ``Seq2SeqTrainer``
and guarantees three things, which is exactly what these tests pin:

1. :class:`~aicls.ml.whisper_ft.AiclsManifestCallback` writes the same ``run_manifest.json`` the
   OCR trainer writes (``test_manifest_callback_writes_the_same_run_manifest``);
2. ``compute_metrics`` reports ``wer`` (``corpus_wer(norm="tr_v1")``) **and** ``wer_raw``
   (``norm="none"``), so an ASR number is comparable with every other number in the project;
3. a Hugging Face checkpoint directory is completed with ``config.yaml``, ``run_manifest.json``
   and ``metrics.json`` (``test_complete_checkpoint_dir_adds_our_three_files``).

**Honesty.** Every WER produced here comes from a hand-made prediction batch or a randomly
initialised tiny Whisper. Those numbers are WIRING CHECKS of the metric path, never ASR results:
real weights sit behind ``WEIGHTS_DOWNLOAD`` and real training behind ``TRAINING`` (ADR-0012).

**Offline.** Nothing here reaches the network. The heavy test builds a tiny ``WhisperConfig``,
saves it to a temporary directory, blocks ``socket`` outright and asserts that the local path is
what ``from_pretrained`` was given, with ``local_files_only=True``.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any

import pytest

from aicls.core.errors import InputValidationError, NotFoundError
from aicls.core.gates import GateError
from aicls.ml.config import TrainConfig, build_config
from aicls.ml.manifest import MANIFEST_FILENAME, RunManifest
from aicls.ml.whisper_ft import (
    CHECKPOINTS_DIRNAME,
    DEFAULT_LANGUAGE,
    DEFAULT_TASK,
    HF_CHECKPOINT_PREFIX,
    LABEL_PAD_ID,
    WER_METRIC_KEYS,
    AiclsManifestCallback,
    DependencyUnavailableError,
    FitReport,
    WhisperDataCollator,
    WhisperDatasetAdapter,
    WhisperFineTuner,
    apply_lora,
    complete_checkpoint_dir,
    freeze_encoder,
    resolve_weights_dir,
)

TINY = {"d_model": 8, "layers": 1, "heads": 2, "ff_dim": 16, "dropout": 0.0, "num_mel_bins": 8}
HAVE_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
HAVE_ACCELERATE = importlib.util.find_spec("accelerate") is not None
HAVE_PEFT = importlib.util.find_spec("peft") is not None


# ==============================================================================================
# fixtures and fakes
# ==============================================================================================
def _cfg(**overrides: Any) -> TrainConfig:
    """A whisper ``TrainConfig`` at tiny size; ``dataset`` defaults to the gate-free synthetic."""
    payload: dict[str, Any] = {
        "run": {"name": "asr_smoke", "seed": 1234, "deterministic": True},
        "data": {
            "dataset": "synthetic_audio",
            "batch_size": 2,
            "num_workers": 0,
            "n_samples": 4,
            "sample_seconds": 0.5,
            "feature_extractor": "whisper",
        },
        "model": {"family": "whisper", "max_len": 16, **TINY},
        "optim": {"lr": 1e-4, "weight_decay": 0.05, "betas": [0.9, 0.98], "warmup_steps": 3},
        "train": {
            "max_steps": 2,
            "grad_accum": 2,
            "eval_every": 2,
            "ckpt_every": 2,
            "log_every": 1,
        },
        "eval": {"beam": 2, "metrics": ["wer"]},
        "checkpoint": {"keep_last": 2, "keep_best": 1, "metric": "wer", "mode": "min"},
    }
    for section, values in overrides.items():
        block = payload.setdefault(section, {})
        assert isinstance(block, dict)
        block.update(values)  # type: ignore[union-attr]  # every override targets a mapping
    return build_config(payload)


def _fake_weights_dir(tmp_path: Path) -> Path:
    """A directory that *looks* like a saved checkpoint, for the paths that never load it."""
    directory = tmp_path / "models" / "pretrained" / "whisper-tiny-local"
    directory.mkdir(parents=True)
    (directory / "config.json").write_text('{"model_type": "whisper"}', encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"not really weights")
    return directory


class _FakeTokenizer:
    """The three tokenizer members ``compute_metrics`` uses, and nothing else.

    Ids are indices into :attr:`words`; a sentence is the space-joined lookup. This is a stand-in
    for a real Whisper tokenizer so the metric path can be exercised without ``transformers``.
    """

    pad_token_id = 0

    def __init__(self, words: dict[int, str]) -> None:
        self.words = words

    def batch_decode(self, sequences: Any, *, skip_special_tokens: bool = True) -> list[str]:
        del skip_special_tokens
        return [
            " ".join(self.words[int(i)] for i in row if int(i) in self.words) for row in sequences
        ]


class _FakeProcessor:
    """``WhisperProcessor``-shaped holder of a tokenizer (no feature extractor needed here)."""

    def __init__(self, tokenizer: _FakeTokenizer) -> None:
        self.tokenizer = tokenizer
        self.feature_extractor = None


class _RecordingTracker:
    """Minimal :class:`aicls.ml.tracking.Tracker` that records the calls it receives."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.metrics: list[tuple[dict[str, float], int, str]] = []
        self.artifacts: list[Path] = []
        self.status: str | None = None

    def start_run(self, name: str, params: Any = None) -> str:
        del params
        self.started.append(name)
        return "run-fake"

    def log_metrics(self, metrics: Any, step: int, *, split: str = "train") -> None:
        self.metrics.append(({str(k): float(v) for k, v in metrics.items()}, step, split))

    def log_artifact(self, path: Path, *, name: str | None = None) -> dict[str, Any]:
        del name
        self.artifacts.append(Path(path))
        return {"path": str(path)}

    def end_run(self, status: str = "finished") -> None:
        self.status = status


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every outbound socket operation raise, so a hidden download cannot pass silently."""

    def _blocked(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("network access attempted in an offline test")

    monkeypatch.setattr(socket.socket, "connect", _blocked, raising=False)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked, raising=False)
    monkeypatch.setattr(socket, "create_connection", _blocked, raising=False)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked, raising=False)


# ==============================================================================================
# the module itself stays cheap and torch-free
# ==============================================================================================
def test_importing_whisper_ft_does_not_import_torch_or_transformers(repo_root: Path) -> None:
    """An ungated process (the CLI listing configs) must not pay for the heavy stack."""
    code = (
        "import json, sys;"
        "import aicls.ml.whisper_ft;"
        "print(json.dumps({m: m in sys.modules for m in ('torch', 'transformers', 'peft')}))"
    )
    env = {**os.environ, "PYTHONPATH": str(repo_root / "src")}

    result = subprocess.run(  # fixed argv, the venv interpreter
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),
        timeout=180,
        check=True,
    )

    assert json.loads(result.stdout.strip()) == {
        "torch": False,
        "transformers": False,
        "peft": False,
    }


# ==============================================================================================
# weight resolution — the WEIGHTS_DOWNLOAD gate, statically and at runtime
# ==============================================================================================
def test_resolve_weights_dir_accepts_a_local_directory(tmp_path: Path) -> None:
    directory = _fake_weights_dir(tmp_path)
    assert resolve_weights_dir(directory) == directory.resolve()


def test_resolve_weights_dir_refuses_a_missing_directory_and_names_the_gate(tmp_path: Path) -> None:
    missing = tmp_path / "models" / "pretrained" / "whisper-small"
    with pytest.raises(NotFoundError) as excinfo:
        resolve_weights_dir(missing)
    details = excinfo.value.details
    assert details["gate"] == "WEIGHTS_DOWNLOAD"
    assert details["path"] == str(missing)
    assert details["reason"] == "weights_dir_missing"


def test_resolve_weights_dir_refuses_a_hub_id() -> None:
    """``org/name`` is a download, not a path: refuse it before ``transformers`` ever sees it."""
    with pytest.raises(InputValidationError) as excinfo:
        resolve_weights_dir("some-org/some-whisper")
    assert excinfo.value.details["reason"] == "hub_id_rejected"
    assert excinfo.value.details["gate"] == "WEIGHTS_DOWNLOAD"


def test_resolve_weights_dir_refuses_nothing_at_all() -> None:
    with pytest.raises(InputValidationError) as excinfo:
        resolve_weights_dir(None)
    assert excinfo.value.details["reason"] == "missing_config_key"
    assert excinfo.value.details["key"] == "model.base_dir"


def test_fine_tuner_falls_back_to_model_base_dir(tmp_path: Path) -> None:
    directory = _fake_weights_dir(tmp_path)
    tuner = WhisperFineTuner(
        _cfg(model={"base_dir": str(directory)}),
        weights_dir=None,
        run_dir=tmp_path / "run",
    )
    assert tuner.weights_dir == directory.resolve()


def test_fine_tuner_refuses_a_non_whisper_family(tmp_path: Path) -> None:
    cfg = build_config({"model": {"family": "seq2seq_image"}, "data": {"dataset": "synthetic_ink"}})
    with pytest.raises(InputValidationError) as excinfo:
        WhisperFineTuner(cfg, weights_dir=_fake_weights_dir(tmp_path), run_dir=tmp_path / "run")
    assert excinfo.value.details["reason"] == "wrong_model_family"


# ==============================================================================================
# training arguments — our settings, without needing accelerate
# ==============================================================================================
def test_training_args_dict_carries_our_settings(tmp_path: Path) -> None:
    tuner = WhisperFineTuner(
        _cfg(), weights_dir=_fake_weights_dir(tmp_path), run_dir=tmp_path / "run"
    )
    args = tuner.training_args_dict()

    # the sharp edges the ADR says Seq2SeqTrainer owns, switched on by us
    assert args["predict_with_generate"] is True
    assert args["generation_num_beams"] == 2
    assert args["generation_max_length"] == 16
    assert args["label_names"] == ["labels"]
    assert args["remove_unused_columns"] is False
    # our config, mapped onto HF names
    assert args["output_dir"] == str(tmp_path / "run" / CHECKPOINTS_DIRNAME)
    assert args["seed"] == 1234
    assert args["data_seed"] == 1234
    assert args["per_device_train_batch_size"] == 2
    assert args["gradient_accumulation_steps"] == 2
    assert args["learning_rate"] == pytest.approx(1e-4)
    assert args["weight_decay"] == pytest.approx(0.05)
    assert args["adam_beta2"] == pytest.approx(0.98)
    assert args["warmup_steps"] == 3
    assert args["lr_scheduler_type"] == "cosine"
    assert args["max_steps"] == 2
    assert args["save_steps"] == 2
    assert args["eval_steps"] == 2
    assert args["logging_steps"] == 1
    assert args["save_total_limit"] == 2
    assert args["metric_for_best_model"] == "wer"
    assert args["greater_is_better"] is False
    assert args["dataloader_num_workers"] == 0
    # ADR-0007: our tracker is the record, HF reports nowhere
    assert args["report_to"] == []


def test_training_args_dict_degrades_to_cpu_fp32_when_cuda_is_absent(tmp_path: Path) -> None:
    """``amp: bf16`` on a CPU-only host must not become a broken run (torch here is +cpu)."""
    tuner = WhisperFineTuner(
        _cfg(train={"amp": "bf16"}),
        weights_dir=_fake_weights_dir(tmp_path),
        run_dir=tmp_path / "run",
    )
    args = tuner.training_args_dict()
    import torch

    if torch.cuda.is_available():  # pragma: no cover - CPU wheel in this environment
        assert args["bf16"] is True
        assert args["use_cpu"] is False
    else:
        assert args["bf16"] is False
        assert args["fp16"] is False
        assert args["use_cpu"] is True


def test_training_args_dict_disables_evaluation_without_an_eval_dataset(tmp_path: Path) -> None:
    tuner = WhisperFineTuner(
        _cfg(), weights_dir=_fake_weights_dir(tmp_path), run_dir=tmp_path / "run"
    )
    assert tuner.training_args_dict(with_eval=False)["eval_strategy"] == "no"
    assert tuner.training_args_dict(with_eval=False)["load_best_model_at_end"] is False
    assert tuner.training_args_dict(with_eval=True)["eval_strategy"] == "steps"


def test_training_args_needs_accelerate_and_says_so(tmp_path: Path) -> None:
    """``Seq2SeqTrainingArguments`` cannot even be constructed without ``accelerate``."""
    tuner = WhisperFineTuner(
        _cfg(), weights_dir=_fake_weights_dir(tmp_path), run_dir=tmp_path / "run"
    )
    if not HAVE_TRANSFORMERS or not HAVE_ACCELERATE:
        with pytest.raises(DependencyUnavailableError) as excinfo:
            tuner.training_args()
        assert excinfo.value.details["package"] in {"accelerate", "transformers"}
        assert "requirements" in excinfo.value.details["hint"]
    else:  # pragma: no cover - accelerate is not installed in this venv
        args = tuner.training_args()
        assert args.predict_with_generate is True
        assert args.max_steps == 2


# ==============================================================================================
# guarantee 2 — one WER definition, reported raw and tr_v1
# ==============================================================================================
def _hand_made_batch() -> tuple[list[list[int]], list[list[int]]]:
    """Predictions and labels for "iki artı iki eşittir dört", with one substitution.

    Ids are indices into :class:`_FakeTokenizer`; ``-100`` is the label padding the HF collator
    writes and ``compute_metrics`` must undo before decoding.
    """
    predictions = [[1, 2, 3, 4, 5], [6, 7, 0, 0, 0]]
    labels = [[1, 2, 3, 4, 8], [6, 7, LABEL_PAD_ID, LABEL_PAD_ID, LABEL_PAD_ID]]
    return predictions, labels


def _fake_metric_processor() -> _FakeProcessor:
    words = {
        1: "iki",
        2: "artı",
        3: "iki",
        4: "eşittir",
        5: "beş",  # the hypothesis' one wrong word
        6: "türevini",
        7: "al",
        8: "dört",  # the reference's last word
    }
    return _FakeProcessor(_FakeTokenizer(words))


def test_compute_metrics_returns_our_wer_keys(tmp_path: Path) -> None:
    """WIRING CHECK of the metric path: hand-made ids, one substitution in seven words."""
    tuner = WhisperFineTuner(
        _cfg(),
        weights_dir=_fake_weights_dir(tmp_path),
        run_dir=tmp_path / "run",
        processor=_fake_metric_processor(),
    )
    predictions, labels = _hand_made_batch()
    metrics = tuner.compute_metrics((predictions, labels))

    assert set(metrics) == set(WER_METRIC_KEYS)
    assert metrics["n_samples"] == 2.0
    # 7 reference words, 1 substitution -> 1/7 under both normalisers
    assert metrics["wer"] == pytest.approx(1.0 / 7.0)
    assert metrics["wer_raw"] == pytest.approx(1.0 / 7.0)


def test_compute_metrics_separates_raw_from_tr_v1(tmp_path: Path) -> None:
    """``tr_v1`` folds casing and punctuation; the raw column must still show the difference."""
    words = {1: "Türevini", 2: "al.", 3: "türevini", 4: "al"}
    tuner = WhisperFineTuner(
        _cfg(),
        weights_dir=_fake_weights_dir(tmp_path),
        run_dir=tmp_path / "run",
        processor=_FakeProcessor(_FakeTokenizer(words)),
    )
    metrics = tuner.compute_metrics(([[1, 2]], [[3, 4]]))
    assert metrics["wer"] == pytest.approx(0.0)
    assert metrics["wer_raw"] > 0.0


def test_compute_metrics_accepts_a_tuple_of_prediction_arrays(tmp_path: Path) -> None:
    """``Seq2SeqTrainer`` hands back a tuple when the model returns more than the sequences."""
    predictions, labels = _hand_made_batch()
    tuner = WhisperFineTuner(
        _cfg(),
        weights_dir=_fake_weights_dir(tmp_path),
        run_dir=tmp_path / "run",
        processor=_fake_metric_processor(),
    )
    metrics = tuner.compute_metrics(((predictions, [[0.0]]), labels))
    assert metrics["wer"] == pytest.approx(1.0 / 7.0)


def test_compute_metrics_on_an_empty_batch_is_zero_not_perfect(tmp_path: Path) -> None:
    tuner = WhisperFineTuner(
        _cfg(),
        weights_dir=_fake_weights_dir(tmp_path),
        run_dir=tmp_path / "run",
        processor=_fake_metric_processor(),
    )
    metrics = tuner.compute_metrics(([], []))
    assert metrics == {"wer": 0.0, "wer_raw": 0.0, "n_samples": 0.0}


def test_compute_metrics_without_a_processor_fails_loudly(tmp_path: Path) -> None:
    tuner = WhisperFineTuner(
        _cfg(), weights_dir=_fake_weights_dir(tmp_path), run_dir=tmp_path / "run"
    )
    # the fake directory holds no tokenizer files, so loading the processor must fail
    with pytest.raises((InputValidationError, NotFoundError, DependencyUnavailableError)):
        tuner.compute_metrics(_hand_made_batch())


# ==============================================================================================
# guarantee 1 — the same run_manifest.json the OCR trainer writes
# ==============================================================================================
def test_manifest_callback_writes_the_same_run_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    callback = AiclsManifestCallback(_cfg(), run_dir, allow_dirty=True, capture_environment=False)
    assert callback.manifest is None

    callback.on_train_begin(None, None, None)

    path = run_dir / MANIFEST_FILENAME
    assert path.is_file()
    document = json.loads(path.read_text(encoding="utf-8"))
    # the field set the OCR trainer's manifest carries (ADR-0007)
    for key in ("git_sha", "dirty", "config_hash", "pip_freeze_sha256", "citable", "run_id"):
        assert key in document
    assert document["run_name"] == "asr_smoke"
    assert document["seed"] == 1234
    # RunManifest.load round-trips what we wrote
    reloaded = RunManifest.load(path)
    assert reloaded.run_id == document["run_id"]
    assert callback.manifest is not None
    assert callback.manifest.run_id == reloaded.run_id


def test_manifest_callback_writes_the_manifest_only_once(tmp_path: Path) -> None:
    callback = AiclsManifestCallback(
        _cfg(), tmp_path / "run", allow_dirty=True, capture_environment=False
    )
    callback.on_train_begin(None, None, None)
    first = callback.manifest
    callback.on_train_begin(None, None, None)
    assert callback.manifest is first


def test_manifest_callback_captures_the_environment_when_asked(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    callback = AiclsManifestCallback(_cfg(), run_dir, allow_dirty=True, capture_environment=True)
    callback.on_train_begin(None, None, None)
    payload = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
    assert payload["pip_freeze_sha256"]


def test_manifest_callback_forwards_metrics_to_the_tracker(tmp_path: Path) -> None:
    tracker = _RecordingTracker()
    callback = AiclsManifestCallback(
        _cfg(), tmp_path / "run", tracker=tracker, allow_dirty=True, capture_environment=False
    )
    callback.on_train_begin(None, None, None)
    callback.on_log(None, _FakeState(step=4), None, logs={"loss": 2.5, "epoch": 0.5})
    callback.on_evaluate(None, _FakeState(step=4), None, metrics={"eval_wer": 0.42, "eval_n": 2})

    assert tracker.started == ["asr_smoke"]
    train_rows = [row for row in tracker.metrics if row[2] == "train"]
    valid_rows = [row for row in tracker.metrics if row[2] == "valid"]
    assert train_rows[0][0]["loss"] == pytest.approx(2.5)
    assert train_rows[0][1] == 4
    assert valid_rows[0][0]["wer"] == pytest.approx(0.42)
    assert callback.metrics["wer"] == pytest.approx(0.42)


def test_manifest_callback_ignores_trainer_events_it_does_not_implement(tmp_path: Path) -> None:
    """HF calls a dozen ``on_*`` hooks; a plain (transformers-free) callback must absorb them."""
    callback = AiclsManifestCallback(
        _cfg(), tmp_path / "run", allow_dirty=True, capture_environment=False
    )
    assert callback.on_step_end(None, None, None) is None
    assert callback.on_prediction_step(None, None, None, extra=1) is None
    with pytest.raises(AttributeError):
        _ = callback.definitely_not_a_trainer_event


class _FakeState:
    """The two ``TrainerState`` fields the callback reads."""

    def __init__(self, step: int = 0, log_history: list[dict[str, Any]] | None = None) -> None:
        self.global_step = step
        self.log_history = log_history or []


class _FakeArgs:
    """The one ``TrainingArguments`` field the callback reads."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = str(output_dir)


# ==============================================================================================
# guarantee 3 — our three files beside the Hugging Face checkpoint files
# ==============================================================================================
def test_complete_checkpoint_dir_adds_our_three_files(tmp_path: Path) -> None:
    cfg = _cfg()
    ckpt = tmp_path / "checkpoints" / "checkpoint-2"
    ckpt.mkdir(parents=True)
    (ckpt / "model.safetensors").write_bytes(b"hf writes this")
    manifest = RunManifest.capture(cfg, allow_dirty=True)

    written = complete_checkpoint_dir(ckpt, cfg=cfg, manifest=manifest, metrics={"wer": 0.25})

    assert {p.name for p in written} == {"config.yaml", MANIFEST_FILENAME, "metrics.json"}
    assert json.loads((ckpt / "metrics.json").read_text(encoding="utf-8")) == {"wer": 0.25}
    assert "whisper" in (ckpt / "config.yaml").read_text(encoding="utf-8")
    assert json.loads((ckpt / MANIFEST_FILENAME).read_text(encoding="utf-8"))["run_id"]
    # the HF file is untouched
    assert (ckpt / "model.safetensors").read_bytes() == b"hf writes this"


def test_manifest_callback_completes_the_checkpoint_directory_on_save(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    ckpt_root = run_dir / CHECKPOINTS_DIRNAME
    ckpt = ckpt_root / f"{HF_CHECKPOINT_PREFIX}-2"
    ckpt.mkdir(parents=True)
    callback = AiclsManifestCallback(
        _cfg(), run_dir, allow_dirty=True, capture_environment=False, checkpoints_dir=ckpt_root
    )
    callback.on_train_begin(None, None, None)
    callback.on_evaluate(None, _FakeState(step=2), None, metrics={"eval_wer": 0.5})

    callback.on_save(_FakeArgs(ckpt_root), _FakeState(step=2), None)

    assert (ckpt / MANIFEST_FILENAME).is_file()
    assert (ckpt / "config.yaml").is_file()
    assert json.loads((ckpt / "metrics.json").read_text(encoding="utf-8"))["wer"] == 0.5


def test_manifest_callback_on_save_is_quiet_when_hf_wrote_nothing(tmp_path: Path) -> None:
    callback = AiclsManifestCallback(
        _cfg(), tmp_path / "run", allow_dirty=True, capture_environment=False
    )
    callback.on_train_begin(None, None, None)
    assert callback.on_save(_FakeArgs(tmp_path / "nowhere"), _FakeState(step=7), None) is None


# ==============================================================================================
# the TRAINING gate (ADR-0012) — the wrapper refuses, it never opens
# ==============================================================================================
def test_fit_refuses_a_real_dataset_without_the_training_gate(
    tmp_path: Path, test_env: dict[str, str]
) -> None:
    del test_env  # the fixture clears every AICLS_GATE_* variable
    cfg = _cfg(data={"dataset": "audio_commands", "manifest": str(tmp_path / "m.jsonl")})
    tuner = WhisperFineTuner(cfg, weights_dir=_fake_weights_dir(tmp_path), run_dir=tmp_path / "run")
    assert cfg.is_gate_free is False
    with pytest.raises(GateError) as excinfo:
        tuner.fit(train_dataset=[])
    assert excinfo.value.exit_code == 3
    assert excinfo.value.details["gate"] == "TRAINING"


def test_require_training_gate_is_skipped_for_synthetic_audio(tmp_path: Path) -> None:
    """``data.dataset: synthetic_*`` is gate-free by ADR-0012; the check must not be reached."""
    tuner = WhisperFineTuner(
        _cfg(), weights_dir=_fake_weights_dir(tmp_path), run_dir=tmp_path / "run"
    )
    assert tuner.require_training_gate() is False


def test_require_training_gate_refuses_a_real_dataset(
    tmp_path: Path, test_env: dict[str, str]
) -> None:
    del test_env
    tuner = WhisperFineTuner(
        _cfg(data={"dataset": "audio_commands"}),
        weights_dir=_fake_weights_dir(tmp_path),
        run_dir=tmp_path / "run",
    )
    with pytest.raises(GateError):
        tuner.require_training_gate(i_have_approval=False)


# ==============================================================================================
# LoRA and encoder freezing — both optional, both lazily imported
# ==============================================================================================
def test_apply_lora_without_peft_raises_a_typed_error() -> None:
    from aicls.ml.config import LoraCfg

    if HAVE_PEFT:  # pragma: no cover - peft is not installed in this venv
        pytest.skip("peft is installed; the missing-dependency path cannot be exercised")
    with pytest.raises(DependencyUnavailableError) as excinfo:
        apply_lora(object(), LoraCfg(enabled=True, r=4, alpha=8))
    assert excinfo.value.details["package"] == "peft"
    assert excinfo.value.details["reason"] == "peft_not_installed"


def test_apply_lora_is_a_no_op_when_disabled() -> None:
    from aicls.ml.config import LoraCfg

    sentinel = object()
    assert apply_lora(sentinel, LoraCfg(enabled=False)) is sentinel


def test_freeze_encoder_freezes_only_the_encoder() -> None:
    torch = pytest.importorskip("torch")

    class _Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(4, 4)
            self.decoder = torch.nn.Linear(4, 4)

        def get_encoder(self) -> Any:
            return self.encoder

    model = _Tiny()
    frozen = freeze_encoder(model)

    assert frozen == sum(p.numel() for p in model.encoder.parameters())
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.decoder.parameters())


def test_freeze_encoder_refuses_a_model_without_one() -> None:
    with pytest.raises(InputValidationError) as excinfo:
        freeze_encoder(object())
    assert excinfo.value.details["reason"] == "encoder_not_found"


# ==============================================================================================
# FitReport
# ==============================================================================================
def test_fit_report_is_a_json_document(tmp_path: Path) -> None:
    report = FitReport(
        run_dir=tmp_path / "run",
        run_id="01J",
        manifest_path=tmp_path / "run" / MANIFEST_FILENAME,
        checkpoint_dir=None,
        global_step=2,
        train_loss=1.5,
        metrics={"wer": 0.5},
        citable=False,
        wall_time_s=0.25,
    )
    document = report.to_dict()
    assert json.loads(json.dumps(document))["global_step"] == 2
    assert document["citable"] is False
    assert document["checkpoint_dir"] is None


# ==============================================================================================
# the rest of the plumbing (all reachable without transformers)
# ==============================================================================================
def test_complete_checkpoint_dir_refuses_to_invent_a_directory(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        complete_checkpoint_dir(tmp_path / "nope", cfg=_cfg())
    assert excinfo.value.details["reason"] == "checkpoint_dir_missing"


def test_complete_checkpoint_dir_accepts_a_manifest_document(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    written = complete_checkpoint_dir(ckpt, manifest={"run_id": "01J", "citable": False})
    assert [p.name for p in written] == [MANIFEST_FILENAME]
    assert json.loads((ckpt / MANIFEST_FILENAME).read_text(encoding="utf-8"))["run_id"] == "01J"


def test_complete_checkpoint_dir_writes_nothing_when_given_nothing(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoint-1"
    ckpt.mkdir()
    assert complete_checkpoint_dir(ckpt) == []
    assert list(ckpt.iterdir()) == []


def test_manifest_callback_records_measured_throughput(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    callback = AiclsManifestCallback(_cfg(), run_dir, allow_dirty=True, capture_environment=False)
    callback.on_train_begin(None, None, None)
    callback.on_log(None, _FakeState(step=8), None, logs={"train_runtime": 4.0})

    callback.on_train_end(None, _FakeState(step=8), None)

    assert callback.manifest is not None
    assert callback.manifest.steps_per_s == pytest.approx(2.0)
    document = json.loads((run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert document["steps_per_s"] == pytest.approx(2.0)


def test_manifest_callback_skips_non_numeric_and_bookkeeping_logs(tmp_path: Path) -> None:
    tracker = _RecordingTracker()
    callback = AiclsManifestCallback(
        _cfg(), tmp_path / "run", tracker=tracker, allow_dirty=True, capture_environment=False
    )
    callback.on_train_begin(None, None, None)
    callback.on_log(None, _FakeState(step=1), None, logs="not a mapping")
    callback.on_log(None, _FakeState(step=1), None, logs={"epoch": 1.0, "should_stop": True})

    assert tracker.metrics == []
    assert callback.metrics == {}


def test_manifest_callback_on_save_without_any_output_dir(tmp_path: Path) -> None:
    callback = AiclsManifestCallback(
        _cfg(), tmp_path / "run", allow_dirty=True, capture_environment=False
    )
    callback.on_train_begin(None, None, None)

    class _NoArgs:
        output_dir = ""

    assert callback.on_save(_NoArgs(), _FakeState(step=1), None) is None


def test_configure_generation_tolerates_a_model_without_a_generation_config() -> None:
    from aicls.ml.whisper_ft import configure_generation

    class _Bare:
        generation_config = None
        config = None

    configure_generation(_Bare())  # must not raise


def test_last_checkpoint_dir_picks_the_highest_step(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    tuner = WhisperFineTuner(_cfg(), weights_dir=_fake_weights_dir(tmp_path), run_dir=run_dir)
    assert tuner.last_checkpoint_dir() is None

    root = run_dir / CHECKPOINTS_DIRNAME
    for name in (f"{HF_CHECKPOINT_PREFIX}-2", f"{HF_CHECKPOINT_PREFIX}-10", "runs"):
        (root / name).mkdir(parents=True)
    assert tuner.last_checkpoint_dir() == root / f"{HF_CHECKPOINT_PREFIX}-10"


def test_last_checkpoint_dir_is_none_when_hf_wrote_none(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / CHECKPOINTS_DIRNAME).mkdir(parents=True)
    tuner = WhisperFineTuner(_cfg(), weights_dir=_fake_weights_dir(tmp_path), run_dir=run_dir)
    assert tuner.last_checkpoint_dir() is None


def test_generation_max_length_falls_back_to_the_project_default(tmp_path: Path) -> None:
    from aicls.ml.whisper_ft import DEFAULT_GENERATION_MAX_LENGTH

    cfg = build_config({"model": {"family": "whisper"}, "data": {"dataset": "synthetic_audio"}})
    tuner = WhisperFineTuner(cfg, weights_dir=_fake_weights_dir(tmp_path), run_dir=tmp_path / "run")
    assert tuner.generation_max_length() == DEFAULT_GENERATION_MAX_LENGTH


def test_compute_metrics_accepts_an_eval_prediction_object(tmp_path: Path) -> None:
    class _EvalPrediction:
        def __init__(self, predictions: Any, label_ids: Any) -> None:
            self.predictions = predictions
            self.label_ids = label_ids

    predictions, labels = _hand_made_batch()
    tuner = WhisperFineTuner(
        _cfg(),
        weights_dir=_fake_weights_dir(tmp_path),
        run_dir=tmp_path / "run",
        processor=_fake_metric_processor(),
    )
    metrics = tuner.compute_metrics(_EvalPrediction(predictions, labels))
    assert metrics["wer"] == pytest.approx(1.0 / 7.0)


def test_compute_metrics_refuses_an_unrecognised_shape(tmp_path: Path) -> None:
    tuner = WhisperFineTuner(
        _cfg(),
        weights_dir=_fake_weights_dir(tmp_path),
        run_dir=tmp_path / "run",
        processor=_fake_metric_processor(),
    )
    with pytest.raises(InputValidationError) as excinfo:
        tuner.compute_metrics(42)
    assert excinfo.value.details["reason"] == "eval_prediction_unrecognised"


def test_dataset_adapter_passes_prepared_rows_through() -> None:
    adapter = WhisperDatasetAdapter([{"input_features": [1.0], "labels": [2, 3]}], processor=None)
    assert adapter[0] == {"input_features": [1.0], "labels": [2, 3]}


def test_dataset_adapter_reads_an_audio_sample_object() -> None:
    class _AudioSample:
        features = [[0.0, 1.0]]
        text = "türevini al"

    class _Tokenized:
        input_ids = [1, 2, 3, 4]

    class _Tokenizer:
        def __call__(self, text: str) -> Any:
            assert text == "türevini al"
            return _Tokenized()

    class _Processor:
        tokenizer = _Tokenizer()

    adapter = WhisperDatasetAdapter([_AudioSample()], _Processor(), max_label_len=3)
    row = adapter[0]
    assert row["input_features"] == [[0.0, 1.0]]
    assert row["labels"] == [1, 2, 3]


def test_dataset_adapter_refuses_an_item_without_a_transcript() -> None:
    adapter = WhisperDatasetAdapter([object()], processor=None)
    with pytest.raises(InputValidationError) as excinfo:
        _ = adapter[0]
    assert excinfo.value.details["reason"] == "audio_item_without_text"


# ==============================================================================================
# the contractual heavy test: a tiny randomly initialised Whisper, offline, from a local path
# ==============================================================================================
def _build_tiny_local_whisper(directory: Path) -> Any:
    """Save a tiny randomly initialised Whisper + processor into ``directory``.

    This stands in for what the ``WEIGHTS_DOWNLOAD`` gate would put under
    ``AICLS_MODELS_ROOT/pretrained/``. Its outputs are noise: any metric measured on it is a
    wiring check, never an ASR result.
    """
    import transformers

    specials = ["<|endoftext|>", "<|startoftranscript|>", "<|tr|>", "<|transcribe|>"]
    vocab = {token: i for i, token in enumerate(specials + [chr(c) for c in range(32, 127)])}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (directory / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    tokenizer = transformers.WhisperTokenizer(
        vocab_file=str(directory / "vocab.json"),
        merges_file=str(directory / "merges.txt"),
        language="turkish",
        task="transcribe",
        pad_token="<|endoftext|>",  # a real Whisper tokenizer pads with its eos token
    )
    extractor = transformers.WhisperFeatureExtractor(
        feature_size=8, num_mel_bins=8, sampling_rate=16000, hop_length=160, chunk_length=1
    )
    processor = transformers.WhisperProcessor(feature_extractor=extractor, tokenizer=tokenizer)
    config = transformers.WhisperConfig(
        vocab_size=len(tokenizer),
        d_model=8,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        num_mel_bins=8,
        max_source_positions=50,
        max_target_positions=16,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=0,
        decoder_start_token_id=1,
    )
    model = transformers.WhisperForConditionalGeneration(config)
    # A real Whisper checkpoint is multilingual: its generation config carries the language and
    # task token maps that `generate` resolves `language="tr"` through. Without them Whisper's
    # `_retrieve_init_tokens` raises, so the stand-in has to carry them too — otherwise this
    # fixture would silently test the *monolingual* path. A generation config built here rather
    # than derived from the model config is the one that survives save/load (`from_pretrained`
    # rebuilds a `_from_model_config` one and drops the extra keys).
    model.generation_config = transformers.GenerationConfig(
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
        decoder_start_token_id=config.decoder_start_token_id,
        lang_to_id={"<|tr|>": vocab["<|tr|>"]},
        task_to_id={"transcribe": vocab["<|transcribe|>"]},
    )
    model.save_pretrained(directory)
    processor.save_pretrained(directory)
    return processor


def test_whisper_ft_builds_from_tiny_random_config_without_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, test_env: dict[str, str], no_network: None
) -> None:
    """The contractual test of ADR-0006: build, arguments, metrics and manifest, all offline.

    Every number this produces is a WIRING CHECK on randomly initialised weights.
    """
    transformers = pytest.importorskip("transformers")
    del no_network
    assert test_env["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

    weights_dir = tmp_path / "models" / "pretrained" / "whisper-tiny-random"
    _build_tiny_local_whisper(weights_dir)

    # record exactly what from_pretrained is asked to load
    seen: list[tuple[str, Any, dict[str, Any]]] = []
    original_model_loader = transformers.WhisperForConditionalGeneration.from_pretrained
    original_processor_loader = transformers.WhisperProcessor.from_pretrained

    def _record_model(path: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(("model", path, dict(kwargs)))
        return original_model_loader(path, *args, **kwargs)

    def _record_processor(path: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(("processor", path, dict(kwargs)))
        return original_processor_loader(path, *args, **kwargs)

    monkeypatch.setattr(
        transformers.WhisperForConditionalGeneration, "from_pretrained", _record_model
    )
    monkeypatch.setattr(transformers.WhisperProcessor, "from_pretrained", _record_processor)

    run_dir = tmp_path / "runs" / "asr_smoke"
    tracker = _RecordingTracker()
    tuner = WhisperFineTuner(
        _cfg(), weights_dir=weights_dir, run_dir=run_dir, tracker=tracker, allow_dirty=True
    )

    # --- it builds, from the LOCAL directory only ------------------------------------------
    model = tuner.model
    assert model.config.d_model == 8
    assert sum(p.numel() for p in model.parameters()) > 0
    assert tuner.processor is not None
    assert [kind for kind, _, _ in seen] == ["model", "processor"]
    for _kind, path, kwargs in seen:
        assert Path(str(path)) == weights_dir.resolve()
        assert kwargs["local_files_only"] is True

    # --- forced Turkish transcription is configured, not re-implemented ---------------------
    assert model.generation_config.language == DEFAULT_LANGUAGE
    assert model.generation_config.task == DEFAULT_TASK

    # --- training arguments carry our settings ---------------------------------------------
    args = tuner.training_args_dict()
    assert args["predict_with_generate"] is True
    assert args["generation_max_length"] <= model.config.max_target_positions
    assert args["output_dir"] == str(run_dir / CHECKPOINTS_DIRNAME)

    # --- compute_metrics on a hand-made batch, through the REAL tokenizer -------------------
    reference = tuner.processor.tokenizer("iki artı iki dört", add_special_tokens=False).input_ids
    hypothesis = tuner.processor.tokenizer("iki artı iki beş", add_special_tokens=False).input_ids
    metrics = tuner.compute_metrics(([hypothesis], [reference]))
    assert set(metrics) == set(WER_METRIC_KEYS)
    assert 0.0 <= metrics["wer"] <= 1.0
    assert metrics["n_samples"] == 1.0

    # --- the collator pads labels with -100 -------------------------------------------------
    collator = tuner.data_collator()
    assert isinstance(collator, WhisperDataCollator)
    batch = collator(
        [
            {"input_features": [[0.1] * 100] * 8, "labels": [4, 5, 6]},
            {"input_features": [[0.2] * 100] * 8, "labels": [4, 5]},
        ]
    )
    assert batch["labels"].shape[0] == 2
    assert int(batch["labels"][1][-1]) == LABEL_PAD_ID

    # --- the manifest callback writes run_manifest.json -------------------------------------
    callback = tuner.manifest_callback()
    callback.on_train_begin(None, None, None)
    manifest_path = run_dir / MANIFEST_FILENAME
    assert manifest_path.is_file()
    assert RunManifest.load(manifest_path).run_name == "asr_smoke"
    assert tracker.started == ["asr_smoke"]


def test_whisper_dataset_adapter_prepares_features_and_labels(tmp_path: Path) -> None:
    """The adapter turns ``aicls.data`` audio samples into what ``Seq2SeqTrainer`` consumes."""
    pytest.importorskip("transformers")
    import numpy as np

    weights_dir = tmp_path / "models" / "pretrained" / "whisper-tiny-random"
    processor = _build_tiny_local_whisper(weights_dir)

    rows = [(np.zeros(8000, dtype=np.float32), "iki artı iki"), (np.zeros(8000, np.float32), "al")]
    adapter = WhisperDatasetAdapter(rows, processor, max_label_len=16)

    assert len(adapter) == 2
    item = adapter[0]
    assert set(item) == {"input_features", "labels"}
    assert len(item["labels"]) <= 16


# ==============================================================================================
# the weight-free smoke path (wave-5 blocking contradiction)
# ==============================================================================================
def test_fine_tuner_accepts_a_gate_free_config_without_any_weights(tmp_path: Path) -> None:
    """``training/configs/asr/smoke_synthetic.yaml`` sets ``model.base_dir: null`` on purpose.

    ``aicls.ml.registry.build_whisper`` has always supported that (a tiny randomly initialised
    model, the CI path); the fine-tuner demanded a local weights directory, so the shipped
    gate-free config could not run through the CLI at all.
    """
    pytest.importorskip("transformers")
    cfg = _cfg(model={"base_dir": None}, data={"feature_extractor": "log_mel"})
    assert cfg.is_gate_free

    tuner = WhisperFineTuner(cfg, run_dir=tmp_path / "run")

    assert tuner.random_init is True
    assert tuner.weights_dir is None
    processor = tuner.processor
    assert processor.tokenizer.vocab_size > 0
    model = tuner.model
    assert model.config.vocab_size == processor.tokenizer.vocab_size
    assert sum(p.numel() for p in model.parameters()) > 0


def test_a_weightless_run_is_refused_when_it_is_not_gate_free(tmp_path: Path) -> None:
    """Only a synthetic run may train random weights; a real one must name its checkpoint."""
    cfg = _cfg(data={"dataset": "audio_commands", "manifest": str(tmp_path / "m.jsonl")})
    assert not cfg.is_gate_free

    with pytest.raises(InputValidationError) as excinfo:
        WhisperFineTuner(cfg, run_dir=tmp_path / "run")
    assert excinfo.value.details["key"] == "model.base_dir"
    assert excinfo.value.details["gate"] == "WEIGHTS_DOWNLOAD"


def test_the_smoke_tokenizer_round_trips_turkish_text() -> None:
    """The stand-in tokenizer is a character model — enough to score a WER, honest about it."""
    from aicls.ml.whisper_ft import build_smoke_processor

    processor = build_smoke_processor(_cfg(model={"base_dir": None}))
    tokenizer = processor.tokenizer
    ids = tokenizer("x karenin türevini al").input_ids

    assert ids[0] == tokenizer.bos_token_id
    assert ids[-1] == tokenizer.eos_token_id
    assert tokenizer.batch_decode([ids])[0] == "x karenin türevini al"
    assert tokenizer.pad_token_id == 0


def test_the_smoke_tokenizer_pads_a_batch_for_the_collator() -> None:
    torch = pytest.importorskip("torch")
    from aicls.ml.whisper_ft import build_smoke_processor

    tokenizer = build_smoke_processor(_cfg(model={"base_dir": None})).tokenizer
    batch = tokenizer.pad([{"input_ids": [4, 5, 6]}, {"input_ids": [7]}], return_tensors="pt")

    assert tuple(batch["input_ids"].shape) == (2, 3)
    assert int(batch["input_ids"][1][2]) == tokenizer.pad_token_id
    assert torch.equal(batch["attention_mask"][1], torch.tensor([1, 0, 0]))


def test_configure_generation_leaves_a_monolingual_model_alone() -> None:
    """Forcing ``language`` on a config with no ``lang_to_id`` makes ``generate`` raise.

    Measured in the cu130 container with transformers 4.57.6: a randomly initialised Whisper
    generates fine until ``generation_config.language`` is set, after which
    ``_retrieve_init_tokens`` fails with ``AttributeError: ... has no attribute 'lang_to_id'``.
    """
    from aicls.ml.whisper_ft import configure_generation

    class _GenerationConfig:
        language = None
        task = None
        forced_decoder_ids = [(1, 2)]

    class _Model:
        generation_config = _GenerationConfig()
        config = None

    model = _Model()
    configure_generation(model)

    assert model.generation_config.language is None, "no language token map, so no language"
    assert model.generation_config.task is None
    assert model.generation_config.forced_decoder_ids is None, "the deprecated field is cleared"


def test_configure_generation_forces_turkish_on_a_multilingual_model() -> None:
    from aicls.ml.whisper_ft import configure_generation

    class _GenerationConfig:
        language = None
        task = None
        forced_decoder_ids = None
        lang_to_id = {"<|tr|>": 3}

    class _Model:
        generation_config = _GenerationConfig()
        config = None

    model = _Model()
    configure_generation(model)

    assert model.generation_config.language == DEFAULT_LANGUAGE
    assert model.generation_config.task == DEFAULT_TASK


def test_the_collator_matches_the_feature_width_the_encoder_demands() -> None:
    """Whisper's encoder accepts exactly ``2 * max_source_positions`` mel frames and nothing else.

    ``data.feature_extractor: log_mel`` produces a frame count that follows the utterance length,
    so without this the first forward pass raises ``expected mel input features of length ...``.
    """
    torch = pytest.importorskip("torch")
    from aicls.ml.whisper_ft import encoder_input_frames

    collator = WhisperDataCollator(
        _FakeProcessor(_FakeTokenizer({})), input_frames=6, decoder_start_token_id=None
    )
    aligned = collator.align_features(
        [
            {"input_features": [[1.0] * 3] * 4, "labels": [1]},
            {"input_features": [[2.0] * 9] * 4, "labels": [1]},
        ]
    )

    assert [row["input_features"].shape for row in aligned] == [(4, 6), (4, 6)]
    assert float(aligned[0]["input_features"][0][4]) == 0.0, "short rows are zero padded"
    assert float(aligned[1]["input_features"][0][5]) == 2.0, "long rows are truncated"
    del torch

    class _Cfg:
        max_source_positions = 100

    assert encoder_input_frames(_Cfg()) == 200
    assert encoder_input_frames(None) is None


def test_silence_hf_printer_removes_the_stdout_callback() -> None:
    """``aicls train --json`` must emit JSON and nothing else; HF's printer breaks that."""
    pytest.importorskip("transformers")
    import transformers

    from aicls.ml.whisper_ft import silence_hf_printer

    removed: list[Any] = []

    class _Trainer:
        def remove_callback(self, callback: Any) -> None:
            removed.append(callback)

    assert silence_hf_printer(_Trainer()) is True
    assert removed == [transformers.PrinterCallback]


def test_the_smoke_tokenizer_marks_what_it_cannot_represent() -> None:
    """A character outside the alphabet becomes ``<unk>``; an id outside the vocabulary is dropped.

    ``skip_special_tokens=True`` then removes the ``<unk>`` marker, exactly as a Hugging Face
    tokenizer does \u2014 one more reason a WER measured through this stand-in is a wiring check:
    reference and hypothesis both lose whatever the alphabet cannot spell.
    """
    from aicls.ml.whisper_ft import SMOKE_SPECIAL_TOKENS, build_smoke_processor

    tokenizer = build_smoke_processor(_cfg(model={"base_dir": None})).tokenizer
    assert len(tokenizer) == tokenizer.vocab_size > len(SMOKE_SPECIAL_TOKENS)

    ids = tokenizer("a\u4e2db").input_ids
    assert ids[2] == tokenizer.unk_token_id
    assert tokenizer.batch_decode([ids], skip_special_tokens=False) == ["<sot>a<unk>b<eot>"]
    assert tokenizer.batch_decode([[*ids, 10**6, -1]]) == ["ab"]
