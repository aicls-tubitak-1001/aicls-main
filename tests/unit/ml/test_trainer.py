"""Unit tests for :mod:`aicls.ml.trainer` (docs/ml/egitim-altyapisi.md §4.1).

Everything here runs on CPU in seconds with a 60k-parameter model over deterministic synthetic
Bezier scribbles, and **every number these tests observe is a wiring check, never an OCR result**:
the weights start random, the "handwriting" is generated from a seed and the same samples are used
for training and validation. The CER < 10 % target of the proposal has no evidence in this file and
cannot have any until the ``DATASET_DOWNLOAD``, ``WEIGHTS_DOWNLOAD``, ``CUDA_INSTALL`` and
``TRAINING`` gates are approved (ADR-0012).

The contractual tests named by the design document are
``test_trainer_two_steps_synthetic_ocr_loss_decreases_cpu`` and ``test_trainer_two_steps_and_resume``.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
import math
from pathlib import Path
from typing import Any

import pytest

from aicls.core.errors import AiclsError
from aicls.core.gates import GateError
from aicls.ml import trainer as t
from aicls.ml.checkpoint import load_checkpoint
from aicls.ml.config import TrainConfig, build_config
from aicls.ml.tracking import read_metrics

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------------------------
def _merge(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive dict merge so a test can override one nested key."""
    out = dict(base)
    for key, value in extra.items():
        current = out.get(key)
        if isinstance(value, Mapping) and isinstance(current, dict):
            out[key] = _merge(current, value)
        else:
            out[key] = value
    return out


def _cfg(**overrides: Any) -> TrainConfig:
    """A gate-free synthetic OCR config small enough to train for a few steps on CPU."""
    document: dict[str, Any] = {
        "run": {"name": "trainer_unit", "seed": 1234, "deterministic": True},
        "data": {
            "dataset": "synthetic_ink",
            "mode": "offline",
            "image_size": [32, 96],
            "preprocess": ["grayscale", "normalize01", "resize_pad"],
            "batch_size": 4,
            "num_workers": 0,
            "n_samples": 4,
            "max_label_len": 12,
        },
        "model": {
            "family": "seq2seq_image",
            "encoder": "cnn_small",
            "decoder": "transformer",
            "d_model": 32,
            "layers": 1,
            "heads": 2,
            "dropout": 0.1,
            "max_len": 12,
        },
        "optim": {"name": "adamw", "lr": 3e-3, "warmup_steps": 0, "scheduler": "cosine"},
        "train": {
            "max_steps": 2,
            "grad_accum": 1,
            "amp": "off",
            "grad_clip": 1.0,
            "eval_every": 1000,
            "ckpt_every": 1000,
            "log_every": 1,
            # A test run is never citable; the repository working tree is dirty by definition
            # while an agent is editing it (ADR-0007).
            "allow_dirty": True,
        },
        "eval": {"beam": 1, "max_n_val": 2, "metrics": ["cer", "exprate"]},
        "checkpoint": {"keep_last": 3, "keep_best": 1, "metric": "val_loss", "mode": "min"},
        "tracking": {"backend": "jsonl", "experiment": "unit"},
    }
    return build_config(_merge(document, overrides))


def _trainer(
    run_dir: Path, cfg: TrainConfig, *, with_val: bool = False, device: str | None = None
) -> t.Trainer:
    """Build the registry objects for ``cfg`` and wire a trainer around them.

    ``device`` is forced only where a test needs bit-exact arithmetic; everywhere else the
    trainer picks what the host has, so the container really exercises CUDA.
    """
    from aicls.ml.registry import build_dataset, build_model
    from aicls.ocr import default_tokenizer

    tokenizer = default_tokenizer()
    model = build_model(cfg, tokenizer.vocab_size)
    train_ds = build_dataset(cfg, "train")
    # NOTE: the synthetic dataset has no splits, so "validation" here is the training data.
    # That makes every metric below a wiring check and nothing else.
    val_ds = build_dataset(cfg, "valid") if with_val else None
    return t.Trainer(
        cfg, model, train_ds, val_ds, run_dir=run_dir, tokenizer=tokenizer, device=device
    )


class _InterruptingDataset:
    """Wraps a dataset and raises ``KeyboardInterrupt`` after ``after`` item reads."""

    def __init__(self, inner: Any, after: int) -> None:
        self._inner = inner
        self._after = int(after)
        self.calls = 0

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int) -> Any:
        self.calls += 1
        if self.calls > self._after:
            raise KeyboardInterrupt
        return self._inner[index]


# --------------------------------------------------------------------------------------------
# device / dtype selection — the single most likely thing to break when the CUDA gate opens
# --------------------------------------------------------------------------------------------
def test_select_device_on_cpu_is_fp32_without_autocast() -> None:
    """A bf16 request on a machine without CUDA degrades to fp32, it does not fail."""
    spec = t.select_device("bf16", probe=t.DeviceProbe())

    assert spec.device == "cpu"
    assert spec.dtype == "float32"
    assert spec.autocast is False
    assert spec.needs_grad_scaler is False
    assert spec.fallback_reason == "amp_requires_cuda"


def test_select_device_off_on_cpu_reports_no_fallback() -> None:
    """``amp: off`` on CPU is exactly what was asked for, so nothing is reported as a fallback."""
    spec = t.select_device("off", probe=t.DeviceProbe())

    assert (spec.device, spec.dtype, spec.autocast, spec.fallback_reason) == (
        "cpu",
        "float32",
        False,
        "",
    )


def test_select_device_with_faked_cuda_enables_bf16_autocast() -> None:
    """The GPU path (unreachable here) is decided by a probe, so it can be tested on CPU."""
    probe = t.DeviceProbe(
        cuda_available=True, bf16_supported=True, device_name="NVIDIA GeForce RTX 5090"
    )
    spec = t.select_device("bf16", probe=probe)

    assert spec.device == "cuda"
    assert spec.is_cuda is True
    assert spec.dtype == "bfloat16"
    assert spec.autocast is True
    assert spec.needs_grad_scaler is False
    assert spec.fallback_reason == ""
    assert spec.device_name == "NVIDIA GeForce RTX 5090"


def test_select_device_falls_back_when_bf16_is_unsupported() -> None:
    """An old CUDA card without bf16 runs fp32 rather than silently producing garbage."""
    spec = t.select_device("bf16", probe=t.DeviceProbe(cuda_available=True, bf16_supported=False))

    assert spec.dtype == "float32"
    assert spec.autocast is False
    assert spec.fallback_reason == "bf16_unsupported"


def test_select_device_fp16_on_cuda_asks_for_a_grad_scaler() -> None:
    """fp16 autocast is only correct with loss scaling."""
    spec = t.select_device("fp16", probe=t.DeviceProbe(cuda_available=True, bf16_supported=True))

    assert (spec.dtype, spec.autocast, spec.needs_grad_scaler) == ("float16", True, True)


def test_select_device_refuses_an_explicit_cuda_request_without_cuda() -> None:
    """Asking for a device that is not there must fail loudly, not fall back."""
    with pytest.raises(AiclsError) as excinfo:
        t.select_device("off", device="cuda", probe=t.DeviceProbe())

    assert excinfo.value.details["reason"] == "cuda_unavailable"


def test_select_device_rejects_an_unknown_amp_mode() -> None:
    with pytest.raises(AiclsError) as excinfo:
        t.select_device("fp8", probe=t.DeviceProbe())

    assert excinfo.value.details["reason"] == "unknown_amp_mode"


def test_probe_device_reports_this_cpu_only_machine() -> None:
    """The real probe on this host: torch 2.11+cpu, so CUDA is absent (CUDA_INSTALL is closed)."""
    probe = t.probe_device()

    assert probe.cuda_available is torch.cuda.is_available()
    if not probe.cuda_available:
        assert probe.bf16_supported is False
        assert probe.device_name == "cpu"


def test_autocast_context_on_cpu_is_a_null_context() -> None:
    """No autocast, therefore no per-step warning spam on the CPU path."""
    spec = t.select_device("bf16", probe=t.DeviceProbe())
    with spec.autocast_context():
        tensor = torch.ones(2, 2)
    assert tensor.dtype == torch.float32


# --------------------------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------------------------
def test_schedule_factor_warms_up_then_decays() -> None:
    warm = [t.schedule_factor(step, warmup=2, total=10) for step in range(2)]
    assert warm == pytest.approx([0.5, 1.0])

    after = [t.schedule_factor(step, warmup=2, total=10) for step in range(2, 11)]
    assert after[0] == pytest.approx(1.0)
    assert after[-1] == pytest.approx(0.0, abs=1e-9)
    assert all(later <= earlier + 1e-12 for earlier, later in pairwise(after))


def test_schedule_factor_kinds() -> None:
    assert t.schedule_factor(5, warmup=0, total=10, kind="constant") == 1.0
    assert t.schedule_factor(5, warmup=0, total=10, kind="linear") == pytest.approx(0.5)
    with pytest.raises(AiclsError) as excinfo:
        t.schedule_factor(1, warmup=0, total=10, kind="triangular")
    assert excinfo.value.details["reason"] == "unknown_scheduler"


# --------------------------------------------------------------------------------------------
# early stopping
# --------------------------------------------------------------------------------------------
def test_early_stopping_counts_only_measured_values() -> None:
    stopper = t.EarlyStopping(2, mode="min")

    assert stopper.update(1.0) is True
    assert stopper.update(None) is False  # not measured: not an improvement, not a bad round
    assert stopper.bad_evals == 0
    assert stopper.update(2.0) is False
    assert stopper.should_stop is False
    assert stopper.update(3.0) is False
    assert stopper.should_stop is True
    assert stopper.best == 1.0


def test_early_stopping_disabled_never_stops() -> None:
    stopper = t.EarlyStopping(None, mode="max")
    for value in (1.0, 0.5, 0.4, 0.3):
        stopper.update(value)
    assert stopper.should_stop is False


# --------------------------------------------------------------------------------------------
# dataloaders
# --------------------------------------------------------------------------------------------
def test_dataloader_order_is_seeded_and_epoch_dependent(tmp_path: Path) -> None:
    """Reproducible batches: same (seed, epoch) is the same order, a new epoch reshuffles."""
    from aicls.ml.registry import build_dataset

    cfg = _cfg(data={"n_samples": 8, "batch_size": 2})
    dataset = build_dataset(cfg, "train")
    collate = t.collate_for(cfg)

    def ids(epoch: int) -> list[str]:
        loader = t.make_dataloader(
            dataset, batch_size=2, shuffle=True, seed=1234, epoch=epoch, collate=collate
        )
        return [name for batch in loader for name in batch["sample_ids"]]

    assert ids(0) == ids(0)
    assert ids(0) != ids(1)
    assert sorted(ids(0)) == sorted(ids(1))


# --------------------------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------------------------
def test_trainer_two_steps_synthetic_ocr_loss_decreases_cpu(tmp_path: Path) -> None:
    """Contractual (design document §10).

    WIRING CHECK, NOT A RESULT: two optimizer steps over four synthetic Bezier scribbles with a
    randomly initialised 60k-parameter model. It proves the batch reaches the model, the loss is
    finite and the optimizer moves the weights in the direction of the gradient. It says nothing
    whatsoever about handwriting recognition or about the CER < 10 % target.
    """
    cfg = _cfg()
    trainer = _trainer(tmp_path / "run", cfg)
    before = [p.detach().clone() for p in trainer.model.parameters()]

    report = trainer.fit()

    assert report.steps == 2
    assert (tmp_path / "run" / "run_manifest.json").is_file()
    assert (tmp_path / "run" / "env.json").is_file()
    # NOT `citable is False`: that held only because the developer's tree happened to be dirty,
    # so the assertion failed on a clean checkout — CI, a fresh clone, or anyone reproducing a
    # reported number, which is precisely the audience the citable flag exists for. `citable` is
    # about git provenance and follows the checkout; whether this run means anything is
    # `wiring_check`, and for two steps of a randomly initialised model on synthetic scribbles
    # that is true no matter how clean the tree is.
    assert report.wiring_check is True
    assert "WIRING CHECK" in report.notes.upper()

    rows = [row for row in read_metrics(tmp_path / "run") if row["split"] == "train"]
    losses = [row["metrics"]["loss"] for row in rows]
    assert len(losses) == 2
    assert all(value is not None and math.isfinite(value) for value in losses)
    assert losses[-1] < losses[0]
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, trainer.model.parameters(), strict=True)
    )


def test_trainer_writes_manifest_before_the_first_optimizer_step(tmp_path: Path) -> None:
    """An interrupted run must still be attributable (design document §4.1, ADR-0007)."""
    cfg = _cfg()
    trainer = _trainer(tmp_path / "run", cfg)
    trainer.train_ds = _InterruptingDataset(trainer.train_ds, after=0)

    with pytest.raises(KeyboardInterrupt):
        trainer.fit()

    assert (tmp_path / "run" / "run_manifest.json").is_file()
    assert trainer.step == 0


def test_trainer_interrupt_writes_an_interrupt_checkpoint_and_reraises(tmp_path: Path) -> None:
    """Ctrl-C keeps the work done so far and does not swallow the signal."""
    cfg = _cfg(data={"n_samples": 12}, train={"max_steps": 3})
    trainer = _trainer(tmp_path / "run", cfg)
    # 4 items per batch, one batch per optimizer step: the 9th read is inside batch 3.
    trainer.train_ds = _InterruptingDataset(trainer.train_ds, after=8)

    with pytest.raises(KeyboardInterrupt):
        trainer.fit()

    assert trainer.step == 2
    checkpoint = load_checkpoint(trainer.interrupt_dir, restore_rng=False)
    assert checkpoint.step == 2
    assert checkpoint.manifest["run_id"] == trainer.manifest.run_id


def test_trainer_gradient_accumulation_consumes_grad_accum_batches_per_step(
    tmp_path: Path,
) -> None:
    """grad_accum=2 means two micro-batches and one optimizer step."""
    cfg = _cfg(
        data={"n_samples": 16, "batch_size": 4},
        train={"max_steps": 2, "grad_accum": 2},
    )
    trainer = _trainer(tmp_path / "run", cfg)

    report = trainer.fit()

    assert report.steps == 2
    assert trainer.samples_seen == 2 * 2 * 4
    assert trainer.scheduler.last_epoch == 2  # one scheduler step per optimizer step


def test_trainer_early_stopping_stops_before_the_step_budget(tmp_path: Path) -> None:
    """A validation metric that keeps getting worse ends the run (design document §4.1)."""
    cfg = _cfg(
        train={"max_steps": 8, "eval_every": 1, "ckpt_every": 8, "early_stop_patience": 1},
        eval={"metrics": ["cer"]},
    )
    trainer = _trainer(tmp_path / "run", cfg, with_val=True)
    values = iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    trainer.evaluate = lambda **_: {"loss": 0.0, "val_loss": next(values)}  # type: ignore[method-assign]

    report = trainer.fit()

    assert report.early_stopped is True
    assert report.steps == 2  # the first value is the best, one worse one exhausts patience 1
    assert report.best_step == 1
    assert report.best_value == 1.0


def test_trainer_two_steps_and_resume(tmp_path: Path) -> None:
    """Contractual (design document §4.1, §10).

    A run resumed from a checkpoint must follow the *same* trajectory as an uninterrupted one:
    same step counter, same weights, same python/numpy/torch RNG state. Dropout is on (0.1), so
    an RNG that was not restored would show up as different weights at step 4.
    """
    cfg = _cfg(data={"n_samples": 12}, train={"max_steps": 4, "ckpt_every": 2})

    first = _trainer(tmp_path / "a", cfg)
    report_a = first.fit(max_steps=4)
    weights_a = [p.detach().clone() for p in first.model.parameters()]
    torch_rng_a = torch.get_rng_state()

    assert report_a.steps == 4
    checkpoint_dir = tmp_path / "a" / "checkpoints" / "step-00000002"
    assert checkpoint_dir.is_dir()

    second = _trainer(tmp_path / "b", cfg)
    restored = second.resume(checkpoint_dir)

    assert restored.step == 2
    assert second.step == 2
    assert restored.rng_restored is True

    report_b = second.fit(max_steps=4)

    assert report_b.steps == 4
    assert second.step == 4
    for original, resumed in zip(weights_a, second.model.parameters(), strict=True):
        assert torch.equal(original, resumed)
    assert torch.equal(torch_rng_a, torch.get_rng_state())


def test_trainer_evaluate_returns_the_requested_metrics(tmp_path: Path) -> None:
    """The validation pass decodes a subset and scores it with aicls.eval.metrics.

    The numbers are meaningless (random weights); only the plumbing is under test.
    """
    cfg = _cfg(train={"max_steps": 1}, eval={"beam": 1, "max_n_val": 2, "metrics": ["cer", "ter"]})
    trainer = _trainer(tmp_path / "run", cfg, with_val=True)

    scores = trainer.evaluate()

    assert set(scores) >= {"loss", "val_loss", "n_val", "cer", "ter"}
    assert scores["n_val"] == 2
    assert scores["loss"] > 0.0
    assert all(math.isfinite(value) for value in scores.values())  # no NaN


def test_trainer_beam_search_path_is_wired(tmp_path: Path) -> None:
    """``eval.beam > 1`` decodes with beam search instead of greedy."""
    cfg = _cfg(train={"max_steps": 1}, eval={"beam": 2, "max_n_val": 1, "metrics": ["exprate"]})
    trainer = _trainer(tmp_path / "run", cfg, with_val=True)

    scores = trainer.evaluate()

    assert "exprate" in scores
    assert scores["n_val"] == 1


def test_trainer_checkpoints_and_reports_the_best_by_metric(tmp_path: Path) -> None:
    cfg = _cfg(
        train={"max_steps": 2, "eval_every": 1, "ckpt_every": 1},
        eval={"metrics": ["cer"]},
        checkpoint={"metric": "val_loss", "mode": "min", "keep_last": 2, "keep_best": 1},
    )
    trainer = _trainer(tmp_path / "run", cfg, with_val=True)
    values = iter([5.0, 1.0])
    trainer.evaluate = lambda **_: {"loss": 0.0, "val_loss": next(values)}  # type: ignore[method-assign]

    report = trainer.fit()

    assert report.best_step == 2
    assert report.best_value == 1.0
    assert Path(report.best_checkpoint or "").name == "step-00000002"
    assert (tmp_path / "run" / "checkpoints" / "best.json").is_file()
    assert (tmp_path / "run" / "fit_report.json").is_file()


def test_fit_report_json_roundtrip(tmp_path: Path) -> None:
    report = t.FitReport(run_id="01J", run_name="unit", steps=2, wiring_check=True)
    path = report.save(tmp_path / "fit_report.json")

    assert path.read_text(encoding="utf-8").endswith("\n")
    assert t.FitReport.model_validate_json(path.read_text(encoding="utf-8")).steps == 2
    assert report.to_dict()["run_id"] == "01J"


# --------------------------------------------------------------------------------------------
# gates and configuration refusals
# --------------------------------------------------------------------------------------------
def test_trainer_refuses_a_real_dataset_without_the_training_gate(
    tmp_path: Path, test_env: dict[str, str]
) -> None:
    """ADR-0012: MathWriting is not gate-free, so ``fit()`` must refuse with exit code 3.

    The refusal happens before the data is touched — ``train_ds`` here is deliberately empty.
    """
    from aicls.core.settings import get_settings

    get_settings.cache_clear()
    cfg = _cfg(data={"dataset": "mathwriting", "root": str(tmp_path / "shards")})
    assert cfg.is_gate_free is False
    trainer = t.Trainer(cfg, model=None, train_ds=[], val_ds=None, run_dir=tmp_path / "run")

    with pytest.raises(GateError) as excinfo:
        trainer.fit()

    get_settings.cache_clear()
    assert excinfo.value.exit_code == 3
    assert excinfo.value.details["gate"] == "TRAINING"
    assert not (tmp_path / "run" / "run_manifest.json").exists()


def test_trainer_refuses_the_whisper_family(tmp_path: Path) -> None:
    """The OCR loop is not the ASR loop; ``WhisperFineTuner`` owns that path (ADR-0006)."""
    cfg = _cfg(
        data={"dataset": "synthetic_audio", "mode": "online"},
        model={"family": "whisper", "encoder": None},
    )
    with pytest.raises(AiclsError) as excinfo:
        t.Trainer(cfg, model=None, train_ds=[], val_ds=None, run_dir=tmp_path / "run")

    assert excinfo.value.details["reason"] == "family_not_supported_by_trainer"


def test_trainer_refuses_a_family_and_mode_mismatch(tmp_path: Path) -> None:
    """An image model fed online stroke features would silently learn nothing."""
    cfg = _cfg(data={"mode": "online"}, model={"family": "seq2seq_image"})
    with pytest.raises(AiclsError) as excinfo:
        t.Trainer(cfg, model=None, train_ds=[], val_ds=None, run_dir=tmp_path / "run")

    assert excinfo.value.details["reason"] == "family_mode_mismatch"


def test_trainer_refuses_lora_for_the_ocr_families(tmp_path: Path) -> None:
    """LoRA belongs to the Whisper path; silently ignoring the key would be a lie."""
    cfg = _cfg(model={"lora": {"enabled": True}})
    with pytest.raises(AiclsError) as excinfo:
        t.Trainer(cfg, model=None, train_ds=[], val_ds=None, run_dir=tmp_path / "run")

    assert excinfo.value.details["reason"] == "lora_not_supported_by_trainer"


def test_require_training_gate_is_silent_for_synthetic_data(tmp_path: Path) -> None:
    """``data.dataset: synthetic_*`` is gate-free by ADR-0012."""
    assert t.require_training_gate(_cfg()) is False


# --------------------------------------------------------------------------------------------
# stroke (online) family
# --------------------------------------------------------------------------------------------
def test_trainer_trains_the_online_stroke_family_too(tmp_path: Path) -> None:
    """The same loop drives ADR-0013's online path; only the collate function changes."""
    cfg = _cfg(
        data={"mode": "online", "n_samples": 4, "batch_size": 2},
        model={"family": "seq2seq_stroke", "encoder": "stroke_transformer"},
        train={"max_steps": 1},
    )
    trainer = _trainer(tmp_path / "run", cfg)

    report = trainer.fit()

    assert report.steps == 1
    assert report.train_loss is not None and report.train_loss > 0.0


# --------------------------------------------------------------------------------------------
# optimizer, freezing and the default run directory
# --------------------------------------------------------------------------------------------
def test_build_optimizer_honours_the_optim_block(tmp_path: Path) -> None:
    """Each ``optim.name`` maps to the torch optimizer the config asked for."""
    model = torch.nn.Linear(4, 3)

    adamw = t.build_optimizer(model, _cfg())
    adam = t.build_optimizer(model, _cfg(optim={"name": "adam"}))
    sgd = t.build_optimizer(model, _cfg(optim={"name": "sgd", "momentum": 0.8}))

    assert isinstance(adamw, torch.optim.AdamW)
    assert isinstance(adam, torch.optim.Adam)
    assert isinstance(sgd, torch.optim.SGD)
    assert sgd.param_groups[0]["momentum"] == 0.8
    assert adamw.param_groups[0]["lr"] == pytest.approx(3e-3)


def test_build_optimizer_refuses_a_fully_frozen_model() -> None:
    """Training a model with nothing to train is a configuration error, not a silent no-op."""
    model = torch.nn.Linear(4, 3)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with pytest.raises(AiclsError) as excinfo:
        t.build_optimizer(model, _cfg())

    assert excinfo.value.details["reason"] == "no_trainable_parameters"


def test_apply_freeze_freezes_the_encoder_only_when_asked(tmp_path: Path) -> None:
    cfg = _cfg(model={"freeze_encoder": True})
    trainer = _trainer(tmp_path / "run", cfg)

    assert t.apply_freeze(trainer.model, _cfg()) == 0  # flag off: nothing happens
    frozen = t.apply_freeze(trainer.model, cfg)

    assert frozen > 0
    assert all(not p.requires_grad for p in trainer.model.encoder.parameters())
    assert any(p.requires_grad for p in trainer.model.decoder.parameters())


def test_apply_freeze_reports_a_model_without_an_encoder() -> None:
    with pytest.raises(AiclsError) as excinfo:
        t.apply_freeze(torch.nn.Linear(2, 2), _cfg(model={"freeze_encoder": True}))

    assert excinfo.value.details["reason"] == "no_encoder_to_freeze"


def test_trainer_default_run_dir_is_unique_under_the_runs_root(test_env: dict[str, str]) -> None:
    """Two runs of the same config never share one ``metrics.jsonl`` (ADR-0007)."""
    from aicls.core.settings import get_settings

    get_settings.cache_clear()
    try:
        first = t.Trainer(_cfg(), model=None, train_ds=[], val_ds=None).run_dir
        second = t.Trainer(_cfg(), model=None, train_ds=[], val_ds=None).run_dir
    finally:
        get_settings.cache_clear()

    assert first != second
    assert first.parent.name == "runs"
    assert first.name.startswith("trainer_unit-")


def test_schedule_factor_rejects_a_negative_step() -> None:
    with pytest.raises(AiclsError) as excinfo:
        t.schedule_factor(-1, warmup=0, total=10)

    assert excinfo.value.details["reason"] == "negative_step"


def test_early_stopping_rejects_an_invalid_patience_or_mode() -> None:
    with pytest.raises(AiclsError) as excinfo:
        t.EarlyStopping(0)
    assert excinfo.value.details["reason"] == "invalid_patience"

    with pytest.raises(AiclsError) as excinfo:
        t.EarlyStopping(1, mode="lowest")  # type: ignore[arg-type]
    assert excinfo.value.details["reason"] == "invalid_mode"


def test_score_predictions_ignores_loss_and_unknown_names() -> None:
    """``loss`` comes from the loop, not from decoded text; unknown names are already validated."""
    pairs = [("x^{2}", "x^{2}"), ("\frac{1}{2}", "\frac{1}{3}")]

    assert t.score_predictions([], ["cer"]) == {}
    scores = t.score_predictions(pairs, ["cer", "ter", "wer", "exprate", "loss"])

    assert set(scores) == {"cer", "ter", "wer", "exprate"}
    assert scores["exprate"] == pytest.approx(0.5)


# --------------------------------------------------------------------------------------------
# symeq — the one metric that runs a CAS over model output
#
# On 2026-09-08 an evaluation called symbolic_equivalence on a decoded hypothesis and never came
# back: 4 h 47 min inside sympy's _eval_power on a power tower the model invented, killed with no
# checkpoint written because the loop evaluates before it saves. The rule that came out of it is
# that the training loop may not call a comparison that does not carry its own bound.
# --------------------------------------------------------------------------------------------
def test_the_shipped_symbolic_comparison_declares_no_bound() -> None:
    """The premise of every test below. If this ever fails, symeq became callable - good news,
    but then these tests are asserting the wrong branch and must be revisited."""
    scorer, reason = t.symeq_scorer()

    assert scorer is None
    assert reason == t.SYMEQ_UNAVAILABLE


def test_score_predictions_omits_symeq_rather_than_running_an_unbounded_cas(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Skipping costs the run one number. Not skipping cost the run five hours."""
    t._SYMEQ_SKIP_ANNOUNCED.clear()
    pairs = [("x^{2}", "x^{2}"), ("\frac{1}{2}", "\frac{1}{3}")]

    scores = t.score_predictions(pairs, ["cer", "symeq"])

    assert set(scores) == {"cer"}, (
        "symeq must be absent, not zero: absent and 0.0 mean opposite things"
    )
    assert "ml.trainer.symeq_skipped" in capsys.readouterr().out


def test_the_symeq_skip_is_announced_once_not_once_per_evaluation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A 30-epoch run evaluates over a hundred times and the answer cannot change between them."""
    t._SYMEQ_SKIP_ANNOUNCED.clear()
    pairs = [("x", "x")]

    for _ in range(3):
        t.score_predictions(pairs, ["symeq"])

    assert capsys.readouterr().out.count("ml.trainer.symeq_skipped") == 1


def test_score_predictions_uses_a_bounded_comparison_as_soon_as_one_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal is about the bound, not about symeq. Declare one and the metric comes back.

    This is what keeps the shipped configurations honest: they still ask for symeq, so the day
    :mod:`aicls.eval.metrics` grows a bounded entry point the metric starts being reported with
    no configuration change anywhere.
    """
    from aicls.eval import metrics as eval_metrics

    seen: list[tuple[str, str]] = []

    def bounded(ref: str, hyp: str) -> str:
        seen.append((ref, hyp))
        return "equal" if ref == hyp else "different"

    monkeypatch.setattr(eval_metrics, "bounded_symbolic_equivalence", bounded, raising=False)
    pairs = [("x^{2}", "x^{2}"), ("\frac{1}{2}", "\frac{1}{3}")]

    scorer, entry = t.symeq_scorer()
    scores = t.score_predictions(pairs, ["symeq"])

    assert scorer is bounded
    assert entry == "bounded_symbolic_equivalence"
    assert scores["symeq"] == pytest.approx(0.5)
    assert seen == pairs, "every pair must reach the bounded comparison"


def test_a_bound_declared_in_place_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A guard may reasonably be added to the existing function rather than under a new name."""
    from aicls.eval import metrics as eval_metrics

    def in_place(ref: str, hyp: str) -> str:  # pragma: no cover - never called here
        return "equal"

    in_place.timeout_s = 2.5  # type: ignore[attr-defined]
    monkeypatch.setattr(eval_metrics, "symbolic_equivalence", in_place)

    scorer, entry = t.symeq_scorer()

    assert scorer is in_place
    assert "bounded in place" in entry


def test_a_module_wide_budget_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Or as a budget the whole module answers to."""
    from aicls.eval import metrics as eval_metrics

    monkeypatch.setattr(eval_metrics, "SYMEQ_TIMEOUT_S", 2.5, raising=False)

    scorer, entry = t.symeq_scorer()

    assert scorer is eval_metrics.symbolic_equivalence
    assert "module budget" in entry


# --------------------------------------------------------------------------------------------
# step budget and failure bookkeeping
# --------------------------------------------------------------------------------------------
def test_a_step_budget_argument_can_only_lower_the_configured_one(tmp_path: Path) -> None:
    """``fit(max_steps=...)`` is a ceiling, not an override, and the direction matters.

    A run's step budget is part of what an approval cites (ADR-0012), so a flag that could raise
    it would let an operator train past what was approved without touching the configuration the
    approval names. Lowering is always safe; raising is not. Until 2026-09-09 the CLI's own help
    text said "overrides the config", which is the opposite of what this asserts.
    """
    trainer = _trainer(tmp_path / "run", _cfg(train={"max_steps": 2}))

    report = trainer.fit(max_steps=10)

    assert report.steps == 2, "the argument raised a budget the configuration had capped"


def test_trainer_runs_one_epoch_when_no_budget_is_configured(tmp_path: Path) -> None:
    """With neither ``max_steps`` nor ``epochs`` the loop does exactly one epoch."""
    cfg = _cfg(
        data={"n_samples": 8, "batch_size": 4},
        train={"max_steps": None, "epochs": None, "ckpt_every": 100},
    )
    trainer = _trainer(tmp_path / "run", cfg)

    report = trainer.fit()

    assert report.steps == 2  # 8 samples / batch 4 = 2 batches = 2 optimizer steps
    assert report.epochs == pytest.approx(1.0)


def test_trainer_epoch_budget_is_converted_to_steps(tmp_path: Path) -> None:
    cfg = _cfg(
        data={"n_samples": 8, "batch_size": 4},
        train={"max_steps": None, "epochs": 2, "ckpt_every": 100},
    )
    trainer = _trainer(tmp_path / "run", cfg)

    report = trainer.fit()

    assert report.steps == 4
    assert report.epochs == pytest.approx(2.0)


class _FailingDataset:
    """Raises a plain error mid-epoch, i.e. not the Ctrl-C path."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls = 0

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int) -> Any:
        self.calls += 1
        if self.calls > 4:
            raise RuntimeError("simulated data failure")
        return self._inner[index]


def test_trainer_marks_the_tracked_run_failed_on_an_error(tmp_path: Path) -> None:
    """A crashed run must not be left recorded as still running (ADR-0007)."""
    import json as _json

    cfg = _cfg(data={"n_samples": 8}, train={"max_steps": 2})
    trainer = _trainer(tmp_path / "run", cfg)
    trainer.train_ds = _FailingDataset(trainer.train_ds)

    with pytest.raises(RuntimeError):
        trainer.fit()

    record = _json.loads((tmp_path / "run" / "params.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"


def test_trainer_warns_that_the_synthetic_warmup_curriculum_is_not_wired(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``data.synthetic_warmup_epochs`` is accepted by the schema but this loop cannot honour it.

    The MathWriting curriculum of design document §1 ("two synthetic-only warm-up epochs, then
    ratio 0.5") needs the dataset rebuilt at an epoch boundary; the trainer is handed an
    already-built dataset. Until that is implemented the key must be loud, not silent.
    """
    cfg = _cfg(data={"synthetic_warmup_epochs": 2, "synthetic_ratio": 0.5})
    t.Trainer(cfg, model=None, train_ds=[], val_ds=None, run_dir=tmp_path / "run")

    assert "ml.trainer.synthetic_warmup_not_implemented" in capsys.readouterr().out


# --------------------------------------------------------------------------------------------
# BatchNorm running statistics — what the first real training run measured instead of the model
#
# runs/ocr_r34_tfm_full2 (RTX 5090, 2026-09-08) reported val_cer 7.34 and ExpRate 0.0 at step
# 18000 while its training loss sat between 0.30 and 0.75. The weights were fine: scored with
# batch statistics the same checkpoint gave a teacher-forced loss of 0.32 against 8.52 with the
# accumulated running statistics, and CER 0.152 once those statistics were re-estimated. The run
# had been early-stopping and picking its "best" checkpoint on the 7.34.
# --------------------------------------------------------------------------------------------
def test_the_fixture_model_carries_batchnorm() -> None:
    """Every test below is vacuous if it does not - so assert it once, loudly, here."""
    from aicls.ml.registry import build_model
    from aicls.ocr import default_tokenizer

    model = build_model(_cfg(), default_tokenizer().vocab_size)
    found = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]

    assert found, "cnn_small no longer has BatchNorm; the recalibration tests below prove nothing"


def test_recalibration_reads_the_training_split_and_never_the_validation_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-estimating on the split about to be scored would tune the model on the test.

    It is the cheap mistake here: the validation loader is already built two lines below, the
    statistics would fit the scored distribution a little better, and every number the project
    reports would be flattered by an amount nobody could see. On the real checkpoint the honest
    choice costs nothing anyway - 0.152 from training data against 0.157 from validation data.
    """
    cfg = _cfg(eval={"recalibrate_norm_samples": 4})
    trainer = _trainer(tmp_path / "run", cfg, with_val=True)
    original = t.Trainer._loader
    seen: list[str] = []

    def spy(self: t.Trainer, dataset: Any, *, shuffle: bool, epoch: int) -> Any:
        if dataset is self.train_ds:
            seen.append("train")
        elif dataset is self.val_ds:
            seen.append("valid")
        else:  # pragma: no cover - would mean the trainer grew a third dataset
            seen.append("other")
        return original(self, dataset, shuffle=shuffle, epoch=epoch)

    monkeypatch.setattr(t.Trainer, "_loader", spy)
    trainer.evaluate()

    assert seen == ["train", "valid"], (
        f"loaders opened in order {seen}; recalibration must read the training split, and the "
        "validation split must be opened once, to be scored"
    )


def test_recalibration_without_a_training_split_refuses_rather_than_using_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A trainer built for scoring alone has no honest source of statistics, so it takes none."""
    from aicls.ml.registry import build_dataset, build_model
    from aicls.ocr import default_tokenizer

    cfg = _cfg(eval={"recalibrate_norm_samples": 4})
    tokenizer = default_tokenizer()
    trainer = t.Trainer(
        cfg,
        build_model(cfg, tokenizer.vocab_size),
        None,
        build_dataset(cfg, "valid"),
        run_dir=tmp_path / "run",
        tokenizer=tokenizer,
    )

    assert trainer._recalibrate_norm() is None
    assert "ml.trainer.norm_statistics_not_recalibrated" in capsys.readouterr().out


def test_recalibration_recovers_a_model_whose_running_statistics_went_stale(
    tmp_path: Path,
) -> None:
    """The defect end to end: displace the statistics, watch the score move, then recover it.

    This fixture pins the *mechanism*, not the magnitude. A 60k-parameter model trained for forty
    steps on eight synthetic scribbles cannot reproduce the 26x loss gap the RTX 5090 produced at
    step 18000 - it has far less to lose. What it does reproduce is the shape: the score depends
    on buffers the optimizer never touched, and re-estimating them undoes the damage completely
    rather than partly. In the real run, as here, the fresh estimate also beat the exponential
    average it replaced.
    """
    settings: dict[str, object] = {"data": {"n_samples": 8}, "train": {"max_steps": 40}}
    cfg = _cfg(eval={"recalibrate_norm_samples": 8}, **settings)
    quiet = _cfg(eval={"recalibrate_norm_samples": None}, **settings)
    trainer = _trainer(tmp_path / "run", cfg, with_val=True)
    modules = trainer._norm_modules()
    # An untrained model is insensitive to its own normalisation - it is already emitting noise,
    # so corrupting the statistics changes nothing measurable. Give it something to lose first.
    trainer.cfg = quiet
    trainer.fit()

    healthy = trainer.evaluate()["val_loss"]
    with torch.no_grad():
        for module in modules:
            module.running_mean.add_(5.0)
            module.running_var.mul_(9.0)
    stale = trainer.evaluate()["val_loss"]
    trainer.cfg = cfg
    recovered = trainer.evaluate()

    assert stale > healthy, (
        f"displacing the statistics left val_loss at {stale:.4f} against {healthy:.4f}; if the "
        "score does not depend on the buffers at all, this fixture cannot show the defect"
    )
    assert recovered["val_loss"] < stale, (
        f"recalibration left val_loss at {recovered['val_loss']:.4f} against a stale {stale:.4f}"
    )
    assert recovered["val_loss"] <= healthy * 1.02, (
        f"recalibration recovered only part of the damage: {recovered['val_loss']:.4f} against an "
        f"uncorrupted {healthy:.4f}. A partial recovery still leaves the loop selecting "
        "checkpoints on a number that describes the statistics."
    )
    assert recovered["norm_recalibrated_n"] == 8.0
    assert "norm_recalibration_seconds" in recovered


def test_a_recalibrated_row_says_so_and_an_uncalibrated_row_stays_silent(tmp_path: Path) -> None:
    """Provenance: a reader of metrics.jsonl must not have to guess which kind of number it is."""
    on = _trainer(tmp_path / "on", _cfg(eval={"recalibrate_norm_samples": 4}), with_val=True)
    off = _trainer(tmp_path / "off", _cfg(eval={"recalibrate_norm_samples": None}), with_val=True)

    assert on.evaluate()["norm_recalibrated_n"] == 4.0
    assert "norm_recalibrated_n" not in off.evaluate()


def test_recalibration_hands_the_training_loop_back_exactly_what_it_borrowed(
    tmp_path: Path,
) -> None:
    """Momentum is restored so continued training keeps its own averaging schedule.

    The model also goes back to training mode, because ``fit()`` calls ``evaluate()`` mid-loop.
    """
    cfg = _cfg(eval={"recalibrate_norm_samples": 4})
    trainer = _trainer(tmp_path / "run", cfg, with_val=True)
    modules = trainer._norm_modules()
    before = [m.momentum for m in modules]
    trainer.model.train()

    trainer.evaluate()

    assert [m.momentum for m in modules] == before
    assert all(m.momentum is not None for m in modules), (
        "momentum=None left behind would make continued training use a cumulative average"
    )
    assert trainer.model.training


def test_recalibration_is_inert_for_a_model_without_batchnorm(tmp_path: Path) -> None:
    """A transformer-only or Whisper model must not pay for a defect it cannot have."""
    cfg = _cfg(eval={"recalibrate_norm_samples": 4})
    trainer = t.Trainer(
        cfg, torch.nn.Linear(2, 2), train_ds=[], val_ds=None, run_dir=tmp_path / "run"
    )

    assert trainer._norm_modules() == []
    assert trainer._recalibrate_norm() is None


def test_scoring_a_batchnorm_model_without_recalibration_is_loud(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The silence is what cost the first run: nothing said the numbers were suspect."""
    cfg = _cfg(eval={"recalibrate_norm_samples": None})
    trainer = _trainer(tmp_path / "run", cfg, with_val=True)

    trainer.evaluate()

    assert "ml.trainer.norm_statistics_not_recalibrated" in capsys.readouterr().out


# --------------------------------------------------------------------------------------------
# the time budget — where a run's wall clock actually goes
# --------------------------------------------------------------------------------------------
def test_time_budget_splits_the_wall_clock_into_phases() -> None:
    """``wall_time_s`` alone hid a run that trained 1.4 s and validated 592 s (RTX 5090)."""
    budget = t.TimeBudget()
    budget.add_eval(300.0, n_val=16)
    budget.add_eval(292.0, n_val=16)
    budget.add_checkpoint(2.0)
    stats = t.TrainStats()
    stats.add(1, 0.7, 0.1)
    stats.add(2, 0.7, 0.1)
    budget.adopt_train(stats)
    budget.wall_time_s = 613.0

    assert budget.train_seconds == pytest.approx(1.4)
    assert budget.eval_seconds == pytest.approx(592.0)
    assert budget.checkpoint_seconds == pytest.approx(2.0)
    assert budget.dataloader_wait_s == pytest.approx(0.2)
    assert budget.evaluations == 2
    assert budget.eval_samples == 32
    assert budget.seconds_per_eval == pytest.approx(296.0)
    assert budget.seconds_per_val_sample == pytest.approx(18.5)
    assert budget.eval_share_pct == pytest.approx(100.0 * 592.0 / 613.0)
    assert budget.other_seconds == pytest.approx(613.0 - 595.4)


def test_time_budget_reports_none_rather_than_a_zero_it_did_not_measure() -> None:
    """No validation pass means no per-sample cost — not ``0.0``, which would price it free."""
    budget = t.TimeBudget()

    assert budget.seconds_per_eval is None
    assert budget.seconds_per_val_sample is None
    assert budget.eval_share_pct is None
    assert budget.other_seconds == 0.0
    assert budget.to_dict()["seconds_per_val_sample"] is None


# --------------------------------------------------------------------------------------------
# validation coverage — a truncated split is a different number, not a cheaper one
# --------------------------------------------------------------------------------------------
def test_val_coverage_records_the_truncation_next_to_the_metric() -> None:
    coverage = t.ValCoverage(n_val=200, n_total=2000, limit=200)

    assert coverage.truncated is True
    assert coverage.coverage_pct == pytest.approx(10.0)
    assert coverage.as_metrics() == {"n_val": 200.0, "n_val_total": 2000.0, "val_truncated": 1.0}


def test_val_coverage_of_a_whole_split_is_not_truncated() -> None:
    coverage = t.ValCoverage(n_val=100, n_total=100, limit=2000)

    assert coverage.truncated is False
    assert coverage.as_metrics()["val_truncated"] == 0.0


def test_val_coverage_omits_a_total_it_does_not_know() -> None:
    """An invented split size would be worse than a missing one."""
    coverage = t.ValCoverage(n_val=8, n_total=None, limit=8)

    assert coverage.truncated is False
    assert coverage.coverage_pct is None
    assert set(coverage.as_metrics()) == {"n_val"}


def test_split_size_returns_none_for_a_dataset_without_a_length() -> None:
    class _Streaming:
        def __getitem__(self, index: int) -> int:
            return index

    assert t.split_size([1, 2, 3]) == 3
    assert t.split_size(_Streaming()) is None


def test_limit_val_dataset_truncates_and_records_the_truncation() -> None:
    """The one-call fix for a loop that is handed a dataset (the ASR path, wave-6 §3.4)."""
    view, coverage = t.limit_val_dataset(list(range(64)), 8)

    assert len(view) == 8
    assert list(view) == list(range(8))
    assert view[-1] == 7
    assert coverage.n_val == 8
    assert coverage.n_total == 64
    assert coverage.truncated is True
    with pytest.raises(IndexError):
        _ = view[8]


def test_limit_val_dataset_leaves_a_short_split_alone() -> None:
    source = list(range(4))

    view, coverage = t.limit_val_dataset(source, 8)

    assert view is source  # nothing is wrapped for nothing
    assert coverage.truncated is False
    assert coverage.n_val == 4


def test_limit_val_dataset_handles_no_cap_and_no_dataset() -> None:
    source = list(range(4))
    kept, coverage = t.limit_val_dataset(source, None)
    empty, none_coverage = t.limit_val_dataset(None, 8)

    assert kept is source
    assert coverage.n_val == 4
    assert empty is None
    assert none_coverage.n_val == 0


def test_limit_val_dataset_refuses_a_negative_cap() -> None:
    with pytest.raises(AiclsError) as excinfo:
        t.limit_val_dataset([1, 2], -1)

    assert excinfo.value.details["reason"] == "negative_max_n_val"


# --------------------------------------------------------------------------------------------
# batched decoding
# --------------------------------------------------------------------------------------------
def test_as_token_ids_accepts_every_shape_the_decoders_return() -> None:
    """Bare ids, ``(ids, score)`` and a best-first beam list all denote one hypothesis."""
    assert t.as_token_ids([4, 5, 6]) == [4, 5, 6]
    assert t.as_token_ids(([4, 5], -0.5)) == [4, 5]
    assert t.as_token_ids([([4, 5], -0.5), ([7], -2.0)]) == [4, 5]
    assert t.as_token_ids([]) == []


def test_as_token_ids_refuses_a_shape_it_does_not_recognise() -> None:
    """A report-grade metric must not be decoded out of something unrecognised."""
    with pytest.raises(AiclsError) as excinfo:
        t.as_token_ids(object())

    assert excinfo.value.details["reason"] == "unknown_decoder_result"
    with pytest.raises(AiclsError):
        t.as_token_ids((1, 2, 3))


def test_batched_greedy_ids_reproduces_per_sample_greedy_decode(tmp_path: Path) -> None:
    """The batching must not change the hypothesis, only the wall clock.

    Batching moves the per-sample loop into the batch dimension, which is where the time goes on a
    GPU (18.5 s per sample was measured on an RTX 5090 with the per-sample loop). The tokens must
    be identical, otherwise every CER moves for a reason that has nothing to do with the model.

    **Exact on CPU only.** On CUDA a matmul over eight rows and a matmul over one row take
    different kernels and different reduction orders, so near-tied logits can pick different
    argmaxes; measured on the RTX 5090, 6 of 8 rows agreed token for token and the other two
    diverged after 84 and 108 tokens of a 160-token random-weight sequence (the same divergence
    appears between ``greedy_decode_batch`` and ``greedy_decode``, so it is arithmetic and not
    logic). This test therefore forces the CPU, where the comparison is exact.
    """
    from aicls.ocr.models.seq2seq import greedy_decode

    cfg = _cfg(data={"n_samples": 6, "batch_size": 3})
    trainer = _trainer(tmp_path / "run", cfg, with_val=True, device="cpu")
    trainer.model.eval()
    loader = trainer._loader(trainer.val_ds, shuffle=False, epoch=0)
    batch = next(iter(loader))
    inputs = batch["images"]

    batched = t.batched_greedy_ids(trainer.model, inputs, max_len=12)
    one_by_one = [greedy_decode(trainer.model, inputs[i : i + 1], max_len=12)[0] for i in range(3)]

    assert len(batched) == 3
    assert batched == one_by_one


def test_batch_decoders_reports_what_the_ocr_package_exposes() -> None:
    """The resolver states a fact about ``aicls.ocr``; it does not decide anything itself."""
    from aicls.ocr.models import seq2seq

    t.batch_decoders.cache_clear()
    greedy, beam = t.batch_decoders()
    t.batch_decoders.cache_clear()

    assert (greedy is None) == (not hasattr(seq2seq, t.BATCH_GREEDY_ENTRY))
    assert (beam is None) == (not hasattr(seq2seq, t.BATCH_BEAM_ENTRY))


def test_evaluate_decodes_in_batches_not_one_sample_at_a_time(tmp_path: Path) -> None:
    """One decoder call per micro-batch, each carrying the whole batch."""
    cfg = _cfg(
        data={"n_samples": 8, "batch_size": 4},
        train={"max_steps": 1},
        eval={"beam": 1, "max_n_val": 8, "metrics": ["cer"]},
    )
    trainer = _trainer(tmp_path / "run", cfg, with_val=True)
    widths: list[int] = []
    inner = trainer._decode_ids

    def spy(inputs: Any) -> list[list[int]]:
        widths.append(int(inputs.shape[0]))
        return inner(inputs)

    trainer._decode_ids = spy  # type: ignore[method-assign]

    scores = trainer.evaluate()

    assert widths == [4, 4]  # two batches, not eight single-sample calls
    assert scores["n_val"] == 8.0


def test_evaluate_publishes_the_truncation_next_to_the_metric(tmp_path: Path) -> None:
    """Contractual: a CER over 2 of 8 samples must not read like a CER over 8."""
    cfg = _cfg(
        data={"n_samples": 8, "batch_size": 2},
        eval={"beam": 1, "max_n_val": 2, "metrics": ["cer"]},
    )
    trainer = _trainer(tmp_path / "run", cfg, with_val=True)

    scores = trainer.evaluate()

    assert scores["n_val"] == 2.0
    assert scores["n_val_total"] == 8.0
    assert scores["val_truncated"] == 1.0
    assert scores["eval_seconds"] > 0.0
    assert trainer.coverage.truncated is True


def test_evaluate_says_so_when_nothing_was_truncated(tmp_path: Path) -> None:
    cfg = _cfg(
        data={"n_samples": 4, "batch_size": 2},
        eval={"beam": 1, "max_n_val": 100, "metrics": ["cer"]},
    )
    trainer = _trainer(tmp_path / "run", cfg, with_val=True)

    scores = trainer.evaluate()

    assert scores["n_val"] == 4.0
    assert scores["n_val_total"] == 4.0
    assert scores["val_truncated"] == 0.0


def test_evaluate_records_its_own_cost_in_the_time_budget(tmp_path: Path) -> None:
    cfg = _cfg(data={"n_samples": 4, "batch_size": 2}, eval={"beam": 1, "max_n_val": 4})
    trainer = _trainer(tmp_path / "run", cfg, with_val=True)

    trainer.evaluate()
    trainer.evaluate()

    assert trainer.budget.evaluations == 2
    assert trainer.budget.eval_samples == 8
    assert trainer.budget.eval_seconds > 0.0
    assert trainer.budget.seconds_per_val_sample is not None
    assert trainer.eval_decoder != ""


def test_evaluate_falls_back_to_per_sample_beam_and_names_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a batched beam entry point the slow loop survives — and says so in the report."""
    monkeypatch.setattr(t, "batch_decoders", lambda: (None, None))
    cfg = _cfg(
        data={"n_samples": 2, "batch_size": 2},
        eval={"beam": 2, "max_n_val": 2, "metrics": ["exprate"]},
    )
    trainer = _trainer(tmp_path / "run", cfg, with_val=True)

    scores = trainer.evaluate()

    assert scores["n_val"] == 2.0
    assert "per sample" in trainer.eval_decoder


def test_evaluate_uses_the_local_batched_greedy_when_ocr_exposes_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The greedy path batches on this side too, so a slow decoder cannot stall a run."""
    monkeypatch.setattr(t, "batch_decoders", lambda: (None, None))
    cfg = _cfg(data={"n_samples": 4, "batch_size": 4}, eval={"beam": 1, "max_n_val": 4})
    trainer = _trainer(tmp_path / "run", cfg, with_val=True)

    scores = trainer.evaluate()

    assert scores["n_val"] == 4.0
    assert trainer.eval_decoder == t.LOCAL_GREEDY_PATH


def test_evaluate_uses_the_batched_entry_points_when_they_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``aicls.ocr`` exposes the batched decoders, this loop must call them once per batch."""
    calls: list[int] = []

    def fake_greedy(model: Any, inputs: Any, *, max_len: int) -> list[tuple[list[int], float]]:
        calls.append(int(inputs.shape[0]))
        return [([5, 6], -0.1) for _ in range(int(inputs.shape[0]))]

    monkeypatch.setattr(t, "batch_decoders", lambda: (fake_greedy, None))
    cfg = _cfg(data={"n_samples": 4, "batch_size": 4}, eval={"beam": 1, "max_n_val": 4})
    trainer = _trainer(tmp_path / "run", cfg, with_val=True)

    scores = trainer.evaluate()

    assert calls == [4]
    assert scores["n_val"] == 4.0
    assert trainer.eval_decoder == f"seq2seq.{t.BATCH_GREEDY_ENTRY}"


def test_fit_report_splits_the_time_budget(tmp_path: Path) -> None:
    """Contractual (wave 7): the report must say where the wall clock went.

    WIRING CHECK: a 60k-parameter model over four generated scribbles on CPU. The *numbers* are
    meaningless; the split and its internal consistency are what is under test.
    """
    import json as _json

    cfg = _cfg(
        data={"n_samples": 4, "batch_size": 2},
        train={"max_steps": 2, "eval_every": 1, "ckpt_every": 1},
        eval={"beam": 1, "max_n_val": 2, "metrics": ["cer"]},
    )
    trainer = _trainer(tmp_path / "run", cfg, with_val=True)

    report = trainer.fit()

    assert report.train_seconds > 0.0
    assert report.eval_seconds > 0.0
    assert report.checkpoint_seconds > 0.0
    assert report.evaluations == 2
    assert report.dataloader_wait_s >= 0.0
    assert report.other_seconds >= 0.0
    total = (
        report.train_seconds
        + report.eval_seconds
        + report.checkpoint_seconds
        + report.other_seconds
    )
    assert total == pytest.approx(report.wall_time_s, rel=1e-6)
    assert report.seconds_per_val_sample == pytest.approx(report.eval_seconds / 4.0)
    assert 0.0 < (report.eval_share_pct or 0.0) <= 100.0

    written = _json.loads((tmp_path / "run" / "fit_report.json").read_text(encoding="utf-8"))
    assert written["eval_seconds"] == report.eval_seconds
    assert written["n_val"] == 2
    assert written["n_val_total"] == 4
    assert written["val_truncated"] is True
    assert written["eval_decoder"] != ""


def test_fit_report_has_no_eval_coverage_without_a_validation_set(tmp_path: Path) -> None:
    """No validation pass, no ``n_val``: a zero would read as "measured on nothing"."""
    trainer = _trainer(tmp_path / "run", _cfg(train={"max_steps": 1}))

    report = trainer.fit()

    assert report.evaluations == 0
    assert report.eval_seconds == 0.0
    assert report.n_val is None
    assert report.n_val_total is None
    assert report.val_truncated is False


def test_evaluate_refuses_a_decoder_that_returns_the_wrong_number_of_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A short decode would pair every later hypothesis with the wrong label."""

    def short_greedy(model: Any, inputs: Any, *, max_len: int) -> list[list[int]]:
        return [[5]]  # one hypothesis for a four-row batch

    monkeypatch.setattr(t, "batch_decoders", lambda: (short_greedy, None))
    cfg = _cfg(data={"n_samples": 4, "batch_size": 4}, eval={"beam": 1, "max_n_val": 4})
    trainer = _trainer(tmp_path / "run", cfg, with_val=True)

    with pytest.raises(AiclsError) as excinfo:
        trainer.evaluate()

    assert excinfo.value.details["reason"] == "decoder_row_count_mismatch"


def test_evaluate_uses_the_real_batched_beam_seam_when_ocr_provides_it(tmp_path: Path) -> None:
    """The seam between this loop and aicls.ocr, unmocked.

    Nothing here is faked: if the OCR package exposes the batched entry points, a beam-4
    validation must go through them, because that is the whole point of the wave. If it does not
    expose them yet, the test says which half is missing instead of silently passing.
    """
    greedy_batch, beam_batch = t.batch_decoders()
    if beam_batch is None:
        pytest.skip(f"aicls.ocr.models.seq2seq has no {t.BATCH_BEAM_ENTRY} yet")
    cfg = _cfg(
        data={"n_samples": 4, "batch_size": 4},
        eval={"beam": 3, "max_n_val": 4, "metrics": ["cer"]},
    )
    trainer = _trainer(tmp_path / "run", cfg, with_val=True)

    scores = trainer.evaluate()

    assert greedy_batch is not None, "the greedy half of the contract is missing"
    assert trainer.eval_decoder == f"seq2seq.{t.BATCH_BEAM_ENTRY}"
    assert scores["n_val"] == 4.0
    assert scores["n_val_total"] == 4.0
