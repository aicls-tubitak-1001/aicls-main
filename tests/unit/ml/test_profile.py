"""Unit tests for :mod:`aicls.ml.profile` (docs/ml/egitim-altyapisi.md §8).

Section 8 refuses hand-written epoch-time estimates: every training budget has to come from a
measured profile run. These tests check the measuring apparatus on CPU with three steps of a tiny
synthetic run — **the throughput numbers they observe describe this CPU test, not the RTX 5090**,
and no budget may be quoted from them. A real profile is the first gated action after
``CUDA_INSTALL`` and ``TRAINING`` are approved (ADR-0012).

The contractual test named by the design document is ``test_profile_report_fields_on_synthetic``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aicls.core.errors import AiclsError
from aicls.core.gates import GateError
from aicls.ml import profile as p
from aicls.ml import trainer as t
from aicls.ml.config import TrainConfig, build_config

torch = pytest.importorskip("torch")


def _cfg(**data_overrides: Any) -> TrainConfig:
    """The gate-free synthetic OCR smoke config of ``training/configs/ocr/smoke_synthetic.yaml``."""
    document: dict[str, Any] = {
        "run": {"name": "profile_unit", "seed": 1234},
        "data": {
            "dataset": "synthetic_ink",
            "mode": "offline",
            "image_size": [32, 96],
            "preprocess": ["grayscale", "normalize01", "resize_pad"],
            "batch_size": 4,
            "num_workers": 0,
            "n_samples": 8,
            "max_label_len": 12,
            **data_overrides,
        },
        "model": {
            "family": "seq2seq_image",
            "encoder": "cnn_small",
            "decoder": "transformer",
            "d_model": 32,
            "layers": 1,
            "heads": 2,
            "max_len": 12,
        },
        "optim": {"name": "adamw", "lr": 1e-3},
        "train": {"amp": "off", "grad_accum": 1, "allow_dirty": True},
        "eval": {"beam": 1, "metrics": ["cer"]},
    }
    return build_config(document)


# --------------------------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------------------------
def test_throughput_excludes_the_warmup_steps() -> None:
    """The first steps carry allocator and cuDNN warm-up; they are measured but not counted."""
    records = [
        t.StepTiming(step=1, seconds=1.0, data_wait_s=0.5),
        t.StepTiming(step=2, seconds=0.5, data_wait_s=0.1),
        t.StepTiming(step=3, seconds=0.5, data_wait_s=0.1),
    ]

    measured, steps_per_s, wait_pct = p.throughput(records, warmup=1)

    assert measured == 2
    assert steps_per_s == pytest.approx(2.0)  # 2 steps in 1.0 s
    assert wait_pct == pytest.approx(20.0)  # 0.2 s waiting out of 1.0 s


def test_throughput_refuses_a_warmup_that_eats_every_step() -> None:
    with pytest.raises(AiclsError) as excinfo:
        p.throughput([t.StepTiming(step=1, seconds=1.0, data_wait_s=0.0)], warmup=1)

    assert excinfo.value.details["reason"] == "no_measured_steps"


def test_peak_vram_is_none_on_cpu() -> None:
    """None, not 0.0: a CPU run has no VRAM figure, and a fake zero would be a false number."""
    assert p.peak_vram_mb(t.select_device("off", probe=t.DeviceProbe())) is None


def test_estimate_epoch_seconds_is_derived_from_a_measurement() -> None:
    """§8: epoch and budget estimates come from steps_per_s, never from a guess."""
    seconds = p.estimate_epoch_seconds(steps_per_s=2.0, n_samples=400, batch_size=8, grad_accum=2)

    assert seconds == pytest.approx(12.5)  # 50 batches / 2 accum = 25 steps / 2 steps per s
    with pytest.raises(AiclsError):
        p.estimate_epoch_seconds(steps_per_s=0.0, n_samples=8, batch_size=4)


# --------------------------------------------------------------------------------------------
# the profile run
# --------------------------------------------------------------------------------------------
def test_profile_report_fields_on_synthetic(tmp_path: Path) -> None:
    """Contractual (design document §8, §10).

    WIRING CHECK: three steps of a 60k-parameter model on synthetic scribbles. The fields must be
    present and internally consistent; the values describe this test machine and are not a budget
    for anything.

    The device assertions follow the host rather than assuming CPU: with no ``device`` argument
    :func:`aicls.ml.trainer.select_device` picks CUDA whenever the machine has it, so inside the
    CUDA container this run is a *CUDA* run and ``peak_vram_mb`` is a number by construction.
    Asserting ``is None`` there failed for a reason that has nothing to do with the profiler
    (measured 2026-09-08 in ``codex-research-cu130``).
    """
    cfg = _cfg()
    on_cpu = not torch.cuda.is_available()

    report = p.run_profile(
        cfg, steps=3, warmup_steps=1, run_dir=tmp_path / "run", out_dir=tmp_path / "metrics"
    )

    assert report.steps == 3
    assert report.warmup_steps == 1
    assert report.measured_steps == 2
    assert report.steps_per_s > 0.0
    assert report.samples_per_s == pytest.approx(report.steps_per_s * 4)
    assert 0.0 <= report.dataloader_wait_pct <= 100.0
    assert report.wall_time_s > 0.0
    assert report.device == ("cpu" if on_cpu else "cuda")
    assert report.dtype == "float32"  # amp: off is fp32 on either device
    assert report.amp == "off"
    assert report.synthetic is True
    assert report.wiring_check is True
    # `citable` is not asserted: it tracks the checkout, not the run. Asserting False here meant
    # the test passed only while the working tree was dirty and failed on a clean one.
    assert len(report.config_hash) == 64
    assert "WIRING CHECK" in report.notes.upper()
    if on_cpu:
        # CPU: no VRAM figure at all, reported as null and never as a fake 0.
        assert report.peak_vram_mb is None
    else:
        assert report.peak_vram_mb is not None and report.peak_vram_mb >= 0.0

    written = tmp_path / "metrics" / "profile_profile_unit.json"
    assert written.is_file()
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["steps_per_s"] == report.steps_per_s
    assert (payload["peak_vram_mb"] is None) is on_cpu


def test_profile_runs_a_config_that_keeps_no_checkpoints(tmp_path: Path) -> None:
    """The shipped ``training/configs/ocr/profile.yaml`` keeps nothing, and that is correct.

    A measurement run has no reason to write a checkpoint, so both profile configs declare
    ``keep_last: 0, keep_best: 0``. ``CheckpointSaver`` refuses that combination, and rightly so
    for a training run — hours of GPU time that retain nothing is a mistake worth stopping. But
    the profiler already disables checkpointing by pushing ``ckpt_every`` past the step budget, so
    the guard was firing on a run that could never save anything anyway. Measured on the RTX 5090
    on 2026-09-08: ``aicls profile --config training/configs/ocr/profile.yaml`` exited 1 with
    ``invalid_retention`` before taking a single step, which is the first gated measurement the
    design document asks for.
    """
    cfg = _cfg()
    cfg = cfg.model_copy(
        update={"checkpoint": cfg.checkpoint.model_copy(update={"keep_last": 0, "keep_best": 0})}
    )

    report = p.run_profile(
        cfg, steps=3, warmup_steps=1, run_dir=tmp_path / "run", out_dir=tmp_path / "metrics"
    )

    assert report.steps_per_s > 0.0
    # `fit()` always writes a final checkpoint, so "keep nothing" is not actually expressible;
    # what the profiler guarantees is that it writes no *periodic* ones. Exactly one survives.
    assert len(list((tmp_path / "run" / "checkpoints").glob("step-*"))) == 1


def test_profile_writes_steps_per_s_into_the_run_manifest(tmp_path: Path) -> None:
    """§7/§8: the measured throughput belongs in ``run_manifest.steps_per_s``."""
    from aicls.ml.manifest import RunManifest

    report = p.run_profile(
        _cfg(), steps=2, warmup_steps=0, run_dir=tmp_path / "run", out_dir=tmp_path / "metrics"
    )
    manifest = RunManifest.load(tmp_path / "run")

    assert manifest.steps_per_s is not None
    assert manifest.steps_per_s == pytest.approx(report.steps_per_s)
    assert manifest.run_id == report.run_id


def test_profile_does_not_write_a_report_when_asked_not_to(tmp_path: Path) -> None:
    report = p.run_profile(
        _cfg(),
        steps=2,
        warmup_steps=0,
        run_dir=tmp_path / "run",
        out_dir=tmp_path / "metrics",
        write_report=False,
    )

    assert report.steps_per_s > 0.0
    assert not (tmp_path / "metrics").exists()


def test_profile_report_json_roundtrip(tmp_path: Path) -> None:
    report = p.ProfileReport(
        name="unit", run_id="01J", steps=2, measured_steps=2, steps_per_s=1.5, wiring_check=True
    )
    path = report.save(tmp_path / "profile_unit.json")

    assert path.name == "profile_unit.json"
    restored = p.ProfileReport.model_validate_json(path.read_text(encoding="utf-8"))
    assert restored.steps_per_s == 1.5
    assert restored.peak_vram_mb is None


def test_run_profile_refuses_a_real_dataset_without_the_training_gate(
    tmp_path: Path, test_env: dict[str, str]
) -> None:
    """§9: ``aicls profile`` on real data is behind the TRAINING gate (exit code 3)."""
    from aicls.core.settings import get_settings

    get_settings.cache_clear()
    cfg = _cfg(dataset="mathwriting", root=str(tmp_path / "shards"))

    with pytest.raises(GateError) as excinfo:
        p.run_profile(cfg, steps=2, run_dir=tmp_path / "run", out_dir=tmp_path / "metrics")

    get_settings.cache_clear()
    assert excinfo.value.exit_code == 3
    assert excinfo.value.details["gate"] == "TRAINING"


def test_run_profile_rejects_a_non_positive_step_budget(tmp_path: Path) -> None:
    with pytest.raises(AiclsError) as excinfo:
        p.run_profile(_cfg(), steps=0, run_dir=tmp_path / "run")

    assert excinfo.value.details["reason"] == "invalid_profile_steps"


def test_profile_filename_is_sanitised(tmp_path: Path) -> None:
    assert p.profile_filename("ocr/offline resnet34") == "profile_ocr_offline_resnet34.json"
    assert p.profile_filename("") == "profile_run.json"


def test_throughput_refuses_a_zero_duration_measurement() -> None:
    """A clock that did not move cannot yield a rate; reporting one would invent a number."""
    with pytest.raises(AiclsError) as excinfo:
        p.throughput([t.StepTiming(step=1, seconds=0.0, data_wait_s=0.0)], warmup=0)

    assert excinfo.value.details["reason"] == "zero_duration"


def test_estimate_epoch_seconds_rejects_an_impossible_epoch_shape() -> None:
    with pytest.raises(AiclsError) as excinfo:
        p.estimate_epoch_seconds(steps_per_s=1.0, n_samples=0, batch_size=4)

    assert excinfo.value.details["reason"] == "invalid_epoch_shape"


def test_profile_report_epoch_seconds_uses_the_measured_rate() -> None:
    """§8: an epoch estimate is a function of the measurement, never of a guess."""
    report = p.ProfileReport(name="unit", steps_per_s=4.0, batch_size=8, grad_accum=1)

    assert report.epoch_seconds(320) == pytest.approx(10.0)


def test_run_profile_refuses_a_warmup_that_leaves_nothing_to_measure(tmp_path: Path) -> None:
    with pytest.raises(AiclsError) as excinfo:
        p.run_profile(_cfg(), steps=2, warmup_steps=2, run_dir=tmp_path / "run")

    assert excinfo.value.details["reason"] == "no_measured_steps"


# --------------------------------------------------------------------------------------------
# the schedule a configuration asks for
# --------------------------------------------------------------------------------------------
def test_resolve_schedule_from_an_explicit_step_budget() -> None:
    """``train.max_steps`` wins and ``eval_every`` decides how many validation passes follow."""
    cfg = _cfg()
    cfg = cfg.model_copy(
        update={"train": cfg.train.model_copy(update={"max_steps": 500, "eval_every": 200})}
    )

    schedule = p.resolve_schedule(cfg, train_size=800, val_size=2000)

    assert schedule.total_steps == 500
    assert schedule.eval_count == 3  # 200, 400 and the final step
    assert schedule.n_val == 2000
    assert schedule.steps_per_epoch == 200  # 800 samples / batch 4
    assert schedule.epochs == pytest.approx(2.5)
    assert schedule.resolved is True


def test_resolve_schedule_from_epochs_needs_the_training_split_size() -> None:
    cfg = _cfg()
    cfg = cfg.model_copy(
        update={
            "train": cfg.train.model_copy(
                update={"max_steps": None, "epochs": 30, "eval_every": 2000}
            )
        }
    )

    with_size = p.resolve_schedule(cfg, train_size=400, val_size=100)
    without = p.resolve_schedule(cfg, train_size=None, val_size=100)

    assert with_size.total_steps == 3000  # 30 epochs x 100 steps
    assert with_size.eval_count == 2
    assert without.total_steps is None
    assert without.reason == "epochs_without_train_size"
    assert without.resolved is False


def test_resolve_schedule_caps_max_n_val_by_the_split_size() -> None:
    """``eval.max_n_val: 2000`` over a 100-sample split decodes 100, not 2000."""
    cfg = _cfg()
    cfg = cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"max_n_val": 2000})})

    assert p.resolve_schedule(cfg, train_size=40, val_size=100).n_val == 100
    assert p.resolve_schedule(cfg, train_size=40, val_size=None).n_val == 2000


# --------------------------------------------------------------------------------------------
# the projection — and its refusal
# --------------------------------------------------------------------------------------------
def test_project_schedule_refuses_without_a_measured_eval_cost() -> None:
    """Contractual (wave 7): no eval measurement, no projection.

    A projection built from ``steps_per_s`` alone prices every validation pass at zero. For the
    run that prompted this work that would have been wrong by a factor of 150 — and quotable.
    """
    cfg = _cfg()
    cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update={"max_steps": 2000})})

    projection = p.project_schedule(
        cfg, steps_per_s=8.66, seconds_per_val_sample=None, train_size=400, val_size=2000
    )

    assert projection.projected is False
    assert projection.reason == "eval_cost_not_measured"
    assert projection.total_seconds is None
    assert projection.eval_seconds is None
    assert projection.total_steps == 2000  # what it refused to price is still stated


def test_project_schedule_prices_both_halves_of_the_schedule() -> None:
    """The measured per-step and per-validation-sample costs, times the configured schedule."""
    cfg = _cfg()
    cfg = cfg.model_copy(
        update={"train": cfg.train.model_copy(update={"max_steps": 2000, "eval_every": 2000})}
    )
    cfg = cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"max_n_val": 2000})})

    projection = p.project_schedule(
        cfg, steps_per_s=8.66, seconds_per_val_sample=18.5, train_size=4000, val_size=2000
    )

    assert projection.projected is True
    assert projection.reason == ""
    assert projection.total_steps == 2000
    assert projection.eval_count == 1
    assert projection.n_val_per_eval == 2000
    assert projection.train_seconds == pytest.approx(2000 / 8.66)
    assert projection.eval_seconds == pytest.approx(37000.0)
    assert projection.total_hours == pytest.approx(projection.total_seconds / 3600.0)
    assert projection.eval_share_pct > 99.0  # the number that decides the schedule


def test_project_schedule_refuses_an_unmeasurable_rate() -> None:
    projection = p.project_schedule(
        _cfg(), steps_per_s=0.0, seconds_per_val_sample=1.0, train_size=40, val_size=8
    )

    assert projection.projected is False
    assert projection.reason == "invalid_steps_per_s"


def test_project_schedule_refuses_an_unresolved_schedule() -> None:
    """It keeps the measured per-unit costs, because they are real, and prices nothing."""
    cfg = _cfg()
    cfg = cfg.model_copy(
        update={"train": cfg.train.model_copy(update={"max_steps": None, "epochs": 30})}
    )

    projection = p.project_schedule(
        cfg, steps_per_s=8.66, seconds_per_val_sample=18.5, train_size=None, val_size=100
    )

    assert projection.projected is False
    assert projection.reason == "epochs_without_train_size"
    assert projection.seconds_per_val_sample == pytest.approx(18.5)
    assert projection.total_seconds is None


# --------------------------------------------------------------------------------------------
# the profile run measures the validation pass too
# --------------------------------------------------------------------------------------------
def test_profile_measures_the_validation_cost_and_projects_from_it(tmp_path: Path) -> None:
    """Contractual (wave 7, design document §8).

    WIRING CHECK on CPU with a 60k-parameter model over generated scribbles: the values describe
    this test machine and are not a budget for anything.
    """
    report = p.run_profile(
        _cfg(),
        steps=3,
        warmup_steps=1,
        eval_samples=4,
        run_dir=tmp_path / "run",
        out_dir=tmp_path / "metrics",
    )

    assert report.eval_measured is True
    assert report.eval_samples == 4
    assert report.eval_seconds is not None and report.eval_seconds > 0.0
    assert report.seconds_per_val_sample == pytest.approx(report.eval_seconds / 4.0)
    assert report.eval_beam == 1
    assert report.eval_decoder != ""
    assert report.train_seconds > 0.0
    assert report.wall_time_s >= report.train_seconds + report.eval_seconds
    assert report.projection.projected is True
    assert report.projection.total_seconds is not None
    assert report.projection.eval_seconds is not None
    assert report.projection.seconds_per_val_sample == report.seconds_per_val_sample

    payload = json.loads(
        (tmp_path / "metrics" / "profile_profile_unit.json").read_text(encoding="utf-8")
    )
    assert payload["projection"]["projected"] is True
    assert payload["eval_seconds"] == report.eval_seconds


def test_profile_without_an_eval_measurement_refuses_to_project(tmp_path: Path) -> None:
    """``eval_samples=0`` is allowed; pretending the resulting budget is one is not."""
    report = p.run_profile(
        _cfg(),
        steps=2,
        warmup_steps=0,
        eval_samples=0,
        run_dir=tmp_path / "run",
        out_dir=tmp_path / "metrics",
    )

    assert report.eval_measured is False
    assert report.eval_seconds is None
    assert report.seconds_per_val_sample is None
    assert report.projection.projected is False
    assert report.projection.reason == "eval_cost_not_measured"


def test_profile_survives_a_validation_split_it_cannot_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half-prepared dataset must still be profilable — with the projection refusing."""
    from aicls.ml import registry

    real_build = registry.build_dataset

    def only_train(cfg: Any, split: str) -> Any:
        if split == "valid":
            raise RuntimeError("no valid shards here")
        return real_build(cfg, split)

    monkeypatch.setattr(registry, "build_dataset", only_train)

    report = p.run_profile(
        _cfg(), steps=2, warmup_steps=0, run_dir=tmp_path / "run", out_dir=tmp_path / "metrics"
    )

    assert report.eval_measured is False
    assert report.eval_val_split_size is None
    assert report.projection.projected is False


def test_profile_report_roundtrip_keeps_the_projection(tmp_path: Path) -> None:
    report = p.ProfileReport(
        name="unit",
        steps_per_s=2.0,
        seconds_per_val_sample=18.5,
        eval_measured=True,
        projection=p.ScheduleProjection(projected=True, total_seconds=10.0, total_hours=10 / 3600),
    )
    path = report.save(tmp_path / "profile_unit.json")

    restored = p.ProfileReport.model_validate_json(path.read_text(encoding="utf-8"))

    assert restored.projection.projected is True
    assert restored.projection.total_seconds == pytest.approx(10.0)
    assert restored.seconds_per_val_sample == pytest.approx(18.5)


def test_profile_measures_the_eval_at_a_full_micro_batch(tmp_path: Path) -> None:
    """A partial batch measures a per-sample cost the real run will never pay.

    Decoding is kernel-launch bound, so 8 rows of a 64-row micro-batch cost nearly what 64 rows
    cost. Measured on the RTX 5090 with the shipped OCR config at beam 4: 0.139 s per sample at 8
    rows against 0.027 s at 64. A projection built from the narrow measurement is 5x too
    pessimistic, so the request is raised to one full batch.
    """
    report = p.run_profile(
        _cfg(batch_size=8, n_samples=16),
        steps=2,
        warmup_steps=0,
        eval_samples=2,
        run_dir=tmp_path / "run",
        out_dir=tmp_path / "metrics",
    )

    assert report.eval_samples == 8  # raised from 2 to one full data.batch_size
    assert report.eval_batch_size == 8
    assert report.eval_measured is True
