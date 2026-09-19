"""Measured training throughput — the only legitimate source of a GPU budget (§8, ADR-0006).

Section 8 of ``docs/ml/egitim-altyapisi.md`` is explicit: hand-written epoch times are not
trustworthy, so this project derives every training budget from a measurement. After the
``CUDA_INSTALL`` and ``TRAINING`` gates open, the first gated action for each configuration is
``aicls profile --config ... --steps 200``, which runs :func:`run_profile`, writes
``reports/metrics/profile_<name>.json`` and records ``steps_per_s`` in the run manifest. Epoch and
budget numbers are then computed with :func:`estimate_epoch_seconds` from that measurement — never
typed into a document by hand.

What is measured:

* ``steps_per_s`` — optimizer steps per second, **excluding** the warm-up steps (allocator, cuDNN
  autotuning and the first kernel launches make the first steps unrepresentative);
* ``seconds_per_val_sample`` — the cost of decoding one validation sample, measured by running a
  real validation pass after the step loop. **This is the term that decides the schedule**: on the
  RTX 5090 against the MathWriting excerpt, 20 optimizer steps cost 1.4 s while validating sixteen
  samples cost ~296 s (run ``ocr_excerpt_wiring``, 2026-09-08). A profile that timed only the step
  loop measured 0.7 % of that run and would have been quoted as a budget;
* ``peak_vram_mb`` — ``torch.cuda.max_memory_allocated`` since the profile started, **decoding
  included**. On a CPU run it is ``None``, not ``0.0``: there is no VRAM figure, and a fake zero in
  a report is a false number;
* ``dataloader_wait_pct`` — the share of the loop spent blocked on the data loader, which is what
  decides whether ``data.num_workers`` or the NVMe layout has to change.

:func:`project_schedule` turns the two measured per-unit costs into a wall-clock projection for the
schedule the configuration actually asks for — ``train.epochs``/``train.max_steps`` steps and
``ceil(steps / train.eval_every)`` validation passes of ``eval.max_n_val`` samples each. It
**refuses to project** when the eval cost was not measured (:attr:`ScheduleProjection.projected` is
false and ``reason`` says why) rather than projecting from the step loop alone: §8 forbids
hand-written epoch estimates, and a measurement that omits the dominant term is worse than none
because it is quotable.

**HONESTY.** A profile taken on this CPU box with a 60k-parameter model on synthetic scribbles
describes *that*, and nothing else. The report carries ``wiring_check: true`` and a note saying so
whenever the data is synthetic; no RTX 5090 budget may be quoted from such a run.

torch is imported lazily inside functions, so importing this module stays cheap and torch-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aicls.core import messages_tr
from aicls.core.errors import InputValidationError
from aicls.core.logging import get_logger
from aicls.ml.config import TrainConfig
from aicls.ml.trainer import (
    WIRING_CHECK_NOTE,
    DeviceProbe,
    DeviceSpec,
    StepTiming,
    Trainer,
    require_training_gate,
    split_size,
)

_log = get_logger(__name__)

#: Step budget of a real profile run (design document §8: ``aicls profile --steps 200``).
DEFAULT_PROFILE_STEPS = 200

#: Steps measured but excluded from the throughput figure (allocator / kernel warm-up).
DEFAULT_WARMUP_STEPS = 5

#: Validation samples decoded to measure the eval cost. Small on purpose — at the 18.5 s/sample
#: this profiler exists to expose, eight samples are already two and a half minutes.
DEFAULT_PROFILE_EVAL_SAMPLES = 8

#: Seconds per hour, so the projection can publish both units without a magic number.
SECONDS_PER_HOUR = 3600.0

#: Where the profile reports live in git, next to the evaluation reports.
DEFAULT_OUT_DIR = Path("reports/metrics")

#: Report file name pattern.
PROFILE_FILENAME_TEMPLATE = "profile_{name}.json"

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True, slots=True)
class Schedule:
    """The run the configuration asks for, in units a projection can price.

    Attributes:
        total_steps: Optimizer steps the schedule budgets, resolved exactly the way
            :meth:`aicls.ml.trainer.Trainer._resolve_total_steps` resolves it — the smallest of
            ``train.max_steps`` and ``train.epochs × steps/epoch``, one epoch when neither is set.
        eval_count: Validation passes at ``train.eval_every`` (the loop also validates on the last
            step, which ``ceil`` already covers).
        n_val: Samples decoded per pass — ``eval.max_n_val`` capped by the split size.
        n_val_total: Samples in the validation split, or ``None`` when it is unknown.
        train_size: Samples in the training split, or ``None``.
        steps_per_epoch: Optimizer steps in one epoch, or ``None`` when ``train_size`` is unknown.
        epochs: Epochs ``total_steps`` corresponds to, or ``None``.
        reason: Empty when the schedule is fully resolved; otherwise why it is not.
    """

    total_steps: int | None = None
    eval_count: int | None = None
    n_val: int | None = None
    n_val_total: int | None = None
    train_size: int | None = None
    steps_per_epoch: int | None = None
    epochs: float | None = None
    reason: str = ""

    @property
    def resolved(self) -> bool:
        """Whether the schedule is complete enough to price."""
        return self.total_steps is not None and self.eval_count is not None


def resolve_schedule(
    cfg: TrainConfig, *, train_size: int | None = None, val_size: int | None = None
) -> Schedule:
    """Work out how many steps and validation passes ``cfg`` asks for.

    Args:
        cfg: The configuration being profiled — the **original** one, not the copy
            :func:`run_profile` shortens.
        train_size: Samples in the training split; needed only for an ``epochs``-based budget.
        val_size: Samples in the validation split; caps ``eval.max_n_val``.

    Returns:
        The :class:`Schedule`. When the step budget cannot be resolved (``train.epochs`` set but
        the training split size unknown), every field that depends on it is ``None`` and
        ``reason`` names the gap instead of a number being invented.

    Note:
        With ``train.max_steps`` **and** ``train.epochs`` set but ``train_size`` unknown, the
        result is the ``max_steps`` bound alone, which the epoch bound can only shorten — so the
        projection is then an upper bound. :func:`run_profile` always knows ``train_size``
        (it built the dataset), so that case only arises when this function is called directly.
    """
    grad_accum = max(int(cfg.train.grad_accum), 1)
    batch_size = max(int(cfg.data.batch_size), 1)
    steps_per_epoch: int | None = None
    if train_size is not None and train_size > 0:
        batches = math.ceil(train_size / batch_size)
        steps_per_epoch = max(math.ceil(batches / grad_accum), 1)

    candidates: list[int] = []
    if cfg.train.max_steps is not None:
        candidates.append(int(cfg.train.max_steps))
    if cfg.train.epochs is not None and steps_per_epoch is not None:
        candidates.append(int(cfg.train.epochs) * steps_per_epoch)
    if not candidates and cfg.train.max_steps is None and steps_per_epoch is not None:
        candidates.append(steps_per_epoch)  # the loop's "one epoch" default

    n_val = _resolve_n_val(cfg.eval.max_n_val, val_size)
    if not candidates:
        reason = "epochs_without_train_size" if cfg.train.epochs is not None else "no_step_budget"
        return Schedule(
            n_val=n_val,
            n_val_total=val_size,
            train_size=train_size,
            steps_per_epoch=steps_per_epoch,
            reason=reason,
        )

    total_steps = max(min(candidates), 1)
    eval_count = math.ceil(total_steps / max(int(cfg.train.eval_every), 1))
    epochs = (total_steps / steps_per_epoch) if steps_per_epoch else None
    return Schedule(
        total_steps=total_steps,
        eval_count=eval_count,
        n_val=n_val,
        n_val_total=val_size,
        train_size=train_size,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
    )


def _resolve_n_val(max_n_val: int | None, val_size: int | None) -> int | None:
    """Samples one validation pass decodes: the cap, the split, or ``None`` when neither says."""
    if max_n_val is None:
        return val_size
    if val_size is None:
        return int(max_n_val)
    return min(int(max_n_val), int(val_size))


class ScheduleProjection(BaseModel):
    """Projected wall clock for the configured schedule, or a refusal to project one.

    §8 forbids hand-written epoch estimates. It equally forbids an estimate that measured only
    the cheap half: the run this module exists to budget spent 1.4 s training and 592 s
    validating, so a projection built from ``steps_per_s`` alone would have been wrong by two
    orders of magnitude *and* quotable. When :attr:`seconds_per_val_sample` was not measured this
    object carries ``projected: false`` and a ``reason``, and no seconds at all.

    Attributes:
        projected: Whether the numbers below exist.
        reason: Why not, when they do not (``eval_cost_not_measured``, ``schedule_unresolved``,
            ``invalid_steps_per_s``).
        total_steps / eval_count / n_val_per_eval: The schedule that was priced.
        seconds_per_step / seconds_per_val_sample: The measured per-unit costs it was priced with.
        train_seconds / eval_seconds / total_seconds / total_hours: The projection.
        eval_share_pct: Share of the projected wall clock spent validating.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    projected: bool = False
    reason: str = ""
    total_steps: int | None = None
    eval_count: int | None = None
    n_val_per_eval: int | None = None
    n_val_total: int | None = None
    epochs: float | None = None
    steps_per_epoch: int | None = None
    beam: int | None = None
    seconds_per_step: float | None = None
    seconds_per_val_sample: float | None = None
    train_seconds: float | None = None
    eval_seconds: float | None = None
    total_seconds: float | None = None
    total_hours: float | None = None
    eval_share_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict of every field."""
        return dict(json.loads(self.model_dump_json()))


def project_schedule(
    cfg: TrainConfig,
    *,
    steps_per_s: float,
    seconds_per_val_sample: float | None,
    train_size: int | None = None,
    val_size: int | None = None,
) -> ScheduleProjection:
    """Price the configured schedule with the two measured per-unit costs.

    Args:
        cfg: The configuration whose schedule is being priced.
        steps_per_s: Measured optimizer steps per second.
        seconds_per_val_sample: Measured cost of decoding one validation sample; ``None`` means
            it was not measured, and then this function **refuses** rather than pricing the
            validation passes at zero.
        train_size: Samples in the training split.
        val_size: Samples in the validation split.

    Returns:
        A :class:`ScheduleProjection`; check :attr:`ScheduleProjection.projected` before quoting
        any field.
    """
    schedule = resolve_schedule(cfg, train_size=train_size, val_size=val_size)
    common: dict[str, Any] = {
        "total_steps": schedule.total_steps,
        "eval_count": schedule.eval_count,
        "n_val_per_eval": schedule.n_val,
        "n_val_total": schedule.n_val_total,
        "epochs": schedule.epochs,
        "steps_per_epoch": schedule.steps_per_epoch,
        "beam": int(cfg.eval.beam),
    }
    if steps_per_s <= 0.0 or not math.isfinite(steps_per_s):
        return ScheduleProjection(reason="invalid_steps_per_s", **common)
    if seconds_per_val_sample is None:
        _log.warning(
            "ml.profile.projection_refused",
            reason="eval_cost_not_measured",
            hint="run the profile with eval_samples > 0; a step-loop-only budget is not a budget",
        )
        return ScheduleProjection(reason="eval_cost_not_measured", **common)
    total_steps, eval_count, n_val = schedule.total_steps, schedule.eval_count, schedule.n_val
    if total_steps is None or eval_count is None or n_val is None:
        return ScheduleProjection(
            reason=schedule.reason or "schedule_unresolved",
            seconds_per_step=1.0 / steps_per_s,
            seconds_per_val_sample=float(seconds_per_val_sample),
            **common,
        )

    train_seconds = total_steps / steps_per_s
    eval_seconds = eval_count * n_val * float(seconds_per_val_sample)
    total = train_seconds + eval_seconds
    return ScheduleProjection(
        projected=True,
        seconds_per_step=1.0 / steps_per_s,
        seconds_per_val_sample=float(seconds_per_val_sample),
        train_seconds=train_seconds,
        eval_seconds=eval_seconds,
        total_seconds=total,
        total_hours=total / SECONDS_PER_HOUR,
        eval_share_pct=(100.0 * eval_seconds / total) if total > 0.0 else None,
        **common,
    )


class ProfileReport(BaseModel):
    """One measured profile run.

    Attributes:
        steps: Steps that were run.
        warmup_steps: Steps excluded from the throughput figure.
        measured_steps: ``steps - warmup_steps`` (what ``steps_per_s`` is computed from).
        steps_per_s: Optimizer steps per second on this machine, with this configuration.
        samples_per_s: ``steps_per_s * batch_size * grad_accum``.
        peak_vram_mb: Peak allocated VRAM, or ``None`` on a CPU run (never a fake ``0.0``).
        dataloader_wait_pct: Share of the loop spent waiting for batches.
        train_seconds / eval_seconds / checkpoint_seconds / dataloader_wait_s: the measured
            :class:`~aicls.ml.trainer.TimeBudget` split of this profile run.
        eval_measured: Whether a validation pass was timed. When false, every eval field below is
            ``None`` and :attr:`projection` refuses to project.
        eval_samples / eval_val_split_size / eval_beam / eval_batch_size: what the eval
            measurement decoded, how big the split it came from is, with which beam width and at
            which micro-batch width — a beam-1 or narrow-batch measurement does not price a
            ``beam: 4``, batch-64 schedule.
        seconds_per_val_sample: The measured cost of decoding one validation sample.
        projection: The projected wall clock for the configured schedule, or a refusal.
        wiring_check: True when the run used synthetic data, i.e. when this measurement is a
            plumbing check and not a budget for a real configuration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 2
    name: str = "run"
    run_id: str = ""
    steps: int = 0
    warmup_steps: int = 0
    measured_steps: int = 0
    steps_per_s: float = 0.0
    samples_per_s: float = 0.0
    peak_vram_mb: float | None = None
    dataloader_wait_pct: float = 0.0
    wall_time_s: float = 0.0
    train_seconds: float = 0.0
    eval_seconds: float | None = None
    checkpoint_seconds: float = 0.0
    dataloader_wait_s: float = 0.0
    eval_measured: bool = False
    eval_samples: int | None = None
    eval_val_split_size: int | None = None
    eval_beam: int | None = None
    eval_batch_size: int | None = None
    eval_decoder: str = ""
    seconds_per_val_sample: float | None = None
    train_split_size: int | None = None
    projection: ScheduleProjection = Field(default_factory=ScheduleProjection)
    batch_size: int = 0
    grad_accum: int = 1
    dataset: str = ""
    model_family: str = ""
    device: str = "cpu"
    device_name: str = "cpu"
    dtype: str = "float32"
    amp: str = "off"
    torch_version: str | None = None
    config_hash: str = ""
    git_sha: str = "unknown"
    citable: bool = False
    synthetic: bool = True
    wiring_check: bool = True
    run_dir: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict of every field."""
        return dict(json.loads(self.model_dump_json()))

    def to_json(self) -> str:
        """Pretty JSON with sorted keys and a trailing newline (UTF-8, LF)."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"

    def save(self, target: Path) -> Path:
        """Write the report to ``target`` (a directory or a ``.json`` file)."""
        path = Path(target)
        if path.suffix != ".json":
            path = path / profile_filename(self.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(self.to_json())
        return path

    def epoch_seconds(self, n_samples: int) -> float:
        """Estimated seconds per epoch for ``n_samples`` at this measured throughput."""
        return estimate_epoch_seconds(
            steps_per_s=self.steps_per_s,
            n_samples=n_samples,
            batch_size=self.batch_size,
            grad_accum=self.grad_accum,
        )


def profile_filename(name: str) -> str:
    """``profile_<sanitised name>.json`` — safe on Windows, stable across runs."""
    cleaned = _UNSAFE_NAME.sub("_", str(name).strip()).strip("_")
    return PROFILE_FILENAME_TEMPLATE.format(name=cleaned or "run")


def throughput(
    records: Sequence[StepTiming], *, warmup: int = DEFAULT_WARMUP_STEPS
) -> tuple[int, float, float]:
    """Summarise per-step timings, dropping the warm-up steps.

    Args:
        records: One :class:`~aicls.ml.trainer.StepTiming` per optimizer step.
        warmup: How many leading steps to exclude.

    Returns:
        ``(measured_steps, steps_per_s, dataloader_wait_pct)``.

    Raises:
        InputValidationError: If nothing is left after the warm-up, or the measured span has no
            positive duration (a clock that did not move cannot yield a rate).
    """
    kept = list(records)[max(int(warmup), 0) :]
    if not kept:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={
                "reason": "no_measured_steps",
                "steps": len(records),
                "warmup": int(warmup),
            },
        )
    seconds = sum(item.seconds for item in kept)
    if seconds <= 0.0:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "zero_duration", "measured_steps": len(kept)},
        )
    waited = sum(item.data_wait_s for item in kept)
    return len(kept), len(kept) / seconds, 100.0 * waited / seconds


def peak_vram_mb(spec: DeviceSpec) -> float | None:
    """Peak allocated VRAM in MiB, or ``None`` on a CPU run.

    ``None`` is deliberate: a CPU run has no VRAM figure at all, and reporting ``0.0`` would put
    a number into a report that no measurement supports.
    """
    if not spec.is_cuda:
        return None
    from aicls.ocr.torch_utils import require_torch

    torch = require_torch()  # pragma: no cover - CUDA_INSTALL is closed
    return float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)  # pragma: no cover


def reset_peak_vram(spec: DeviceSpec) -> None:
    """Clear the CUDA peak-memory counter before a measurement (no-op on CPU)."""
    if not spec.is_cuda:
        return
    from aicls.ocr.torch_utils import require_torch

    require_torch().cuda.reset_peak_memory_stats()  # pragma: no cover - CUDA_INSTALL is closed


def estimate_epoch_seconds(
    *, steps_per_s: float, n_samples: int, batch_size: int, grad_accum: int = 1
) -> float:
    """Seconds per epoch derived from a measured ``steps_per_s`` (§8).

    Args:
        steps_per_s: Measured optimizer steps per second.
        n_samples: Samples in one epoch.
        batch_size: Micro-batch size.
        grad_accum: Micro-batches per optimizer step.

    Returns:
        The estimated epoch duration in seconds.

    Raises:
        InputValidationError: On a non-positive rate or batch shape — an estimate without a
            measurement behind it is exactly what §8 forbids.
    """
    if steps_per_s <= 0.0 or not math.isfinite(steps_per_s):
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "invalid_steps_per_s", "steps_per_s": float(steps_per_s)},
        )
    if n_samples <= 0 or batch_size <= 0 or grad_accum <= 0:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={
                "reason": "invalid_epoch_shape",
                "n_samples": int(n_samples),
                "batch_size": int(batch_size),
                "grad_accum": int(grad_accum),
            },
        )
    batches = math.ceil(n_samples / batch_size)
    steps = math.ceil(batches / grad_accum)
    return steps / steps_per_s


def run_profile(
    cfg: TrainConfig,
    *,
    steps: int = DEFAULT_PROFILE_STEPS,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    eval_samples: int | None = DEFAULT_PROFILE_EVAL_SAMPLES,
    run_dir: Path | None = None,
    out_dir: Path | None = None,
    model: Any = None,
    train_ds: Any = None,
    val_ds: Any = None,
    tracker: Any = None,
    device: str | None = None,
    probe: DeviceProbe | None = None,
    i_have_approval: bool = False,
    ledger_path: Path | None = None,
    write_report: bool = True,
) -> ProfileReport:
    """Measure the throughput **and the validation cost** of ``cfg``.

    Two measurements, in this order:

    1. ``steps`` optimizer steps through a real :class:`~aicls.ml.trainer.Trainer` — same
       manifest, same seed — with validation and checkpointing switched off inside ``fit()``, so
       ``steps_per_s`` describes the step loop and nothing else;
    2. one real validation pass over ``eval_samples`` samples, timed separately. Skipping it was
       how a profile of the run that spent 99 % of its wall clock decoding could report a number
       that looked like a budget.

    On real data the whole thing is gated (``TRAINING``); a ``synthetic_*`` dataset is gate-free
    (ADR-0012).

    Args:
        cfg: The configuration to profile.
        steps: Optimizer steps to run (200 in a real profile).
        warmup_steps: Leading steps excluded from ``steps_per_s``.
        eval_samples: Validation samples to decode for the eval measurement; ``None`` or ``0``
            skips it, and then :attr:`ProfileReport.projection` refuses to project. A non-zero
            request is raised to at least one full ``data.batch_size`` micro-batch, because a
            partial batch measures a per-sample cost the real run will never pay.
        run_dir: Run directory; defaults to the trainer's own default.
        out_dir: Where ``profile_<name>.json`` is written (default ``reports/metrics``).
        model: Pre-built model; built from the registry when omitted.
        train_ds: Pre-built dataset; built from the registry when omitted.
        val_ds: Pre-built validation dataset; built from the registry when omitted. A split that
            cannot be built is not an error: the eval measurement is skipped and said so.
        tracker: Experiment tracker override.
        device: Force ``"cpu"``/``"cuda"``.
        probe: Hardware probe override (tests).
        i_have_approval: Whether the caller passed ``--i-have-approval``.
        ledger_path: Approval-ledger override (tests only).
        write_report: Whether to write the JSON report.

    Returns:
        The :class:`ProfileReport`.

    Raises:
        InputValidationError: If ``steps`` is not positive or the warm-up leaves nothing.
        GateError: If the dataset is not synthetic and the ``TRAINING`` gate is closed.
    """
    if int(steps) < 1:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "invalid_profile_steps", "steps": int(steps)},
        )
    require_training_gate(cfg, i_have_approval=i_have_approval, ledger_path=ledger_path)

    budget = int(steps)
    warmup = max(int(warmup_steps), 0)
    if warmup >= budget:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "no_measured_steps", "steps": budget, "warmup": warmup},
        )
    # Decoding 8 rows of a 64-row micro-batch costs almost what decoding all 64 costs: the loop is
    # kernel-launch bound, not arithmetic bound. Measured on the RTX 5090 with the shipped OCR
    # config (beam 4): 0.139 s per sample at 8 rows against 0.027 s at 64 rows. A measurement taken
    # on a partial batch therefore over-states the per-sample cost of the configured batch size by
    # 5x, so the request is rounded up to one full micro-batch -- nearly free, and it is the width
    # the real run will use.
    requested_eval = max(int(eval_samples or 0), 0)
    wanted_eval = max(requested_eval, int(cfg.data.batch_size)) if requested_eval else 0
    if wanted_eval != requested_eval:
        _log.info(
            "ml.profile.eval_samples_raised_to_one_batch",
            requested=requested_eval,
            measured=wanted_eval,
            batch_size=int(cfg.data.batch_size),
        )
    profile_cfg = _profile_config(cfg, budget)
    if model is None or train_ds is None:
        from aicls.ml.registry import DatasetSpec, build_dataset, build_model

        tokenizer = DatasetSpec.from_config(profile_cfg).tokenizer()
        model = model if model is not None else build_model(profile_cfg, tokenizer.vocab_size)
        train_ds = train_ds if train_ds is not None else build_dataset(profile_cfg, "train")
    if val_ds is None and wanted_eval > 0:
        val_ds = _build_val_split(profile_cfg)

    trainer = Trainer(
        profile_cfg,
        model,
        train_ds,
        None,
        tracker=tracker,
        run_dir=run_dir,
        device=device,
        probe=probe,
        i_have_approval=i_have_approval,
        ledger_path=ledger_path,
    )
    trainer.record_step_times = True
    reset_peak_vram(trainer.device_spec)
    fit = trainer.fit(max_steps=budget)
    measured, steps_per_s, wait_pct = throughput(trainer.stats.records, warmup=warmup)

    # The validation pass is measured AFTER `fit()` and outside it: the trainer is built with
    # `val_ds=None` so that the step loop stays uncontaminated (it validates on its last step
    # whenever a validation set is present), and the dataset is attached here for one timed pass.
    eval_seconds: float | None = None
    decoded: int | None = None
    seconds_per_sample: float | None = None
    if val_ds is not None and wanted_eval > 0:
        trainer.val_ds = val_ds
        trainer.evaluate(max_n=wanted_eval)
        eval_seconds = trainer.budget.eval_seconds
        decoded = trainer.coverage.n_val
        seconds_per_sample = trainer.budget.seconds_per_val_sample

    projection = project_schedule(
        cfg,
        steps_per_s=steps_per_s,
        seconds_per_val_sample=seconds_per_sample,
        train_size=split_size(train_ds),
        val_size=split_size(val_ds),
    )
    report = ProfileReport(
        name=profile_cfg.run.name or "run",
        run_id=fit.run_id,
        steps=trainer.step,
        warmup_steps=warmup,
        measured_steps=measured,
        steps_per_s=steps_per_s,
        samples_per_s=steps_per_s * profile_cfg.data.batch_size * profile_cfg.train.grad_accum,
        peak_vram_mb=peak_vram_mb(trainer.device_spec),
        dataloader_wait_pct=wait_pct,
        wall_time_s=fit.wall_time_s + (eval_seconds or 0.0),
        train_seconds=trainer.budget.train_seconds,
        eval_seconds=eval_seconds,
        checkpoint_seconds=trainer.budget.checkpoint_seconds,
        dataloader_wait_s=trainer.budget.dataloader_wait_s,
        eval_measured=seconds_per_sample is not None,
        eval_samples=decoded,
        eval_val_split_size=split_size(val_ds),
        eval_beam=int(cfg.eval.beam) if decoded is not None else None,
        eval_batch_size=(
            min(int(profile_cfg.data.batch_size), decoded) if decoded is not None else None
        ),
        eval_decoder=trainer.eval_decoder,
        seconds_per_val_sample=seconds_per_sample,
        train_split_size=split_size(train_ds),
        projection=projection,
        batch_size=profile_cfg.data.batch_size,
        grad_accum=profile_cfg.train.grad_accum,
        dataset=profile_cfg.data.dataset,
        model_family=profile_cfg.model.family,
        device=trainer.device_spec.device,
        device_name=trainer.device_spec.device_name,
        dtype=trainer.device_spec.dtype,
        amp=trainer.device_spec.amp,
        torch_version=_torch_version(),
        config_hash=fit.config_hash,
        git_sha=fit.git_sha,
        citable=fit.citable,
        synthetic=profile_cfg.is_gate_free,
        wiring_check=profile_cfg.is_gate_free,
        run_dir=str(trainer.run_dir),
        notes=WIRING_CHECK_NOTE if profile_cfg.is_gate_free else "",
    )
    if trainer.manifest is not None:
        trainer.manifest.set_steps_per_s(steps_per_s, run_dir=trainer.run_dir)
    if write_report:
        path = report.save(Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR)
        _log.info("ml.profile.report_written", path=str(path), steps_per_s=steps_per_s)
    _log.info(
        "ml.profile.measured",
        name=report.name,
        steps=report.steps,
        measured_steps=report.measured_steps,
        steps_per_s=report.steps_per_s,
        peak_vram_mb=report.peak_vram_mb,
        dataloader_wait_pct=report.dataloader_wait_pct,
        eval_measured=report.eval_measured,
        eval_samples=report.eval_samples,
        seconds_per_val_sample=report.seconds_per_val_sample,
        projected_hours=report.projection.total_hours,
        projection_reason=report.projection.reason or None,
        device=report.device,
        wiring_check=report.wiring_check,
    )
    return report


def _build_val_split(cfg: TrainConfig) -> Any:
    """The validation dataset of ``cfg``, or ``None`` when it cannot be built.

    A configuration whose validation split is missing must still be profilable for its step loop:
    the eval measurement is skipped, ``eval_measured`` stays false and the projection refuses. The
    alternative — raising — would make ``aicls profile`` unusable exactly where it is most needed,
    on a half-prepared dataset.
    """
    from aicls.ml.registry import build_dataset

    try:
        return build_dataset(cfg, "valid")
    except Exception as exc:
        _log.warning(
            "ml.profile.val_split_unavailable",
            dataset=cfg.data.dataset,
            error=str(exc),
            hint="the eval cost stays unmeasured and the schedule projection will refuse",
        )
        return None


def _profile_config(cfg: TrainConfig, steps: int) -> TrainConfig:
    """A copy of ``cfg`` with validation, checkpointing and chatty logging switched off."""
    loop = cfg.train.model_copy(
        update={
            "max_steps": steps,
            "epochs": None,
            "eval_every": steps + 1,
            "ckpt_every": steps + 1,
            "log_every": max(1, steps // 10),
            "early_stop_patience": None,
        }
    )
    update: dict[str, Any] = {"train": loop}
    if cfg.checkpoint.keep_last + cfg.checkpoint.keep_best < 1:
        # Both profile configs declare "keep nothing", which is right for a measurement and which
        # CheckpointSaver refuses — that guard exists so a long training run cannot retain nothing.
        # The profiler never reaches a save (ckpt_every is past the budget above), so give the
        # saver a retention it will accept rather than weakening the guard for real runs.
        update["checkpoint"] = cfg.checkpoint.model_copy(update={"keep_last": 1})
    return cfg.model_copy(update=update)


def _torch_version() -> str | None:
    """The installed torch version, or ``None`` when torch is absent."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is installed in this environment
        return None
    return str(torch.__version__)


__all__ = [
    "DEFAULT_OUT_DIR",
    "DEFAULT_PROFILE_EVAL_SAMPLES",
    "DEFAULT_PROFILE_STEPS",
    "DEFAULT_WARMUP_STEPS",
    "PROFILE_FILENAME_TEMPLATE",
    "SECONDS_PER_HOUR",
    "ProfileReport",
    "Schedule",
    "ScheduleProjection",
    "estimate_epoch_seconds",
    "peak_vram_mb",
    "profile_filename",
    "project_schedule",
    "reset_peak_vram",
    "resolve_schedule",
    "run_profile",
    "throughput",
]
