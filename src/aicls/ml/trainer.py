"""The OCR training loop (docs/ml/egitim-altyapisi.md §4.1, ADR-0006).

:class:`Trainer` is the ~400-line loop ADR-0006 chose for the two OCR families
(``seq2seq_image``, ``seq2seq_stroke``); the ASR side keeps Hugging Face's ``Seq2SeqTrainer``
behind :mod:`aicls.ml.whisper_ft`, because rewriting Whisper's sharp corners would produce a WER
that cannot be compared with the literature. Both paths share one reporting contract: the same
``run_manifest.json``, the same checkpoint layout and the same metric definitions.

``fit()`` runs, in this order:

1. :func:`aicls.ml.manifest.seed_everything`;
2. :meth:`aicls.ml.manifest.RunManifest.capture` written to ``run_manifest.json`` **before the
   first optimizer step**, so a run that is killed after ten minutes is still attributable, plus
   ``env.json`` (ADR-0007);
3. data loaders with a seeded ``generator`` and ``worker_init_fn``, so an epoch is reproducible;
4. the step loop: autocast forward, loss, gradient accumulation, clipping, AdamW with a
   warmup-then-cosine schedule;
5. every ``train.eval_every`` steps a validation subset is decoded **in batches** (greedy or beam
   from :mod:`aicls.ocr.models.seq2seq`) and scored with :mod:`aicls.eval.metrics`;
6. :class:`aicls.ml.checkpoint.CheckpointSaver` keeps the best-by-metric and the last checkpoint;
7. :class:`EarlyStopping`;
8. ``KeyboardInterrupt`` writes an ``interrupt/`` checkpoint and is re-raised.

:meth:`Trainer.resume` restores model, optimizer, scheduler, step counter and the RNG state of
python, numpy, torch and CUDA, and skips the batches the interrupted run had already consumed, so
a resumed run follows the same trajectory as an uninterrupted one
(``test_trainer_two_steps_and_resume``).

**The validation pass is the expensive half of a run, and the report has to say so.** Measured on
the RTX 5090 against the official MathWriting excerpt on 2026-09-08 (run ``ocr_excerpt_wiring``):
20 optimizer steps cost 1.4 s in total while each validation of sixteen samples cost ~296 s — 18.5
seconds *per sample*, because decoding ran one sample at a time and recomputed the whole prefix at
every token. Extrapolated to the shipped ``eval.max_n_val: 2000`` that is a 10-hour validation pass
against a four-minute training interval. Two things follow, and both live in this module:

* :meth:`Trainer.evaluate` decodes the subset **in batches** through
  :func:`aicls.ocr.models.seq2seq.greedy_decode_batch` / ``beam_search_batch``, falling back to
  :func:`batched_greedy_ids` (this module) and finally to per-sample decoding, always recording
  which path produced the number in :attr:`FitReport.eval_decoder`;
* :class:`TimeBudget` splits the wall clock into ``train_seconds`` / ``eval_seconds`` /
  ``checkpoint_seconds`` / ``dataloader_wait_s``. ``wall_time_s 613`` next to ``steps_per_s 8.66``
  are both true of a run that trained for 1.4 s and validated for 592 s, and together they are
  misleading; the split is what makes the report checkable.

**A truncated validation set is a different number, not a cheaper one.** ``eval.max_n_val`` caps
the decoded subset, so a CER over 200 of 2000 samples is not the CER of that model. Every metrics
row and every fit report therefore carries :class:`ValCoverage` — ``n_val``, ``n_val_total`` and
``val_truncated`` — next to the metric it qualifies.

**Device and dtype.** :func:`select_device` is deliberately a small pure function over a
:class:`DeviceProbe`: bf16 autocast is only correct on CUDA, and the machine this code was written
on runs ``torch 2.11.0+cpu`` behind the ``CUDA_INSTALL`` gate. On CPU the loop runs fp32 with no
autocast context at all — no per-step warning, no silent dtype surprise — and the fallback is
recorded once in ``DeviceSpec.fallback_reason``. Separating the probe from the decision is what
makes the GPU branch testable before the gate opens.

**HONESTY.** Everything this module can produce today runs on randomly initialised weights over
deterministic synthetic Bezier scribbles. A loss that falls for two steps is a WIRING CHECK: it
shows that batches reach the model and that the optimizer moves the weights. It is not an OCR
result and must never be quoted against the proposal's CER < 10 % target. Real numbers need the
``DATASET_DOWNLOAD``, ``WEIGHTS_DOWNLOAD``, ``CUDA_INSTALL`` and ``TRAINING`` gates (ADR-0012);
until then :attr:`FitReport.wiring_check` is true and :attr:`FitReport.notes` says so in the file.

torch is imported lazily inside functions so that importing this module (a CLI listing configs,
a test collecting names) stays cheap and torch-free.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
import functools
import json
import math
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from aicls.core import messages_tr
from aicls.core.errors import InputValidationError
from aicls.core.gates import GateName, require_gate
from aicls.core.logging import get_logger
from aicls.ml.checkpoint import CheckpointSaver, load_checkpoint, save_checkpoint
from aicls.ml.config import DEFAULT_MAX_LEN, AmpMode, TrainConfig, resolved_dict
from aicls.ml.manifest import RunManifest, capture_env, seed_everything
from aicls.ml.tracking import JsonlTracker
from aicls.ocr.tokenizer import BOS_ID, EOS_ID, PAD_ID
from aicls.ocr.torch_utils import require_torch

_log = get_logger(__name__)

#: Sub-directory of the run directory that holds the step checkpoints.
CHECKPOINTS_DIRNAME = "checkpoints"

#: Checkpoint directory written when the run is interrupted with Ctrl-C (design document §4.1).
INTERRUPT_DIRNAME = "interrupt"

#: File the fit report is written to inside the run directory.
FIT_REPORT_FILENAME = "fit_report.json"

#: Approval gate every non-synthetic training run has to pass (ADR-0012).
TRAINING_GATE: GateName = "TRAINING"

#: Metric key the checkpoint selector uses by default (``CheckpointCfg.metric``).
VAL_LOSS_KEY = "val_loss"

#: Prefix every validation metric is *also* published under, so ``checkpoint.metric: val_cer``
#: (what the shipped OCR configurations ask for) resolves against :meth:`Trainer.evaluate`.
VAL_PREFIX = "val_"

#: Keys :meth:`Trainer.evaluate` publishes to describe the *subset* a metric was computed on
#: rather than the model. They are metrics-row citizens on purpose: a CER without its ``n_val``
#: and ``n_val_total`` cannot be checked (``docs/ml/egitim-altyapisi.md`` §11.4).
COVERAGE_KEYS: tuple[str, ...] = ("n_val", "n_val_total", "val_truncated")

#: Name of the batched greedy entry point :mod:`aicls.ocr.models.seq2seq` is expected to expose.
BATCH_GREEDY_ENTRY = "greedy_decode_batch"

#: Name of the batched beam entry point :mod:`aicls.ocr.models.seq2seq` is expected to expose.
BATCH_BEAM_ENTRY = "beam_search_batch"

#: Value :attr:`FitReport.eval_decoder` takes when this module batched the greedy decode itself.
LOCAL_GREEDY_PATH = "aicls.ml.trainer.batched_greedy_ids"

#: ``train.amp`` value to the dtype autocast would use.
AMP_DTYPES: dict[str, str] = {"off": "float32", "bf16": "bfloat16", "fp16": "float16"}

#: Batch key holding the model input, per ``data.mode``.
INPUT_KEYS: dict[str, str] = {"offline": "images", "online": "features"}

#: OCR families this loop drives; ``whisper``/``wav2vec2`` belong to ``WhisperFineTuner``.
TRAINER_FAMILIES: tuple[str, ...] = ("seq2seq_image", "seq2seq_stroke")

#: ``model.family`` to the ``data.mode`` it consumes.
FAMILY_MODES: dict[str, str] = {"seq2seq_image": "offline", "seq2seq_stroke": "online"}

#: Written into every report produced from synthetic data, so no reader can mistake it evidence.
WIRING_CHECK_NOTE = (
    "WIRING CHECK, NOT A RESULT: this run used synthetic data (and, before the WEIGHTS_DOWNLOAD "
    "gate, randomly initialised weights). Every metric here shows that the pipeline is wired, "
    "not that anything was recognised. See docs/ml/egitim-altyapisi.md §9 and ADR-0012."
)

#: Wall-clock floor for checkpointing, in seconds.
#:
#: ``train.ckpt_every`` counts optimizer steps, so it prices the work at risk in *steps*: a slow
#: epoch, a stalling data loader or a bigger model can push an hour of GPU time between two
#: saves without any configuration being wrong. This floor prices it in *time* instead. Measured
#: basis (RTX 5090, 2026-09-08, ``runs/ocr_r34_tfm_full``): the shipped OCR configuration trained
#: 2000 steps in 184 s, so its ``ckpt_every: 2000`` saved about every three minutes and this
#: floor never fires; it fires when a run is five times slower than that measurement, which is
#: exactly when a step count stops bounding anything.
DEFAULT_CKPT_MIN_INTERVAL_S = 900.0

#: Wall-clock budget for one call of :meth:`Trainer.evaluate`, in seconds.
#:
#: Measured basis (RTX 5090, 2026-09-08): a validation pass over 2000 real MathWriting samples
#: costs 38.5 s at ``data.batch_size 64`` and ``eval.beam 4``. 900 s is a ~23x margin, so it
#: cannot fire on a healthy pass; what it does is bound the damage of an unhealthy one. On
#: 2026-09-08 an evaluation of a 2000-step model hung for 4 h 47 min inside SymPy and the run had
#: to be killed with no checkpoint at all. With this budget the same stall costs 15 minutes of
#: *monitoring* and the training loop keeps running.
DEFAULT_EVAL_TIMEOUT_S = 900.0

#: Metrics-row key that says an evaluation was cut short by its budget. ``1.0`` means "the
#: numbers of this pass were abandoned", never "the model got worse": an abandoned pass
#: publishes no model-quality metric at all and cannot select a checkpoint.
EVAL_ABORTED_KEY = "eval_aborted"

#: Metrics-row key carrying the evaluation budget that was in force, in seconds.
EVAL_TIMEOUT_KEY = "eval_timeout_s"

#: Metrics-row key counting the decoded-text metrics that finished before the budget ran out.
EVAL_METRICS_COMPUTED_KEY = "eval_metrics_computed"

#: Decoded-text metrics :func:`score_predictions` knows how to compute (``loss`` comes from the
#: loop, not from text, and is deliberately absent).
SCORED_METRICS: tuple[str, ...] = ("cer", "ter", "wer", "exprate", "symeq")

#: The one metric of :data:`SCORED_METRICS` that runs a CAS over model output.
SYMEQ_METRIC = "symeq"

#: Names a **bounded** symbolic comparison may be exposed under by :mod:`aicls.eval.metrics`,
#: best first. See :func:`symeq_scorer` for why the trainer refuses to call an unbounded one.
BOUNDED_SYMEQ_ENTRIES: tuple[str, ...] = (
    "bounded_symbolic_equivalence",
    "symbolic_equivalence_bounded",
    "safe_symbolic_equivalence",
    "guarded_symbolic_equivalence",
)

#: Attributes on ``aicls.eval.metrics.symbolic_equivalence`` that mark it as already bounded (a
#: guard added in place rather than under a new name).
SYMEQ_BOUNDED_MARKERS: tuple[str, ...] = ("bounded", "is_bounded", "sandboxed", "timeout_s")

#: Module-level constants in :mod:`aicls.eval.metrics` whose presence means the same thing.
SYMEQ_TIMEOUT_CONSTANTS: tuple[str, ...] = (
    "SYMEQ_TIMEOUT_S",
    "DEFAULT_SYMEQ_TIMEOUT_S",
    "SYMBOLIC_EQUIVALENCE_TIMEOUT_S",
)

#: What :attr:`FitReport.symeq_entry` says when no bounded comparison could be resolved.
SYMEQ_UNAVAILABLE = "unavailable: symeq skipped (no bounded entry point in aicls.eval.metrics)"

#: Guards the ``symeq`` skip warning so it is emitted once per process rather than once per
#: evaluation. A module-level set rather than a bool so a test can clear it.
_SYMEQ_SKIP_ANNOUNCED: set[str] = set()

BestMode = Literal["min", "max"]


def _past(deadline: float | None) -> bool:
    """Whether a :func:`time.perf_counter` deadline has been reached (``None`` = no deadline)."""
    return deadline is not None and perf_counter() >= deadline


def _resolve_budget(
    explicit: float | None, configured: Any, default: float, *, field: str
) -> float:
    """Resolve a wall-clock budget: the argument wins, then the config, then ``default``.

    ``math.inf`` is the documented way to switch a budget off, which is why it is accepted here
    and a non-positive number is not: a zero budget would abandon every evaluation before it
    started, and a run whose monitoring is silently always-abandoned is worse than one with none.

    Args:
        explicit: What the caller passed; ``None`` means "not specified".
        configured: What the configuration says, or ``None``. ``TrainConfig`` has no such key
            today; it is read through :func:`getattr` so that adding one later needs no change
            here.
        default: The module default.
        field: Name used in the error details.

    Returns:
        The budget in seconds (possibly ``math.inf``).

    Raises:
        InputValidationError: If the resolved value is not a positive number.
    """
    value = explicit if explicit is not None else configured
    seconds = float(default if value is None else value)
    if not seconds > 0.0:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={
                "reason": "non_positive_budget",
                "field": field,
                "value": seconds,
                "hint": "pass math.inf to switch the budget off; 0 would abandon every pass",
            },
        )
    return seconds


# ----------------------------------------------------------------------------------------------
# device and dtype
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DeviceProbe:
    """What the machine can do, as plain data.

    Keeping the hardware query out of :func:`select_device` is what makes the CUDA branch
    testable on a CPU-only box: a test constructs the probe it wants.

    Attributes:
        cuda_available: Whether a usable CUDA device is present.
        bf16_supported: Whether that device supports bfloat16 matmuls.
        device_name: Human-readable device name (``"cpu"`` when there is no GPU).
        capability: CUDA compute capability, e.g. ``(12, 0)`` for Blackwell sm_120.
    """

    cuda_available: bool = False
    bf16_supported: bool = False
    device_name: str = "cpu"
    capability: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    """The device and dtype decision for one run.

    Attributes:
        device: ``"cpu"`` or ``"cuda"``.
        dtype: The autocast dtype name, or ``"float32"`` when autocast is off.
        autocast: Whether the forward pass runs inside ``torch.autocast``.
        amp: The ``train.amp`` value that was requested.
        fallback_reason: Empty when the request was honoured; otherwise why it was not
            (``"amp_requires_cuda"``, ``"bf16_unsupported"``).
        device_name: The probed device name, for the manifest and the logs.
    """

    device: str = "cpu"
    dtype: str = "float32"
    autocast: bool = False
    amp: str = "off"
    fallback_reason: str = ""
    device_name: str = "cpu"

    @property
    def is_cuda(self) -> bool:
        """Whether the run lives on a CUDA device."""
        return self.device == "cuda"

    @property
    def needs_grad_scaler(self) -> bool:
        """fp16 autocast needs loss scaling; bf16 and fp32 do not."""
        return self.autocast and self.dtype == "float16"

    def torch_dtype(self) -> Any:
        """The ``torch`` dtype object named by :attr:`dtype`."""
        return getattr(require_torch(), self.dtype)

    def autocast_context(self) -> AbstractContextManager[Any]:
        """The context the forward pass runs in — a null context whenever autocast is off."""
        if not self.autocast:
            return nullcontext()
        torch = require_torch()  # pragma: no cover - autocast is CUDA-only here
        return torch.autocast(device_type=self.device, dtype=self.torch_dtype())  # pragma: no cover


def probe_device(torch_module: Any = None) -> DeviceProbe:
    """Ask torch what this machine offers.

    Args:
        torch_module: Injected ``torch`` module (tests); imported lazily when omitted.

    Returns:
        The :class:`DeviceProbe`. On the CPU wheel installed here every CUDA field is false.
    """
    torch = torch_module if torch_module is not None else require_torch()
    if not torch.cuda.is_available():
        return DeviceProbe()
    # Everything below needs a CUDA build; it is unreachable until the CUDA_INSTALL gate opens
    # and must be re-verified the first time this runs on the RTX 5090 (sm_120, cu128 wheel).
    index = torch.cuda.current_device()  # pragma: no cover - no CUDA wheel in this environment
    capability = tuple(torch.cuda.get_device_capability(index))  # pragma: no cover
    return DeviceProbe(  # pragma: no cover
        cuda_available=True,
        bf16_supported=bool(torch.cuda.is_bf16_supported()),
        device_name=str(torch.cuda.get_device_name(index)),
        capability=(int(capability[0]), int(capability[1])),
    )


def select_device(
    amp: AmpMode | str = "off",
    *,
    device: str | None = None,
    probe: DeviceProbe | None = None,
) -> DeviceSpec:
    """Decide device and dtype for a run — the one function the CUDA gate will exercise first.

    The rules, in order:

    * ``amp: off`` runs fp32 everywhere;
    * mixed precision on anything but CUDA degrades to fp32 (``amp_requires_cuda``) rather than
      raising, because the same config file has to run on the CPU box and on the workstation;
    * ``bf16`` on a CUDA device that has no bf16 support degrades to fp32 (``bf16_unsupported``);
    * an *explicit* ``device="cuda"`` on a machine without CUDA is an error, not a fallback.

    Args:
        amp: ``"off"``, ``"bf16"`` or ``"fp16"`` (``cfg.train.amp``).
        device: Force a device; ``None`` picks CUDA when it is available.
        probe: Hardware description; :func:`probe_device` is used when omitted.

    Returns:
        The :class:`DeviceSpec` the trainer runs with.

    Raises:
        InputValidationError: On an unknown ``amp`` mode, an unknown device string, or an
            explicit CUDA request on a machine without CUDA.
    """
    hardware = probe if probe is not None else probe_device()
    requested = str(amp or "off")
    if requested not in AMP_DTYPES:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={
                "reason": "unknown_amp_mode",
                "amp": requested,
                "allowed": sorted(AMP_DTYPES),
            },
        )
    if device is None:
        resolved = "cuda" if hardware.cuda_available else "cpu"
    else:
        resolved = str(device).split(":", 1)[0]
        if resolved not in ("cpu", "cuda"):
            raise InputValidationError(
                messages_tr.MSG_VALIDATION_ERROR,
                details={"reason": "unknown_device", "device": str(device)},
            )
        if resolved == "cuda" and not hardware.cuda_available:
            raise InputValidationError(
                messages_tr.MSG_VALIDATION_ERROR,
                details={
                    "reason": "cuda_unavailable",
                    "device": str(device),
                    "hint": "the cu128 wheel sits behind the CUDA_INSTALL gate (ADR-0012)",
                },
            )

    reason = ""
    if requested == "off":
        autocast = False
    elif resolved != "cuda":
        autocast, reason = False, "amp_requires_cuda"
    elif requested == "bf16" and not hardware.bf16_supported:
        autocast, reason = False, "bf16_unsupported"
    else:  # pragma: no cover - CUDA autocast is unreachable in this environment
        autocast = True
    spec = DeviceSpec(
        device=resolved,
        dtype=AMP_DTYPES[requested] if autocast else "float32",
        autocast=autocast,
        amp=requested,
        fallback_reason=reason,
        device_name=hardware.device_name,
    )
    # Logged once per trainer, never inside the step loop: no warning spam on the CPU path.
    _log.info(
        "ml.trainer.device_selected",
        device=spec.device,
        dtype=spec.dtype,
        autocast=spec.autocast,
        amp_requested=spec.amp,
        fallback_reason=spec.fallback_reason or None,
        device_name=spec.device_name,
    )
    return spec


# ----------------------------------------------------------------------------------------------
# schedule
# ----------------------------------------------------------------------------------------------
def schedule_factor(step: int, *, warmup: int = 0, total: int = 1, kind: str = "cosine") -> float:
    """Learning-rate multiplier at ``step`` (linear warmup, then decay).

    Args:
        step: Optimizer steps already taken (0-based; the value ``LambdaLR`` passes).
        warmup: Warmup steps; the factor ramps ``1/warmup ... 1``.
        total: Total step budget the decay is spread over.
        kind: ``"cosine"``, ``"linear"`` or ``"constant"`` (``cfg.optim.scheduler``).

    Returns:
        A multiplier in ``[0, 1]``.

    Raises:
        InputValidationError: If ``step`` is negative or ``kind`` is unknown.
    """
    if step < 0:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "negative_step", "step": int(step)},
        )
    if kind not in ("cosine", "linear", "constant"):
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "unknown_scheduler", "scheduler": str(kind)},
        )
    warm = max(int(warmup), 0)
    budget = max(int(total), 1)
    if warm and step < warm:
        return float(step + 1) / float(warm)
    if kind == "constant":
        return 1.0
    span = max(budget - warm, 1)
    progress = min(max(float(step - warm) / float(span), 0.0), 1.0)
    if kind == "linear":
        return max(1.0 - progress, 0.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ----------------------------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------------------------
def seed_worker(worker_id: int) -> None:
    """Seed ``random`` and ``numpy`` inside a DataLoader worker from torch's per-worker seed."""
    torch = require_torch()
    worker_seed = (int(torch.initial_seed()) + int(worker_id)) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def loader_generator(seed: int, epoch: int) -> Any:
    """A torch generator seeded from ``(seed, epoch)``, so every epoch is reproducible."""
    torch = require_torch()
    generator = torch.Generator()
    generator.manual_seed((int(seed) * 1_000_003 + int(epoch)) % (2**31 - 1))
    return generator


def collate_for(cfg: TrainConfig) -> Callable[[Sequence[Any]], dict[str, Any]]:
    """The :mod:`aicls.data.collate` function matching ``data.mode`` (tensors, not numpy)."""
    from aicls.data.collate import collate_image, collate_ink

    function = collate_image if cfg.data.mode == "offline" else collate_ink
    return functools.partial(function, to_torch=True)


def make_dataloader(
    dataset: Any,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 1234,
    epoch: int = 0,
    collate: Callable[[Sequence[Any]], dict[str, Any]] | None = None,
    drop_last: bool = False,
) -> Any:
    """Build a ``DataLoader`` whose batch order is a function of ``(seed, epoch)``.

    Args:
        dataset: An :mod:`aicls.data` map-style dataset (``to_torch()`` is used when present) or
            any object with ``__len__``/``__getitem__``.
        batch_size: Samples per micro-batch.
        shuffle: Whether to shuffle (training yes, validation no).
        num_workers: Worker processes; ``0`` on Windows in tests (no ``fork``).
        seed: ``run.seed``.
        epoch: Epoch index; a new epoch reshuffles deterministically.
        collate: Collate function; see :func:`collate_for`.
        drop_last: Drop an incomplete final batch.

    Returns:
        The ``torch.utils.data.DataLoader``.
    """
    require_torch()
    from torch.utils.data import DataLoader

    source = dataset.to_torch() if hasattr(dataset, "to_torch") else dataset
    workers = max(int(num_workers), 0)
    return DataLoader(
        source,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=workers,
        collate_fn=collate,
        generator=loader_generator(seed, epoch),
        worker_init_fn=seed_worker if workers else None,
        drop_last=bool(drop_last),
    )


# ----------------------------------------------------------------------------------------------
# early stopping and timing
# ----------------------------------------------------------------------------------------------
class EarlyStopping:
    """Stop after ``patience`` validations without an improvement (design document §4.1).

    Args:
        patience: Validations to tolerate; ``None`` disables early stopping.
        mode: ``"min"`` (lower is better, e.g. CER) or ``"max"``.
        min_delta: How much better a value has to be to count as an improvement.

    Raises:
        InputValidationError: On a non-positive patience or an unknown mode.
    """

    def __init__(
        self, patience: int | None, *, mode: BestMode = "min", min_delta: float = 0.0
    ) -> None:
        if patience is not None and patience < 1:
            raise InputValidationError(
                messages_tr.MSG_VALIDATION_ERROR,
                details={"reason": "invalid_patience", "patience": patience},
            )
        if mode not in ("min", "max"):
            raise InputValidationError(
                messages_tr.MSG_VALIDATION_ERROR,
                details={"reason": "invalid_mode", "mode": str(mode)},
            )
        self.patience = patience
        self.mode: BestMode = mode
        self.min_delta = float(min_delta)
        self.best: float | None = None
        self.bad_evals = 0

    def update(self, value: float | None) -> bool:
        """Record one validation result.

        ``None`` means the metric was **not measured** (the run has no validation set, or the
        metric was not requested). That is not counted as a bad round: a configuration mistake
        must not look like a diverging model.

        Returns:
            Whether this value improved on the best one so far.
        """
        if value is None:
            return False
        number = float(value)
        if self.best is None:
            self.best = number
            self.bad_evals = 0
            return True
        improved = (
            number < self.best - self.min_delta
            if self.mode == "min"
            else number > self.best + self.min_delta
        )
        if improved:
            self.best = number
            self.bad_evals = 0
            return True
        self.bad_evals += 1
        return False

    @property
    def should_stop(self) -> bool:
        """Whether the patience is exhausted."""
        return self.patience is not None and self.bad_evals >= self.patience


@dataclass(frozen=True, slots=True)
class StepTiming:
    """Wall time of one optimizer step (the profiler's raw material).

    Attributes:
        step: The optimizer step this describes.
        seconds: Total wall time of the step.
        data_wait_s: Part of it spent waiting for the data loader.
    """

    step: int
    seconds: float
    data_wait_s: float


@dataclass(slots=True)
class TrainStats:
    """Running totals of the step loop.

    Attributes:
        steps: Optimizer steps measured.
        seconds: Wall time inside the step loop.
        data_wait_s: Wall time spent waiting for batches.
        records: Per-step timings; only filled when ``Trainer.record_step_times`` is set (the
            profiler does), so a 100k-step run does not accumulate a list of that size.
    """

    steps: int = 0
    seconds: float = 0.0
    data_wait_s: float = 0.0
    records: list[StepTiming] = field(default_factory=list)

    def add(self, step: int, seconds: float, data_wait_s: float, *, record: bool = False) -> None:
        """Account for one optimizer step."""
        self.steps += 1
        self.seconds += float(seconds)
        self.data_wait_s += float(data_wait_s)
        if record:
            self.records.append(
                StepTiming(step=int(step), seconds=float(seconds), data_wait_s=float(data_wait_s))
            )

    @property
    def steps_per_s(self) -> float | None:
        """Measured optimizer steps per second, or ``None`` when nothing was measured."""
        if self.steps <= 0 or self.seconds <= 0.0:
            return None
        return self.steps / self.seconds

    @property
    def dataloader_wait_pct(self) -> float:
        """Share of the loop spent waiting for data, in percent."""
        if self.seconds <= 0.0:
            return 0.0
        return 100.0 * self.data_wait_s / self.seconds


@dataclass(slots=True)
class TimeBudget:
    """Where the wall clock of one :meth:`Trainer.fit` went, phase by phase.

    ``fit_report.json`` used to publish ``wall_time_s: 613`` beside ``steps_per_s: 8.66`` for a
    run that spent 1.4 s training and 592 s validating (``ocr_excerpt_wiring``, RTX 5090,
    2026-09-08). Both numbers were true and nobody reading them could see where the time went.
    Section 8 of the design document forbids hand-written epoch estimates and requires the budget
    to come from a measurement — and a measurement that omits the dominant term is worse than
    none, because it is quotable.

    Attributes:
        train_seconds: Wall time inside the optimizer-step loop.
        eval_seconds: Wall time inside :meth:`Trainer.evaluate`, decoding included — abandoned
            passes too, because the clock has to add up.
        checkpoint_seconds: Wall time spent writing checkpoints.
        dataloader_wait_s: Part of ``train_seconds`` spent blocked on the data loader.
        evaluations: Validation passes that **completed**.
        eval_samples: Validation samples decoded across those.
        evaluations_abandoned: Validation passes cut short by the watchdog.
        eval_seconds_abandoned: The part of ``eval_seconds`` those cost. It is subtracted from
            the per-unit costs below: a stalled pass is not evidence of what a pass costs, and
            leaving it in would poison the schedule projection it feeds.
        checkpoints: Checkpoint writes performed.
        wall_time_s: Total wall time of the call, set once at the end.
    """

    train_seconds: float = 0.0
    eval_seconds: float = 0.0
    checkpoint_seconds: float = 0.0
    dataloader_wait_s: float = 0.0
    evaluations: int = 0
    eval_samples: int = 0
    evaluations_abandoned: int = 0
    eval_seconds_abandoned: float = 0.0
    checkpoints: int = 0
    wall_time_s: float = 0.0

    def add_eval(self, seconds: float, *, n_val: int, aborted: bool = False) -> None:
        """Account for one validation pass that decoded ``n_val`` samples.

        Args:
            seconds: Wall time the pass took.
            n_val: Samples it decoded.
            aborted: Whether the watchdog cut it short. The seconds are still charged to
                ``eval_seconds`` (they were really spent) but the pass counts towards
                ``evaluations_abandoned`` instead of ``evaluations``, and its samples are not
                counted at all — half a decode is not a measurement of a decode.
        """
        self.eval_seconds += float(seconds)
        if aborted:
            self.evaluations_abandoned += 1
            self.eval_seconds_abandoned += float(seconds)
            return
        self.evaluations += 1
        self.eval_samples += int(n_val)

    def add_checkpoint(self, seconds: float) -> None:
        """Account for one checkpoint write."""
        self.checkpoint_seconds += float(seconds)
        self.checkpoints += 1

    def adopt_train(self, stats: TrainStats) -> None:
        """Copy the step loop's totals in (they are measured by :class:`TrainStats`)."""
        self.train_seconds = float(stats.seconds)
        self.dataloader_wait_s = float(stats.data_wait_s)

    @property
    def accounted_seconds(self) -> float:
        """Seconds this budget explains."""
        return self.train_seconds + self.eval_seconds + self.checkpoint_seconds

    @property
    def other_seconds(self) -> float:
        """Wall time the three phases do not explain (setup, manifest, git, teardown)."""
        return max(self.wall_time_s - self.accounted_seconds, 0.0)

    @property
    def completed_eval_seconds(self) -> float:
        """Seconds spent in validation passes that actually finished."""
        return max(self.eval_seconds - self.eval_seconds_abandoned, 0.0)

    @property
    def seconds_per_eval(self) -> float | None:
        """Mean cost of one **completed** validation pass, or ``None`` when none was measured."""
        if self.evaluations <= 0:
            return None
        return self.completed_eval_seconds / self.evaluations

    @property
    def seconds_per_val_sample(self) -> float | None:
        """Mean cost of decoding one validation sample — the number that decides the schedule."""
        if self.eval_samples <= 0 or self.completed_eval_seconds <= 0.0:
            return None
        return self.completed_eval_seconds / self.eval_samples

    @property
    def eval_share_pct(self) -> float | None:
        """Share of the wall clock spent validating, or ``None`` before it is known."""
        if self.wall_time_s <= 0.0:
            return None
        return 100.0 * self.eval_seconds / self.wall_time_s

    def to_dict(self) -> dict[str, float | int | None]:
        """JSON-compatible view, derived values included."""
        return {
            "train_seconds": self.train_seconds,
            "eval_seconds": self.eval_seconds,
            "checkpoint_seconds": self.checkpoint_seconds,
            "dataloader_wait_s": self.dataloader_wait_s,
            "other_seconds": self.other_seconds,
            "wall_time_s": self.wall_time_s,
            "evaluations": self.evaluations,
            "eval_samples": self.eval_samples,
            "evaluations_abandoned": self.evaluations_abandoned,
            "eval_seconds_abandoned": self.eval_seconds_abandoned,
            "checkpoints": self.checkpoints,
            "seconds_per_eval": self.seconds_per_eval,
            "seconds_per_val_sample": self.seconds_per_val_sample,
            "eval_share_pct": self.eval_share_pct,
        }


# ----------------------------------------------------------------------------------------------
# validation coverage
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ValCoverage:
    """How much of a validation split one metric was computed on.

    ``eval.max_n_val`` makes validation affordable by shortening it, which silently changes the
    number being reported: a WER over 200 of 2000 samples is not the WER over 2000. Recording the
    truncation next to the metric is what makes the report checkable, so this record travels with
    every metrics row and every fit report.

    Attributes:
        n_val: Samples the metric was actually computed on.
        n_total: Samples in the full split, or ``None`` when the dataset cannot say.
        limit: The cap that was applied (``eval.max_n_val`` or an explicit argument).
    """

    n_val: int
    n_total: int | None = None
    limit: int | None = None

    @property
    def truncated(self) -> bool:
        """Whether samples of the split were left out of the number."""
        return self.n_total is not None and self.n_val < self.n_total

    @property
    def coverage_pct(self) -> float | None:
        """Share of the split the metric covers, or ``None`` when the split size is unknown."""
        if self.n_total is None or self.n_total <= 0:
            return None
        return 100.0 * self.n_val / self.n_total

    def as_metrics(self) -> dict[str, float]:
        """The :data:`COVERAGE_KEYS` as a metrics row fragment.

        ``n_val_total`` and ``val_truncated`` are omitted when the split size is unknown: an
        invented total would be worse than a missing one.
        """
        row: dict[str, float] = {"n_val": float(self.n_val)}
        if self.n_total is not None:
            row["n_val_total"] = float(self.n_total)
            row["val_truncated"] = 1.0 if self.truncated else 0.0
        return row


def split_size(dataset: Any) -> int | None:
    """Number of samples in ``dataset``, or ``None`` when it cannot say.

    Args:
        dataset: Any dataset; iterable-style datasets have no length and that is not an error.

    Returns:
        The length, or ``None``.
    """
    try:
        size = len(dataset)
    except TypeError:
        return None
    return int(size) if size >= 0 else None


@dataclass(frozen=True, slots=True)
class SubsetView:
    """The first ``n`` samples of a map-style dataset, as a map-style dataset.

    :meth:`Trainer.evaluate` truncates while it decodes and needs no view, but a loop that is
    handed a *dataset* — Hugging Face's ``Seq2SeqTrainer`` behind
    :mod:`aicls.ml.whisper_ft` — cannot honour ``eval.max_n_val`` any other way. That path
    evaluated all 64 samples of a config asking for ``max_n_val: 8`` (measured on the RTX 5090,
    2026-09-08); :func:`limit_val_dataset` is the one-call fix.

    Attributes:
        dataset: The wrapped dataset.
        n: How many leading samples this view exposes.
    """

    dataset: Any
    n: int

    def __len__(self) -> int:
        """Number of samples in the view."""
        return self.n

    def __getitem__(self, index: int) -> Any:
        """Sample ``index`` of the view.

        Raises:
            IndexError: If ``index`` is outside the view.
        """
        position = index + self.n if index < 0 else index
        if not 0 <= position < self.n:
            raise IndexError(f"dataset index out of range: {index}")
        return self.dataset[position]

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the exposed samples."""
        for position in range(self.n):
            yield self.dataset[position]


def limit_val_dataset(dataset: Any, max_n: int | None) -> tuple[Any, ValCoverage]:
    """Truncate a validation dataset to ``max_n`` samples and record the truncation.

    Args:
        dataset: The validation dataset; ``None`` is returned untouched with a zero coverage.
        max_n: ``eval.max_n_val``; ``None`` means "the whole split".

    Returns:
        ``(dataset_or_view, coverage)``. The dataset is returned unchanged when no truncation is
        needed, so nothing is wrapped for nothing.

    Raises:
        InputValidationError: If ``max_n`` is negative.
    """
    if max_n is not None and int(max_n) < 0:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "negative_max_n_val", "max_n_val": int(max_n)},
        )
    if dataset is None:
        return None, ValCoverage(n_val=0, n_total=None, limit=max_n)
    total = split_size(dataset)
    if max_n is None or total is None or total <= int(max_n):
        return dataset, ValCoverage(
            n_val=total if total is not None else 0, n_total=total, limit=max_n
        )
    kept = int(max_n)
    _log.info(
        "ml.trainer.val_split_truncated",
        n_val=kept,
        n_val_total=total,
        max_n_val=int(max_n),
        hint="the reported metric describes this subset, not the whole split",
    )
    return SubsetView(dataset, kept), ValCoverage(n_val=kept, n_total=total, limit=max_n)


# ----------------------------------------------------------------------------------------------
# batched decoding
# ----------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def batch_decoders() -> tuple[Any | None, Any | None]:
    """The batched decoding entry points of :mod:`aicls.ocr.models.seq2seq`, when they exist.

    The batched, incrementally-cached decoders live in ``aicls.ocr`` (that is where the model is);
    this loop only has to use them. Resolving them by name instead of importing them keeps the
    trainer working against the pre-batching decoder too, at the cost of one slow path that
    :attr:`FitReport.eval_decoder` names in the report.

    Returns:
        ``(greedy_batch, beam_batch)``; either is ``None`` when that entry point is absent.
    """
    from aicls.ocr.models import seq2seq

    greedy = getattr(seq2seq, BATCH_GREEDY_ENTRY, None)
    beam = getattr(seq2seq, BATCH_BEAM_ENTRY, None)
    resolved = (greedy if callable(greedy) else None, beam if callable(beam) else None)
    _log.info(
        "ml.trainer.batch_decoders_resolved",
        greedy=BATCH_GREEDY_ENTRY if resolved[0] is not None else None,
        beam=BATCH_BEAM_ENTRY if resolved[1] is not None else None,
    )
    return resolved


def as_token_ids(result: Any) -> list[int]:
    """Normalise one decoder result into a plain list of token ids.

    The three shapes the decoding entry points use are accepted: ``[ids]`` (a bare sequence),
    ``(ids, score)`` (:func:`aicls.ocr.models.seq2seq.greedy_decode`) and
    ``[(ids, score), ...]`` sorted best-first (:func:`aicls.ocr.models.seq2seq.beam_search`).

    Args:
        result: One sample's decoder output.

    Returns:
        The token ids of the best hypothesis.

    Raises:
        InputValidationError: On a shape none of the entry points produces — better than
            decoding a report-grade metric out of something unrecognised.
    """
    item: Any = result
    if isinstance(item, list) and item and isinstance(item[0], tuple):
        item = item[0]  # beam hypotheses, best first
    if isinstance(item, tuple):
        if len(item) != 2:
            raise InputValidationError(
                messages_tr.MSG_VALIDATION_ERROR,
                details={"reason": "unknown_decoder_result", "arity": len(item)},
            )
        item = item[0]  # (ids, score)
    if not isinstance(item, list | tuple):
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "unknown_decoder_result", "type": type(result).__name__},
        )
    return [int(token) for token in item]


def batched_greedy_ids(
    model: Any,
    inputs: Any,
    *,
    max_len: int = DEFAULT_MAX_LEN,
    bos_id: int = BOS_ID,
    eos_id: int = EOS_ID,
) -> list[list[int]]:
    """Greedy-decode a whole batch: one encoder pass, one decoder call per position.

    A **fallback**, used only while :func:`aicls.ocr.models.seq2seq.greedy_decode_batch` is
    absent, and written against the model's public ``encode``/``decode`` pair alone so that it
    works with any decoder the registry can build. It reproduces
    :func:`aicls.ocr.models.seq2seq.greedy_decode` row by row — ``argmax`` over the final
    position's logits, stop at ``</s>`` — with the per-sample loop moved into the batch
    dimension, which is where the time goes on a GPU: the cost is dominated by kernel-launch
    latency, not by arithmetic.

    Args:
        model: A model exposing ``encode(inputs)`` and ``decode(memory, tgt_ids)``.
        inputs: ``(B, ...)`` batch of encoder inputs.
        max_len: Maximum generated tokens per row.
        bos_id: Start token id.
        eos_id: End token id.

    Returns:
        One list of token ids per row, ``<s>``/``</s>`` excluded.
    """
    torch = require_torch()
    with torch.no_grad():
        memory = model.encode(inputs)
        rows = int(memory.shape[0])
        tokens = torch.full((rows, 1), int(bos_id), dtype=torch.long, device=memory.device)
        done = torch.zeros(rows, dtype=torch.bool, device=memory.device)
        for _ in range(max(int(max_len), 0)):
            logits = model.decode(memory, tokens)[:, -1, :]
            step = torch.argmax(logits, dim=-1)
            # A finished row is fed `</s>` forever: its own prefix stops mattering and no other
            # row can see it, so the surviving rows decode exactly as they would alone.
            step = torch.where(done, torch.full_like(step, int(eos_id)), step)
            tokens = torch.cat([tokens, step.unsqueeze(1)], dim=1)
            done = done | (step == int(eos_id))
            if bool(done.all()):
                break
    return [_until_eos(row, int(eos_id)) for row in tokens[:, 1:].tolist()]


def _until_eos(row: Sequence[int], eos_id: int) -> list[int]:
    """The generated ids of one row up to (not including) the first ``</s>``."""
    out: list[int] = []
    for token in row:
        if int(token) == eos_id:
            break
        out.append(int(token))
    return out


# ----------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------
def symeq_scorer() -> tuple[Any | None, str]:
    """Resolve a **bounded** symbolic comparison, or explain why there is none.

    ``symeq`` is the one entry of :data:`SCORED_METRICS` that runs a computer algebra system
    over model output, and model output is arbitrary text. On 2026-09-08 an evaluation of a
    2000-step OCR model called :func:`aicls.eval.metrics.symbolic_equivalence` on a decoded
    hypothesis and did not return: the run hung for 4 h 47 min and had to be killed, with no
    checkpoint written, because the loop evaluates before it saves. py-spy put the whole time in
    ``sympy.core.numbers._eval_power`` under ``visit_pow`` -- a power tower the model invented,
    which nothing in the path bounds.

    A metric that can run for an unbounded time is not a metric the training loop may call. So
    this resolves a comparison that carries its own bound and refuses to substitute one that does
    not. A bound counts as declared when :mod:`aicls.eval.metrics` exposes it under one of
    :data:`BOUNDED_SYMEQ_ENTRIES`, when ``symbolic_equivalence`` itself carries one of
    :data:`SYMEQ_BOUNDED_MARKERS`, or when the module defines one of
    :data:`SYMEQ_TIMEOUT_CONSTANTS` -- three shapes because the bound may reasonably be added
    under a new name, in place, or as a module-wide budget.

    The check is deliberately by declaration rather than by measurement: no test can prove a CAS
    call terminates, so the module that owns the comparison has to say so, and the trainer takes
    it at its word. Skipping the metric is the safe failure. A skipped ``symeq`` costs the run
    one number; an unbounded one costs the run.

    Returns:
        ``(callable, name)`` where ``callable(ref, hyp)`` answers ``"equal"`` for a match, or
        ``(None, reason)`` when no bounded comparison is available -- the reason is
        :data:`SYMEQ_UNAVAILABLE`, and it belongs in the report next to the missing metric.
    """
    from aicls.eval import metrics as _metrics

    for name in BOUNDED_SYMEQ_ENTRIES:
        found = getattr(_metrics, name, None)
        if callable(found):
            return found, name
    in_place = getattr(_metrics, "symbolic_equivalence", None)
    if in_place is not None and any(
        getattr(in_place, marker, None) for marker in SYMEQ_BOUNDED_MARKERS
    ):
        return in_place, "symbolic_equivalence (bounded in place)"
    if in_place is not None and any(
        getattr(_metrics, const, None) for const in SYMEQ_TIMEOUT_CONSTANTS
    ):
        return in_place, "symbolic_equivalence (module budget)"
    return None, SYMEQ_UNAVAILABLE


def score_predictions(pairs: Sequence[tuple[str, str]], metrics: Sequence[str]) -> dict[str, float]:
    """Score ``(reference, hypothesis)`` pairs with :mod:`aicls.eval.metrics`.

    ``loss`` is ignored here — it comes from the loop, not from decoded text. Unknown names are
    ignored as well; :class:`aicls.ml.config.EvalCfg` has already validated them.

    Args:
        pairs: Decoded validation pairs.
        metrics: Metric names from ``cfg.eval.metrics``.

    Returns:
        Metric name to value (empty when there are no pairs).
    """
    if not pairs:
        return {}
    from aicls.eval.metrics import corpus_cer, corpus_ter, corpus_wer, exact_match_rate

    rows = list(pairs)
    scores: dict[str, float] = {}
    for name in metrics:
        if name == "cer":
            scores["cer"] = corpus_cer(rows)
        elif name == "ter":
            scores["ter"] = corpus_ter(rows)
        elif name == "wer":
            scores["wer"] = corpus_wer(rows)
        elif name == "exprate":
            scores["exprate"] = exact_match_rate(rows)
        elif name == SYMEQ_METRIC:
            compare, entry = symeq_scorer()
            if compare is None:
                # Not an error and not a silent drop: the row simply carries no symeq, and the
                # log says why. See symeq_scorer() for the run this rule was written from. Said
                # once per process - a long run evaluates a hundred times and the answer cannot
                # change between them, so repeating it would only bury the rest of the log.
                if not _SYMEQ_SKIP_ANNOUNCED:
                    _SYMEQ_SKIP_ANNOUNCED.add(entry)
                    _log.warning("ml.trainer.symeq_skipped", reason=entry)
                continue
            equal = sum(1 for ref, hyp in rows if compare(ref, hyp) == "equal")
            scores[SYMEQ_METRIC] = equal / len(rows)
    return scores


def _move_model_to_device(model: Any, device: str) -> bool:
    """Move ``model`` onto ``device`` if it is a module at all.

    ``Trainer`` accepts ``model=None`` (the configuration-only tests construct it that way) and
    a stub object is a legitimate argument in unit tests, so this refuses to be the reason a
    non-module trainer cannot be built. Everything that *is* a module is moved eagerly, which is
    what keeps ``evaluate()`` and ``resume()`` usable before ``fit()`` has run.

    Args:
        model: The model, ``None``, or any object.
        device: The torch device string chosen by :func:`select_device`.

    Returns:
        ``True`` when the move happened, ``False`` when the object has no ``.to``.
    """
    move = getattr(model, "to", None)
    if not callable(move):
        return False
    move(device)
    return True


# ----------------------------------------------------------------------------------------------
# gate
# ----------------------------------------------------------------------------------------------
def require_training_gate(
    cfg: TrainConfig, *, i_have_approval: bool = False, ledger_path: Path | None = None
) -> bool:
    """Enforce the ``TRAINING`` gate unless the run is synthetic (ADR-0012, §9).

    Args:
        cfg: The run configuration; ``cfg.is_gate_free`` is decided by the dataset name alone,
            so a config cannot declare its way past the gate.
        i_have_approval: Whether the caller passed ``--i-have-approval``.
        ledger_path: Approval-ledger override (tests only).

    Returns:
        Whether the gate was actually checked (``False`` for a synthetic run).

    Raises:
        GateError: If the run needs approval and any of the three conditions is missing
            (exit code 3).
    """
    if cfg.is_gate_free:
        _log.info("ml.trainer.gate_free", dataset=cfg.data.dataset)
        return False
    require_gate(TRAINING_GATE, cli_flag=bool(i_have_approval), ledger_path=ledger_path)
    return True


# ----------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------
class FitReport(BaseModel):
    """What one :meth:`Trainer.fit` produced.

    Attributes:
        steps: Optimizer steps performed (absolute, resumes included).
        epochs: Fractional epochs the steps correspond to.
        samples_seen: Micro-batch samples pushed through the model in this call.
        train_loss: Loss of the last optimizer step.
        metrics: The last validation metrics.
        best_metric / best_value / best_step: Checkpoint selection state.
        early_stopped: Whether :class:`EarlyStopping` ended the run.
        citable: Copied from the run manifest — false for a dirty tree (ADR-0007).
        wiring_check: True when the run used synthetic data, i.e. when no number in this report
            is evidence for anything (see :data:`WIRING_CHECK_NOTE`).
        train_seconds / eval_seconds / checkpoint_seconds / dataloader_wait_s / other_seconds:
            The :class:`TimeBudget` split. ``wall_time_s`` alone hid a run that trained for 1.4 s
            and validated for 592 s; these four say where the clock went.
        evaluations / seconds_per_eval / seconds_per_val_sample / eval_share_pct: the measured
            per-unit cost of validating, which is what a schedule projection needs.
        n_val / n_val_total / val_truncated: :class:`ValCoverage` of the metrics above — how much
            of the validation split they were computed on.
        eval_decoder: Which decoding entry point produced them (batched or per-sample).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    run_id: str = ""
    run_name: str = ""
    steps: int = 0
    epochs: float = 0.0
    samples_seen: int = 0
    train_loss: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    best_metric: str = VAL_LOSS_KEY
    best_value: float | None = None
    best_step: int | None = None
    best_checkpoint: str | None = None
    last_checkpoint: str | None = None
    early_stopped: bool = False
    interrupted: bool = False
    device: str = "cpu"
    dtype: str = "float32"
    amp: str = "off"
    wall_time_s: float = 0.0
    steps_per_s: float | None = None
    dataloader_wait_pct: float | None = None
    train_seconds: float = 0.0
    eval_seconds: float = 0.0
    checkpoint_seconds: float = 0.0
    dataloader_wait_s: float = 0.0
    other_seconds: float = 0.0
    evaluations: int = 0
    seconds_per_eval: float | None = None
    seconds_per_val_sample: float | None = None
    eval_share_pct: float | None = None
    n_val: int | None = None
    n_val_total: int | None = None
    val_truncated: bool = False
    eval_decoder: str = ""
    run_dir: str = ""
    config_hash: str = ""
    git_sha: str = "unknown"
    citable: bool = False
    wiring_check: bool = True
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
            path = path / FIT_REPORT_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(self.to_json())
        return path


# ----------------------------------------------------------------------------------------------
# the trainer
# ----------------------------------------------------------------------------------------------
class Trainer:
    """The OCR training loop (design document §4.1).

    Args:
        cfg: The resolved run configuration.
        model: A module built by :func:`aicls.ml.registry.build_model`.
        train_ds: Training dataset (map-style, from :mod:`aicls.data`).
        val_ds: Validation dataset; ``None`` disables the validation pass entirely.
        metrics: Metric names to score the decoded validation subset with; defaults to
            ``cfg.eval.metrics``.
        tracker: Experiment tracker; defaults to a :class:`~aicls.ml.tracking.JsonlTracker` in
            the run directory (ADR-0007).
        run_dir: Where ``run_manifest.json``, ``metrics.jsonl`` and ``checkpoints/`` are written;
            defaults to ``AICLS_DATA_ROOT/runs/<run.name>-<ulid>``.
        tokenizer: LaTeX tokenizer for decoding; resolved from ``model.vocab_path`` when omitted.
        repo_root: Directory the manifest's ``git`` commands run in.
        i_have_approval: Forwarded to :func:`require_training_gate`; **only** a CLI that received
            ``--i-have-approval`` may set this.
        ledger_path: Approval-ledger override (tests only).
        device: Force ``"cpu"`` or ``"cuda"``; ``None`` picks what the machine has.
        probe: Hardware probe override (tests).

    Raises:
        InputValidationError: If the model family is not an OCR family, if the family and
            ``data.mode`` disagree, or if LoRA is requested (that is the Whisper path).
    """

    def __init__(
        self,
        cfg: TrainConfig,
        model: Any,
        train_ds: Any,
        val_ds: Any = None,
        *,
        metrics: Sequence[str] | None = None,
        tracker: Any = None,
        run_dir: Path | None = None,
        tokenizer: Any = None,
        repo_root: Path | None = None,
        i_have_approval: bool = False,
        ledger_path: Path | None = None,
        device: str | None = None,
        probe: DeviceProbe | None = None,
    ) -> None:
        self._check_config(cfg)
        self.cfg = cfg
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.metrics: tuple[str, ...] = (
            tuple(metrics) if metrics is not None else tuple(cfg.eval.metrics)
        )
        self.run_dir = Path(run_dir) if run_dir is not None else _default_run_dir(cfg)
        self.tracker = tracker if tracker is not None else JsonlTracker(self.run_dir)
        self.repo_root = repo_root
        self.i_have_approval = bool(i_have_approval)
        self.ledger_path = ledger_path
        self.device_spec = select_device(cfg.train.amp, device=device, probe=probe)
        # The weights move to the selected device HERE, not in `fit()`. Reproduced on the RTX
        # 5090, 2026-09-08: with the move deferred to `fit()`, `evaluate()` or `beam_search()`
        # on a fresh trainer raised "Input type (torch.cuda.FloatTensor) and weight type
        # (torch.FloatTensor) should be the same", and `resume()` built the optimizer over CPU
        # parameters -- torch casts the loaded optimizer state to the parameters' current
        # device, so the later `fit()` move left Adam holding CPU exp_avg tensors against CUDA
        # parameters. `training/scripts/train.py --resume` does exactly that. `fit()` keeps its
        # own `.to()` call, which is now idempotent.
        _move_model_to_device(self.model, self.device_spec.device)
        self.saver = CheckpointSaver(
            self.run_dir / CHECKPOINTS_DIRNAME,
            keep_last=cfg.checkpoint.keep_last,
            keep_best=cfg.checkpoint.keep_best,
            metric=cfg.checkpoint.metric,
            mode=cfg.checkpoint.mode,
            cfg=cfg,
        )
        self.stats = TrainStats()
        self.budget = TimeBudget()
        self.coverage = ValCoverage(n_val=0, n_total=split_size(val_ds), limit=cfg.eval.max_n_val)
        self.eval_decoder = ""
        self.record_step_times = False
        self.step = 0
        self.epoch = 0
        self.samples_seen = 0
        self.manifest: RunManifest | None = None
        self.optimizer: Any = None
        self.scheduler: Any = None
        self._scaler: Any = None
        self._norm_warned = False
        self._tokenizer = tokenizer
        self._total_steps = 1
        self._resumed = False

    # ------------------------------------------------------------------------------ properties
    @property
    def tokenizer(self) -> Any:
        """The LaTeX tokenizer used for decoding (resolved lazily from the config)."""
        if self._tokenizer is None:
            from aicls.ml.registry import DatasetSpec

            self._tokenizer = DatasetSpec.from_config(self.cfg).tokenizer()
        return self._tokenizer

    @property
    def vocab(self) -> dict[str, Any]:
        """The vocabulary document stored next to every checkpoint."""
        return dict(self.tokenizer.to_dict())

    @property
    def interrupt_dir(self) -> Path:
        """Where a ``KeyboardInterrupt`` checkpoint is written."""
        return self.run_dir / CHECKPOINTS_DIRNAME / INTERRUPT_DIRNAME

    # ---------------------------------------------------------------------------------- setup
    @staticmethod
    def _check_config(cfg: TrainConfig) -> None:
        """Refuse configurations this loop cannot honour, instead of ignoring keys silently."""
        family = cfg.model.family
        if family not in TRAINER_FAMILIES:
            raise InputValidationError(
                messages_tr.MSG_VALIDATION_ERROR,
                details={
                    "reason": "family_not_supported_by_trainer",
                    "family": family,
                    "supported": list(TRAINER_FAMILIES),
                    "hint": "speech families are fine-tuned by aicls.ml.whisper_ft (ADR-0006)",
                },
            )
        expected = FAMILY_MODES[family]
        if cfg.data.mode != expected:
            raise InputValidationError(
                messages_tr.MSG_VALIDATION_ERROR,
                details={
                    "reason": "family_mode_mismatch",
                    "family": family,
                    "mode": cfg.data.mode,
                    "expected_mode": expected,
                },
            )
        if cfg.model.lora.enabled:
            raise InputValidationError(
                messages_tr.MSG_VALIDATION_ERROR,
                details={
                    "reason": "lora_not_supported_by_trainer",
                    "hint": "model.lora applies to the Whisper path only (design document §3)",
                },
            )
        if cfg.data.synthetic_warmup_epochs:
            # The "two synthetic-only warm-up epochs, then synthetic_ratio 0.5" curriculum of §1
            # needs the *dataset* to be rebuilt at an epoch boundary, and the trainer receives an
            # already-built dataset. Not implemented: say so loudly rather than let a config key
            # look effective while it does nothing.
            _log.warning(
                "ml.trainer.synthetic_warmup_not_implemented",
                synthetic_warmup_epochs=cfg.data.synthetic_warmup_epochs,
                synthetic_ratio=cfg.data.synthetic_ratio,
                hint="the ratio is fixed at dataset build time; the warm-up switch is unwired",
            )

    def _prepare(self) -> None:
        """Build the optimizer, scheduler and (fp16 only) the gradient scaler, once."""
        if self.optimizer is None:
            apply_freeze(self.model, self.cfg)
            self.optimizer = build_optimizer(self.model, self.cfg)
        if self.scheduler is None:
            self.scheduler = self._build_scheduler(self.optimizer)
        if self._scaler is None and self.device_spec.needs_grad_scaler:
            torch = require_torch()  # pragma: no cover - fp16 needs CUDA
            self._scaler = torch.amp.GradScaler(self.device_spec.device)  # pragma: no cover

    def _build_scheduler(self, optimizer: Any) -> Any:
        """A ``LambdaLR`` reading :attr:`_total_steps` at call time, so a resume can change it."""
        require_torch()
        from torch.optim.lr_scheduler import LambdaLR

        trainer = self

        def factor(step: int) -> float:
            return schedule_factor(
                step,
                warmup=trainer.cfg.optim.warmup_steps,
                total=trainer._total_steps,
                kind=trainer.cfg.optim.scheduler,
            )

        return LambdaLR(optimizer, factor)

    # ---------------------------------------------------------------------------------- resume
    def resume(self, ckpt_dir: Path) -> Any:
        """Restore model, optimizer, scheduler, step counter and RNG state from a checkpoint.

        The RNG of python, numpy, torch and CUDA is put back, and a following :meth:`fit` does
        **not** re-seed (that would throw the restored state away) and skips the batches the
        interrupted run had already consumed.

        Args:
            ckpt_dir: A checkpoint directory written by
                :class:`aicls.ml.checkpoint.CheckpointSaver`.

        Returns:
            The :class:`aicls.ml.checkpoint.Checkpoint` that was loaded.

        Raises:
            NotFoundError: If the directory or its weight file is missing.
            InputValidationError: If the directory is incomplete or the weight hash disagrees.
        """
        self._prepare()
        checkpoint = load_checkpoint(
            Path(ckpt_dir),
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device_spec.device,
            restore_rng=True,
        )
        self.step = int(checkpoint.step)
        self._resumed = True
        self.saver.last_step = self.step
        _log.info(
            "ml.trainer.resumed",
            path=str(ckpt_dir),
            step=self.step,
            rng_restored=checkpoint.rng_restored,
        )
        return checkpoint

    # ------------------------------------------------------------------------------------ fit
    def fit(self, *, max_steps: int | None = None) -> FitReport:
        """Run the training loop.

        Args:
            max_steps: Absolute step budget (the same meaning as ``train.max_steps``: the loop
                stops when ``trainer.step`` reaches it, resumed steps included). The smallest of
                this argument, ``train.max_steps`` and ``train.epochs`` wins; with none of them
                set the loop runs exactly one epoch.

        Returns:
            The :class:`FitReport`. On synthetic data ``wiring_check`` is true and ``notes``
            carries :data:`WIRING_CHECK_NOTE`.

        Raises:
            GateError: If the dataset is not synthetic and the ``TRAINING`` gate is closed.
            KeyboardInterrupt: Re-raised after an ``interrupt/`` checkpoint has been written.
        """
        cfg = self.cfg
        require_training_gate(
            cfg, i_have_approval=self.i_have_approval, ledger_path=self.ledger_path
        )
        started = perf_counter()
        if not self._resumed:
            seed_everything(cfg.run.seed, deterministic=cfg.run.deterministic)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        manifest = self._capture_manifest()
        capture_env(self.run_dir)

        require_torch()
        batches_per_epoch = max(len(self._loader(self.train_ds, shuffle=True, epoch=0)), 1)
        total = self._resolve_total_steps(max_steps, batches_per_epoch)
        self._total_steps = total
        self._prepare()
        self.tracker.start_run(cfg.run.name or self.run_dir.name, resolved_dict(cfg))

        # Idempotent: `__init__` already moved the weights, which is what makes `evaluate()`
        # and `resume()` usable before this call. Kept so that a caller who swapped `.model`
        # after construction still gets a correctly placed module.
        _move_model_to_device(self.model, self.device_spec.device)
        self.model.train()
        stream = self._batch_stream(batches_per_epoch)
        stopper = EarlyStopping(cfg.train.early_stop_patience, mode=cfg.checkpoint.mode)
        metrics: dict[str, float] = {}
        train_loss: float | None = None
        interrupted = False
        _log.info(
            "ml.trainer.fit_started",
            run_dir=str(self.run_dir),
            run_id=manifest.run_id,
            total_steps=total,
            start_step=self.step,
            batches_per_epoch=batches_per_epoch,
            device=self.device_spec.device,
            dtype=self.device_spec.dtype,
            synthetic=cfg.is_gate_free,
        )
        try:
            while self.step < total:
                step_started = perf_counter()
                waited = 0.0
                accumulated = 0.0
                self.optimizer.zero_grad(set_to_none=True)
                for _ in range(cfg.train.grad_accum):
                    batch, wait = self._next_batch(stream)
                    waited += wait
                    loss = self._forward_loss(batch)
                    self._backward(loss / cfg.train.grad_accum)
                    accumulated += float(loss.detach().item())
                    self.samples_seen += self._batch_size(batch)
                grad_norm = self._optimizer_step()
                self.step += 1
                train_loss = accumulated / cfg.train.grad_accum
                self.stats.add(
                    self.step,
                    perf_counter() - step_started,
                    waited,
                    record=self.record_step_times,
                )
                is_last = self.step >= total
                if self.step % cfg.train.log_every == 0 or is_last:
                    self._log_train(train_loss, grad_norm)
                improved = False
                if self.val_ds is not None and (self.step % cfg.train.eval_every == 0 or is_last):
                    metrics = self.evaluate()
                    self.tracker.log_metrics(metrics, self.step, split="valid")
                    # The saver decides "best" from the same numbers the stopper sees, and an
                    # improving validation is always checkpointed: otherwise "best-by-CER" would
                    # only ever mean "best among the steps that happened to hit ckpt_every".
                    improved = self.saver.is_best(metrics)
                    stopper.update(metrics.get(cfg.checkpoint.metric))
                if (
                    improved
                    or is_last
                    or stopper.should_stop
                    or self.step % cfg.train.ckpt_every == 0
                ):
                    ckpt_started = perf_counter()
                    self._save(metrics)
                    self.budget.add_checkpoint(perf_counter() - ckpt_started)
                if stopper.should_stop:
                    _log.info(
                        "ml.trainer.early_stopped",
                        step=self.step,
                        metric=cfg.checkpoint.metric,
                        best=stopper.best,
                        patience=cfg.train.early_stop_patience,
                    )
                    break
        except KeyboardInterrupt:
            interrupted = True
            ckpt_started = perf_counter()
            self._save_interrupt(metrics)
            self.budget.add_checkpoint(perf_counter() - ckpt_started)
            self.tracker.end_run("killed")
            _log.warning(
                "ml.trainer.interrupted",
                step=self.step,
                checkpoint=str(self.interrupt_dir),
                run_dir=str(self.run_dir),
            )
            raise
        except Exception:
            self.tracker.end_run("failed")
            raise

        report = self._build_report(
            manifest=manifest,
            started=started,
            train_loss=train_loss,
            metrics=metrics,
            early_stopped=stopper.should_stop,
            interrupted=interrupted,
            batches_per_epoch=batches_per_epoch,
        )
        path = report.save(self.run_dir)
        self.tracker.log_artifact(path)
        self.tracker.end_run("finished")
        _log.info(
            "ml.trainer.fit_finished",
            run_id=report.run_id,
            steps=report.steps,
            train_loss=report.train_loss,
            steps_per_s=report.steps_per_s,
            train_seconds=report.train_seconds,
            eval_seconds=report.eval_seconds,
            checkpoint_seconds=report.checkpoint_seconds,
            eval_share_pct=report.eval_share_pct,
            n_val=report.n_val,
            n_val_total=report.n_val_total,
            wiring_check=report.wiring_check,
        )
        return report

    # ----------------------------------------------------------------------------- evaluation
    def _norm_modules(self) -> list[Any]:
        """Every BatchNorm module in the model."""
        torch = require_torch()
        return [
            m for m in self.model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
        ]

    def _recalibrate_norm(self) -> tuple[int, float] | None:
        """Re-estimate BatchNorm running statistics on training data, before a validation pass.

        BatchNorm keeps two things: weights, learned by the optimizer, and running statistics,
        accumulated as an exponential average of the batch statistics seen while training. In
        ``train()`` the layer normalises with the batch's own statistics; in ``eval()`` it uses
        the accumulated ones. When the two disagree, evaluation measures the statistics rather
        than the model, and it does so silently -- the training loss stays clean because training
        never reads the running statistics.

        That is not hypothetical here. On the RTX 5090, ``runs/ocr_r34_tfm_full2`` step 18000
        (2026-09-08): identical weights, identical batch, teacher-forced loss **0.32** with batch
        statistics and **8.52** with the accumulated ones; reported CER **7.34**, and **0.152**
        once the statistics were re-estimated on 4096 samples. The run had been early-stopping and
        selecting its best checkpoint on the 7.34.

        Three details are deliberate:

        * **Training data only.** The estimate is taken on :attr:`train_ds`, never on
          :attr:`val_ds`. Re-estimating on the split about to be scored would tune a part of the
          model on the evaluation data -- a leak, and one that would flatter every number the
          project reports. Measured both ways on the checkpoint above, training data costs
          nothing anyway: CER 0.152 against 0.157.
        * **Only the BatchNorm layers run in training mode.** Dropout stays off; leaving it on
          would inflate the variance estimate with noise the evaluation will not contain.
        * **A cumulative average, not an exponential one.** ``momentum=None`` for the pass, so all
          the samples weigh the same, and the original momentum is restored afterwards so that
          continued training keeps its own schedule.

        The new statistics are **kept**: they go on to be written into the checkpoint, so an
        exported model carries statistics that describe it. Nothing on the gradient path is
        touched.

        Returns:
            ``(samples, seconds)``, or ``None`` when the model has no BatchNorm, when the knob is
            off, or when there is no training split to estimate from.
        """
        modules = self._norm_modules()
        if not modules:
            return None
        n_target = self.cfg.eval.recalibrate_norm_samples
        if n_target is None:
            if not self._norm_warned:
                self._norm_warned = True
                _log.warning(
                    "ml.trainer.norm_statistics_not_recalibrated",
                    n_batchnorm_layers=len(modules),
                    reason=(
                        "the model carries BatchNorm but eval.recalibrate_norm_samples is unset; "
                        "validation may describe the running statistics, not the weights"
                    ),
                )
            return None
        if self.train_ds is None:
            if not self._norm_warned:
                self._norm_warned = True
                _log.warning(
                    "ml.trainer.norm_statistics_not_recalibrated",
                    n_batchnorm_layers=len(modules),
                    reason=(
                        "no training split to estimate from; the split being scored must not be "
                        "used as its own reference"
                    ),
                )
            return None

        torch = require_torch()
        key = INPUT_KEYS[self.cfg.data.mode]
        was_training = bool(self.model.training)
        saved = [(m, m.momentum) for m in modules]
        started = perf_counter()
        self.model.eval()
        for module in modules:
            module.reset_running_stats()
            module.momentum = None
            module.train()
        seen = 0
        try:
            with torch.no_grad():
                for batch in self._loader(self.train_ds, shuffle=True, epoch=self.epoch):
                    self._forward_loss(batch)
                    seen += int(batch[key].shape[0])
                    if seen >= int(n_target):
                        break
        finally:
            for module, momentum in saved:
                module.momentum = momentum
            if was_training:
                self.model.train()
            else:
                self.model.eval()
        seconds = perf_counter() - started
        _log.info(
            "ml.trainer.norm_statistics_recalibrated",
            step=self.step,
            samples=seen,
            requested=int(n_target),
            layers=len(modules),
            seconds=seconds,
            split="train",
        )
        return seen, seconds

    def evaluate(self, *, max_n: int | None = None) -> dict[str, float]:
        """Score a validation subset: teacher-forced loss plus decoded-text metrics.

        Decoding is **batched**: every micro-batch of the validation loader is handed to
        :func:`aicls.ocr.models.seq2seq.greedy_decode_batch` / ``beam_search_batch`` in one call
        (see :meth:`_decode_ids` for the fallbacks), and the pairs are scored with
        :func:`score_predictions`. The sample-at-a-time loop this replaced cost 18.5 s per sample
        on an RTX 5090 against the MathWriting excerpt, which put a 2000-sample validation pass at
        ten hours next to four minutes of training.

        The pass is timed into :attr:`budget` and the truncation is recorded in
        :attr:`coverage`, so both survive into ``metrics.jsonl`` and ``fit_report.json``.

        Args:
            max_n: Cap on decoded samples; defaults to ``eval.max_n_val`` (``None`` = all).

        Returns:
            ``{"loss", "val_loss", "n_val", "n_val_total", "val_truncated", "eval_seconds",
            ...requested metrics}``; empty without a validation set. When the normalisation
            statistics were re-estimated first (:meth:`_recalibrate_norm`) the row also carries
            ``norm_recalibrated_n`` and ``norm_recalibration_seconds``, so a reader can tell which
            kind of statistics produced the number. ``n_val_total`` and
            ``val_truncated`` are omitted when the dataset cannot report its length. Every
            decoded metric is published twice, bare (``cer``) and with the ``val_`` prefix
            (``val_cer``), and ``val_loss`` aliases ``loss``, so that any ``checkpoint.metric``
            spelling the shipped configurations use — ``val_loss``, ``val_cer``, ``val_wer`` —
            selects a best checkpoint. Without the aliases
            :meth:`aicls.ml.checkpoint.CheckpointSaver.is_best` never finds its key and
            best-by-CER selection is silently inert;
            ``tests/integration/test_ml_contracts.py`` pins the contract.

            **The loss is a subset loss too**: it averages the batches the decode consumed, not
            the whole split.
        """
        if self.val_ds is None:
            return {}
        torch = require_torch()
        cfg = self.cfg
        limit = max_n if max_n is not None else cfg.eval.max_n_val
        n_total = split_size(self.val_ds)
        key = INPUT_KEYS[cfg.data.mode]
        recalibrated = self._recalibrate_norm()
        loader = self._loader(self.val_ds, shuffle=False, epoch=0)
        was_training = bool(self.model.training)
        self.model.eval()
        started = perf_counter()
        total_loss = 0.0
        n_batches = 0
        pairs: list[tuple[str, str]] = []
        try:
            with torch.no_grad():
                for batch in loader:
                    total_loss += float(self._forward_loss(batch).item())
                    n_batches += 1
                    references = [str(text) for text in batch["labels_text"]]
                    take = (
                        len(references)
                        if limit is None
                        else min(len(references), int(limit) - len(pairs))
                    )
                    if take <= 0:
                        break
                    inputs = batch[key][:take].to(self.device_spec.device)
                    hypotheses = self._decode_batch(inputs)
                    pairs.extend(zip(references[:take], hypotheses, strict=True))
                    if limit is not None and len(pairs) >= int(limit):
                        break
        finally:
            if was_training:
                self.model.train()
        seconds = perf_counter() - started
        self.coverage = ValCoverage(n_val=len(pairs), n_total=n_total, limit=limit)
        self.budget.add_eval(seconds, n_val=len(pairs))
        loss = total_loss / max(n_batches, 1)
        scores: dict[str, float] = {"loss": loss, VAL_LOSS_KEY: loss, "eval_seconds": seconds}
        if recalibrated is not None:
            # Provenance, not decoration: a reader of metrics.jsonl must be able to tell whether
            # a row was produced against re-estimated normalisation statistics or accumulated
            # ones. The two answered 0.152 and 7.34 for the same weights.
            samples, recal_seconds = recalibrated
            scores["norm_recalibrated_n"] = float(samples)
            scores["norm_recalibration_seconds"] = recal_seconds
        scores.update(self.coverage.as_metrics())
        decoded = score_predictions(pairs, self.metrics)
        scores.update(decoded)
        scores.update({f"{VAL_PREFIX}{name}": value for name, value in decoded.items()})
        _log.info(
            "ml.trainer.evaluated",
            step=self.step,
            n_val=self.coverage.n_val,
            n_val_total=self.coverage.n_total,
            truncated=self.coverage.truncated,
            eval_seconds=seconds,
            seconds_per_val_sample=seconds / max(len(pairs), 1),
            decoder=self.eval_decoder,
            beam=cfg.eval.beam,
        )
        return scores

    def _decode_batch(self, inputs: Any) -> list[str]:
        """Decode a ``(B, ...)`` batch of encoder inputs into LaTeX strings."""
        return [str(self.tokenizer.decode(ids)) for ids in self._decode_ids(inputs)]

    def _decode_ids(self, inputs: Any) -> list[list[int]]:
        """Token ids for every row of ``inputs``, through the fastest decoder available.

        Three paths, and :attr:`eval_decoder` always says which one produced the number:

        1. the batched entry points of :mod:`aicls.ocr.models.seq2seq` (the intended path);
        2. :func:`batched_greedy_ids` — this module's own batching over ``encode``/``decode``,
           for a greedy run against a decoder module that has not grown a batched entry point;
        3. the per-sample :func:`aicls.ocr.models.seq2seq.beam_search`, which is the loop that
           made a validation pass cost hours. It is kept because a wrong CER is worse than a slow
           one, and it is named in the report so a slow run can be explained.

        Raises:
            InputValidationError: If a decoder returns a different number of hypotheses than the
                batch had rows. Silently zipping those onto the references would pair every
                remaining hypothesis with the wrong label and move every metric.
        """
        from aicls.ocr.models.seq2seq import beam_search

        greedy_batch, beam_batch = batch_decoders()
        max_len = self.cfg.model.max_len or self.cfg.data.max_label_len or DEFAULT_MAX_LEN
        beam = int(self.cfg.eval.beam)
        rows = int(inputs.shape[0])
        if beam <= 1:
            if greedy_batch is not None:
                self.eval_decoder = f"seq2seq.{BATCH_GREEDY_ENTRY}"
                decoded = [
                    as_token_ids(item) for item in greedy_batch(self.model, inputs, max_len=max_len)
                ]
            else:
                self.eval_decoder = LOCAL_GREEDY_PATH
                decoded = batched_greedy_ids(self.model, inputs, max_len=max_len)
        elif beam_batch is not None:
            self.eval_decoder = f"seq2seq.{BATCH_BEAM_ENTRY}"
            decoded = [
                as_token_ids(item)
                for item in beam_batch(self.model, inputs, beam=beam, max_len=max_len)
            ]
        else:
            self.eval_decoder = "seq2seq.beam_search (per sample)"
            decoded = [
                as_token_ids(
                    beam_search(self.model, inputs[row : row + 1], beam=beam, max_len=max_len)
                )
                for row in range(rows)
            ]
        if len(decoded) != rows:
            raise InputValidationError(
                messages_tr.MSG_VALIDATION_ERROR,
                details={
                    "reason": "decoder_row_count_mismatch",
                    "rows": rows,
                    "decoded": len(decoded),
                    "decoder": self.eval_decoder,
                },
            )
        return decoded

    # --------------------------------------------------------------------------- loop internals
    def _loader(self, dataset: Any, *, shuffle: bool, epoch: int) -> Any:
        """A data loader for ``dataset`` at ``epoch``."""
        return make_dataloader(
            dataset,
            batch_size=self.cfg.data.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.data.num_workers,
            seed=self.cfg.run.seed,
            epoch=epoch,
            collate=collate_for(self.cfg),
        )

    def _batch_stream(self, batches_per_epoch: int) -> Iterator[Any]:
        """Yield training batches forever, rebuilding a freshly seeded loader every epoch.

        A resumed run skips the batches the interrupted one had already consumed, so the two
        trajectories stay identical.
        """
        consumed = self.step * self.cfg.train.grad_accum
        self.epoch = consumed // batches_per_epoch
        skip = consumed % batches_per_epoch
        if skip:
            _log.info("ml.trainer.resume_skip", epoch=self.epoch, batches=skip)
        while True:
            for batch in self._loader(self.train_ds, shuffle=True, epoch=self.epoch):
                if skip:
                    skip -= 1
                    continue
                yield batch
            self.epoch += 1

    @staticmethod
    def _next_batch(stream: Iterator[Any]) -> tuple[Any, float]:
        """Pull one batch, returning it with the seconds spent waiting for it."""
        started = perf_counter()
        batch = next(stream)
        return batch, perf_counter() - started

    def _forward_loss(self, batch: Mapping[str, Any]) -> Any:
        """Teacher-forced cross entropy over the batch (``<pad>`` positions ignored)."""
        torch = require_torch()
        device = self.device_spec.device
        inputs = batch[INPUT_KEYS[self.cfg.data.mode]].to(device)
        labels = batch["labels"].to(device)
        if labels.shape[1] < 2:
            raise InputValidationError(
                messages_tr.MSG_VALIDATION_ERROR,
                details={"reason": "label_too_short", "length": int(labels.shape[1])},
            )
        with self.device_spec.autocast_context():
            logits = self.model(inputs, labels[:, :-1])
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=PAD_ID,
            )
        return loss

    def _backward(self, loss: Any) -> None:
        """Backward pass, scaled when fp16 autocast is on."""
        if self._scaler is not None:  # pragma: no cover - fp16 needs CUDA
            self._scaler.scale(loss).backward()  # pragma: no cover
        else:
            loss.backward()

    def _optimizer_step(self) -> float | None:
        """Clip, step the optimizer and the schedule; return the pre-clip gradient norm."""
        torch = require_torch()
        if self._scaler is not None:  # pragma: no cover - fp16 needs CUDA
            self._scaler.unscale_(self.optimizer)  # pragma: no cover
        norm: float | None = None
        if self.cfg.train.grad_clip is not None:
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.cfg.train.grad_clip)
                )
            )
        if self._scaler is not None:  # pragma: no cover - fp16 needs CUDA
            self._scaler.step(self.optimizer)  # pragma: no cover
            self._scaler.update()  # pragma: no cover
        else:
            self.optimizer.step()
        self.scheduler.step()
        return norm

    @staticmethod
    def _batch_size(batch: Mapping[str, Any]) -> int:
        """Number of samples in a collated batch."""
        return len(batch["labels_text"])

    def _current_lr(self) -> float:
        """Learning rate of the first parameter group."""
        return float(self.optimizer.param_groups[0]["lr"])

    def _log_train(self, train_loss: float, grad_norm: float | None) -> None:
        """Send one training row to the tracker."""
        row: dict[str, float] = {
            "loss": train_loss,
            "lr": self._current_lr(),
            "epoch": float(self.epoch),
        }
        if grad_norm is not None:
            row["grad_norm"] = grad_norm
        self.tracker.log_metrics(row, self.step, split="train")

    # ------------------------------------------------------------------------------- artefacts
    def _capture_manifest(self) -> RunManifest:
        """Capture and write ``run_manifest.json`` before the first optimizer step.

        The manifest adopts the tracker's run id when the tracker has one. Both objects mint a
        ULID of their own, and letting them keep separate ids would put two different ``run_id``
        values in one run directory — ``run_manifest.json`` saying one thing and
        ``metrics.jsonl`` another — so nothing could join the metrics to the provenance record
        except by directory. ``tests/integration/test_ml_contracts.py`` pins the single id.
        """
        manifest = RunManifest.capture(
            self.cfg,
            allow_dirty=self.cfg.train.allow_dirty,
            repo_root=self.repo_root,
            run_name=self.cfg.run.name or self.run_dir.name,
        )
        tracker_run_id = getattr(self.tracker, "run_id", None)
        if isinstance(tracker_run_id, str) and tracker_run_id:
            manifest.run_id = tracker_run_id
        manifest.save(self.run_dir)
        self.manifest = manifest
        self.saver.manifest = manifest
        return manifest

    def _save(self, metrics: Mapping[str, float]) -> None:
        """Write a step checkpoint through the retention policy."""
        self.saver.save(
            self.step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            metrics=dict(metrics),
            manifest=self.manifest,
            vocab=self.vocab,
        )

    def _save_interrupt(self, metrics: Mapping[str, float]) -> None:
        """Write the ``interrupt/`` checkpoint (best effort: never mask the Ctrl-C)."""
        try:
            save_checkpoint(
                self.interrupt_dir,
                model=self.model,
                step=self.step,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                cfg=self.cfg,
                manifest=self.manifest,
                metrics=dict(metrics),
                vocab=self.vocab,
            )
        except Exception as exc:  # pragma: no cover - disk failure during Ctrl-C
            _log.error("ml.trainer.interrupt_checkpoint_failed", error=str(exc))

    def _resolve_total_steps(self, max_steps: int | None, batches_per_epoch: int) -> int:
        """The absolute step budget: the smallest of the argument, ``max_steps`` and ``epochs``."""
        cfg = self.cfg
        per_epoch = max(1, math.ceil(batches_per_epoch / cfg.train.grad_accum))
        candidates: list[int] = []
        if max_steps is not None:
            candidates.append(int(max_steps))
        if cfg.train.max_steps is not None:
            candidates.append(int(cfg.train.max_steps))
        if cfg.train.epochs is not None:
            candidates.append(int(cfg.train.epochs) * per_epoch)
        if not candidates:
            candidates.append(per_epoch)
            _log.info("ml.trainer.default_budget", steps=per_epoch, reason="one epoch")
        total = min(candidates)
        if total < 1:
            raise InputValidationError(
                messages_tr.MSG_VALIDATION_ERROR,
                details={"reason": "empty_step_budget", "total": total},
            )
        return total

    def _build_report(
        self,
        *,
        manifest: RunManifest,
        started: float,
        train_loss: float | None,
        metrics: Mapping[str, float],
        early_stopped: bool,
        interrupted: bool,
        batches_per_epoch: int,
    ) -> FitReport:
        """Assemble the :class:`FitReport` and record the measured throughput in the manifest."""
        steps_per_s = self.stats.steps_per_s
        if steps_per_s is not None:
            manifest.set_steps_per_s(steps_per_s, run_dir=self.run_dir)
        epochs = (self.step * self.cfg.train.grad_accum) / max(batches_per_epoch, 1)
        best_dir = self.saver.best_dir
        last_dir = self.saver.last_dir
        budget = self.budget
        budget.adopt_train(self.stats)
        budget.wall_time_s = perf_counter() - started
        coverage = self.coverage if budget.evaluations else None
        return FitReport(
            run_id=manifest.run_id,
            run_name=manifest.run_name,
            steps=self.step,
            epochs=epochs,
            samples_seen=self.samples_seen,
            train_loss=train_loss,
            metrics={str(k): float(v) for k, v in metrics.items()},
            best_metric=self.cfg.checkpoint.metric,
            best_value=self.saver.best_value,
            best_step=self.saver.best_step,
            best_checkpoint=str(best_dir) if best_dir is not None else None,
            last_checkpoint=str(last_dir) if last_dir is not None else None,
            early_stopped=early_stopped,
            interrupted=interrupted,
            device=self.device_spec.device,
            dtype=self.device_spec.dtype,
            amp=self.device_spec.amp,
            wall_time_s=budget.wall_time_s,
            steps_per_s=steps_per_s,
            dataloader_wait_pct=self.stats.dataloader_wait_pct,
            train_seconds=budget.train_seconds,
            eval_seconds=budget.eval_seconds,
            checkpoint_seconds=budget.checkpoint_seconds,
            dataloader_wait_s=budget.dataloader_wait_s,
            other_seconds=budget.other_seconds,
            evaluations=budget.evaluations,
            seconds_per_eval=budget.seconds_per_eval,
            seconds_per_val_sample=budget.seconds_per_val_sample,
            eval_share_pct=budget.eval_share_pct,
            n_val=coverage.n_val if coverage is not None else None,
            n_val_total=coverage.n_total if coverage is not None else None,
            val_truncated=bool(coverage.truncated) if coverage is not None else False,
            eval_decoder=self.eval_decoder,
            run_dir=str(self.run_dir),
            config_hash=manifest.config_hash,
            git_sha=manifest.git_sha,
            citable=manifest.citable,
            wiring_check=self.cfg.is_gate_free,
            notes=WIRING_CHECK_NOTE if self.cfg.is_gate_free else "",
        )


# ----------------------------------------------------------------------------------------------
# optimizer helpers
# ----------------------------------------------------------------------------------------------
def apply_freeze(model: Any, cfg: TrainConfig) -> int:
    """Freeze the encoder when ``model.freeze_encoder`` is set.

    Args:
        model: The module being trained.
        cfg: The run configuration.

    Returns:
        The number of parameters that were frozen (0 when the flag is off).
    """
    if not cfg.model.freeze_encoder:
        return 0
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "no_encoder_to_freeze", "family": cfg.model.family},
        )
    frozen = 0
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
        frozen += int(parameter.numel())
    _log.info("ml.trainer.encoder_frozen", parameters=frozen, family=cfg.model.family)
    return frozen


def build_optimizer(model: Any, cfg: TrainConfig) -> Any:
    """Build the optimizer described by ``cfg.optim`` over the trainable parameters.

    Args:
        model: The module being trained.
        cfg: The run configuration.

    Returns:
        A ``torch.optim.Optimizer``.

    Raises:
        InputValidationError: If the model has no trainable parameter left.
    """
    torch = require_torch()
    optim = cfg.optim
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"reason": "no_trainable_parameters", "family": cfg.model.family},
        )
    if optim.name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=optim.lr,
            momentum=optim.momentum,
            weight_decay=optim.weight_decay,
        )
    factory = torch.optim.Adam if optim.name == "adam" else torch.optim.AdamW
    return factory(
        parameters,
        lr=optim.lr,
        betas=(float(optim.betas[0]), float(optim.betas[1])),
        weight_decay=optim.weight_decay,
    )


def _default_run_dir(cfg: TrainConfig) -> Path:
    """``AICLS_DATA_ROOT/runs/<run.name>-<ulid>`` — unique so two runs never share a JSONL file."""
    from aicls.core.ids import ulid
    from aicls.core.paths import runs_root

    name = cfg.run.name or "run"
    return runs_root() / f"{name}-{ulid()}"


__all__ = [
    "AMP_DTYPES",
    "BATCH_BEAM_ENTRY",
    "BATCH_GREEDY_ENTRY",
    "CHECKPOINTS_DIRNAME",
    "COVERAGE_KEYS",
    "FAMILY_MODES",
    "FIT_REPORT_FILENAME",
    "INPUT_KEYS",
    "INTERRUPT_DIRNAME",
    "LOCAL_GREEDY_PATH",
    "TRAINER_FAMILIES",
    "TRAINING_GATE",
    "VAL_LOSS_KEY",
    "VAL_PREFIX",
    "WIRING_CHECK_NOTE",
    "DeviceProbe",
    "DeviceSpec",
    "EarlyStopping",
    "FitReport",
    "StepTiming",
    "SubsetView",
    "TimeBudget",
    "TrainStats",
    "Trainer",
    "ValCoverage",
    "apply_freeze",
    "as_token_ids",
    "batch_decoders",
    "batched_greedy_ids",
    "build_optimizer",
    "collate_for",
    "limit_val_dataset",
    "loader_generator",
    "make_dataloader",
    "probe_device",
    "require_training_gate",
    "schedule_factor",
    "score_predictions",
    "seed_worker",
    "select_device",
    "split_size",
]
