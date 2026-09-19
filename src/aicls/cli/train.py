"""``aicls train`` and ``aicls profile`` — the two gated commands (ADR-0012, ADR-0006).

The gate is the point of this module. Both commands call
:func:`aicls.ml.trainer.require_training_gate`, which refuses unless **either**

* the run is on synthetic data — ``--synthetic``, or a config whose ``data.dataset`` already
  starts with ``synthetic_`` (that is how CI and a developer exercise the loop), **or**
* all three conditions of the triple lock hold: ``--i-have-approval`` **and**
  ``AICLS_GATE_TRAINING=approved`` **and** an ``approved`` row in
  ``docs/gelistirme/onay-defteri.md``. Anything else exits ``3``.

``--synthetic`` does **not** wave a real run past the gate: it rewrites ``data.dataset`` to the
synthetic sibling of the model family, so the run that follows genuinely uses generated data and
is gate-free for that reason and no other. The decision is always
:attr:`aicls.ml.config.TrainConfig.is_gate_free`, which reads ``data.dataset`` alone — a config
cannot declare its way past the gate.

Honesty: every number a synthetic run produces is a **wiring check**, not a result. The reports
carry ``wiring_check=true`` and the CLI repeats it on stdout, because a loss curve from a
randomly initialised network on generated scribbles says nothing about the İP-3 CER target.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from aicls.cli.common import cli_logger, echo, emit, use_json

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps torch out of the import path
    from aicls.core.settings import GateName
    from aicls.ml.config import TrainConfig

TRAIN_HELP = "Train a model from a YAML config (TRAINING gate; --synthetic is gate-free)."
PROFILE_HELP = "Measure training throughput over N steps (TRAINING gate; synthetic is gate-free)."

#: Synthetic dataset each model family falls back to under ``--synthetic``.
SYNTHETIC_DATASETS: dict[str, str] = {
    "seq2seq_image": "synthetic_ink",
    "seq2seq_stroke": "synthetic_ink",
    "whisper": "synthetic_audio",
    "wav2vec2": "synthetic_audio",
}

#: Families the OCR :class:`aicls.ml.trainer.Trainer` drives.
TRAINER_FAMILIES = ("seq2seq_image", "seq2seq_stroke")

#: The approval gate every non-synthetic run of this command sits behind (ADR-0012).
TRAINING_GATE: GateName = "TRAINING"

#: The one sentence that must accompany every synthetic figure this CLI prints.
WIRING_NOTE = (
    "WIRING CHECK: synthetic data and (usually) randomly initialised weights. "
    "No number from this run is evidence for a contractual metric."
)


def resolve_config(config: Path, overrides: list[str] | None, *, synthetic: bool) -> TrainConfig:
    """Load ``config``, apply ``--override`` pairs and, for ``--synthetic``, the synthetic dataset.

    Args:
        config: Path of the YAML configuration.
        overrides: ``key=value`` strings, applied in order.
        synthetic: Whether ``--synthetic`` was given. A config that is already gate-free is left
            alone; otherwise ``data.dataset`` is rewritten to the family's synthetic sibling and
            the file is re-validated, so the run really does use generated data.

    Returns:
        The validated configuration.

    Raises:
        aicls.core.errors.InputValidationError: If the file, an override or the resulting
            configuration is invalid, or the family has no synthetic sibling.
    """
    from aicls.core.errors import InputValidationError
    from aicls.core.messages_tr import MSG_VALIDATION_ERROR
    from aicls.ml.config import load_config

    given = list(overrides or [])
    cfg = load_config(config, given)
    if not synthetic or cfg.is_gate_free:
        return cfg
    dataset = SYNTHETIC_DATASETS.get(cfg.model.family)
    if dataset is None:  # pragma: no cover - every declared family has a sibling
        raise InputValidationError(
            MSG_VALIDATION_ERROR,
            details={"reason": "no_synthetic_dataset_for_family", "family": cfg.model.family},
        )
    cli_logger(__name__).info(
        "cli.train.synthetic_substitution", family=cfg.model.family, dataset=dataset
    )
    return load_config(config, [*given, f"data.dataset={dataset}"])


def plan_payload(cfg: TrainConfig, *, gate_checked: bool, run_dir: Path | None) -> dict[str, Any]:
    """The resolved plan of a run: what would be trained, on what, and how the gate decided."""
    from aicls.ml.config import config_hash

    return {
        "run_name": cfg.run.name,
        "config_hash": config_hash(cfg),
        "dataset": cfg.data.dataset,
        "model_family": cfg.model.family,
        "encoder": cfg.model.encoder,
        "batch_size": cfg.data.batch_size,
        "max_steps": cfg.train.max_steps,
        "epochs": cfg.train.epochs,
        "amp": cfg.train.amp,
        "gate": "TRAINING",
        "gate_free": cfg.is_gate_free,
        "gate_checked": gate_checked,
        "wiring_check": cfg.is_gate_free,
        "run_dir": run_dir.as_posix() if run_dir is not None else None,
        "notes": WIRING_NOTE if cfg.is_gate_free else "",
    }


def train(
    *,
    config: Annotated[Path, typer.Option("--config", "-c", help="Training YAML configuration.")],
    override: Annotated[
        list[str] | None,
        typer.Option(
            "--override",
            "--set",
            "-o",
            help="Config override 'key=value'; repeatable. --set is the same flag.",
        ),
    ] = None,
    synthetic: Annotated[
        bool,
        typer.Option("--synthetic", help="Run on generated data; swaps in a synthetic dataset."),
    ] = False,
    allow_dirty: Annotated[
        bool,
        typer.Option(
            "--allow-dirty",
            help="Run with a dirty working tree; the manifest is then citable=false (ADR-0007).",
        ),
    ] = False,
    i_have_approval: Annotated[
        bool,
        typer.Option("--i-have-approval", help="One of the three TRAINING gate conditions."),
    ] = False,
    run_dir: Annotated[
        Path | None,
        typer.Option("--run-dir", help="Run directory; AICLS_DATA_ROOT/runs/<name>-<ulid>."),
    ] = None,
    resume: Annotated[
        Path | None,
        typer.Option("--resume", help="Checkpoint directory to continue from."),
    ] = None,
    queue: Annotated[
        bool,
        typer.Option(
            "--queue/--no-queue",
            help="Take the workstation's single-job GPU lock (default: for gated runs).",
        ),
    ] = True,
    force_queue: Annotated[
        bool,
        typer.Option(
            "--force-queue",
            help="Seize a lock left by a job that died; refused while the GPU is in use.",
        ),
    ] = False,
    max_steps: Annotated[
        int,
        typer.Option(
            "--max-steps",
            help="Shorten the run to this many steps (0=off). It can only lower the budget: "
            "Trainer takes the smallest of this, train.max_steps and train.epochs, so a flag "
            "cannot train past what the configuration - and the approval that cites it - says.",
        ),
    ] = 0,
    device: Annotated[
        str, typer.Option("--device", help="Force 'cpu' or 'cuda'; auto-detected when empty.")
    ] = "",
    validate: Annotated[
        bool, typer.Option("--val/--no-val", help="Build and score a validation split.")
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Resolve the config, report the gate status, then stop."),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Train a model.

    Raises:
        aicls.core.gates.GateError: If the run needs approval and any condition is missing
            (exit code 3). Nothing is built and no run directory is created in that case.
        aicls.core.errors.InputValidationError: If the model family has no trainer here.
    """
    from aicls.core.gates import gate_status
    from aicls.ml.trainer import require_training_gate

    use_json(json_output)
    # Same mechanism the script uses: the flag is a configuration override, so one code path
    # decides what a dirty tree means and the manifest reaches its own verdict from it.
    extra = [*(override or []), "train.allow_dirty=true"] if allow_dirty else override
    cfg = resolve_config(config, extra, synthetic=synthetic)
    if dry_run:
        # A dry run never asks for the gate: the plan it prints is what an operator attaches to
        # the approval PR (docs/gelistirme/onay-defteri.md "Satır eklemeden önce"), so demanding
        # the approval first would make the procedure circular. It reports the lock's current
        # state instead, exactly like ``training/scripts/train.py --dry-run``. Nothing is built,
        # no directory is created and no invocation is logged.
        missing = [] if cfg.is_gate_free else gate_status(TRAINING_GATE, cli_flag=i_have_approval)
        plan = plan_payload(cfg, gate_checked=False, run_dir=run_dir)
        payload = {**plan, "dry_run": True, "missing_conditions": missing}
        emit(payload, *_plan_lines(plan, missing=missing), "dry run: nothing was trained")
        return
    # First thing for real, before a model, a dataset or a directory exists.
    gate_checked = require_training_gate(cfg, i_have_approval=i_have_approval)
    plan = plan_payload(cfg, gate_checked=gate_checked, run_dir=run_dir)
    echo(f"training {plan['run_name']} on {plan['dataset']} ({plan['model_family']})")
    if cfg.is_gate_free:
        echo(WIRING_NOTE)

    budget = int(max_steps) or None
    with _queue_for(cfg, use_queue=queue, force=force_queue):
        return _dispatch(
            cfg,
            plan=plan,
            budget=budget,
            device=device,
            run_dir=run_dir,
            resume=resume,
            i_have_approval=i_have_approval,
            validate=validate,
        )


def _queue_for(cfg: TrainConfig, *, use_queue: bool, force: bool) -> Any:
    """The workstation's single-job GPU lock, for the runs that need it.

    ``aicls train`` did not take this lock until 2026-09-09 while ``training/scripts/train.py``
    always did, so the protection against two trainings on one GPU could be walked straight past
    by using the other door. Gate-free synthetic runs still skip it, matching the script: probing
    nvidia-smi and touching training/queue.lock from a two-step CPU smoke run is noise, not
    safety.
    """
    if not use_queue or cfg.is_gate_free:
        return nullcontext()
    from aicls.ml.queue import TrainingQueue

    lock = TrainingQueue(job=cfg.run.name)
    if force:
        lock.take_over(reason="aicls train --force-queue")
    return lock


def _dispatch(
    cfg: TrainConfig,
    *,
    plan: dict[str, Any],
    budget: int | None,
    device: str,
    run_dir: Path | None,
    resume: Path | None,
    i_have_approval: bool,
    validate: bool,
) -> None:
    """Run the loop the model family names, then emit the report."""
    if cfg.model.family in TRAINER_FAMILIES:
        report = _run_trainer(
            cfg,
            run_dir=run_dir,
            max_steps=budget,
            device=device or None,
            i_have_approval=i_have_approval,
            validate=validate,
            resume=resume,
        )
    else:
        if resume is not None:
            # The Whisper path fine-tunes through Hugging Face, which owns its own resume; there
            # is no Trainer checkpoint to hand it. Saying so beats accepting the flag and
            # continuing from the beginning, which is what silently ignoring it would do.
            from aicls.core.errors import InputValidationError
            from aicls.core.messages_tr import MSG_VALIDATION_ERROR

            raise InputValidationError(
                MSG_VALIDATION_ERROR,
                details={"reason": "resume_unsupported_for_family", "family": cfg.model.family},
            )
        report = _run_whisper(
            cfg,
            run_dir=run_dir,
            device=device or None,
            i_have_approval=i_have_approval,
            validate=validate,
        )
    payload = {"plan": plan, "report": report}
    emit(payload, *_report_lines(report))


def _plan_lines(plan: dict[str, Any], *, missing: list[str] | None = None) -> list[str]:
    """Human-readable rendering of :func:`plan_payload`.

    Args:
        plan: The payload from :func:`plan_payload`.
        missing: Missing gate conditions of a dry run (``None`` for a run that passed the gate).
    """
    if plan["gate_free"]:
        gate = "free (synthetic)"
    elif missing is None:
        gate = "checked and approved"
    else:
        gate = f"closed, missing: {', '.join(missing)}" if missing else "would open"
    lines = [
        f"run {plan['run_name']} ({plan['model_family']} on {plan['dataset']})",
        f"config_hash {plan['config_hash']}",
        f"gate TRAINING: {gate}",
    ]
    if plan["notes"]:
        lines.append(plan["notes"])
    return lines


def _report_lines(report: dict[str, Any]) -> list[str]:
    """Human-readable rendering of a fit report.

    The two loops name the step count differently — :class:`aicls.ml.trainer.FitReport` calls it
    ``steps`` and :class:`aicls.ml.whisper_ft.FitReport` ``global_step`` — so the ASR path used to
    print ``steps None`` after a run that had in fact taken its steps.
    """
    steps = report.get("steps", report.get("global_step"))
    lines = [
        f"steps {steps} loss {report.get('train_loss')}",
        f"metrics {report.get('metrics')}",
        f"run_dir {report.get('run_dir')}",
    ]
    if report.get("wiring_check"):
        lines.append(WIRING_NOTE)
    return lines


def _run_trainer(
    cfg: TrainConfig,
    *,
    run_dir: Path | None,
    max_steps: int | None,
    device: str | None,
    i_have_approval: bool,
    validate: bool,
    resume: Path | None = None,
) -> dict[str, Any]:
    """Drive :class:`aicls.ml.trainer.Trainer` for the two OCR families.

    ``resume`` continues from a checkpoint directory, restoring weights, optimizer, schedule and
    RNG state. Until 2026-09-09 only ``training/scripts/train.py`` could do this, so continuing
    an interrupted run meant leaving the CLI for the script -- and a nine-hour run interrupted at
    step 18000 is exactly when nobody wants to discover that.
    """
    from aicls.cli.common import jsonable
    from aicls.ml.registry import DatasetSpec, build_dataset, build_model
    from aicls.ml.trainer import Trainer

    tokenizer = DatasetSpec.from_config(cfg).tokenizer()
    model = build_model(cfg, tokenizer.vocab_size)
    train_ds = build_dataset(cfg, "train")
    val_ds = build_dataset(cfg, "valid") if validate else None
    trainer = Trainer(
        cfg,
        model,
        train_ds,
        val_ds,
        run_dir=run_dir,
        tokenizer=tokenizer,
        i_have_approval=i_have_approval,
        device=device,
    )
    resumed_from: str | None = None
    if resume is not None:
        checkpoint = trainer.resume(Path(resume))
        resumed_from = Path(resume).as_posix()
        echo(f"resumed from {resumed_from} at step {checkpoint.step}")
    report = trainer.fit(max_steps=max_steps)
    payload: dict[str, Any] = dict(jsonable(report))
    payload["run_dir"] = trainer.run_dir.as_posix()
    if resumed_from is not None:
        payload["resumed_from"] = resumed_from
    return payload


def _run_whisper(
    cfg: TrainConfig,
    *,
    run_dir: Path | None,
    device: str | None = None,
    i_have_approval: bool,
    validate: bool,
) -> dict[str, Any]:
    """Drive :class:`aicls.ml.whisper_ft.WhisperFineTuner` for the ASR family.

    Args:
        cfg: The resolved run configuration.
        run_dir: Run directory; ``<AICLS_DATA_ROOT>/runs/<name>-<ulid>`` when omitted.
        device: What the operator asked for with ``--device``. Hugging Face's ``Trainer``
            selects the device itself, so this route cannot honour the request; it is accepted
            and **logged as ignored** rather than silently dropped, which is what made a
            ``--device cpu`` ASR run execute on the GPU without saying so (measured 2026-09-08).
        i_have_approval: One of the three ``TRAINING`` gate conditions.
        validate: Whether to build the validation split.

    Raises:
        aicls.core.errors.InputValidationError: For ``wav2vec2``, which has no fine-tuning
            wrapper yet — refusing is better than pretending the family is supported.
    """
    from aicls.cli.common import jsonable
    from aicls.core.errors import InputValidationError
    from aicls.core.logging import get_logger
    from aicls.core.messages_tr import MSG_VALIDATION_ERROR
    from aicls.ml.registry import build_dataset
    from aicls.ml.tracking import JsonlTracker
    from aicls.ml.trainer import _default_run_dir
    from aicls.ml.whisper_ft import WhisperDatasetAdapter, WhisperFineTuner

    if cfg.model.family != "whisper":
        raise InputValidationError(
            MSG_VALIDATION_ERROR,
            details={
                "reason": "family_not_supported_by_cli",
                "family": cfg.model.family,
                "supported": [*TRAINER_FAMILIES, "whisper"],
            },
        )
    # `<name>-<ulid>`, the same rule the OCR loop uses — `_default_run_dir` is private to the ml
    # package but shared here on purpose, because a second copy of the rule would drift. Without
    # the ULID a second ASR run overwrites the first's `run_manifest.json`: provenance lost.
    target = Path(run_dir) if run_dir is not None else _default_run_dir(cfg)
    if device:
        get_logger(__name__).warning(
            "cli.train.device_ignored",
            requested=device,
            family=cfg.model.family,
            reason="transformers.Trainer selects the device itself; --device is not wired here",
        )
    # The same tracker the OCR loop gets. Without it an ASR run directory carried only
    # `run_manifest.json` and `env.json`, so `metrics.jsonl` and `params.json` -- the ADR-0007
    # tracking record every report tool reads -- existed for one of the two loops only.
    tuner = WhisperFineTuner(
        cfg,
        run_dir=target,
        tracker=JsonlTracker(target),
        allow_dirty=cfg.train.allow_dirty,
    )
    processor = tuner.processor
    # `data.max_label_len` is a hard limit, not a hint: a label longer than the decoder's
    # `max_target_positions` indexes past its positional embedding and the run dies on the first
    # batch. The trainer loop applies it through the dataset; this path has to apply it here.
    cap = cfg.data.max_label_len
    train_ds = WhisperDatasetAdapter(build_dataset(cfg, "train"), processor, max_label_len=cap)
    eval_ds = (
        WhisperDatasetAdapter(build_dataset(cfg, "valid"), processor, max_label_len=cap)
        if validate
        else None
    )
    report = tuner.fit(
        train_dataset=train_ds, eval_dataset=eval_ds, i_have_approval=i_have_approval
    )
    payload: dict[str, Any] = dict(jsonable(report))
    payload.setdefault("run_dir", target.as_posix())
    payload.setdefault("wiring_check", cfg.is_gate_free)
    return payload


def profile(
    *,
    config: Annotated[Path, typer.Option("--config", "-c", help="Training YAML configuration.")],
    steps: Annotated[int, typer.Option("--steps", help="Optimizer steps to time.")] = 200,
    warmup: Annotated[int, typer.Option("--warmup", help="Leading steps excluded.")] = 5,
    override: Annotated[
        list[str] | None,
        typer.Option("--override", "-o", help="Config override 'key=value'; repeatable."),
    ] = None,
    synthetic: Annotated[
        bool, typer.Option("--synthetic", help="Profile on generated data (gate-free).")
    ] = False,
    i_have_approval: Annotated[
        bool, typer.Option("--i-have-approval", help="One of the three TRAINING gate conditions.")
    ] = False,
    run_dir: Annotated[
        Path | None, typer.Option("--run-dir", help="Run directory of the profiling run.")
    ] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="Report directory; reports/metrics by default.")
    ] = None,
    device: Annotated[
        str, typer.Option("--device", help="Force 'cpu' or 'cuda'; auto-detected when empty.")
    ] = "",
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Measure steps/s, samples/s and peak VRAM for a configuration.

    ``peak_vram_mb`` is ``null`` on CPU — never a fake ``0.0`` — and the epoch estimate that
    follows from ``steps_per_s`` describes the machine the profile ran on and no other.
    """
    from aicls.cli.common import jsonable
    from aicls.ml.profile import run_profile

    use_json(json_output)
    cfg = resolve_config(config, override, synthetic=synthetic)
    report = run_profile(
        cfg,
        steps=steps,
        warmup_steps=warmup,
        run_dir=run_dir,
        out_dir=out,
        device=device or None,
        i_have_approval=i_have_approval,
    )
    payload: dict[str, Any] = dict(jsonable(report))
    lines = [
        f"{report.name}: {report.steps_per_s:.3f} steps/s, {report.samples_per_s:.1f} samples/s",
        f"device {report.device} ({report.device_name}) dtype {report.dtype} amp {report.amp}",
        f"peak VRAM {report.peak_vram_mb if report.peak_vram_mb is not None else 'n/a (CPU)'}",
    ]
    if report.wiring_check:
        lines.append(WIRING_NOTE)
    emit(payload, *lines)


__all__ = [
    "PROFILE_HELP",
    "SYNTHETIC_DATASETS",
    "TRAINER_FAMILIES",
    "TRAINING_GATE",
    "TRAIN_HELP",
    "WIRING_NOTE",
    "plan_payload",
    "profile",
    "resolve_config",
    "train",
]
