"""Run a training job — **GATED: TRAINING**, except on synthetic data (ADR-0012, §9).

The gate rule, and why it is shaped this way:

* the run is gate-free **iff the resolved configuration's dataset is synthetic**
  (:attr:`aicls.ml.config.TrainConfig.is_gate_free`, which looks only at ``data.dataset``).
  ``--synthetic`` therefore does not *declare* a run gate-free — it **rewrites the configuration**
  to the synthetic sibling dataset and lets the same property decide. A flag can never talk a
  MathWriting run past the gate;
* every other run needs all three conditions together (``--i-have-approval`` **and**
  ``AICLS_GATE_TRAINING=approved`` **and** an ``approved`` row in
  ``docs/gelistirme/onay-defteri.md``). A refusal exits **3** and names the missing conditions in
  Turkish. CI always refuses (:func:`aicls.core.gates.is_ci_environment`);
* the gate is checked **before torch is imported**, before a dataset is opened and before the run
  directory is created: a refused run leaves nothing behind. :class:`aicls.ml.trainer.Trainer`
  checks it a second time, so a caller that skips this script is refused too.

``--dry-run`` prints the resolved plan (config hash, dataset, model, budget, run directory, gate
status, device) and exits 0 without asking for the gate and without writing anything.

Usage (from the repository root, project venv)::

    python training/scripts/train.py --synthetic                       # gate-free CPU smoke run
    python training/scripts/train.py --config training/configs/ocr/profile.yaml --dry-run
    python training/scripts/train.py --config <cfg> --set optim.lr=1e-4 --i-have-approval

**Honesty.** Any run whose dataset is ``synthetic_*`` produces a *wiring check*, never evidence:
random weights on generated fixtures, with the validation set generated from the same seed as the
training set. CER/WER evidence needs the DATASET_DOWNLOAD, WEIGHTS_DOWNLOAD, CUDA_INSTALL and
TRAINING gates open, and real data behind them.

Exit codes: ``0`` success or dry run, ``1`` a typed AI-CLS error, ``3`` the approval gate refused.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:  # plain checkout without an editable install
    sys.path.insert(0, str(_SRC))

from aicls.core.errors import AiclsError, InputValidationError, InternalError  # noqa: E402
from aicls.core.gates import (  # noqa: E402
    CLI_FLAG,
    GateError,
    default_ledger_path,
    is_ci_environment,
    is_gate_approved_in_ledger,
    read_gate_ledger,
)
from aicls.core.ids import ulid  # noqa: E402
from aicls.core.logging import get_logger  # noqa: E402
from aicls.core.messages_tr import msg  # noqa: E402
from aicls.core.settings import GateName, get_settings  # noqa: E402
from aicls.ml.config import TrainConfig, config_hash, load_config  # noqa: E402
from aicls.ml.trainer import WIRING_CHECK_NOTE, require_training_gate  # noqa: E402

_log = get_logger(__name__)

#: Approval gate this script sits behind (skipped only for a synthetic dataset).
GATE: GateName = "TRAINING"

#: Shipped gate-free configurations, per component.
SMOKE_CONFIGS: dict[str, str] = {
    "ocr": "training/configs/ocr/smoke_synthetic.yaml",
    "asr": "training/configs/asr/smoke_synthetic.yaml",
}

#: Synthetic sibling dataset of every model family (what ``--synthetic`` rewrites the config to).
SYNTHETIC_DATASETS: dict[str, str] = {
    "seq2seq_image": "synthetic_ink",
    "seq2seq_stroke": "synthetic_ink",
    "whisper": "synthetic_audio",
    "wav2vec2": "synthetic_audio",
}

#: Families fine-tuned through :class:`aicls.ml.whisper_ft.WhisperFineTuner`.
ASR_FAMILIES: frozenset[str] = frozenset({"whisper", "wav2vec2"})


@dataclass(frozen=True, slots=True)
class TrainPlan:
    """The resolved run, printed before anything happens.

    Attributes:
        cfg: The resolved configuration.
        config_path: File it came from.
        run_dir: Where the run would write (``None`` lets the trainer pick).
        gate_free: Whether the dataset makes this run gate-free.
        missing: Missing gate conditions (empty when the gate is open or not needed).
        use_queue: Whether the single-job lock would be taken.
        notes: Human-readable notes about rewrites this script applied.
    """

    cfg: TrainConfig
    config_path: Path
    run_dir: Path | None
    gate_free: bool
    missing: tuple[str, ...]
    use_queue: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON document of the plan (``--json``)."""
        cfg = self.cfg
        return {
            "action": "train",
            "config": str(self.config_path),
            "config_hash": config_hash(cfg),
            "run_name": cfg.run.name,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "dataset": cfg.data.dataset,
            "mode": cfg.data.mode,
            "family": cfg.model.family,
            "encoder": cfg.model.encoder,
            "batch_size": cfg.data.batch_size,
            "grad_accum": cfg.train.grad_accum,
            "max_steps": cfg.train.max_steps,
            "epochs": cfg.train.epochs,
            "amp": cfg.train.amp,
            "gate": GATE,
            "gate_required": not self.gate_free,
            "missing_conditions": list(self.missing),
            "queue": self.use_queue,
            "wiring_check": self.gate_free,
            "notes": list(self.notes),
        }

    def render(self) -> str:
        """Human-readable plan."""
        cfg = self.cfg
        budget = (
            f"max_steps {cfg.train.max_steps}"
            if cfg.train.max_steps is not None
            else f"epochs {cfg.train.epochs}"
        )
        gate_line = (
            f"NOT REQUIRED — dataset '{cfg.data.dataset}' is synthetic (ADR-0012)"
            if self.gate_free
            else ("open" if not self.missing else f"closed — missing: {', '.join(self.missing)}")
        )
        lines = [
            f"plan: train ({self.config_path})",
            f"  run name     : {cfg.run.name}   seed {cfg.run.seed}",
            f"  config hash  : {config_hash(cfg)}",
            f"  dataset      : {cfg.data.dataset} (mode {cfg.data.mode}, split {cfg.data.split})",
            f"  model        : {cfg.model.family} / {cfg.model.encoder or '-'} / "
            f"{cfg.model.decoder}  d_model {cfg.model.d_model} layers {cfg.model.layers} "
            f"heads {cfg.model.heads}",
            f"  budget       : {budget}, batch {cfg.data.batch_size} x "
            f"grad_accum {cfg.train.grad_accum}, amp {cfg.train.amp}",
            f"  run dir      : {self.run_dir or '(AICLS_DATA_ROOT/runs/<name>-<ulid>)'}",
            f"  tracking     : {cfg.tracking.backend} (experiment {cfg.tracking.experiment})",
            f"  checkpoint   : best by {cfg.checkpoint.metric} ({cfg.checkpoint.mode})",
            f"  queue lock   : {'yes' if self.use_queue else 'skipped'}",
            f"  gate         : {GATE} — {gate_line}",
        ]
        lines.extend(f"  note         : {note}" for note in self.notes)
        if self.gate_free:
            lines.append(f"  HONESTY      : {WIRING_CHECK_NOTE}")
        return "\n".join(lines)


def gate_status(name: str, *, cli_flag: bool, ledger_path: Path | None = None) -> list[str]:
    """Read-only preview of the triple lock: which conditions are currently missing.

    Reports only; it never approves anything and never writes the invocation log.

    Args:
        name: Gate name.
        cli_flag: Whether ``--i-have-approval`` was given.
        ledger_path: Ledger override (tests); the repository ledger by default.

    Returns:
        The missing conditions, in the order :func:`aicls.core.gates.require_gate` lists them.
    """
    missing: list[str] = []
    if is_ci_environment():
        missing.append("ci")
    if not cli_flag:
        missing.append("cli_flag")
    try:
        approved_env = get_settings().is_gate_env_approved(name)
    except (AiclsError, ValueError):  # pragma: no cover - a broken .env
        approved_env = False
    if not approved_env:
        missing.append("env")
    ledger = ledger_path or default_ledger_path()
    try:
        entries = read_gate_ledger(ledger) if ledger.is_file() else []
    except AiclsError:  # a malformed ledger is a closed gate, never an open one
        entries = []
    if not is_gate_approved_in_ledger(name, entries):
        missing.append("ledger")
    return missing


def resolve_config(args: argparse.Namespace) -> tuple[TrainConfig, Path, list[str]]:
    """Load the configuration, applying ``--set`` overrides and the ``--synthetic`` rewrite.

    ``--synthetic`` is implemented as *configuration rewriting*, not as a gate exemption: it
    replaces ``data.dataset`` with the synthetic sibling of the model family, so the run is
    gate-free because of what it actually trains on.

    Args:
        args: Parsed command line.

    Returns:
        ``(cfg, config_path, notes)`` where ``notes`` describes every rewrite that was applied.

    Raises:
        InputValidationError: If no config was given and ``--synthetic`` was not used either, or
            if the file, an override or a value is invalid.
        InternalError: If ``--synthetic`` somehow failed to produce a gate-free config (a bug
            here would be a safety hole, so it is checked rather than assumed).
    """
    overrides = list(args.overrides)
    if args.allow_dirty:
        overrides.append("train.allow_dirty=true")
    if args.config is not None:
        path = Path(args.config)
    elif args.synthetic:
        path = REPO_ROOT / SMOKE_CONFIGS[args.family]
    else:
        raise InputValidationError(
            msg("VALIDATION_ERROR"),
            details={
                "reason": "config_required",
                "hint": "pass --config <path> (or --synthetic for the shipped smoke config)",
            },
        )

    cfg = load_config(path, overrides)
    notes: list[str] = []
    if args.synthetic:
        extra: list[str] = []
        target = SYNTHETIC_DATASETS[cfg.model.family]
        if cfg.data.dataset != target:
            extra.append(f"data.dataset={target}")
            notes.append(f"--synthetic rewrote data.dataset to {target}")
        vocab = cfg.model.vocab_path
        if vocab is not None and not Path(vocab).is_file():
            extra.append("model.vocab_path=null")
            notes.append(
                f"--synthetic cleared model.vocab_path ({vocab} is absent); default tokenizer used"
            )
        if extra:
            cfg = load_config(path, [*overrides, *extra])
        if not cfg.is_gate_free:  # pragma: no cover - defensive; the rewrite above guarantees it
            raise InternalError(
                msg("INTERNAL"),
                details={"reason": "synthetic_rewrite_failed", "dataset": cfg.data.dataset},
            )
    return cfg, path, notes


def build_plan(
    cfg: TrainConfig,
    config_path: Path,
    args: argparse.Namespace,
    *,
    ledger_path: Path | None = None,
) -> TrainPlan:
    """Describe the run without starting it."""
    gate_free = cfg.is_gate_free
    missing = (
        ()
        if gate_free
        else tuple(gate_status(GATE, cli_flag=bool(args.i_have_approval), ledger_path=ledger_path))
    )
    return TrainPlan(
        cfg=cfg,
        config_path=config_path,
        run_dir=Path(args.run_dir) if args.run_dir else None,
        gate_free=gate_free,
        missing=missing,
        use_queue=use_queue(args, gate_free),
        notes=tuple(args.notes),
    )


def use_queue(args: argparse.Namespace, gate_free: bool) -> bool:
    """Whether to take the workstation's single-job lock (docs/ml/egitim-altyapisi.md §8).

    Real runs take it by default; a gate-free CPU smoke run does not, because probing
    ``nvidia-smi`` and touching ``training/queue.lock`` from a unit test is noise, not safety.
    """
    if args.no_queue:
        return False
    return True if args.queue else not gate_free


def device_line(cfg: TrainConfig) -> str:
    """One line describing the device that would be selected (imports torch; dry run only)."""
    try:
        from aicls.ml.trainer import select_device

        spec = select_device(cfg.train.amp)
    except Exception as exc:  # a device preview must never fail the command
        return f"  device       : unknown ({type(exc).__name__}: {exc})"
    fallback = f"   [fallback: {spec.fallback_reason}]" if spec.fallback_reason else ""
    return f"  device       : {spec.device} ({spec.device_name}) {spec.dtype}{fallback}"


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the script."""
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Run a training job (GATED: TRAINING; --synthetic is gate-free, ADR-0012).",
    )
    parser.add_argument("--config", type=Path, default=None, help="training configuration YAML")
    parser.add_argument(
        "--set",
        "--override",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted-path override, repeatable (optim.lr=1e-4)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="rewrite the config onto its synthetic dataset; the run is then gate-free",
    )
    parser.add_argument(
        "--family",
        choices=sorted(SMOKE_CONFIGS),
        default="ocr",
        help="which shipped smoke config --synthetic uses when --config is omitted",
    )
    parser.add_argument(
        CLI_FLAG,
        dest="i_have_approval",
        action="store_true",
        help="one of the three approval conditions (ADR-0012); never set it for someone else",
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="run directory override")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="absolute step budget for this call (resumes included)",
    )
    parser.add_argument("--resume", type=Path, default=None, help="checkpoint directory to resume")
    parser.add_argument("--no-val", action="store_true", help="skip the validation dataset")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run with a dirty working tree; the manifest is marked citable=false",
    )
    parser.add_argument(
        "--queue", action="store_true", help="always take the training queue lock (§8)"
    )
    parser.add_argument(
        "--device",
        default="",
        help="force 'cpu' or 'cuda'; auto-detected when empty (same flag as `aicls train`)",
    )
    parser.add_argument("--no-queue", action="store_true", help="never take the queue lock")
    parser.add_argument(
        "--force-queue",
        action="store_true",
        help="seize a lock left by a job that died (refused while the GPU is in use)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved plan and exit 0 without asking for the gate",
    )
    parser.add_argument("--json", action="store_true", help="print plan and report as JSON")
    return parser


def main(argv: list[str] | None = None, *, ledger_path: Path | None = None) -> int:
    """Entry point.

    Args:
        argv: Command line (``sys.argv[1:]`` when omitted).
        ledger_path: Approval-ledger override. Deliberately **not** a command-line flag; only
            tests point it elsewhere.

    Returns:
        Process exit code: 0 ok/dry run, 1 typed error, 3 gate refused.
    """
    args = build_parser().parse_args(argv)
    try:
        cfg, config_path, notes = resolve_config(args)
    except AiclsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(f"details: {exc.details}", file=sys.stderr)
        return int(exc.exit_code)

    args.notes = notes
    plan = build_plan(cfg, config_path, args, ledger_path=ledger_path)
    print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False) if args.json else plan.render())
    if args.dry_run:
        print(device_line(cfg))
        print("dry run: nothing was written and no gate was asked for.")
        return 0

    # The gate is the first thing that happens for real. Nothing above imported torch, opened a
    # dataset or created a directory, so a refusal leaves the filesystem untouched.
    try:
        checked = require_training_gate(
            cfg, i_have_approval=bool(args.i_have_approval), ledger_path=ledger_path
        )
    except GateError as exc:
        print(f"\n{exc}", file=sys.stderr)
        print(
            f"hint: {CLI_FLAG} + AICLS_GATE_{GATE}=approved + a row in "
            "docs/gelistirme/onay-defteri.md, or run on synthetic data with --synthetic "
            "(ADR-0012).",
            file=sys.stderr,
        )
        return int(exc.exit_code)
    _log.info(
        "training.train.gate_resolved",
        gate=GATE,
        checked=checked,
        dataset=cfg.data.dataset,
        gate_free=cfg.is_gate_free,
    )

    try:
        report = _run(cfg, args, plan, ledger_path=ledger_path)
    except AiclsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(f"details: {exc.details}", file=sys.stderr)
        return int(exc.exit_code)
    except KeyboardInterrupt:
        print("\ninterrupted: an interrupt checkpoint was written under <run_dir>/checkpoints.")
        return 1

    print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else _render(report, plan))
    return 0


def _run(
    cfg: TrainConfig,
    args: argparse.Namespace,
    plan: TrainPlan,
    *,
    ledger_path: Path | None,
) -> dict[str, Any]:
    """Take the queue (when applicable) and dispatch to the right training loop."""
    if plan.use_queue:
        from aicls.ml.queue import TrainingQueue

        queue: Any = TrainingQueue(job=cfg.run.name)
        if getattr(args, "force_queue", False):
            # Explicit, and only ever explicit. take_over() still refuses while the GPU is in
            # use, so the flag cannot produce the two-jobs-on-one-GPU case; what it overrides is
            # a lock file whose writer is unreachable -- which, in a container, is every lock a
            # dead run leaves behind.
            queue.take_over(reason="--force-queue")
    else:
        queue = nullcontext()
    with queue:
        if cfg.model.family in ASR_FAMILIES:
            return _fit_asr(cfg, args)
        return _fit_ocr(cfg, args, ledger_path=ledger_path)


def _fit_ocr(
    cfg: TrainConfig, args: argparse.Namespace, *, ledger_path: Path | None
) -> dict[str, Any]:
    """OCR path: :class:`aicls.ml.trainer.Trainer` over an :mod:`aicls.data` dataset."""
    from aicls.ml.registry import DatasetSpec, build_dataset, build_model
    from aicls.ml.trainer import Trainer

    tokenizer = DatasetSpec.from_config(cfg).tokenizer()
    model = build_model(cfg, int(tokenizer.vocab_size))
    train_ds = build_dataset(cfg, "train")
    val_ds = None if args.no_val else _optional_dataset(cfg, "valid")
    trainer = Trainer(
        cfg,
        model,
        train_ds,
        val_ds,
        run_dir=Path(args.run_dir) if args.run_dir else None,
        tokenizer=tokenizer,
        i_have_approval=bool(args.i_have_approval),
        ledger_path=ledger_path,
        device=getattr(args, "device", "") or None,
    )
    if args.resume is not None:
        checkpoint = trainer.resume(Path(args.resume))
        print(f"resumed from {args.resume} at step {checkpoint.step}")
    report = trainer.fit(max_steps=args.max_steps)
    document: dict[str, Any] = dict(report.to_dict())
    document["run_dir"] = str(trainer.run_dir)
    return document


def _fit_asr(cfg: TrainConfig, args: argparse.Namespace) -> dict[str, Any]:
    """ASR path: :class:`aicls.ml.whisper_ft.WhisperFineTuner` (Hugging Face ``Seq2SeqTrainer``).

    Unproven on this machine: ``accelerate`` is not installed, so ``Seq2SeqTrainingArguments``
    cannot even be constructed here and the wrapper raises a typed dependency error. The path is
    written, and it is the first thing to re-verify once the CUDA_INSTALL gate opens.
    """
    from aicls.ml.registry import build_dataset
    from aicls.ml.whisper_ft import WhisperDatasetAdapter, WhisperFineTuner

    settings = get_settings()
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else settings.runs_root / f"{cfg.run.name or 'run'}-{ulid()}"
    )
    tuner = WhisperFineTuner(cfg, run_dir=run_dir, allow_dirty=cfg.train.allow_dirty)
    processor = tuner.processor
    train_ds = WhisperDatasetAdapter(
        build_dataset(cfg, "train"), processor, max_label_len=cfg.data.max_label_len
    )
    eval_ds = (
        None if args.no_val else _optional_adapter(cfg, "valid", processor, WhisperDatasetAdapter)
    )
    report = tuner.fit(
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        i_have_approval=bool(args.i_have_approval),
    )
    return dict(report.to_dict())


def _optional_dataset(cfg: TrainConfig, split: str) -> Any:
    """Build ``split``, or return ``None`` with a printed reason (a missing split is not fatal)."""
    from aicls.ml.registry import build_dataset

    try:
        return build_dataset(cfg, split)
    except AiclsError as exc:
        print(f"warning: no '{split}' dataset ({exc.details}); validation disabled")
        _log.warning("training.train.no_validation_split", split=split, details=exc.details)
        return None


def _optional_adapter(cfg: TrainConfig, split: str, processor: Any, adapter: Any) -> Any:
    """Same as :func:`_optional_dataset`, wrapped for the Hugging Face trainer."""
    dataset = _optional_dataset(cfg, split)
    if dataset is None:
        return None
    return adapter(dataset, processor, max_label_len=cfg.data.max_label_len)


def _render(report: dict[str, Any], plan: TrainPlan) -> str:
    """Human-readable end-of-run summary."""
    lines = ["run finished"]
    for key in (
        "run_id",
        "steps",
        "global_step",
        "epochs",
        "train_loss",
        "best_value",
        "best_step",
        "best_checkpoint",
        "last_checkpoint",
        "checkpoint_dir",
        "device",
        "dtype",
        "steps_per_s",
        "wall_time_s",
        "citable",
        "run_dir",
    ):
        if key in report and report[key] is not None:
            lines.append(f"  {key:16s}: {report[key]}")
    metrics = report.get("metrics") or {}
    for name, value in metrics.items():
        lines.append(f"  metric {name:9s}: {value}")
    if plan.gate_free or report.get("wiring_check"):
        lines.append(f"  HONESTY         : {WIRING_CHECK_NOTE}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
