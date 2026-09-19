"""Atomic checkpoints and retention (docs/ml/egitim-altyapisi.md §5, ADR-0006).

A checkpoint directory is self-describing — the weights *and* everything needed to explain them::

    <AICLS_MODELS_ROOT>/<component>-<version>/
        model.safetensors     weights (torch.save fallback: model.pt)
        optimizer.pt          last checkpoint only
        scheduler.pt
        config.yaml           the resolved TrainConfig
        run_manifest.json     git SHA, config hash, data fingerprints (ADR-0007)
        vocab.json            tokenizer vocabulary
        metrics.json          the metrics this checkpoint was selected by
        rng_state.pt          python / numpy / torch / cuda RNG for an exact resume
        checkpoint.json       this module's metadata, written **last**

``checkpoint.json`` carries the SHA-256 of the weight file — ``aicls models import`` copies it
into the ``model_versions`` table, so the API can prove which bytes it is serving — and it doubles
as the *completion marker*: :func:`save_checkpoint` builds the whole directory under a temporary
name and only then ``os.replace``s it into place, and :func:`load_checkpoint` refuses a directory
without that file or with a weight file whose hash does not match. A crash mid-write therefore
leaves the previous checkpoint untouched and nothing half-written that could be loaded
(``test_checkpoint_atomic_write``).

torch is imported lazily, so importing this module on a machine without torch is harmless.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import random
import re
import shutil
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import yaml  # type: ignore[import-untyped]  # PyYAML ships no stubs and types-PyYAML is absent

from aicls.core.errors import InputValidationError, NotFoundError
from aicls.core.hashing import sha256_file
from aicls.core.ids import ulid
from aicls.core.logging import get_logger
from aicls.core.paths import long_path
from aicls.ml.manifest import MANIFEST_FILENAME, RunManifest, config_document

_log = get_logger(__name__)

#: Weight file written when ``safetensors`` is available (the documented layout).
MODEL_SAFETENSORS_FILENAME = "model.safetensors"
#: Weight file written when ``safetensors`` cannot serialise the state dict.
MODEL_TORCH_FILENAME = "model.pt"
OPTIMIZER_FILENAME = "optimizer.pt"
SCHEDULER_FILENAME = "scheduler.pt"
CONFIG_FILENAME = "config.yaml"
VOCAB_FILENAME = "vocab.json"
METRICS_FILENAME = "metrics.json"
RNG_FILENAME = "rng_state.pt"
#: Metadata and completion marker; written last, checked first.
META_FILENAME = "checkpoint.json"
#: Pointers written by :class:`CheckpointSaver` next to the checkpoint directories.
BEST_POINTER_FILENAME = "best.json"
LAST_POINTER_FILENAME = "last.json"
#: Directory name pattern used by :class:`CheckpointSaver` (``step-00002000``).
STEP_DIR_TEMPLATE = "step-{step:08d}"
_STEP_DIR_RE = re.compile(r"^step-(\d{1,12})$")
#: Characters of a ULID used to make a staging directory unique. The staging name lives in the
#: same directory as the checkpoint it replaces, so every character of it eats into the Windows
#: ``MAX_PATH`` budget: ``.step-00002000.t01J8R2ZQ`` is 11 characters longer than the target,
#: where the old ``.step-00002000.tmp-<pid>-<full ulid>`` was 38 longer and pushed a legal run
#: directory over 260 (KNOWN_ISSUES medium-priority item 2, reproduced 2026-09-08).
_TMP_TOKEN_LEN = 8

#: Selection direction of the "best" checkpoint.
BestMode = Literal["min", "max"]
#: Serialisation used for the weight file.
WeightFormat = Literal["safetensors", "torch"]


def _torch() -> Any:
    """Import torch lazily (typed error instead of ``ImportError`` when it is missing)."""
    from aicls.ocr.torch_utils import require_torch

    return require_torch()


# ---------------------------------------------------------------------------------- documents
class CheckpointMeta(BaseModel):
    """``checkpoint.json``: what this checkpoint is and how to verify it.

    Attributes:
        step: Optimizer step the weights were taken at.
        model_file: Name of the weight file inside the directory.
        model_sha256: SHA-256 of that file — the identity of the weights in reports and in the
            ``model_versions`` table.
        files: Every other file written, so a truncated copy is visible.
        metrics: The metrics recorded with the checkpoint (selection evidence).
        run_id / git_sha / config_hash: Copied from the run manifest for quick lookups.
    """

    model_config = ConfigDict(extra="forbid")

    SCHEMA_VERSION: ClassVar[int] = 1

    schema_version: int = SCHEMA_VERSION
    step: int
    created_at: str
    format: WeightFormat = "safetensors"
    model_file: str = MODEL_SAFETENSORS_FILENAME
    model_sha256: str = ""
    files: list[str] = Field(default_factory=list)
    has_optimizer: bool = False
    has_scheduler: bool = False
    has_rng: bool = False
    has_vocab: bool = False
    metrics: dict[str, float] = Field(default_factory=dict)
    run_id: str = ""
    git_sha: str = ""
    config_hash: str = ""

    def to_json(self) -> str:
        """Pretty JSON with a trailing newline (UTF-8, LF)."""
        payload = json.loads(self.model_dump_json())
        return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """What :func:`load_checkpoint` restored.

    Attributes:
        path: The checkpoint directory.
        step: Optimizer step to resume from.
        meta: The parsed ``checkpoint.json``.
        config: The resolved configuration (``{}`` when the checkpoint carries none).
        manifest: The run manifest as a plain dict (``{}`` when absent).
        metrics: The metrics stored with the checkpoint.
        vocab: The tokenizer vocabulary, or ``None``.
        rng_restored: Whether the RNG state was put back into the process.
    """

    path: Path
    step: int
    meta: CheckpointMeta
    config: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    vocab: dict[str, Any] | None = None
    rng_restored: bool = False

    @property
    def model_sha256(self) -> str:
        """SHA-256 of the weight file that was loaded."""
        return self.meta.model_sha256


# ------------------------------------------------------------------------------------ RNG state
def rng_state() -> dict[str, Any]:
    """Capture the python, numpy, torch (and CUDA) RNG state for an exact resume."""
    torch = _torch()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():  # pragma: no cover - CPU build in this environment
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Put a state captured by :func:`rng_state` back into the process."""
    torch = _torch()
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():  # pragma: no cover - no CUDA wheel yet
        torch.cuda.set_rng_state_all(state["cuda"])


# ------------------------------------------------------------------------------------- saving
def save_checkpoint(
    ckpt_dir: Path,
    *,
    model: Any,
    step: int,
    optimizer: Any = None,
    scheduler: Any = None,
    cfg: Any = None,
    manifest: RunManifest | Mapping[str, Any] | None = None,
    metrics: Mapping[str, float] | None = None,
    vocab: Mapping[str, Any] | None = None,
    save_rng: bool = True,
) -> CheckpointMeta:
    """Write a complete checkpoint directory atomically.

    The payload is written into a sibling temporary directory and moved into place with
    ``os.replace``; an existing checkpoint is renamed away first and only deleted after the move
    succeeded. If anything raises, the temporary directory is removed and the previous checkpoint
    is left exactly as it was.

    Args:
        ckpt_dir: Target directory (created/replaced as a whole).
        model: Module (or plain tensor mapping) whose ``state_dict`` is stored.
        step: Optimizer step being checkpointed.
        optimizer: Optimizer whose state is stored (``optimizer.pt``, last checkpoint only).
        scheduler: LR scheduler whose state is stored.
        cfg: Resolved configuration, written as ``config.yaml``.
        manifest: :class:`~aicls.ml.manifest.RunManifest` (or its dict) for ``run_manifest.json``.
        metrics: Metrics this checkpoint was selected by.
        vocab: Tokenizer vocabulary.
        save_rng: Whether to store the RNG state for an exact resume.

    Returns:
        The :class:`CheckpointMeta` that was written as ``checkpoint.json``.

    Raises:
        InputValidationError: If ``model`` exposes no state dict.
        TorchUnavailableError: If torch is not installed.
    """
    target = long_path(ckpt_dir)
    tmp_dir = target.with_name(_staging_name(target.name, "t"))
    published = False
    try:
        meta = _write_payload(
            tmp_dir,
            model=model,
            step=step,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            manifest=manifest,
            metrics=metrics,
            vocab=vocab,
            save_rng=save_rng,
        )
        _publish(tmp_dir, target)
        published = True
    finally:
        if not published:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    _log.info(
        "ml.checkpoint.saved",
        path=str(Path(ckpt_dir)),
        step=step,
        sha256=meta.model_sha256[:16],
        format=meta.format,
    )
    return meta


def _staging_name(target_name: str, kind: str) -> str:
    """Name of the hidden sibling directory a save stages into (``t``) or backs up to (``o``).

    Kept deliberately short: it shares the parent directory with ``target_name``, so the extra
    characters count against the Windows ``MAX_PATH`` budget of every file written inside it.
    The trailing characters of a ULID are its random component, which is what uniqueness needs.
    """
    return f".{target_name}.{kind}{ulid()[-_TMP_TOKEN_LEN:]}"


def _write_payload(
    tmp_dir: Path,
    *,
    model: Any,
    step: int,
    optimizer: Any,
    scheduler: Any,
    cfg: Any,
    manifest: RunManifest | Mapping[str, Any] | None,
    metrics: Mapping[str, float] | None,
    vocab: Mapping[str, Any] | None,
    save_rng: bool,
) -> CheckpointMeta:
    """Fill the temporary directory; ``checkpoint.json`` is written last, by :func:`_write_meta`."""
    torch = _torch()
    tmp_dir.mkdir(parents=True, exist_ok=False)
    model_file, weight_format = _save_weights(tmp_dir, model, step)
    written: list[str] = [model_file]

    if optimizer is not None:
        torch.save(optimizer.state_dict(), tmp_dir / OPTIMIZER_FILENAME)
        written.append(OPTIMIZER_FILENAME)
    if scheduler is not None:
        torch.save(scheduler.state_dict(), tmp_dir / SCHEDULER_FILENAME)
        written.append(SCHEDULER_FILENAME)
    if cfg is not None:
        _write_text(tmp_dir / CONFIG_FILENAME, _config_yaml(cfg))
        written.append(CONFIG_FILENAME)
    manifest_document = _manifest_document(manifest)
    if manifest_document is not None:
        _write_json(tmp_dir / MANIFEST_FILENAME, manifest_document)
        written.append(MANIFEST_FILENAME)
    if vocab is not None:
        _write_json(tmp_dir / VOCAB_FILENAME, dict(vocab))
        written.append(VOCAB_FILENAME)
    if metrics is not None:
        _write_json(tmp_dir / METRICS_FILENAME, {str(k): float(v) for k, v in metrics.items()})
        written.append(METRICS_FILENAME)
    if save_rng:
        torch.save(rng_state(), tmp_dir / RNG_FILENAME)
        written.append(RNG_FILENAME)

    document = manifest_document or {}
    meta = CheckpointMeta(
        step=int(step),
        created_at=datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        format=weight_format,
        model_file=model_file,
        model_sha256=sha256_file(tmp_dir / model_file),
        files=sorted(written),
        has_optimizer=optimizer is not None,
        has_scheduler=scheduler is not None,
        has_rng=save_rng,
        has_vocab=vocab is not None,
        metrics={str(k): float(v) for k, v in (metrics or {}).items()},
        run_id=str(document.get("run_id", "")),
        git_sha=str(document.get("git_sha", "")),
        config_hash=str(document.get("config_hash", "")),
    )
    _write_meta(tmp_dir, meta)
    return meta


def _write_meta(tmp_dir: Path, meta: CheckpointMeta) -> None:
    """Write ``checkpoint.json`` — the last step of a save and the completion marker."""
    _write_text(tmp_dir / META_FILENAME, meta.to_json())


def _publish(tmp_dir: Path, target: Path) -> None:
    """Move ``tmp_dir`` onto ``target`` atomically, keeping the old directory until it lands."""
    target.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if target.exists():
        backup = target.with_name(_staging_name(target.name, "o"))
        os.replace(target, backup)
    try:
        os.replace(tmp_dir, target)
    except OSError:
        if backup is not None:  # pragma: no cover - only on a filesystem-level failure
            os.replace(backup, target)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def _safetensors_save_file() -> Any:
    """``safetensors.torch.save_file``, or ``None`` when the optional package is absent."""
    try:
        from safetensors.torch import save_file
    except ImportError:  # pragma: no cover - safetensors ships with the dev extra
        return None
    return save_file


def _save_weights(tmp_dir: Path, model: Any, step: int) -> tuple[str, WeightFormat]:
    """Write the state dict, preferring ``safetensors`` and falling back to ``torch.save``."""
    torch = _torch()
    state_dict = model.state_dict() if hasattr(model, "state_dict") else model
    if not isinstance(state_dict, Mapping):
        raise InputValidationError(
            details={"reason": "model_has_no_state_dict", "type": type(model).__name__}
        )
    tensors = {key: value.detach().cpu().contiguous() for key, value in state_dict.items()}
    save_file = _safetensors_save_file()
    if save_file is not None:
        try:
            save_file(tensors, str(tmp_dir / MODEL_SAFETENSORS_FILENAME), {"step": str(step)})
        except (RuntimeError, ValueError) as exc:  # pragma: no cover - tied/shared tensors
            _log.warning("ml.checkpoint.safetensors_failed", error=str(exc))
        else:
            return MODEL_SAFETENSORS_FILENAME, "safetensors"
    torch.save(tensors, tmp_dir / MODEL_TORCH_FILENAME)  # pragma: no cover - fallback path
    return MODEL_TORCH_FILENAME, "torch"  # pragma: no cover - fallback path


def _manifest_document(
    manifest: RunManifest | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if manifest is None:
        return None
    if isinstance(manifest, RunManifest):
        return manifest.to_dict()
    return dict(manifest)


def _config_yaml(cfg: Any) -> str:
    """The resolved configuration as YAML (sorted keys, UTF-8, LF)."""
    document = config_document(cfg)
    dumped = yaml.safe_dump(document, sort_keys=True, allow_unicode=True, default_flow_style=False)
    return str(dumped)


def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    document = json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True)
    _write_text(path, document + "\n")


# ------------------------------------------------------------------------------------ loading
def load_checkpoint(
    ckpt_dir: Path,
    *,
    model: Any = None,
    optimizer: Any = None,
    scheduler: Any = None,
    map_location: str = "cpu",
    restore_rng: bool = True,
    verify_sha: bool = True,
    strict: bool = True,
) -> Checkpoint:
    """Read a checkpoint directory and restore the objects that were passed in.

    Args:
        ckpt_dir: The checkpoint directory.
        model: Module to load the weights into (skipped when ``None``).
        optimizer: Optimizer to restore (skipped when the file is absent).
        scheduler: LR scheduler to restore.
        map_location: Device the tensors are read onto (``"cpu"`` by default: a checkpoint from
            the GPU box must load on a CPU box).
        restore_rng: Whether to put the stored RNG state back into the process. Training resumes
            pass ``True``; inference callers pass ``False`` so the process RNG is left alone.
        verify_sha: Re-hash the weight file and compare with ``checkpoint.json``.
        strict: Passed to ``model.load_state_dict``.

    Returns:
        The :class:`Checkpoint` record (step, metadata, config, manifest, metrics, vocab).

    Raises:
        NotFoundError: If the directory or the weight file is missing.
        InputValidationError: If ``checkpoint.json`` is absent or unreadable (a half-written or
            hand-assembled directory) or the weight hash does not match.
        TorchUnavailableError: If torch is not installed and something has to be deserialised.
    """
    given = Path(ckpt_dir)
    # Filesystem calls go through the long-path form; the record keeps the caller's own path so
    # reports and logs never carry a `\\?\` prefix.
    path = long_path(given)
    if not path.is_dir():
        raise NotFoundError(details={"reason": "checkpoint_missing", "path": str(given)})
    meta = _read_meta(path)
    weight_path = path / meta.model_file
    if not weight_path.is_file():
        raise NotFoundError(
            details={"reason": "checkpoint_weights_missing", "path": str(weight_path)}
        )
    if verify_sha and meta.model_sha256:
        actual = sha256_file(weight_path)
        if actual != meta.model_sha256:
            raise InputValidationError(
                details={
                    "reason": "checkpoint_sha_mismatch",
                    "path": str(weight_path),
                    "expected": meta.model_sha256,
                    "actual": actual,
                }
            )
    if model is not None:
        model.load_state_dict(_read_weights(weight_path, meta.format, map_location), strict=strict)
    if optimizer is not None and (path / OPTIMIZER_FILENAME).is_file():
        optimizer.load_state_dict(_torch_load(path / OPTIMIZER_FILENAME, map_location))
    if scheduler is not None and (path / SCHEDULER_FILENAME).is_file():
        scheduler.load_state_dict(_torch_load(path / SCHEDULER_FILENAME, map_location))
    rng_restored = False
    if restore_rng and (path / RNG_FILENAME).is_file():
        restore_rng_state(_torch_load(path / RNG_FILENAME, "cpu"))
        rng_restored = True
    checkpoint = Checkpoint(
        path=given,
        step=meta.step,
        meta=meta,
        config=_read_yaml(path / CONFIG_FILENAME),
        manifest=_read_json(path / MANIFEST_FILENAME),
        metrics={str(k): float(v) for k, v in _read_json(path / METRICS_FILENAME).items()},
        vocab=_read_json(path / VOCAB_FILENAME) if (path / VOCAB_FILENAME).is_file() else None,
        rng_restored=rng_restored,
    )
    _log.info(
        "ml.checkpoint.loaded", path=str(given), step=checkpoint.step, rng_restored=rng_restored
    )
    return checkpoint


def _read_meta(path: Path) -> CheckpointMeta:
    meta_path = path / META_FILENAME
    if not meta_path.is_file():
        raise InputValidationError(
            details={
                "reason": "checkpoint_incomplete",
                "path": str(path),
                "missing": META_FILENAME,
                "hint": "the directory was never published by save_checkpoint",
            }
        )
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        return CheckpointMeta.model_validate(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise InputValidationError(
            details={"reason": "checkpoint_meta_invalid", "path": str(meta_path), "error": str(exc)}
        ) from exc


def _read_weights(path: Path, weight_format: WeightFormat, map_location: str) -> dict[str, Any]:
    if weight_format == "safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path), device=map_location))
    return dict(_torch_load(path, map_location))  # pragma: no cover - fallback path


def _torch_load(path: Path, map_location: str) -> Any:
    """``torch.load`` on a file this package wrote.

    ``weights_only=False`` is required for the optimizer, scheduler and RNG payloads (they carry
    plain Python objects). These files are produced by :func:`save_checkpoint` under
    ``AICLS_MODELS_ROOT`` and are trusted; a downloaded checkpoint is never read this way.
    """
    torch = _torch()
    return torch.load(path, map_location=map_location, weights_only=False)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InputValidationError(
            details={"reason": "checkpoint_json_invalid", "path": str(path), "error": str(exc)}
        ) from exc
    return dict(payload) if isinstance(payload, dict) else {}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise InputValidationError(
            details={"reason": "checkpoint_yaml_invalid", "path": str(path), "error": str(exc)}
        ) from exc
    return dict(payload) if isinstance(payload, dict) else {}


# -------------------------------------------------------------------------------- retention
class CheckpointSaver:
    """Periodic checkpointing with ``keep_last`` + ``keep_best`` retention (§5).

    The saver writes ``<root>/step-<step>/`` directories, keeps the ``keep_last`` most recent
    ones plus the best one by ``metric``, and maintains two pointer files (``best.json``,
    ``last.json``) so a resume does not have to guess. Every write goes through
    :func:`save_checkpoint`, so pruning can never race with a half-written directory.

    Args:
        root: Directory the checkpoints live in.
        keep_last: How many recent checkpoints to keep. ``0`` (with ``keep_best=1``) keeps only
            the best one, which is what ``cfg.checkpoint.keep_last: 0`` means.
        keep_best: 1 to protect the best checkpoint from pruning, 0 to disable. ``keep_last`` and
            ``keep_best`` may not both be 0 — that would keep nothing at all.
        metric: Metric name used to select the best checkpoint. The defaults of this class are
            the defaults of :class:`aicls.ml.config.CheckpointCfg`, so a trainer that forwards
            ``cfg.checkpoint`` and one that constructs the saver bare behave the same; OCR runs
            set ``checkpoint.metric: cer`` and ASR runs ``wer``.
        mode: ``"min"`` (lower is better) or ``"max"``.
        cfg / manifest / vocab: Defaults passed to every :func:`save_checkpoint` call.

    Raises:
        InputValidationError: On an invalid ``keep_last``, ``keep_best``, retention
            combination or ``mode``.
    """

    def __init__(
        self,
        root: Path,
        *,
        keep_last: int = 2,
        keep_best: int = 1,
        metric: str = "val_loss",
        mode: BestMode = "min",
        cfg: Any = None,
        manifest: RunManifest | Mapping[str, Any] | None = None,
        vocab: Mapping[str, Any] | None = None,
    ) -> None:
        if mode not in ("min", "max"):
            raise InputValidationError(details={"reason": "invalid_mode", "mode": str(mode)})
        if keep_last < 0:
            raise InputValidationError(
                details={"reason": "invalid_keep_last", "keep_last": keep_last}
            )
        if keep_best not in (0, 1):
            raise InputValidationError(
                details={"reason": "invalid_keep_best", "keep_best": keep_best}
            )
        if keep_last + keep_best < 1:
            raise InputValidationError(
                details={
                    "reason": "invalid_retention",
                    "keep_last": keep_last,
                    "keep_best": keep_best,
                    "hint": "keep_last + keep_best must keep at least one checkpoint",
                }
            )
        self.root = Path(root)
        self.keep_last = keep_last
        self.keep_best = keep_best
        self.metric = metric
        self.mode: BestMode = mode
        self.cfg = cfg
        self.manifest = manifest
        self.vocab = vocab
        self.best_step: int | None = None
        self.best_value: float | None = None
        self.last_step: int | None = None

    # ------------------------------------------------------------------------------ pointers
    @property
    def best_dir(self) -> Path | None:
        """Directory of the best checkpoint so far, or ``None``."""
        return None if self.best_step is None else self.dir_for(self.best_step)

    @property
    def last_dir(self) -> Path | None:
        """Directory of the most recent checkpoint, or ``None``."""
        return None if self.last_step is None else self.dir_for(self.last_step)

    def dir_for(self, step: int) -> Path:
        """``<root>/step-XXXXXXXX`` for ``step``."""
        return self.root / STEP_DIR_TEMPLATE.format(step=step)

    def is_best(self, metrics: Mapping[str, float]) -> bool:
        """Whether ``metrics`` improves on the best value seen so far."""
        value = metrics.get(self.metric)
        if value is None:
            return False
        if self.best_value is None:
            return True
        return value < self.best_value if self.mode == "min" else value > self.best_value

    # -------------------------------------------------------------------------------- saving
    def save(
        self,
        step: int,
        *,
        model: Any,
        optimizer: Any = None,
        scheduler: Any = None,
        metrics: Mapping[str, float] | None = None,
        cfg: Any = None,
        manifest: RunManifest | Mapping[str, Any] | None = None,
        vocab: Mapping[str, Any] | None = None,
        save_rng: bool = True,
    ) -> CheckpointMeta:
        """Write a checkpoint for ``step``, update the pointers and prune.

        Returns:
            The metadata of the checkpoint that was written.
        """
        values = dict(metrics or {})
        improved = self.is_best(values)
        meta = save_checkpoint(
            self.dir_for(step),
            model=model,
            step=step,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg if cfg is not None else self.cfg,
            manifest=manifest if manifest is not None else self.manifest,
            metrics=values,
            vocab=vocab if vocab is not None else self.vocab,
            save_rng=save_rng,
        )
        self.last_step = step
        if improved:
            self.best_step = step
            self.best_value = float(values[self.metric])
            self._write_pointer(BEST_POINTER_FILENAME, step, self.best_value)
        self._write_pointer(LAST_POINTER_FILENAME, step, values.get(self.metric))
        self.prune()
        return meta

    def prune(self) -> list[Path]:
        """Delete checkpoints outside ``keep_last`` (and the protected best one).

        Returns:
            The directories that were removed.
        """
        existing = sorted(self.existing_steps())
        keep = set(existing[-self.keep_last :]) if self.keep_last else set()
        if self.keep_best and self.best_step is not None:
            keep.add(self.best_step)
        removed: list[Path] = []
        for step in existing:
            if step in keep:
                continue
            path = self.dir_for(step)
            shutil.rmtree(long_path(path), ignore_errors=True)
            removed.append(path)
            _log.info("ml.checkpoint.pruned", path=str(path), step=step)
        return removed

    def existing_steps(self) -> list[int]:
        """Steps of the checkpoint directories currently on disk."""
        root = long_path(self.root)
        if not root.is_dir():
            return []
        steps: list[int] = []
        for child in root.iterdir():
            match = _STEP_DIR_RE.match(child.name)
            if child.is_dir() and match:
                steps.append(int(match.group(1)))
        return sorted(steps)

    def _write_pointer(self, name: str, step: int, value: float | None) -> None:
        _write_json(
            long_path(self.root) / name,
            {
                "step": step,
                "dir": self.dir_for(step).name,
                "metric": self.metric,
                "mode": self.mode,
                "value": value,
            },
        )


__all__ = [
    "BEST_POINTER_FILENAME",
    "CONFIG_FILENAME",
    "LAST_POINTER_FILENAME",
    "META_FILENAME",
    "METRICS_FILENAME",
    "MODEL_SAFETENSORS_FILENAME",
    "MODEL_TORCH_FILENAME",
    "OPTIMIZER_FILENAME",
    "RNG_FILENAME",
    "SCHEDULER_FILENAME",
    "STEP_DIR_TEMPLATE",
    "VOCAB_FILENAME",
    "BestMode",
    "Checkpoint",
    "CheckpointMeta",
    "CheckpointSaver",
    "WeightFormat",
    "load_checkpoint",
    "restore_rng_state",
    "rng_state",
    "save_checkpoint",
]
