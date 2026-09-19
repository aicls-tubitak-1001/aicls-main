"""Experiment tracking: the :class:`Tracker` protocol, JSONL and MLflow (ADR-0007, §6).

A training run writes its numbers through a :class:`Tracker`. Two implementations exist and the
protocol is what the trainer depends on:

* :class:`JsonlTracker` — the **default**. No dependencies, no server, nothing leaves the
  workstation (KVKK): ``<run_dir>/metrics.jsonl`` (one JSON object per line),
  ``<run_dir>/params.json`` (the resolved configuration and the run record) and
  ``<run_dir>/artifacts.jsonl`` plus an ``artifacts/`` copy of every logged file.
* :class:`MlflowTracker` — optional. ``mlflow`` is imported lazily inside the call that first
  needs it, so importing this module never requires the dependency; when it is missing the
  caller gets an :class:`~aicls.core.errors.AiclsError` naming it instead of an ``ImportError``.

The primary citation key of a report stays ``run_manifest.json`` (git SHA + config hash); an
MLflow ``run_id`` is secondary. There is no TensorBoard (ADR-0007: two overlapping outputs, not
three).

**The line schema of ``metrics.jsonl`` is a contract** (``test_jsonl_tracker_schema``): reporting
code reads these files. Non-finite values (a diverged loss) are written as ``null`` so the file
always parses as strict JSON, and the metric is still visible in the row.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import shutil
from time import perf_counter
from types import TracebackType
from typing import Any, Literal, Protocol, runtime_checkable

from aicls.core.errors import InputValidationError, InternalError, NotFoundError
from aicls.core.hashing import sha256_file
from aicls.core.ids import ulid
from aicls.core.logging import get_logger

_log = get_logger(__name__)

#: Schema version of both ``metrics.jsonl`` lines and ``params.json``.
SCHEMA_VERSION = 1
#: One JSON object per logged metric group.
METRICS_FILENAME = "metrics.jsonl"
#: The resolved configuration and the run record.
PARAMS_FILENAME = "params.json"
#: One JSON object per logged artifact.
ARTIFACTS_FILENAME = "artifacts.jsonl"
#: Directory inside the run directory where external artifacts are copied.
ARTIFACTS_DIRNAME = "artifacts"
#: Sub-directory of ``AICLS_DATA_ROOT`` used as the local MLflow store.
MLRUNS_DIRNAME = "mlruns"

#: Lifecycle status of a run.
RunStatus = Literal["running", "finished", "failed", "killed"]
#: MLflow's spelling of the terminal statuses.
_MLFLOW_STATUS: dict[str, str] = {
    "finished": "FINISHED",
    "failed": "FAILED",
    "killed": "KILLED",
    "running": "RUNNING",
}


@runtime_checkable
class Tracker(Protocol):
    """What the trainer needs from an experiment tracker (docs/mimari/moduller.md).

    Implementations must tolerate being used as a context manager and must not raise from
    :meth:`end_run` when the run never started.
    """

    def start_run(self, name: str, params: Mapping[str, Any] | None = None) -> str:
        """Begin a run and return its id."""
        ...

    def log_metrics(self, metrics: Mapping[str, float], step: int, *, split: str = "train") -> None:
        """Record one group of metrics measured at ``step`` on ``split``."""
        ...

    def log_artifact(self, path: Path, *, name: str | None = None) -> dict[str, Any]:
        """Attach a file to the run and return the record written for it."""
        ...

    def end_run(self, status: RunStatus = "finished") -> None:
        """Close the run; calling it twice keeps the first terminal status."""
        ...


class _BaseTracker:
    """Context-manager plumbing shared by the implementations."""

    def end_run(self, status: RunStatus = "finished") -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def __enter__(self) -> Any:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """End the run, marking it ``failed`` when the body raised."""
        self.end_run("failed" if exc_type is not None else "finished")


def flatten_params(params: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a nested configuration into dotted ``str`` pairs (MLflow's parameter shape).

    Args:
        params: Nested mapping, e.g. the resolved :class:`~aicls.ml.config.TrainConfig` dump.
        prefix: Prefix for the recursion (callers pass nothing).

    Returns:
        ``{"optim.lr": "0.0003", ...}`` — values are stringified because MLflow parameters are
        strings; the untouched nested document lives in ``params.json``.
    """
    flat: dict[str, str] = {}
    for key, value in params.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, prefix=f"{dotted}."))
        else:
            flat[dotted] = str(value)
    return flat


def _numeric(name: str, value: Any) -> float | None:
    """Coerce a metric value to ``float``; non-finite values become ``None``.

    Raises:
        InputValidationError: If the value is not a real number (a string metric is a bug in the
            caller, not a diverged run).
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InputValidationError(
            details={"reason": "metric_not_numeric", "metric": name, "type": type(value).__name__}
        )
    number = float(value)
    if not math.isfinite(number):
        _log.warning("ml.tracking.non_finite_metric", metric=name, value=str(value))
        return None
    return number


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    """Append one strict-JSON line (LF, UTF-8) and flush, so a crash keeps what was logged."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


class JsonlTracker(_BaseTracker):
    """Dependency-free tracker writing JSONL and JSON into the run directory.

    Args:
        run_dir: Directory of the run; created on :meth:`start_run`.
        run_id: Explicit run id (default: a fresh ULID). Pass the
            :attr:`aicls.ml.manifest.RunManifest.run_id` to tie the two records together.
    """

    def __init__(self, run_dir: Path, *, run_id: str | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id or ulid()
        self.run_name = ""
        self.status: RunStatus = "running"
        self._params: dict[str, Any] = {}
        self._started_at: str | None = None
        self._started_perf: float | None = None
        self._ended = False

    # ------------------------------------------------------------------------------ lifecycle
    def start_run(self, name: str, params: Mapping[str, Any] | None = None) -> str:
        """Create the run directory and write ``params.json``.

        Args:
            name: Human-readable run name (``cfg.run.name``).
            params: The resolved configuration, stored verbatim.

        Returns:
            The run id.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = name
        self._params = dict(params or {})
        self._started_at = _utc_now_iso()
        self._started_perf = perf_counter()
        self._ended = False
        self.status = "running"
        self._write_params(ended_at=None, duration_s=None)
        _log.info("ml.tracking.run_started", run_id=self.run_id, run_name=name, backend="jsonl")
        return self.run_id

    def log_metrics(self, metrics: Mapping[str, float], step: int, *, split: str = "train") -> None:
        """Append one row to ``metrics.jsonl``.

        Args:
            metrics: Metric name to value; non-finite values are written as ``null``.
            step: Optimizer step the measurement belongs to.
            split: ``"train"``, ``"valid"``, ``"test"`` or any other corpus label.

        Raises:
            InputValidationError: If the run has not been started or a value is not numeric.
        """
        if self._started_perf is None:
            raise InputValidationError(details={"reason": "run_not_started", "step": step})
        row = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "step": int(step),
            "split": str(split),
            "wall_time": round(perf_counter() - self._started_perf, 6),
            "ts": _utc_now_iso(),
            "metrics": {str(k): _numeric(str(k), v) for k, v in metrics.items()},
        }
        _append_jsonl(self.run_dir / METRICS_FILENAME, row)

    def log_artifact(self, path: Path, *, name: str | None = None) -> dict[str, Any]:
        """Record a file produced by the run, copying it in when it lives elsewhere.

        Args:
            path: The file to attach.
            name: Name to store it under (default: the file name).

        Returns:
            The record appended to ``artifacts.jsonl``.

        Raises:
            NotFoundError: If ``path`` is not an existing file.
        """
        source = Path(path)
        if not source.is_file():
            raise NotFoundError(details={"reason": "artifact_missing", "path": str(source)})
        self.run_dir.mkdir(parents=True, exist_ok=True)
        label = name or source.name
        stored = self._store_artifact(source, label)
        record = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "name": label,
            "path": stored,
            "size": source.stat().st_size,
            "sha256": sha256_file(source),
            "ts": _utc_now_iso(),
        }
        _append_jsonl(self.run_dir / ARTIFACTS_FILENAME, record)
        return record

    def end_run(self, status: RunStatus = "finished") -> None:
        """Write the terminal status into ``params.json`` (idempotent)."""
        if self._started_perf is None or self._ended:
            return
        self._ended = True
        self.status = status
        duration = round(perf_counter() - self._started_perf, 3)
        self._write_params(ended_at=_utc_now_iso(), duration_s=duration)
        _log.info("ml.tracking.run_ended", run_id=self.run_id, status=status, duration_s=duration)

    # -------------------------------------------------------------------------------- helpers
    def _store_artifact(self, source: Path, label: str) -> str:
        """Return the recorded path, copying the file into ``artifacts/`` when it is external."""
        try:
            return source.resolve().relative_to(self.run_dir.resolve()).as_posix()
        except ValueError:
            target_dir = self.run_dir / ARTIFACTS_DIRNAME
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target_dir / label)
            return (Path(ARTIFACTS_DIRNAME) / label).as_posix()

    def _write_params(self, *, ended_at: str | None, duration_s: float | None) -> None:
        document = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "started_at": self._started_at,
            "ended_at": ended_at,
            "status": self.status,
            "duration_s": duration_s,
            "params": self._params,
        }
        path = self.run_dir / PARAMS_FILENAME
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")


class MlflowTracker(_BaseTracker):
    """Optional MLflow-backed tracker (local file store, no server, no cloud).

    Args:
        experiment: MLflow experiment name (``cfg.tracking.experiment``).
        tracking_uri: Store URI; defaults to ``file:<AICLS_DATA_ROOT>/mlruns`` (ADR-0007 — data
            never leaves the institution).
        run_id: Ignored by MLflow, kept so both trackers construct alike.

    Raises:
        MlflowUnavailableError: From the first call that needs ``mlflow`` when it is not
            installed. Constructing the tracker never imports it.
    """

    def __init__(
        self,
        experiment: str = "aicls",
        *,
        tracking_uri: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.experiment = experiment
        self.tracking_uri = tracking_uri
        self.run_id = run_id or ""
        self.run_name = ""
        self.status: RunStatus = "running"
        self._active = False

    def start_run(self, name: str, params: Mapping[str, Any] | None = None) -> str:
        """Start an MLflow run and log the flattened parameters."""
        mlflow = _require_mlflow()
        mlflow.set_tracking_uri(self.tracking_uri or default_mlflow_uri())
        mlflow.set_experiment(self.experiment)
        run = mlflow.start_run(run_name=name)
        self.run_id = str(run.info.run_id)
        self.run_name = name
        self.status = "running"
        self._active = True
        if params:
            mlflow.log_params(flatten_params(params))
        _log.info("ml.tracking.run_started", run_id=self.run_id, run_name=name, backend="mlflow")
        return self.run_id

    def log_metrics(self, metrics: Mapping[str, float], step: int, *, split: str = "train") -> None:
        """Log metrics as ``<split>.<name>`` so both trackers keep the split visible."""
        mlflow = _require_mlflow()
        payload: dict[str, float] = {}
        for key, value in metrics.items():
            number = _numeric(str(key), value)
            if number is not None:
                payload[f"{split}.{key}"] = number
        if payload:
            mlflow.log_metrics(payload, step=int(step))

    def log_artifact(self, path: Path, *, name: str | None = None) -> dict[str, Any]:
        """Upload a file into the run's artifact store.

        Raises:
            NotFoundError: If ``path`` is not an existing file.
        """
        source = Path(path)
        if not source.is_file():
            raise NotFoundError(details={"reason": "artifact_missing", "path": str(source)})
        mlflow = _require_mlflow()
        mlflow.log_artifact(str(source))
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "name": name or source.name,
            "path": str(source),
            "size": source.stat().st_size,
            "sha256": sha256_file(source),
            "ts": _utc_now_iso(),
        }

    def end_run(self, status: RunStatus = "finished") -> None:
        """Close the MLflow run (no-op when none was started)."""
        if not self._active:
            return
        self._active = False
        self.status = status
        _require_mlflow().end_run(status=_MLFLOW_STATUS.get(status, "FINISHED"))
        _log.info("ml.tracking.run_ended", run_id=self.run_id, status=status, backend="mlflow")


class MlflowUnavailableError(InternalError):
    """``mlflow`` is not installed (it is an optional dependency in ``requirements/dev.txt``)."""


def _require_mlflow() -> Any:
    """Import ``mlflow`` lazily.

    Raises:
        MlflowUnavailableError: If the optional dependency is missing.
    """
    try:
        import mlflow
    except ImportError as exc:
        raise MlflowUnavailableError(
            details={
                "reason": "mlflow_not_installed",
                "error": str(exc),
                "hint": "pip install mlflow (requirements/dev.txt); JsonlTracker needs nothing",
            }
        ) from exc
    return mlflow


def default_mlflow_uri() -> str:
    """``file:<AICLS_DATA_ROOT>/mlruns`` — the local, offline MLflow store (ADR-0007)."""
    from aicls.core.settings import get_settings

    return f"file:{(get_settings().data_root / MLRUNS_DIRNAME).as_posix()}"


# ------------------------------------------------------------------------------------- readers
def iter_metrics(run_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield the rows of ``metrics.jsonl``; an absent file yields nothing.

    Raises:
        InputValidationError: If a line is not valid JSON (a truncated file is a real failure
            mode of a killed run and must not be silently ignored by a report).
    """
    path = Path(run_dir) / METRICS_FILENAME if Path(run_dir).is_dir() else Path(run_dir)
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputValidationError(
                    details={
                        "reason": "metrics_line_not_json",
                        "path": str(path),
                        "line": line_no,
                        "error": str(exc),
                    }
                ) from exc
            yield row


def read_metrics(run_dir: Path) -> list[dict[str, Any]]:
    """All rows of ``metrics.jsonl`` (empty list when the run logged nothing)."""
    return list(iter_metrics(run_dir))


def read_params(run_dir: Path) -> dict[str, Any]:
    """The ``params.json`` document of a run.

    Raises:
        NotFoundError: If the file does not exist.
        InputValidationError: If it is not valid JSON.
    """
    path = Path(run_dir)
    if path.is_dir():
        path = path / PARAMS_FILENAME
    if not path.is_file():
        raise NotFoundError(details={"reason": "params_missing", "path": str(path)})
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputValidationError(
            details={"reason": "params_not_json", "path": str(path), "error": str(exc)}
        ) from exc
    if not isinstance(payload, dict):  # pragma: no cover - written by this module only
        raise InputValidationError(details={"reason": "params_not_object", "path": str(path)})
    return payload


__all__ = [
    "ARTIFACTS_DIRNAME",
    "ARTIFACTS_FILENAME",
    "METRICS_FILENAME",
    "MLRUNS_DIRNAME",
    "PARAMS_FILENAME",
    "SCHEMA_VERSION",
    "JsonlTracker",
    "MlflowTracker",
    "MlflowUnavailableError",
    "RunStatus",
    "Tracker",
    "default_mlflow_uri",
    "flatten_params",
    "iter_metrics",
    "read_metrics",
    "read_params",
]
