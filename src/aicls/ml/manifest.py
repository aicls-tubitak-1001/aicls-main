"""Run manifest, seeding and environment capture (ADR-0006, ADR-0007; egitim-altyapisi.md §7).

Every number that reaches a TÜBİTAK progress report has to be traceable to the exact code, data
and machine that produced it. :class:`RunManifest` is that record. The trainer writes it to
``<run_dir>/run_manifest.json`` **before the first optimizer step**; it names the git commit, the
working-tree state, the resolved-configuration hash, the dataset fingerprints, a hash of the
installed packages, the interpreter, torch/CUDA/cuDNN, the GPU and its driver, the host and the
start time. ``steps_per_s`` is the one field the trainer fills in afterwards (§8: budgets come
from a measured profile run, never from a hand-written estimate).

Two rules are enforced here instead of being left to discipline:

#. **A dirty working tree refuses the run.** ``RunManifest.capture(cfg)`` raises
   :class:`DirtyWorkingTreeError` unless the caller passes ``allow_dirty=True`` (``--allow-dirty``
   / ``train.allow_dirty``), and such a run is marked ``citable=False`` for ``poe report-check``.
#. **``citable`` is derived, never supplied.** It is true only when git answered *and* the tree
   was clean; a manifest constructed with ``citable=True`` over a dirty tree comes back false.

``data_fingerprints`` names the *data* half of that record. For a prepared shards root it holds
one entry per split (``mathwriting.train``, ``mathwriting.valid``, ...) read from the per-split
``shards_manifest.json`` that the preparation wrote, plus one aggregate ``mathwriting`` key a
report can cite. The layout knowledge lives in :mod:`aicls.data.shards`, not here.

Git state comes from real ``git`` subprocesses (this is the reproducibility backbone and the
workstation always has git). A missing or failing ``git`` is recorded as the ``"unknown"`` state
and makes the run non-citable — it never crashes a training run. :mod:`aicls.eval.provenance`
makes the opposite trade for the evaluation harness: it reads ``.git`` files without a subprocess.

torch is imported lazily inside the functions that need it, so importing this module (and with it
:mod:`aicls.ml`) stays cheap and works on a machine without torch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
from dataclasses import dataclass
from datetime import UTC, datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import socket
import subprocess
import sys
from types import ModuleType
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aicls.core.errors import AiclsError, InputValidationError, NotFoundError
from aicls.core.hashing import (
    FINGERPRINT_PREFIX,
    canonical_json,
    fingerprint_jsonl,
    sha256_bytes,
    sha256_file,
    sha256_json,
)
from aicls.core.ids import ulid
from aicls.core.logging import get_logger

_log = get_logger(__name__)

#: File name of the manifest inside a run directory.
MANIFEST_FILENAME = "run_manifest.json"
#: File name of the full environment dump written next to the manifest.
ENV_FILENAME = "env.json"
#: Recorded value of every git field that could not be determined.
UNKNOWN = "unknown"
#: Project-wide default seed (docs/gelistirme/test-stratejisi.md).
DEFAULT_SEED = 1234
#: Largest seed numpy accepts.
MAX_SEED = 2**32 - 1
#: Environment variable cuBLAS reads for its deterministic workspace configuration.
CUBLAS_ENV_VAR = "CUBLAS_WORKSPACE_CONFIG"
#: Value required for deterministic cuBLAS GEMMs (docs/ml/egitim-altyapisi.md §7).
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
#: Default timeout (seconds) for the small probe processes started here.
DEFAULT_PROCESS_TIMEOUT_S = 15.0
#: Timeout for ``git diff HEAD``, which can be large on a very dirty tree.
DIFF_TIMEOUT_S = 60.0

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
#: Environment variables whose *value* is never written to ``env.json``.
_SECRET_ENV_RE = re.compile(r"(SECRET|TOKEN|PASSWORD|PASSWD|API_KEY|JWT|CREDENTIAL)", re.IGNORECASE)
#: Non-``AICLS_`` environment variables worth recording for reproducibility.
_RECORDED_ENV_NAMES: tuple[str, ...] = (
    "CONDA_DEFAULT_ENV",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "HF_HUB_OFFLINE",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "PYTHONHASHSEED",
    "TRANSFORMERS_OFFLINE",
    "VIRTUAL_ENV",
)


class DirtyWorkingTreeError(InputValidationError):
    """The git working tree carries uncommitted changes and ``allow_dirty`` is false (ADR-0007).

    The Turkish user text is the generic validation message from
    :mod:`aicls.core.messages_tr`; the machine-readable ``details["reason"]`` is
    ``"dirty_working_tree"`` and ``details`` also carries the commit and the diff hash.
    """


# --------------------------------------------------------------------------------- subprocesses
@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Outcome of a probe process.

    Attributes:
        args: The command line that was run.
        returncode: Process exit status.
        stdout: Captured standard output (UTF-8, undecodable bytes replaced).
        stderr: Captured standard error.
    """

    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        """True when the process exited with status 0."""
        return self.returncode == 0


def run_process(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout_s: float = DEFAULT_PROCESS_TIMEOUT_S,
) -> ProcessResult | None:
    """Run a local command and capture its output; never raise.

    Only fixed command lines built in this package are passed here (``git``, ``nvidia-smi``); no
    shell is involved and no argument comes from user input.

    Args:
        args: Program and arguments.
        cwd: Working directory for the child process.
        timeout_s: Seconds before the child is killed.

    Returns:
        The :class:`ProcessResult`, or ``None`` when the executable is missing, could not be
        started, or timed out. Callers treat ``None`` as "this tool is not available here".
    """
    try:
        # Fixed argv built in this module; shell=False, no user input reaches the command.
        completed = subprocess.run(
            list(args),
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        _log.debug("ml.process.unavailable", command=next(iter(args), ""), error=str(exc))
        return None
    return ProcessResult(
        args=tuple(args),
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


# ------------------------------------------------------------------------------------- git state
@dataclass(frozen=True, slots=True)
class GitState:
    """Commit identity and working-tree state of a training run.

    Attributes:
        sha: Full commit hash, or :data:`UNKNOWN` when git did not answer.
        branch: Branch name, ``"HEAD"`` for a detached head, or :data:`UNKNOWN`.
        dirty: Whether the tree carried uncommitted or untracked changes.
        diff_sha256: SHA-256 over ``git status --porcelain`` plus ``git diff HEAD`` when dirty,
            so that untracked-only dirt still produces a distinct hash; ``None`` when clean.
        available: Whether ``git`` answered at all.
    """

    sha: str = UNKNOWN
    branch: str = UNKNOWN
    dirty: bool = False
    diff_sha256: str | None = None
    available: bool = False

    @property
    def short_sha(self) -> str:
        """First seven characters of :attr:`sha` (``"unknown"`` stays as it is)."""
        return self.sha[:7] if _SHA_RE.match(self.sha) else self.sha


def capture_git_state(repo_root: Path | None = None) -> GitState:
    """Ask ``git`` for the commit, the branch and the working-tree state.

    Args:
        repo_root: Directory the git commands run in; defaults to the repository that holds this
            source file (``aicls.core.paths.find_repo_root``), falling back to the process CWD.

    Returns:
        The :class:`GitState`. When ``git`` is absent, fails or the directory is not a checkout,
        every field keeps its "unknown" default and ``available`` is false — the run is then not
        citable, but it is not stopped.
    """
    cwd = repo_root if repo_root is not None else _default_repo_root()
    head = run_process(["git", "rev-parse", "HEAD"], cwd=cwd)
    if head is None or not head.ok:
        _log.warning("ml.manifest.git_unavailable", cwd=str(cwd))
        return GitState()
    sha = head.stdout.strip()
    branch_result = run_process(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    branch = branch_result.stdout.strip() if branch_result and branch_result.ok else UNKNOWN
    status = run_process(["git", "status", "--porcelain"], cwd=cwd)
    if status is None or not status.ok:
        # `available=False`, deliberately. `citable = git_available and not dirty`, and `dirty`
        # defaults to False, so reporting `available=True` here published **citable=True for a
        # run whose working-tree state is unknown** -- the one thing the manifest exists to
        # prevent. It is not hypothetical: `git status --porcelain` over a /mnt path in WSL was
        # measured at 10.2 s against DEFAULT_PROCESS_TIMEOUT_S = 15.0 s (2026-09-08), so a
        # loaded box crosses it. The sha and branch are kept -- they are still true and useful
        # -- and the run is not stopped; it simply may not be cited.
        _log.warning(
            "ml.manifest.git_status_failed",
            cwd=str(cwd),
            sha=sha,
            citable=False,
            reason="working tree state unknown; the run may not be cited",
        )
        return GitState(sha=sha or UNKNOWN, branch=branch or UNKNOWN, available=False)
    status_text = status.stdout
    dirty = bool(status_text.strip())
    diff_sha256: str | None = None
    if dirty:
        diff = run_process(["git", "diff", "HEAD"], cwd=cwd, timeout_s=DIFF_TIMEOUT_S)
        diff_text = diff.stdout if diff is not None and diff.ok else ""
        diff_sha256 = sha256_bytes((status_text + "\n" + diff_text).encode("utf-8"))
    return GitState(
        sha=sha or UNKNOWN,
        branch=branch or UNKNOWN,
        dirty=dirty,
        diff_sha256=diff_sha256,
        available=True,
    )


def _default_repo_root() -> Path:
    from aicls.core.paths import find_repo_root

    try:
        return find_repo_root()
    except AiclsError:  # pragma: no cover - only outside a checkout
        return Path.cwd()


# ---------------------------------------------------------------------------------- environment
def pip_freeze_lines() -> list[str]:
    """Installed distributions as sorted, unique ``name==version`` lines.

    Built from :mod:`importlib.metadata` rather than by running ``pip freeze``: same content for
    the reproducibility record, no subprocess and no chance of a network call.
    """
    seen: set[str] = set()
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"] if dist.metadata else None
        if not name:  # pragma: no cover - malformed distribution metadata
            continue
        seen.add(f"{name}=={dist.version}")
    return sorted(seen)


def pip_freeze_sha256(lines: Sequence[str] | None = None) -> str:
    """SHA-256 over the :func:`pip_freeze_lines` document (one line each, LF, trailing newline)."""
    body = "\n".join(lines if lines is not None else pip_freeze_lines()) + "\n"
    return sha256_bytes(body.encode("utf-8"))


def _try_import_torch() -> ModuleType | None:
    """Import torch, or return ``None`` when it is not installed (lazy by design)."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a dev dependency here
        _log.warning("ml.manifest.torch_missing")
        return None
    return torch


def torch_versions() -> dict[str, Any]:
    """torch, CUDA and cuDNN versions plus visible devices; all ``None``/empty without torch."""
    info: dict[str, Any] = {
        "torch": None,
        "cuda": None,
        "cudnn": None,
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
    }
    torch = _try_import_torch()
    if torch is None:  # pragma: no cover - torch is installed in this environment
        return info
    info["torch"] = str(torch.__version__)
    cuda_version = getattr(torch.version, "cuda", None)
    info["cuda"] = str(cuda_version) if cuda_version else None
    try:
        cudnn_version = torch.backends.cudnn.version()
    except (AttributeError, RuntimeError, OSError):  # pragma: no cover - CPU build returns None
        cudnn_version = None
    info["cudnn"] = str(cudnn_version) if cudnn_version else None
    try:
        available = bool(torch.cuda.is_available())
    except (AssertionError, RuntimeError, OSError):  # pragma: no cover - broken driver
        available = False
    info["cuda_available"] = available
    if available:  # pragma: no cover - no CUDA wheel until the CUDA_INSTALL gate opens
        count = int(torch.cuda.device_count())
        info["device_count"] = count
        info["devices"] = [str(torch.cuda.get_device_name(i)) for i in range(count)]
    return info


def nvidia_smi_devices(
    *, timeout_s: float = DEFAULT_PROCESS_TIMEOUT_S
) -> list[dict[str, Any]] | None:
    """Query ``nvidia-smi`` for the installed GPUs.

    Args:
        timeout_s: Seconds before the probe is killed.

    Returns:
        One dict per GPU with ``index``, ``name``, ``driver`` and ``memory_total_mb``, or ``None``
        when ``nvidia-smi`` is not installed or failed (a CPU box is not an error).
    """
    result = run_process(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=timeout_s,
    )
    if result is None or not result.ok:
        return None
    devices: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != 4:
            continue
        index, name, driver, memory = cells
        devices.append(
            {
                "index": _as_int(index),
                "name": name,
                "driver": driver,
                "memory_total_mb": _as_float(memory),
            }
        )
    return devices


def _as_int(text: str) -> int | None:
    try:
        return int(text)
    except ValueError:  # pragma: no cover - nvidia-smi always prints an index
        return None


def _as_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:  # pragma: no cover - nvidia-smi always prints a number
        return None


def safe_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The reproducibility-relevant environment variables, with secrets removed.

    ``AICLS_*`` variables and a small allow-list (thread counts, ``CUDA_VISIBLE_DEVICES``,
    offline switches) are recorded; any name that looks like a secret (``*SECRET*``, ``*TOKEN*``,
    ``*API_KEY*``, ``*JWT*``, ...) is dropped entirely — value *and* name.
    """
    source = os.environ if environ is None else environ
    recorded = {
        name: value
        for name, value in source.items()
        if (name.startswith("AICLS_") or name in _RECORDED_ENV_NAMES)
        and not _SECRET_ENV_RE.search(name)
    }
    return dict(sorted(recorded.items()))


def capture_env(run_dir: Path) -> dict[str, Any]:
    """Write ``env.json`` into ``run_dir`` and return the same document.

    The manifest carries only the *hash* of the installed packages; this file carries the full
    list plus the interpreter, the platform, the torch/CUDA/cuDNN versions, the GPUs reported by
    ``nvidia-smi`` and the sanitised environment variables.

    Args:
        run_dir: Run directory (created when missing).

    Returns:
        The JSON document that was written.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    packages = pip_freeze_lines()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": _utc_now_iso(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "prefix": sys.prefix,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": socket.gethostname(),
            "cpu_count": os.cpu_count(),
        },
        "torch": torch_versions(),
        "gpus": nvidia_smi_devices() or [],
        "packages": packages,
        "pip_freeze_sha256": pip_freeze_sha256(packages),
        "env": safe_env(),
    }
    _write_json_atomic(run_dir / ENV_FILENAME, payload)
    _log.info("ml.manifest.env_captured", run_dir=str(run_dir), packages=len(packages))
    return payload


# --------------------------------------------------------------------------------------- seeding
def seed_everything(seed: int = DEFAULT_SEED, *, deterministic: bool = True) -> int:
    """Seed ``random``, ``numpy``, ``torch`` and ``torch.cuda`` and request deterministic kernels.

    With ``deterministic=True`` this also sets ``CUBLAS_WORKSPACE_CONFIG=:4096:8``, calls
    ``torch.use_deterministic_algorithms(True, warn_only=True)`` and turns cuDNN benchmarking off.
    The cuBLAS variable only takes effect when it is set **before** the CUDA context is created,
    so ``seed_everything`` must run at the very start of a training process.

    Args:
        seed: Seed in ``[0, 2**32 - 1]``.
        deterministic: Whether to request deterministic algorithms as well as seeding.

    Returns:
        The seed, so a caller can log or record what was actually used.

    Raises:
        InputValidationError: If ``seed`` is not an integer in range.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise InputValidationError(
            details={"reason": "seed_out_of_range", "seed": repr(seed), "max": MAX_SEED}
        )
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        os.environ[CUBLAS_ENV_VAR] = CUBLAS_WORKSPACE_CONFIG
    torch = _try_import_torch()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - CPU build in this environment
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    _log.debug("ml.seed", seed=seed, deterministic=deterministic)
    return seed


# ------------------------------------------------------------------------------ config handling
def config_document(cfg: Any) -> dict[str, Any]:
    """Return ``cfg`` as a plain JSON-compatible dict.

    Accepts a pydantic model (``model_dump(mode="json")``), a dataclass, or any mapping whose
    values are JSON-serialisable (``Path``, ``set`` and ``bytes`` are normalised by
    :func:`aicls.core.hashing.canonical_json`).

    Raises:
        InputValidationError: If ``cfg`` is none of those, or contains unserialisable values.
    """
    if hasattr(cfg, "model_dump"):
        dumped = cfg.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return dataclasses.asdict(cfg)
    if isinstance(cfg, Mapping):
        decoded = json.loads(canonical_json(cfg))
        if isinstance(decoded, dict):
            return decoded
    raise InputValidationError(details={"reason": "unsupported_config", "type": type(cfg).__name__})


def config_fingerprint(cfg: Any) -> str:
    """Hash of the resolved configuration, recorded in :attr:`RunManifest.config_hash`.

    Delegates to :func:`aicls.ml.config.config_hash` for a real
    :class:`~aicls.ml.config.TrainConfig` (so a manifest and a run directory always agree) and
    falls back to the canonical-JSON SHA-256 for the plain mappings used in tests and scripts.
    """
    hash_fn, train_config = _config_module_api()
    if hash_fn is not None and train_config is not None and isinstance(cfg, train_config):
        return str(hash_fn(cfg))
    return sha256_json(config_document(cfg))


def _config_module_api() -> tuple[Any | None, Any | None]:
    """``(config_hash, TrainConfig)`` from :mod:`aicls.ml.config`, or ``(None, None)``."""
    try:
        from aicls.ml import config as config_module
    except ImportError:  # pragma: no cover - config.py is a sibling module of this wave
        return None, None
    return getattr(config_module, "config_hash", None), getattr(config_module, "TrainConfig", None)


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    """Read a dotted path (``"run.name"``) from a mapping or an attribute tree."""
    current: Any = cfg
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return default
            current = current[part]
        else:
            if not hasattr(current, part):
                return default
            current = getattr(current, part)
    return default if current is None else current


def collect_data_fingerprints(cfg: Any, *, splits: Sequence[str] | None = None) -> dict[str, str]:
    """Fingerprints of the data ``cfg`` reads (``{"mathwriting": "sha256:...", ...}``).

    Four sources are recorded when they exist:

    * ``data.root`` as a **prepared shards root** — one entry per split, keyed
      ``<name>.<split>`` (``mathwriting.train``), read from the per-split
      ``shards_manifest.json`` that :func:`aicls.data.shards.prepare_shards_from_source` wrote.
      This is the normal MathWriting case;
    * ``data.root`` as a :class:`aicls.data.manifest.DataManifest` root (a ``manifest.json`` at
      the root, what ``prepare_mathwriting.py --manifest`` and ``aicls data verify`` use), keyed
      by the dataset name. When there is no such file but there are split fingerprints, the
      ``<name>`` key is derived from them with
      :func:`aicls.data.shards.shards_root_fingerprint`, so a report always has one value to
      cite as ``EvalReport.dataset_fingerprint``;
    * ``data.manifest`` — the JSONL export manifest of an audio run, keyed ``<name>.manifest``;
    * ``data.splits_path`` — the speaker-disjoint split file, keyed ``<name>.splits`` (§2: its
      SHA goes into every ASR report).

    Args:
        cfg: The resolved configuration (or a mapping with the same shape).
        splits: Restrict the shard entries to these splits — what a run that opens only
            ``train`` and ``valid`` should record. By default every prepared split under
            ``data.root`` is fingerprinted.

    Returns:
        The fingerprints; ``{}`` when the configuration names no dataset or nothing was found.

    Nothing here is fatal: a synthetic run has no data root at all. The consequence of an empty
    result is visible in the manifest — an evaluation report cannot cite data it cannot
    fingerprint, and ``ml.manifest.data_fingerprint_missing`` says which root was empty.
    """
    name = str(_cfg_get(cfg, "data.dataset", "") or "")
    if not name:
        return {}
    fingerprints: dict[str, str] = {}
    root = _cfg_get(cfg, "data.root", "")
    if root:
        fingerprints.update(_root_fingerprints(Path(str(root)), name, splits))
    jsonl_manifest = _cfg_get(cfg, "data.manifest", "")
    if jsonl_manifest and Path(str(jsonl_manifest)).is_file():
        fingerprints[f"{name}.manifest"] = fingerprint_jsonl(Path(str(jsonl_manifest)))
    splits_path = _cfg_get(cfg, "data.splits_path", "")
    if splits_path and Path(str(splits_path)).is_file():
        fingerprints[f"{name}.splits"] = FINGERPRINT_PREFIX + sha256_file(Path(str(splits_path)))
    return fingerprints


def _root_fingerprints(root: Path, name: str, splits: Sequence[str] | None) -> dict[str, str]:
    """Fingerprint ``root`` both ways: a ``DataManifest`` root and a prepared shards root.

    The two layouts coexist on purpose. ``manifest.json`` is a full re-hash of every byte below
    the root and is written only on request (``--manifest``); ``<split>/shards_manifest.json``
    is written by every preparation run and already carries the digest of each shard. Looking
    for the first alone is what left ``data_fingerprints: {}`` in the first real MathWriting run
    — the run recorded which code it was, but not which data.
    """
    from aicls.data.manifest import load_manifest
    from aicls.data.shards import shard_fingerprints, shards_root_fingerprint

    found: dict[str, str] = {}
    try:
        found[name] = load_manifest(root).fingerprint
    except AiclsError as exc:
        _log.debug("ml.manifest.data_root_manifest_absent", dataset=name, reason=str(exc))
    per_split = shard_fingerprints(root, splits)
    for split, fingerprint in per_split.items():
        found[f"{name}.{split}"] = fingerprint
    if per_split and not found.get(name):
        found[name] = shards_root_fingerprint(per_split)
    if not found:
        _log.info(
            "ml.manifest.data_fingerprint_missing",
            dataset=name,
            root=str(root),
            reason="no manifest.json and no <split>/shards_manifest.json below the data root",
        )
    else:
        _log.info(
            "ml.manifest.data_fingerprints",
            dataset=name,
            root=str(root),
            splits=sorted(per_split),
            fingerprint=found.get(name, ""),
        )
    return found


# ---------------------------------------------------------------------------------- the manifest
class RunManifest(BaseModel):
    """Provenance of one training run (docs/ml/egitim-altyapisi.md §7, ADR-0007).

    Field names follow the contract in ``docs/mimari/moduller.md``. Everything except
    :attr:`steps_per_s` is filled in by :meth:`capture` before the first optimizer step; the
    trainer (or ``aicls profile``) writes the measured throughput afterwards with
    :meth:`set_steps_per_s`.
    """

    model_config = ConfigDict(extra="forbid")

    #: Schema version of the manifest document itself.
    SCHEMA_VERSION: ClassVar[int] = 1

    schema_version: int = SCHEMA_VERSION
    run_id: str
    run_name: str = "run"
    seed: int | None = None
    started_at: datetime

    # provenance
    git_sha: str = UNKNOWN
    branch: str = UNKNOWN
    dirty: bool = False
    diff_sha256: str | None = None
    git_available: bool = False
    allow_dirty: bool = False
    citable: bool = False

    # inputs
    config_hash: str = ""
    data_fingerprints: dict[str, str] = Field(default_factory=dict)

    # environment
    pip_freeze_sha256: str = ""
    python: str = ""
    platform: str = ""
    hostname: str = ""
    torch: str | None = None
    cuda: str | None = None
    cudnn: str | None = None
    cuda_available: bool = False
    device_count: int = 0
    gpu_name: str | None = None
    driver: str | None = None
    gpu_memory_total_mb: float | None = None

    # measured during the run
    steps_per_s: float | None = None

    @model_validator(mode="after")
    def _derive_citable(self) -> RunManifest:  # noqa: N804 - "after" validators take self
        """``citable`` is recomputed, so a caller cannot claim a dirty run is citable."""
        self.citable = self.git_available and not self.dirty
        return self

    # ------------------------------------------------------------------------------- capture
    @classmethod
    def capture(
        cls,
        cfg: Any,
        *,
        allow_dirty: bool = False,
        repo_root: Path | None = None,
        run_name: str | None = None,
        data_fingerprints: Mapping[str, str] | None = None,
        data_splits: Sequence[str] | None = None,
    ) -> RunManifest:
        """Record the current commit, configuration, data and machine.

        Args:
            cfg: The resolved :class:`~aicls.ml.config.TrainConfig` (or any mapping with the same
                shape) that the run will use.
            allow_dirty: Permit a run from a dirty working tree. Such a run is recorded with
                ``citable=False`` and must not appear in a report. ``cfg.train.allow_dirty`` has
                the same effect (docs/ml/egitim-altyapisi.md §7); the manifest records the
                effective value.
            repo_root: Directory the ``git`` commands run in (default: this checkout).
            run_name: Overrides ``cfg.run.name``.
            data_fingerprints: Explicit dataset fingerprints; by default they are collected from
                ``cfg`` by :func:`collect_data_fingerprints`.
            data_splits: The splits the run will actually open (``("train", "valid")`` for the
                OCR loop). Only those shard fingerprints are recorded; by default every prepared
                split under ``cfg.data.root`` is. Ignored when ``data_fingerprints`` is given.

        Returns:
            The manifest, ready to :meth:`save` into the run directory.

        Raises:
            DirtyWorkingTreeError: If the tree is dirty and ``allow_dirty`` is false.
            InputValidationError: If ``cfg`` cannot be turned into a configuration document.
        """
        git = capture_git_state(repo_root)
        allowed = bool(allow_dirty) or bool(_cfg_get(cfg, "train.allow_dirty", False))
        if git.dirty and not allowed:
            _log.error("ml.manifest.dirty_tree_refused", git_sha=git.sha)
            raise DirtyWorkingTreeError(
                details={
                    "reason": "dirty_working_tree",
                    "git_sha": git.sha,
                    "diff_sha256": git.diff_sha256,
                    "hint": "commit or stash, or pass allow_dirty=True for a non-citable run",
                }
            )
        if git.dirty:
            _log.warning("ml.manifest.dirty_tree_allowed", git_sha=git.sha, citable=False)
        torch_info = torch_versions()
        devices = nvidia_smi_devices() or []
        first = devices[0] if devices else {}
        manifest = cls(
            run_id=ulid(),
            run_name=str(run_name or _cfg_get(cfg, "run.name", "") or "run"),
            seed=_as_optional_int(_cfg_get(cfg, "run.seed")),
            started_at=datetime.now(UTC),
            git_sha=git.sha,
            branch=git.branch,
            dirty=git.dirty,
            diff_sha256=git.diff_sha256,
            git_available=git.available,
            allow_dirty=allowed,
            config_hash=config_fingerprint(cfg),
            data_fingerprints=dict(data_fingerprints)
            if data_fingerprints is not None
            else collect_data_fingerprints(cfg, splits=data_splits),
            pip_freeze_sha256=pip_freeze_sha256(),
            python=platform.python_version(),
            platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
            hostname=socket.gethostname(),
            torch=torch_info["torch"],
            cuda=torch_info["cuda"],
            cudnn=torch_info["cudnn"],
            cuda_available=bool(torch_info["cuda_available"]),
            device_count=int(torch_info["device_count"]),
            gpu_name=first.get("name") or _first_torch_device(torch_info),
            driver=first.get("driver"),
            gpu_memory_total_mb=first.get("memory_total_mb"),
        )
        _log.info(
            "ml.manifest.captured",
            run_id=manifest.run_id,
            run_name=manifest.run_name,
            git_sha=git.short_sha,
            dirty=manifest.dirty,
            citable=manifest.citable,
            config_hash=manifest.config_hash[:16],
            data_fingerprints=sorted(manifest.data_fingerprints),
        )
        return manifest

    # ------------------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict of every field."""
        return dict(json.loads(self.model_dump_json()))

    def to_json(self) -> str:
        """Pretty JSON with sorted keys and a trailing newline (UTF-8, LF)."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"

    def save(self, target: Path) -> Path:
        """Write ``run_manifest.json``.

        Args:
            target: A run directory (the file is written as ``<target>/run_manifest.json``) or
                the target file path itself.

        Returns:
            The path that was written.
        """
        target = Path(target)
        path = target if target.suffix == ".json" else target / MANIFEST_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_text_atomic(path, self.to_json())
        return path

    @classmethod
    def load(cls, path: Path) -> RunManifest:
        """Read a manifest written by :meth:`save`.

        Args:
            path: The manifest file, or the run directory containing one.

        Raises:
            NotFoundError: If the file does not exist.
            InputValidationError: If it is not a valid manifest document.
        """
        path = Path(path)
        if path.is_dir():
            path = path / MANIFEST_FILENAME
        if not path.is_file():
            raise NotFoundError(details={"reason": "manifest_missing", "path": str(path)})
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise InputValidationError(
                details={"reason": "manifest_not_json", "path": str(path), "error": str(exc)}
            ) from exc
        try:
            return cls.model_validate(payload)
        except ValueError as exc:
            raise InputValidationError(
                details={"reason": "manifest_invalid", "path": str(path), "error": str(exc)}
            ) from exc

    def set_steps_per_s(self, value: float, *, run_dir: Path | None = None) -> None:
        """Record the measured training throughput (docs/ml/egitim-altyapisi.md §8).

        Args:
            value: Optimizer steps per second, strictly positive and finite.
            run_dir: When given, the manifest is rewritten there.

        Raises:
            InputValidationError: If ``value`` is not a positive finite number.
        """
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise InputValidationError(details={"reason": "invalid_steps_per_s", "value": value})
        self.steps_per_s = number
        if run_dir is not None:
            self.save(run_dir)


def _first_torch_device(torch_info: Mapping[str, Any]) -> str | None:
    devices = torch_info.get("devices") or []
    return str(devices[0]) if devices else None


def _as_optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _utc_now_iso() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` through a temporary file and ``os.replace`` (never a half-written file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


__all__ = [
    "CUBLAS_ENV_VAR",
    "CUBLAS_WORKSPACE_CONFIG",
    "DEFAULT_SEED",
    "ENV_FILENAME",
    "MANIFEST_FILENAME",
    "MAX_SEED",
    "UNKNOWN",
    "DirtyWorkingTreeError",
    "GitState",
    "ProcessResult",
    "RunManifest",
    "capture_env",
    "capture_git_state",
    "collect_data_fingerprints",
    "config_document",
    "config_fingerprint",
    "nvidia_smi_devices",
    "pip_freeze_lines",
    "pip_freeze_sha256",
    "run_process",
    "safe_env",
    "seed_everything",
    "torch_versions",
]
