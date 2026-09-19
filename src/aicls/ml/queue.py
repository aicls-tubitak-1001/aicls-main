"""Single-GPU training queue (docs/ml/egitim-altyapisi.md §8).

The project has **one** RTX 5090. Two training processes on it do not halve the throughput, they
crash with a CUDA out-of-memory error somewhere in the middle of an epoch — usually after hours.
:class:`TrainingQueue` makes that impossible to do by accident: a job takes a lock file
(``training/queue.lock`` by default) and, before it does, asks ``nvidia-smi`` whether another
process is already holding GPU memory.

**The lock file is git-ignored and the queue never deletes a file it did not write.** Both halves
matter: the lock is taken *around* the run, so it exists while :class:`aicls.ml.manifest.
RunManifest` captures the git state, and a tracked (or newly created, tracked-adjacent) lock would
make every queued run ``citable=false``. Earlier the queue also treated any unparsable lock as
stale and ``unlink``ed it, which deleted the committed ``training/queue.txt`` placeholder on the
first queued run; it now overwrites in place instead (:meth:`TrainingQueue._write_lock`).

Two properties matter for a machine that is also somebody's desktop:

* **A stale lock never blocks the box forever.** The lock records the PID and the host; when the
  holder is gone the lock is taken over. Liveness is checked without signalling anything — on
  Windows ``os.kill(pid, 0)`` *terminates* the process, so this module uses ``OpenProcess`` +
  ``GetExitCodeProcess`` there and ``os.kill(pid, 0)`` only on POSIX.
* **A missing ``nvidia-smi`` is not an error.** The CPU development box has no GPU at all, and
  the CUDA wheel is still behind the ``CUDA_INSTALL`` gate; the queue then relies on the lock
  file alone.

A lock from a *different host* is never assumed stale — the PID means nothing there — so a shared
data directory cannot cause two workstations to grab the same job.

That rule has a cost this box actually pays. Training runs inside a container (ADR-0020), and a
container's hostname is a fresh random id, so **every** lock is written by a "different host" and
the takeover above can never fire. An abrupt death — a Windows Update restart, ``docker kill``, an
OOM — therefore leaves a lock that blocks every later run, with an error telling the operator to
wait for a job that no longer exists. Observed on 2026-09-09: the machine rebooted mid-run and the
next run refused to start, holder ``20684ee4158c`` pid 1.

Automatic expiry is the wrong repair here. The failure the lock exists to prevent — two trainings
on one GPU — is unrecoverable and silent, while the failure it just caused is loud and safe, so
trading the second for a chance of the first is a bad bargain. :meth:`TrainingQueue.take_over` is
the deliberate way out instead: it refuses while any process is using the GPU, records who is being
displaced, and is only ever reached through an explicit flag typed by somebody who looked.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import socket
import sys
from time import monotonic, sleep
from types import TracebackType
from typing import Any

from aicls.core import messages_tr
from aicls.core.errors import AiclsError, ErrorCode
from aicls.core.logging import get_logger
from aicls.ml.manifest import run_process

_log = get_logger(__name__)

#: Name of the lock file. It is **git-ignored** on purpose: the lock is written while the run
#: captures its manifest, so a tracked lock file would dirty the working tree and record every
#: queued run as ``citable=false`` (ADR-0007). ``training/queue.txt`` next to it is a committed
#: description of this mechanism and is never written to.
QUEUE_LOCK_FILENAME = "queue.lock"
#: Lock file, relative to the repository root.
QUEUE_RELATIVE_PATH = Path("training") / QUEUE_LOCK_FILENAME
#: A compute process using at least this much GPU memory makes the GPU "busy".
DEFAULT_GPU_BUSY_MB = 512.0
#: Seconds between two attempts while :meth:`TrainingQueue.acquire` waits.
DEFAULT_POLL_S = 2.0
#: Timeout for the ``nvidia-smi`` probe.
GPU_PROBE_TIMEOUT_S = 15.0
#: Windows ``OpenProcess`` access right and "still running" exit code.
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5


class TrainingQueueBusyError(AiclsError):
    """Another job (or another process) already owns the GPU.

    Carries ``details["reason"]`` — ``"training_queue_busy"`` when a live lock file exists,
    ``"gpu_busy"`` when ``nvidia-smi`` reports a foreign compute process. The code is
    :attr:`~aicls.core.errors.ErrorCode.TRAINING_BUSY` and the Turkish text is
    :data:`~aicls.core.messages_tr.MSG_TRAINING_QUEUE_BUSY`: the operator is not an API client
    that sent too many requests (which is what the previously borrowed ``RATE_LIMITED`` text
    said), the one RTX 5090 is simply taken. This is **not** an approval gate, so it is not a
    :class:`~aicls.core.gates.GateError` and the exit code stays 1.
    """

    default_code = ErrorCode.TRAINING_BUSY

    def __init__(self, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(self.default_code, messages_tr.MSG_TRAINING_QUEUE_BUSY, details, None)


@dataclass(frozen=True, slots=True)
class QueueLock:
    """The record stored in the lock file.

    Attributes:
        pid: Process id of the holder.
        host: Host name of the holder (a PID is only meaningful together with it).
        started_at: UTC ISO-8601 timestamp with a ``Z`` suffix.
        job: Free-form job label, usually ``cfg.run.name``.
        user: OS user that started the job.
    """

    pid: int
    host: str
    started_at: str
    job: str = ""
    user: str = ""

    def to_json(self) -> str:
        """The record as one JSON object (what the lock file contains)."""
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> QueueLock | None:
        """Parse a lock file's content; ``None`` when it is empty or malformed."""
        stripped = text.strip()
        if not stripped:
            return None
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict) or "pid" not in payload or "host" not in payload:
            return None
        try:
            return cls(
                pid=int(payload["pid"]),
                host=str(payload["host"]),
                started_at=str(payload.get("started_at", "")),
                job=str(payload.get("job", "")),
                user=str(payload.get("user", "")),
            )
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True, slots=True)
class GpuProcess:
    """One compute process reported by ``nvidia-smi --query-compute-apps``."""

    pid: int
    used_memory_mb: float | None
    name: str


def default_queue_path() -> Path:
    """``<repo>/training/queue.lock`` — the single-job lock of this workstation."""
    from aicls.core.paths import find_repo_root

    try:
        root = find_repo_root()
    except AiclsError:  # pragma: no cover - only outside a checkout
        root = Path.cwd()
    return root / QUEUE_RELATIVE_PATH


def hostname() -> str:
    """This machine's host name, as recorded in the lock."""
    return socket.gethostname()


def _username() -> str:
    return os.environ.get("USERNAME") or os.environ.get("USER") or ""


def process_is_alive(pid: int) -> bool:
    """Whether ``pid`` is a running process on **this** host, without signalling it.

    On Windows ``os.kill(pid, 0)`` calls ``TerminateProcess`` — it would kill the very job we are
    asking about — so the check goes through ``OpenProcess``/``GetExitCodeProcess`` there.

    Args:
        pid: Process id (non-positive ids are never alive).

    Returns:
        True when the process exists and has not exited. An access-denied answer counts as alive:
        the process exists, it just belongs to somebody else.
    """
    if pid <= 0:
        return False
    # `os.name` (not `sys.platform`) so a type checker keeps both branches reachable.
    if os.name == "nt":
        return _windows_process_is_alive(pid)
    try:  # pragma: no cover - POSIX branch; this project runs on Windows
        os.kill(pid, 0)
    except ProcessLookupError:  # pragma: no cover - POSIX branch
        return False
    except PermissionError:  # pragma: no cover - POSIX branch
        return True
    return True  # pragma: no cover - POSIX branch


def _windows_process_is_alive(pid: int) -> bool:
    """``OpenProcess`` + ``GetExitCodeProcess``; never signals the process."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # Declare the prototypes: the default `c_int` result would truncate a 64-bit HANDLE.
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True  # pragma: no cover - the handle is open, so assume it is alive
            return exit_code.value == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    else:  # pragma: no cover - the dispatcher tests os.name and never sends POSIX here
        # The if/else is for mypy: it skips whichever branch sys.platform rules out, so
        # ``ctypes.WinDLL`` (absent on Linux) is checked on Windows and ignored on Linux.
        raise RuntimeError("_windows_process_is_alive called on a non-Windows platform")


def probe_gpu_processes(*, timeout_s: float = GPU_PROBE_TIMEOUT_S) -> list[GpuProcess] | None:
    """Compute processes currently holding GPU memory.

    Args:
        timeout_s: Seconds before the ``nvidia-smi`` probe is killed.

    Returns:
        One :class:`GpuProcess` per compute application, or ``None`` when ``nvidia-smi`` is not
        installed or failed — a CPU box, not an error. Graphics-only processes are not listed by
        ``--query-compute-apps``, which is exactly what we want: a desktop does not block a run.
    """
    result = run_process(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=timeout_s,
    )
    if result is None or not result.ok:
        _log.debug("ml.queue.nvidia_smi_unavailable")
        return None
    processes: list[GpuProcess] = []
    for line in result.stdout.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) < 3:
            continue
        try:
            pid = int(cells[0])
        except ValueError:
            continue
        try:
            memory: float | None = float(cells[1])
        except ValueError:
            memory = None
        processes.append(GpuProcess(pid=pid, used_memory_mb=memory, name=cells[2]))
    return processes


class TrainingQueue:
    """One-job-at-a-time lock over the workstation's GPU.

    Args:
        lock_path: Lock file (default: :func:`default_queue_path`).
        job: Label written into the lock (usually ``cfg.run.name``).
        check_gpu: Whether to probe ``nvidia-smi`` before taking the lock.
        gpu_busy_mb: A foreign compute process using at least this much memory blocks the queue.
        gpu_probe: Injected probe returning :class:`GpuProcess` records (tests and profiling);
            defaults to :func:`probe_gpu_processes`.

    Example:
        >>> with TrainingQueue(job="ocr_offline"):  # doctest: +SKIP
        ...     trainer.fit()
    """

    def __init__(
        self,
        lock_path: Path | None = None,
        *,
        job: str = "",
        check_gpu: bool = True,
        gpu_busy_mb: float = DEFAULT_GPU_BUSY_MB,
        gpu_probe: Callable[[], list[GpuProcess] | None] | None = None,
    ) -> None:
        self.lock_path = Path(lock_path) if lock_path is not None else default_queue_path()
        self.job = job
        self.check_gpu = check_gpu
        self.gpu_busy_mb = gpu_busy_mb
        self.gpu_probe = gpu_probe
        self._lock: QueueLock | None = None

    # ------------------------------------------------------------------------------ inspection
    @property
    def held(self) -> bool:
        """Whether *this* instance currently holds the lock."""
        return self._lock is not None

    @property
    def lock(self) -> QueueLock | None:
        """The lock record this instance holds, or ``None``."""
        return self._lock

    def read_lock(self) -> QueueLock | None:
        """Parse the lock file; ``None`` when it is absent, empty or malformed."""
        try:
            text = self.lock_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        return QueueLock.from_json(text)

    def is_busy(self) -> bool:
        """Whether a *live* job currently holds the lock."""
        return self._live_holder() is not None

    # -------------------------------------------------------------------------------- locking
    def acquire(self, *, timeout_s: float = 0.0, poll_s: float = DEFAULT_POLL_S) -> QueueLock:
        """Take the queue, waiting at most ``timeout_s`` seconds.

        Args:
            timeout_s: How long to keep retrying before refusing (0: refuse immediately).
            poll_s: Delay between attempts.

        Returns:
            The :class:`QueueLock` that was written (or the one already held by this instance).

        Raises:
            TrainingQueueBusyError: If another live job holds the lock, or ``nvidia-smi`` reports
                a foreign process holding GPU memory, until the timeout expires.
        """
        if self._lock is not None:
            return self._lock
        deadline = monotonic() + max(timeout_s, 0.0)
        while True:
            holder = self._live_holder()
            busy_gpu = [] if holder is not None else self._gpu_blockers()
            if holder is None and not busy_gpu:
                lock = self._write_lock()
                if lock is not None:
                    self._lock = lock
                    _log.info(
                        "ml.queue.acquired", pid=lock.pid, job=lock.job, lock=str(self.lock_path)
                    )
                    return lock
            if monotonic() >= deadline:
                raise self._busy_error(holder, busy_gpu)
            sleep(max(poll_s, 0.0))

    def take_over(self, *, reason: str = "") -> QueueLock:
        """Seize the queue from a holder that cannot be verified, and say so in the log.

        For the lock left behind by a run that died without releasing it. The check that keeps
        this honest is the GPU, not the lock file: whoever wrote the lock is unreachable by
        definition, but a training job that is genuinely still running is holding GPU memory, and
        that is observable. So this refuses exactly when it matters and proceeds otherwise.

        It is never called automatically. A caller reaches it only through an explicit flag, which
        keeps a person between "the lock looks wrong" and "two jobs on one GPU".

        Args:
            reason: What the operator checked, recorded in the log beside the displaced holder.

        Returns:
            The lock now held.

        Raises:
            TrainingQueueBusyError: If ``nvidia-smi`` reports a process holding GPU memory. A
                seizure while the GPU is busy is the one case that produces the corruption the
                queue exists to prevent.
        """
        if self._lock is not None:
            return self._lock
        blockers = self._gpu_blockers()
        if blockers:
            raise self._busy_error(None, blockers)
        displaced = self.read_lock()
        self.lock_path.unlink(missing_ok=True)
        lock = self._write_lock()
        if lock is None:  # pragma: no cover - another process won the empty slot
            raise self._busy_error(self.read_lock(), [])
        self._lock = lock
        _log.warning(
            "ml.queue.taken_over",
            reason=reason or "unspecified",
            displaced_pid=getattr(displaced, "pid", None),
            displaced_host=getattr(displaced, "host", None),
            displaced_job=getattr(displaced, "job", None),
            displaced_started_at=getattr(displaced, "started_at", None),
            pid=lock.pid,
            lock=str(self.lock_path),
        )
        return lock

    def release(self) -> bool:
        """Give the queue back.

        Returns:
            True when this instance held the lock and the file was removed; False when it held
            nothing (releasing somebody else's lock is never done).
        """
        if self._lock is None:
            return False
        current = self.read_lock()
        if current is not None and (current.pid, current.host) != (
            self._lock.pid,
            self._lock.host,
        ):  # pragma: no cover - only after a forced takeover
            _log.warning("ml.queue.foreign_lock_kept", pid=current.pid, host=current.host)
            self._lock = None
            return False
        self.lock_path.unlink(missing_ok=True)
        _log.info("ml.queue.released", pid=self._lock.pid, lock=str(self.lock_path))
        self._lock = None
        return True

    def __enter__(self) -> QueueLock:
        return self.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()

    # -------------------------------------------------------------------------------- internals
    def _live_holder(self) -> QueueLock | None:
        """The lock's owner when it is still running; ``None`` when the slot is free.

        This method **inspects only**. It used to ``unlink`` an unparsable or stale lock, which
        deleted whatever file happened to sit at ``lock_path`` — including the committed
        ``training/queue.txt`` placeholder, on the first queued run. Taking over a free slot is
        :meth:`_write_lock`'s job and it overwrites rather than removes.
        """
        existing = self.read_lock()
        if existing is None:
            if self.lock_path.exists():
                _log.warning("ml.queue.unreadable_lock_ignored", lock=str(self.lock_path))
            return None
        if existing.host != hostname():
            # A PID from another machine cannot be checked; assume the job is running.
            return existing
        if process_is_alive(existing.pid):
            return existing
        _log.warning(
            "ml.queue.stale_lock_taken_over", pid=existing.pid, started_at=existing.started_at
        )
        return None

    def _gpu_blockers(self) -> list[GpuProcess]:
        """Foreign compute processes big enough to block a training run."""
        if not self.check_gpu:
            return []
        probe = self.gpu_probe or probe_gpu_processes
        processes = probe()
        if processes is None:
            return []
        mine = os.getpid()
        return [
            process
            for process in processes
            if process.pid != mine and (process.used_memory_mb or 0.0) >= self.gpu_busy_mb
        ]

    def _write_lock(self) -> QueueLock | None:
        """Take the slot: create the lock exclusively, or overwrite a lock nobody holds.

        ``O_EXCL`` is the fast path and the race detector. When the file already exists the
        holder is re-checked: a *live* holder means another process won the race (``None``), and
        a dead or unparsable one means the slot is free, so the file is truncated and rewritten.
        Overwriting — never removing — is what keeps a file the queue did not create from
        disappearing from the working tree.

        Returns:
            The lock that was written, or ``None`` when another live job owns the file.
        """
        lock = QueueLock(
            pid=os.getpid(),
            host=hostname(),
            started_at=datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            job=self.job,
            user=_username(),
        )
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if self._live_holder() is not None:  # pragma: no cover - needs a real race
                return None
            handle = os.open(self.lock_path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY)
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(lock.to_json() + "\n")
        return lock

    def _busy_error(
        self, holder: QueueLock | None, gpu_blockers: list[GpuProcess]
    ) -> TrainingQueueBusyError:
        if holder is not None:
            _log.warning("ml.queue.refused", reason="training_queue_busy", holder_pid=holder.pid)
            return TrainingQueueBusyError(
                details={
                    "reason": "training_queue_busy",
                    "lock": str(self.lock_path),
                    "holder_pid": holder.pid,
                    "holder_host": holder.host,
                    "holder_job": holder.job,
                    "started_at": holder.started_at,
                    "hint": (
                        "if that job is gone (a reboot, docker kill, OOM), check nvidia-smi and "
                        "re-run with --force-queue"
                    ),
                }
            )
        pids = [process.pid for process in gpu_blockers]
        _log.warning("ml.queue.refused", reason="gpu_busy", gpu_pids=pids)
        return TrainingQueueBusyError(
            details={
                "reason": "gpu_busy",
                "lock": str(self.lock_path),
                "gpu_pids": pids,
                "gpu_processes": [
                    {"pid": p.pid, "used_memory_mb": p.used_memory_mb, "name": p.name}
                    for p in gpu_blockers
                ],
                "threshold_mb": self.gpu_busy_mb,
            }
        )


__all__ = [
    "DEFAULT_GPU_BUSY_MB",
    "DEFAULT_POLL_S",
    "QUEUE_RELATIVE_PATH",
    "GpuProcess",
    "QueueLock",
    "TrainingQueue",
    "TrainingQueueBusyError",
    "default_queue_path",
    "hostname",
    "probe_gpu_processes",
    "process_is_alive",
]
