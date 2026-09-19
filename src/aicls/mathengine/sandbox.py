"""Warm spawn worker pool for CAS execution (ADR-0016).

Why a pool of processes at all: ``sympy.integrate``/``simplify`` can run for minutes on some
inputs, and a Python **thread** cannot be stopped, so an in-process timeout leaks a burning
core per bad request. Why a *warm* pool: on Windows every process starts with ``spawn``, and
importing SymPy plus building the Lark grammar costs 1-2 seconds -- most of the 2.5 s budget --
so the import has to happen once per worker, in the initializer, not once per request.

How it works::

    parent                                  worker (spawn, one Pipe each)
    ------                                  -----------------------------
    start()  --------------------------->   import sympy, build the grammar
                                            memory cap = usage now + headroom
             <---- ("ready", probe) ------
    run(cmd) ----- ("task", payload) --->   compute(cmd)
             <---- ("ok" | "error", ...) -
    (no answer within timeout_s)
      process.kill(); restarts += 1
      respawn on a background thread

Design notes worth keeping in mind when editing this file:

* **Everything the child touches must be importable by name.** ``spawn`` re-imports the module
  in a fresh interpreter, so :func:`_worker_entry` and :func:`run_command_payload` are
  module-level functions and the payload crossing the pipe is plain JSON-compatible data
  (``cmd.model_dump(mode="json")``), never a live object.
* **One pipe per worker, discarded on kill.** After a timeout the killed worker may have left a
  half-written message in the pipe, so the connection is closed and the replacement gets a new
  one. No answer can ever be attributed to the wrong request.
* **The memory cap is relative, not absolute, and it is installed after the warm-up.** A
  Windows Job Object (``ctypes``, commit charge) or POSIX ``RLIMIT_AS`` (address space) caps the
  worker at *what it uses once it is warm* plus :data:`DEFAULT_MEMORY_HEADROOM_BYTES`. An
  absolute 1 GB cap was the design until 2026-09-08 and it was measured to be unusable on
  Linux: ``libopenblas`` (pulled in by NumPy through :mod:`aicls.mathengine.plotting`) reserves
  **128 MB of address space per CPU** at import time -- 1032 MB on this 8-core container, 2 GB
  on the 16-core workstation in the procurement spec -- while the worker's resident set stays
  at 119 MB. A 1 GB ``RLIMIT_AS`` therefore sits *below the floor*: the warm-up died with
  ``MemoryError`` and every request answered ``INTERNAL``, and when the cap was installed before
  NumPy was imported the OpenBLAS initialiser did not fail at all but spun forever, so the
  parent only noticed after ``WORKER_START_TIMEOUT_S``. Measuring the baseline instead of
  guessing it removes the whole class: the cap adapts to core count, BLAS build and container.
  Measured on this host (``codex-research-cu130``, 8 CPU, numpy 2.3.4): warm baseline
  ``VmSize`` 1329 MB, cap 2353 MB, a runaway 3 GiB allocation stopped with ``MemoryError``.
  It is still best effort -- a locked-down host may refuse both mechanisms, and then the real
  protections are the timeout, the 2048 character input limit and worker recycling.
"""

from __future__ import annotations

from collections.abc import Sequence
import contextlib
from dataclasses import dataclass, field
import importlib
import multiprocessing
from multiprocessing.process import BaseProcess
import os
import queue
import sys
import threading
import time
from typing import Any, Protocol, cast

from aicls.core.errors import AiclsError, InternalError
from aicls.core.logging import get_logger
from aicls.core.settings import get_settings
from aicls.mathengine.errors import CasTimeout, error_to_payload, payload_to_error
from aicls.mathengine.models import COMMAND_ADAPTER, CasCommand, Result, check_input_size

#: Memory a worker may take **on top of its warm baseline** before the OS refuses it
#: (ADR-0016: 1 GB). See the module docstring for why the cap is relative on both platforms.
DEFAULT_MEMORY_HEADROOM_BYTES = 1 << 30

#: Historic name of :data:`DEFAULT_MEMORY_HEADROOM_BYTES`, kept because it is re-exported and
#: quoted in ADR-0016. The number is unchanged; what changed is that 1 GB is now head-room over
#: a measured baseline rather than the worker's whole address space.
DEFAULT_MEMORY_LIMIT_BYTES = DEFAULT_MEMORY_HEADROOM_BYTES

#: Page size assumed when ``os.sysconf`` cannot be asked (it is 4096 on every platform this
#: project runs on; only used to turn ``/proc/self/statm`` pages into bytes).
_FALLBACK_PAGE_SIZE = 4096

#: Requests a worker serves before it is retired and replaced (guards slow memory growth).
DEFAULT_MAX_TASKS_PER_WORKER = 500

#: How long :meth:`PoolExecutor.start` waits for a worker to finish its warm-up.
WORKER_START_TIMEOUT_S = 90.0

#: How long :meth:`PoolExecutor.run` waits for a free worker before giving up.
ACQUIRE_TIMEOUT_S = 30.0

#: How long a killed or retired worker is joined for before it is left to the OS.
JOIN_TIMEOUT_S = 5.0

#: Dotted ``module:function`` path of the worker entry point. Overridable so that fault
#: injection tests can drive a worker that never answers (and so that a future back end -- a
#: Mathematica bridge, say -- can be dropped in without touching the pool).
DEFAULT_TARGET = "aicls.mathengine.sandbox:run_command_payload"

_logger = get_logger(__name__)


class PipeEnd(Protocol):
    """One end of the duplex pipe to a worker.

    ``multiprocessing`` hands back a ``Connection`` on POSIX and a ``PipeConnection`` on
    Windows. The two are siblings rather than subclasses, so the surface the pool relies on is
    described structurally instead of naming either concrete class.
    """

    def send(self, obj: Any) -> None:
        """Send one picklable object."""
        ...

    def recv(self) -> Any:
        """Receive one object, blocking until it arrives."""
        ...

    def poll(self, timeout: float | None = ...) -> bool:
        """Whether an object is ready to be received within ``timeout`` seconds."""
        ...

    def close(self) -> None:
        """Close this end of the pipe."""
        ...

    @property
    def closed(self) -> bool:
        """Whether this end has been closed."""
        ...


class WorkerTarget(Protocol):
    """What the worker loop calls for every task: JSON in, JSON out."""

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle one task payload and return the reply payload."""
        ...


@dataclass(frozen=True, slots=True)
class WorkerProbe:
    """What a worker reports about itself once it is warm.

    Attributes:
        pid: Operating-system process id.
        sympy_imported: Whether ``sympy`` is in the worker's ``sys.modules``.
        grammar_ready: Whether the Lark parser has been built in the worker.
        memory_limit_applied: Whether the best-effort memory cap could be installed.
        memory_limit_bytes: The cap actually installed (warm baseline + head-room), or 0 when
            none is in force. Reported so that "the sandbox has a memory limit" is an
            observable fact rather than a claim in a document.
        warm_ms: Milliseconds the warm-up took.
        tasks_done: Number of tasks this worker has served.
    """

    pid: int
    sympy_imported: bool
    grammar_ready: bool
    memory_limit_applied: bool
    warm_ms: float
    memory_limit_bytes: int = 0
    tasks_done: int = 0


# ---------------------------------------------------------------------------------------------
# worker side
# ---------------------------------------------------------------------------------------------
def process_memory_bytes() -> int | None:
    """This process's memory usage **in the unit the platform's cap counts**.

    The two mechanisms do not count the same thing, so neither does this function:

    * POSIX -- virtual address space (``/proc/self/statm`` field 0), what ``RLIMIT_AS`` limits.
    * Windows -- private commit charge (``PROCESS_MEMORY_COUNTERS_EX.PrivateUsage``), what a Job
      Object's ``ProcessMemoryLimit`` limits.

    Returns:
        The current usage in bytes, or None where it cannot be read (a kernel without
        ``/proc``, a refused ``psapi`` call). None means "do not install a relative cap": a cap
        computed from a guessed baseline is how the Linux warm-up failure of 2026-09-08
        happened, and no cap at all is at least honest about it.
    """
    if os.name == "nt":
        return _windows_private_usage()
    return _posix_address_space()


def _posix_address_space() -> int | None:
    """Virtual address size of this process from ``/proc/self/statm``, in bytes."""
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            pages = int(handle.read().split()[0])
    except (OSError, ValueError, IndexError) as exc:  # pragma: no cover - no /proc (macOS, BSD)
        _logger.debug("mathengine.sandbox.statm_unavailable", error=type(exc).__name__)
        return None
    # Fetched by name: ``os.sysconf`` is POSIX-only, and a direct call makes the Windows type
    # check fail on a branch that never runs there.
    sysconf = getattr(os, "sysconf", None)
    try:
        page_size = int(sysconf("SC_PAGE_SIZE")) if sysconf is not None else _FALLBACK_PAGE_SIZE
    except (OSError, ValueError):  # pragma: no cover - sysconf always answers on Linux
        page_size = _FALLBACK_PAGE_SIZE
    return pages * page_size


def install_memory_cap(headroom_bytes: int) -> tuple[bool, int]:
    """Cap this process at *what it uses right now* plus ``headroom_bytes``, best effort.

    Call this **after** the warm-up imports, from inside the worker: the baseline then includes
    SymPy, the Lark tables, Matplotlib and -- on Linux -- the ~128 MB per CPU that OpenBLAS
    reserves, none of which a computation should have to pay for out of its own budget.

    Args:
        headroom_bytes: What a computation may add on top of the warm baseline; values below 1
            disable the cap entirely.

    Returns:
        ``(installed, cap_bytes)``; ``(False, 0)`` when the cap was disabled, the baseline could
        not be measured or the OS refused it.
    """
    if headroom_bytes < 1:
        return False, 0
    baseline = process_memory_bytes()
    if baseline is None:
        _logger.warning("mathengine.sandbox.memory_baseline_unavailable", os_name=os.name)
        return False, 0
    cap = baseline + headroom_bytes
    if not apply_memory_limit(cap):
        return False, 0
    _logger.debug("mathengine.sandbox.memory_cap", baseline=baseline, cap=cap)
    return True, cap


def apply_memory_limit(limit_bytes: int) -> bool:
    """Cap this process's memory at ``limit_bytes`` absolutely, best effort.

    Windows has no ``setrlimit``; a Job Object with ``ProcessMemoryLimit`` is the closest
    equivalent and is installed through ``ctypes``. On POSIX ``RLIMIT_AS`` is used. Failure is
    expected on locked-down hosts and is reported, not raised.

    Only the **soft** limit is lowered: the hard limit is left where the parent put it, so the
    call is reversible inside the process (a test can restore it) and an outer sandbox keeps
    the last word. A soft limit that is already stricter than ``limit_bytes`` is left alone and
    reported as installed -- it is, just by somebody else.

    Args:
        limit_bytes: The cap; values below 1 disable the attempt. Prefer
            :func:`install_memory_cap`, which computes this number by measurement.

    Returns:
        True when a limit is in force.
    """
    if limit_bytes < 1:
        return False
    if os.name == "nt":
        return _apply_windows_job_limit(limit_bytes)
    try:
        # Imported dynamically: ``resource`` is POSIX-only, and a static import would make a
        # type check on Windows fail on a branch that never runs there.
        resource = importlib.import_module("resource")
        infinity = resource.RLIM_INFINITY
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        target = limit_bytes if hard == infinity else min(limit_bytes, hard)
        if soft != infinity and soft <= target:
            _logger.debug("mathengine.sandbox.rlimit_already_stricter", soft=soft, wanted=target)
            return True
        resource.setrlimit(resource.RLIMIT_AS, (target, hard))
    except Exception as exc:
        _logger.debug("mathengine.sandbox.rlimit_failed", error=type(exc).__name__)
        return False
    return True


def _declare(dll: Any, name: str, argtypes: tuple[Any, ...], restype: Any) -> None:
    """Give one ``ctypes`` entry point its real prototype.

    Without this, ctypes assumes ``c_int`` for the result, which truncates every 64-bit
    ``HANDLE`` the Win32 API returns -- the defect that kept the sandbox's memory cap from ever
    installing on Windows. Failures are suppressed so that a stubbed ``kernel32`` (the
    fault-injection tests) still behaves like the real one.
    """
    function = getattr(dll, name)
    with contextlib.suppress(AttributeError, TypeError):
        function.argtypes = argtypes
        function.restype = restype


def _windows_private_usage() -> int | None:
    """Commit charge of this process (``PrivateUsage``), the quantity a Job Object limits."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCountersEx(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            # K32GetProcessMemoryInfo lives in kernel32 since Windows 7; psapi.dll only forwards to
            # it. Going straight to kernel32 avoids the psapi/psapi_version split entirely.
            # The prototypes are declared: ctypes' default `c_int` return would truncate the 64-bit
            # pseudo-handle, and the call then fails with ERROR_INVALID_HANDLE.
            _declare(kernel32, "GetCurrentProcess", (), wintypes.HANDLE)
            _declare(
                kernel32,
                "K32GetProcessMemoryInfo",
                (wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCountersEx), wintypes.DWORD),
                wintypes.BOOL,
            )
            counters = ProcessMemoryCountersEx()
            counters.cb = ctypes.sizeof(ProcessMemoryCountersEx)
            ok = kernel32.K32GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            )
            if not ok:
                _logger.debug(
                    "mathengine.sandbox.private_usage_failed", error=ctypes.get_last_error()
                )
                return None
            return int(counters.PrivateUsage)
        except Exception as exc:  # pragma: no cover - only on a host without kernel32
            _logger.debug("mathengine.sandbox.private_usage_failed", error=type(exc).__name__)
            return None
    else:  # pragma: no cover - the dispatcher tests os.name and never sends POSIX here
        # The if/else is for mypy: it skips whichever branch sys.platform rules out, so
        # ``ctypes.WinDLL`` (absent on Linux) is checked on Windows and ignored on Linux.
        return None


#: Job Object handles are kept alive for the life of the process: closing the last handle to a
#: job that has no kill-on-close flag simply detaches it, and we want the limit to persist.
_JOB_HANDLES: list[int] = []


def _apply_windows_job_limit(limit_bytes: int) -> bool:
    """Install a Windows Job Object memory limit on the current process."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class IoCounters(ctypes.Structure):
                _fields_ = [
                    ("ReadOperationCount", ctypes.c_ulonglong),
                    ("WriteOperationCount", ctypes.c_ulonglong),
                    ("OtherOperationCount", ctypes.c_ulonglong),
                    ("ReadTransferCount", ctypes.c_ulonglong),
                    ("WriteTransferCount", ctypes.c_ulonglong),
                    ("OtherTransferCount", ctypes.c_ulonglong),
                ]

            class BasicLimitInformation(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class ExtendedLimitInformation(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", BasicLimitInformation),
                    ("IoInfo", IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            job_object_extended_limit_information = 9
            job_object_limit_process_memory = 0x00000100

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            # The prototypes are declared, and that is not cosmetic: with ctypes' default
            # `c_int` result, GetCurrentProcess()'s pseudo-handle (HANDLE)-1 came back as the
            # Python int -1 and reached AssignProcessToJobObject as a 32-bit value, which failed
            # every time with ERROR_INVALID_HANDLE (6). The cap was therefore never installed on
            # Windows -- silently, because ADR-0016 calls it best effort and the call only ever
            # returned False. Measured 2026-09-08; with these prototypes all three calls succeed.
            _declare(
                kernel32, "CreateJobObjectW", (wintypes.LPVOID, wintypes.LPCWSTR), wintypes.HANDLE
            )
            _declare(
                kernel32,
                "SetInformationJobObject",
                (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD),
                wintypes.BOOL,
            )
            _declare(
                kernel32,
                "AssignProcessToJobObject",
                (wintypes.HANDLE, wintypes.HANDLE),
                wintypes.BOOL,
            )
            _declare(kernel32, "GetCurrentProcess", (), wintypes.HANDLE)
            _declare(kernel32, "CloseHandle", (wintypes.HANDLE,), wintypes.BOOL)

            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                return False
            info = ExtendedLimitInformation()
            info.BasicLimitInformation.LimitFlags = job_object_limit_process_memory
            info.ProcessMemoryLimit = limit_bytes
            ok = kernel32.SetInformationJobObject(
                handle,
                job_object_extended_limit_information,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not ok:
                kernel32.CloseHandle(handle)
                return False
            if not kernel32.AssignProcessToJobObject(handle, kernel32.GetCurrentProcess()):
                _logger.debug("mathengine.sandbox.assign_job_failed", error=ctypes.get_last_error())
                kernel32.CloseHandle(handle)
                return False
            _JOB_HANDLES.append(int(handle))
        except Exception as exc:
            _logger.debug("mathengine.sandbox.job_object_failed", error=type(exc).__name__)
            return False
        return True
    else:  # pragma: no cover - the dispatcher tests os.name and never sends POSIX here
        # The if/else is for mypy: it skips whichever branch sys.platform rules out, so
        # ``ctypes.WinDLL`` (absent on Linux) is checked on Windows and ignored on Linux.
        return False


def resolve_target(dotted: str) -> WorkerTarget:
    """Import the ``"module:function"`` worker entry point named by ``dotted``.

    Args:
        dotted: A ``package.module:function`` string.

    Returns:
        The callable.

    Raises:
        aicls.core.errors.InternalError: If the string is malformed or the import fails.
    """
    module_name, separator, function_name = dotted.partition(":")
    if not separator or not module_name or not function_name:
        raise InternalError(details={"target": dotted, "reason": "expected 'module:function'"})
    try:
        module = importlib.import_module(module_name)
        target = getattr(module, function_name)
    except Exception as exc:
        raise InternalError(
            details={"target": dotted, "error": type(exc).__name__, "message": str(exc)[:200]}
        ) from exc
    if not callable(target):
        raise InternalError(details={"target": dotted, "reason": "not callable"})
    return cast(WorkerTarget, target)


def run_command_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one serialised command in the worker (the default :data:`DEFAULT_TARGET`).

    Args:
        payload: ``cmd.model_dump(mode="json")`` of a
            :data:`~aicls.mathengine.models.CasCommand`.

    Returns:
        ``result.model_dump()`` of the computed :class:`~aicls.mathengine.models.Result`.

    Raises:
        aicls.core.errors.AiclsError: Every failure mode of :func:`aicls.mathengine.ops.compute`;
            the worker loop converts it into an ``("error", ...)`` reply.
    """
    from aicls.mathengine.ops import compute

    command: CasCommand = COMMAND_ADAPTER.validate_python(payload)
    check_input_size(command)
    return compute(command).model_dump()


def _warm_up() -> tuple[bool, bool, float]:
    """Import SymPy, build the Lark grammar and load the plot backend.

    This is the whole point of a *warm* pool (ADR-0016): importing SymPy and building the LALR
    parser costs 1-2 s under ``spawn``, which is most of the 2.5 s request budget. Matplotlib is
    loaded here too, because :mod:`aicls.mathengine.plotting` deliberately imports it lazily.

    Returns:
        ``(sympy imported, grammar built, milliseconds spent)``.
    """
    started = time.perf_counter()
    grammar_ready = False
    try:
        import sympy  # noqa: F401 - imported for its side effect: the warm-up itself

        from aicls.latex2sympy.parser import lark_parser
        from aicls.mathengine.ops import compute  # noqa: F401 - the per-task import, paid once
        from aicls.mathengine.plotting import load_backend

        lark_parser()
        load_backend()
        grammar_ready = True
    except Exception as exc:
        _logger.warning(
            "mathengine.sandbox.warmup_failed", error=type(exc).__name__, message=str(exc)[:200]
        )
    return "sympy" in sys.modules, grammar_ready, (time.perf_counter() - started) * 1000.0


def worker_loop(
    conn: PipeEnd,
    *,
    memory_headroom_bytes: int = DEFAULT_MEMORY_HEADROOM_BYTES,
    target: str = DEFAULT_TARGET,
) -> None:
    """Warm up, install the memory cap, announce readiness, then serve tasks until told to stop.

    Runs in the spawned child; it is also called directly (on a thread) by the unit tests, which
    is why it takes a connection instead of creating one.

    The cap goes on **after** the warm-up on purpose. Under ``spawn`` the child re-imports
    :mod:`aicls.mathengine`, and that alone pulls in NumPy and its BLAS; a cap installed before
    those imports has to be big enough to cover a floor nobody measured, which is exactly the
    bug this ordering removes. Nothing user-controlled is imported during the warm-up -- the
    payload is a validated :class:`~aicls.mathengine.models.CasCommand`, never code -- so the
    uncapped window covers our own modules only.

    Protocol -- parent to worker: ``("task", payload)``, ``("probe", None)``, ``("stop", None)``.
    Worker to parent: ``("ready", probe_dict)`` once, then ``("ok", result_dict)``,
    ``("error", error_payload)`` or ``("probe", probe_dict)`` per message.

    Args:
        conn: The worker's end of the duplex pipe.
        memory_headroom_bytes: What a computation may allocate on top of the warm baseline;
            0 disables the cap.
        target: Dotted path of the per-task callable.
    """
    try:
        handler = resolve_target(target)
    except AiclsError as exc:
        conn.send(("error", error_to_payload(exc)))
        return
    sympy_imported, grammar_ready, warm_ms = _warm_up()
    memory_limit_applied, memory_limit_bytes = install_memory_cap(memory_headroom_bytes)
    tasks_done = 0

    def probe() -> dict[str, Any]:
        return {
            "pid": os.getpid(),
            "sympy_imported": sympy_imported,
            "grammar_ready": grammar_ready,
            "memory_limit_applied": memory_limit_applied,
            "memory_limit_bytes": memory_limit_bytes,
            "warm_ms": warm_ms,
            "tasks_done": tasks_done,
        }

    conn.send(("ready", probe()))
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            # The parent went away (or killed us mid-wait); nothing left to serve.
            return
        kind, body = message
        if kind == "stop":
            return
        if kind == "probe":
            conn.send(("probe", probe()))
            continue
        try:
            reply: tuple[str, dict[str, Any]] = ("ok", handler(body))
        except AiclsError as exc:
            reply = ("error", error_to_payload(exc))
        except Exception as exc:
            # A worker must always answer; an unexpected failure becomes an INTERNAL payload.
            internal = InternalError(
                details={"error": type(exc).__name__, "message": str(exc)[:200]}
            )
            reply = ("error", error_to_payload(internal))
        tasks_done += 1
        try:
            conn.send(reply)
        except (BrokenPipeError, OSError):
            return


def _worker_entry(conn: PipeEnd, memory_headroom_bytes: int, target: str) -> None:
    """``multiprocessing`` entry point: run :func:`worker_loop`, then close the pipe."""
    try:
        worker_loop(conn, memory_headroom_bytes=memory_headroom_bytes, target=target)
    finally:
        with contextlib.suppress(OSError):
            conn.close()


# ---------------------------------------------------------------------------------------------
# parent side
# ---------------------------------------------------------------------------------------------
@dataclass
class _Worker:
    """A live worker process and the parent's end of its pipe."""

    process: BaseProcess
    conn: PipeEnd
    probe: WorkerProbe
    tasks_done: int = 0
    closed: bool = field(default=False)

    def close(self, *, kill: bool) -> None:
        """Stop the worker: kill it outright, or ask it to finish and leave."""
        if self.closed:
            return
        self.closed = True
        try:
            if kill:
                self.process.kill()
            elif self.process.is_alive():
                self.conn.send(("stop", None))
        except (OSError, ValueError, BrokenPipeError, EOFError) as exc:
            _logger.debug("mathengine.sandbox.close_failed", error=type(exc).__name__)
        with contextlib.suppress(OSError):
            self.conn.close()
        self.process.join(timeout=JOIN_TIMEOUT_S)
        if self.process.is_alive():  # pragma: no cover - only on a wedged OS handle
            self.process.kill()
            self.process.join(timeout=JOIN_TIMEOUT_S)


class PoolExecutor:
    """A warm pool of spawned SymPy workers (production executor, ADR-0016).

    Args:
        workers: Number of workers; defaults to ``settings.sandbox_workers``.
        timeout_s: Default per-task budget; defaults to ``settings.compute_timeout_s``.
            :meth:`run` takes the budget as an argument, so this is only the fallback.
        max_tasks_per_worker: Requests a worker serves before it is recycled.
        memory_headroom_bytes: Best-effort per-worker memory cap, expressed as head-room over
            the worker's warm baseline; 0 disables it.
        target: Dotted ``module:function`` path the workers call per task.
        start_timeout_s: How long to wait for each worker's warm-up.
        acquire_timeout_s: How long :meth:`run` waits for a free worker.

    Attributes:
        restarts: How many workers were killed after a timeout and respawned. Exposed by
            ``GET /health/ready`` and included in the latency report (ADR-0016).
        recycles: How many workers were retired after ``max_tasks_per_worker`` tasks.

    Example:
        >>> from aicls.mathengine import Diff, PoolExecutor  # doctest: +SKIP
        >>> with PoolExecutor(workers=1) as pool:  # doctest: +SKIP
        ...     pool.run(Diff(op="diff", expr="x^{2}"), 2.5).latex
        '2 \\\\cdot x'
    """

    def __init__(
        self,
        *,
        workers: int | None = None,
        timeout_s: float | None = None,
        max_tasks_per_worker: int = DEFAULT_MAX_TASKS_PER_WORKER,
        memory_headroom_bytes: int = DEFAULT_MEMORY_HEADROOM_BYTES,
        target: str = DEFAULT_TARGET,
        start_timeout_s: float = WORKER_START_TIMEOUT_S,
        acquire_timeout_s: float = ACQUIRE_TIMEOUT_S,
    ) -> None:
        settings = get_settings()
        self._workers = max(1, workers if workers is not None else settings.sandbox_workers)
        self._timeout_s = timeout_s if timeout_s is not None else settings.compute_timeout_s
        self._max_tasks = max(1, max_tasks_per_worker)
        self._memory_headroom_bytes = memory_headroom_bytes
        self._target = target
        self._start_timeout_s = start_timeout_s
        self._acquire_timeout_s = acquire_timeout_s
        self._context = multiprocessing.get_context("spawn")
        self._idle: queue.Queue[_Worker] = queue.Queue()
        self._live: list[_Worker] = []
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._started = False
        self.restarts = 0
        self.recycles = 0

    # -- lifecycle ---------------------------------------------------------------------------
    @property
    def workers(self) -> int:
        """Configured number of workers."""
        return self._workers

    @property
    def timeout_s(self) -> float:
        """Fallback per-task budget in seconds."""
        return self._timeout_s

    @property
    def started(self) -> bool:
        """Whether :meth:`start` has run and :meth:`stop` has not."""
        return self._started

    @property
    def prewarm_diagnostics(self) -> list[WorkerProbe]:
        """The :class:`WorkerProbe` each worker reported *before* serving any task."""
        with self._lock:
            return [worker.probe for worker in self._live]

    def start(self) -> None:
        """Spawn and warm every worker; idempotent.

        Raises:
            aicls.core.errors.InternalError: If a worker fails to warm up in
                ``start_timeout_s`` seconds or dies during warm-up.
        """
        if self._started:
            return
        for _ in range(self._workers):
            worker = self._spawn()
            with self._lock:
                self._live.append(worker)
            self._idle.put(worker)
        self._started = True
        _logger.info("mathengine.sandbox.started", workers=self._workers, timeout_s=self._timeout_s)

    def stop(self) -> None:
        """Ask every worker to exit, then join it; idempotent."""
        self._started = False
        for thread in list(self._threads):
            thread.join(timeout=JOIN_TIMEOUT_S)
        self._threads.clear()
        with self._lock:
            workers, self._live = list(self._live), []
        for worker in workers:
            worker.close(kill=False)
        while True:
            try:
                self._idle.get_nowait()
            except queue.Empty:
                break
        _logger.info("mathengine.sandbox.stopped", restarts=self.restarts)

    def __enter__(self) -> PoolExecutor:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # -- worker management -------------------------------------------------------------------
    def _spawn(self) -> _Worker:
        parent_conn, child_conn = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_worker_entry,
            args=(child_conn, self._memory_headroom_bytes, self._target),
            name="aicls-cas-worker",
            daemon=True,
        )
        process.start()
        child_conn.close()  # the parent keeps only its own end
        try:
            if not parent_conn.poll(self._start_timeout_s):
                raise InternalError(
                    details={"reason": "worker_warmup_timeout", "timeout_s": self._start_timeout_s}
                )
            kind, body = parent_conn.recv()
        except (EOFError, OSError) as exc:
            process.kill()
            raise InternalError(details={"reason": "worker_died_during_warmup"}) from exc
        except InternalError:
            process.kill()
            raise
        if kind == "error":
            process.kill()
            raise payload_to_error(body)
        probe = WorkerProbe(**body)
        _logger.debug("mathengine.sandbox.worker_ready", pid=probe.pid, warm_ms=probe.warm_ms)
        return _Worker(process=process, conn=parent_conn, probe=probe)

    def _replace_async(self, dead: _Worker, *, kill: bool = True) -> None:
        """Retire ``dead`` and spawn its replacement on a background thread (ADR-0016).

        Args:
            dead: The worker to retire.
            kill: True after a timeout or a broken pipe (the worker is mid-computation and must
                be stopped); False when recycling a healthy worker, which is asked to exit.
        """

        def _work() -> None:
            dead.close(kill=kill)
            if not self._started:
                return
            try:
                worker = self._spawn()
            except AiclsError as exc:
                _logger.error("mathengine.sandbox.respawn_failed", error=str(exc))
                return
            with self._lock:
                self._live.append(worker)
            self._idle.put(worker)

        with self._lock:
            self._live = [worker for worker in self._live if worker is not dead]
        thread = threading.Thread(target=_work, name="aicls-cas-respawn", daemon=True)
        self._threads = [item for item in self._threads if item.is_alive()]
        self._threads.append(thread)
        thread.start()

    def _acquire(self) -> _Worker:
        if not self._started:
            self.start()
        try:
            return self._idle.get(timeout=self._acquire_timeout_s)
        except queue.Empty as exc:
            raise CasTimeout(self._acquire_timeout_s, details={"reason": "no_free_worker"}) from exc

    def _release(self, worker: _Worker) -> None:
        if worker.tasks_done >= self._max_tasks:
            with self._lock:
                self.recycles += 1
            _logger.info("mathengine.sandbox.recycle", pid=worker.probe.pid)
            self._replace_async(worker, kill=False)
            return
        self._idle.put(worker)

    # -- execution ---------------------------------------------------------------------------
    def run(self, cmd: CasCommand, timeout_s: float | None = None) -> Result:
        """Execute ``cmd`` on a warm worker (see :meth:`Executor.run`).

        Args:
            cmd: The validated command.
            timeout_s: Budget in seconds; defaults to :attr:`timeout_s`.

        Returns:
            The computed :class:`~aicls.mathengine.models.Result`.

        Raises:
            aicls.mathengine.errors.CasTimeout: If the worker does not answer in time. It is
                killed, :attr:`restarts` is incremented and a replacement is spawned
                asynchronously, so the *next* request finds a warm pool again.
            aicls.core.errors.AiclsError: Whatever the computation raised, rebuilt from the
                worker's error payload.
        """
        check_input_size(cmd)
        budget = self._timeout_s if timeout_s is None else timeout_s
        worker = self._acquire()
        payload = cmd.model_dump(mode="json")
        try:
            worker.conn.send(("task", payload))
            ready = worker.conn.poll(budget)
        except (OSError, BrokenPipeError, ValueError) as exc:
            with self._lock:
                self.restarts += 1
            self._replace_async(worker)
            raise InternalError(
                details={"reason": "worker_pipe_broken", "op": cmd.op, "error": type(exc).__name__}
            ) from exc
        if not ready:
            with self._lock:
                self.restarts += 1
            _logger.warning(
                "mathengine.sandbox.timeout", op=cmd.op, timeout_s=budget, pid=worker.probe.pid
            )
            self._replace_async(worker)
            raise CasTimeout(
                budget, details={"op": cmd.op, "executor": "pool", "restarts": self.restarts}
            )
        try:
            kind, body = worker.conn.recv()
        except (EOFError, OSError) as exc:
            with self._lock:
                self.restarts += 1
            self._replace_async(worker)
            raise InternalError(
                details={"reason": "worker_died", "op": cmd.op, "error": type(exc).__name__}
            ) from exc
        worker.tasks_done += 1
        self._release(worker)
        if kind == "error":
            raise payload_to_error(body)
        return Result.model_validate(body)

    def probe(self, timeout_s: float = WORKER_START_TIMEOUT_S) -> WorkerProbe:
        """Ask one worker to describe its live state (used by the readiness endpoint).

        Args:
            timeout_s: How long to wait for the answer.

        Returns:
            The worker's current :class:`WorkerProbe`.

        Raises:
            aicls.mathengine.errors.CasTimeout: If no worker answers in time.
        """
        worker = self._acquire()
        try:
            worker.conn.send(("probe", None))
            if not worker.conn.poll(timeout_s):
                raise CasTimeout(timeout_s, details={"reason": "probe_timeout"})
            kind, body = worker.conn.recv()
        except (EOFError, OSError, BrokenPipeError) as exc:
            self._replace_async(worker)
            raise InternalError(details={"reason": "probe_failed"}) from exc
        except CasTimeout:
            with self._lock:
                self.restarts += 1
            self._replace_async(worker)
            raise
        self._idle.put(worker)
        if kind == "error":
            raise payload_to_error(body)
        return WorkerProbe(**body)


#: ``SandboxPoolExecutor`` is the name used in the İP-4 work-package notes; ADR-0016 and
#: docs/mimari/moduller.md call the same class ``PoolExecutor``. Both are supported.
SandboxPoolExecutor = PoolExecutor


def pool_from_settings(**overrides: Any) -> PoolExecutor:
    """Build a :class:`PoolExecutor` from ``AICLS_*`` settings (the FastAPI lifespan helper).

    Args:
        **overrides: Any :class:`PoolExecutor` keyword argument, to override the settings.

    Returns:
        An unstarted pool.
    """
    settings = get_settings()
    kwargs: dict[str, Any] = {
        "workers": settings.sandbox_workers,
        "timeout_s": settings.compute_timeout_s,
    }
    kwargs.update(overrides)
    return PoolExecutor(**kwargs)


def worker_probe_fields() -> Sequence[str]:
    """Field names of :class:`WorkerProbe` (used by the readiness payload schema)."""
    return tuple(WorkerProbe.__dataclass_fields__)


__all__ = [
    "ACQUIRE_TIMEOUT_S",
    "DEFAULT_MAX_TASKS_PER_WORKER",
    "DEFAULT_MEMORY_HEADROOM_BYTES",
    "DEFAULT_MEMORY_LIMIT_BYTES",
    "DEFAULT_TARGET",
    "JOIN_TIMEOUT_S",
    "WORKER_START_TIMEOUT_S",
    "PipeEnd",
    "PoolExecutor",
    "SandboxPoolExecutor",
    "WorkerProbe",
    "WorkerTarget",
    "apply_memory_limit",
    "install_memory_cap",
    "pool_from_settings",
    "process_memory_bytes",
    "resolve_target",
    "run_command_payload",
    "worker_loop",
    "worker_probe_fields",
]
