"""The executor protocol and the in-process implementation (ADR-0016).

An :class:`Executor` is the *where* of a computation; :class:`~aicls.mathengine.MathEngine` is
the *what*. Two implementations exist:

* :class:`InProcessExecutor` -- tests and the CLI. It runs SymPy on a worker **thread** so that
  a budget can still be measured, but a thread cannot be stopped: when the budget expires the
  call returns :class:`~aicls.mathengine.errors.CasTimeout` while the computation keeps
  burning a core in the background. That is precisely the failure mode ADR-0016 rejects for
  the server, and the reason the class carries this warning instead of pretending otherwise.
* :class:`~aicls.mathengine.sandbox.PoolExecutor` -- production. A killed process really stops.

The thread is discarded (not reused) after a timeout so the next call does not queue behind the
runaway computation.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
import threading
from typing import Protocol, runtime_checkable

from aicls.core.logging import get_logger
from aicls.mathengine.errors import CasTimeout
from aicls.mathengine.models import CasCommand, Result
from aicls.mathengine.ops import compute

_logger = get_logger(__name__)


@runtime_checkable
class Executor(Protocol):
    """Where a :data:`~aicls.mathengine.models.CasCommand` is actually computed."""

    def run(self, cmd: CasCommand, timeout_s: float) -> Result:
        """Compute ``cmd``, giving up after ``timeout_s`` seconds.

        Args:
            cmd: The validated command.
            timeout_s: Wall-clock budget in seconds.

        Returns:
            The computed :class:`~aicls.mathengine.models.Result`.

        Raises:
            aicls.mathengine.errors.CasTimeout: If the budget is exceeded.
        """
        ...


class InProcessExecutor:
    """Run SymPy in this process (tests and CLI).

    Args:
        enforce_timeout: When True (default) the computation runs on a worker thread and the
            budget is enforced *for the caller*. When False the command runs inline on the
            calling thread with no budget at all -- which is what a sandbox worker wants, since
            the parent process already owns the timeout.

    Warning:
        A Python thread cannot be cancelled. With ``enforce_timeout=True`` a timed-out
        computation keeps running until SymPy finishes; only
        :class:`~aicls.mathengine.sandbox.PoolExecutor` can really stop one. Never use this
        class to serve requests.
    """

    def __init__(self, *, enforce_timeout: bool = True) -> None:
        self._enforce_timeout = enforce_timeout
        self._pool: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()
        self.abandoned = 0

    @property
    def enforce_timeout(self) -> bool:
        """Whether :meth:`run` enforces the budget on a worker thread."""
        return self._enforce_timeout

    def start(self) -> None:
        """Pay the cold imports now rather than inside the first request's budget.

        The pool does exactly this in every worker (``sandbox._warm_up``: SymPy, the Lark
        grammar, the Matplotlib backend), so a production request never meets a cold
        interpreter. This class used to skip it, and the first ``plot`` of a test process then
        paid the Matplotlib import against a 2 s budget: fine on the workstation, a
        ``CAS_TIMEOUT`` on GitHub's two-vCPU runner with two xdist workers on it (2026-09-10).
        Idempotent: a second call finds everything in ``sys.modules``.
        """
        from aicls.mathengine.sandbox import _warm_up

        _warm_up()

    def stop(self) -> None:
        """Shut the worker thread down; safe to call more than once."""
        with self._lock:
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def _submit(self, cmd: CasCommand) -> Future[Result]:
        with self._lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="aicls-cas-inproc"
                )
            return self._pool.submit(compute, cmd)

    def run(self, cmd: CasCommand, timeout_s: float) -> Result:
        """Compute ``cmd`` (see :meth:`Executor.run`).

        Raises:
            aicls.mathengine.errors.CasTimeout: If ``timeout_s`` elapses first. The abandoned
                thread is dropped and :attr:`abandoned` is incremented.
        """
        if not self._enforce_timeout:
            return compute(cmd)
        future = self._submit(cmd)
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeoutError as exc:
            self.abandoned += 1
            self.stop()  # the runaway thread keeps the old pool alive; the next call gets a new one
            _logger.warning(
                "mathengine.inprocess.timeout", op=cmd.op, timeout_s=timeout_s, abandoned=True
            )
            raise CasTimeout(timeout_s, details={"op": cmd.op, "executor": "in_process"}) from exc

    def __enter__(self) -> InProcessExecutor:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


__all__ = ["Executor", "InProcessExecutor"]
