"""``MathEngine`` -- the object the rest of the platform talks to.

It owns three concerns and nothing else:

* **validation** -- a raw ``{"op": ...}`` mapping from the API is turned into a typed command,
  and the 2048-character input contract is enforced before anything is spawned or parsed;
* **caching** -- an LRU keyed by the canonical command (İP-4 risk plan). A hit costs a
  dictionary lookup, which is what keeps repeated classroom commands off the CPU entirely;
* **timing** -- a :class:`~aicls.core.timing.StageTimer` so ``ComputeOut.timings`` and the
  ``Server-Timing`` header report the same numbers the latency report is built from.

*Where* the computation happens is the executor's business, and that is the whole point of the
split: the same engine serves tests through :class:`~aicls.mathengine.executor.InProcessExecutor`
and requests through :class:`~aicls.mathengine.sandbox.PoolExecutor`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from aicls.core import messages_tr
from aicls.core.errors import InputValidationError
from aicls.core.logging import get_logger
from aicls.core.settings import get_settings
from aicls.core.timing import StageTimer
from aicls.mathengine.cache import LruCache
from aicls.mathengine.executor import Executor, InProcessExecutor
from aicls.mathengine.models import (
    CAS_COMMAND_TYPES,
    COMMAND_ADAPTER,
    CasCommand,
    Result,
    canonical_key,
    check_input_size,
)

_logger = get_logger(__name__)


class MathEngine:
    """Validate, cache and execute typed CAS commands.

    Args:
        executor: Where computations run; defaults to a fresh
            :class:`~aicls.mathengine.executor.InProcessExecutor` (tests and CLI). Production
            passes a started :class:`~aicls.mathengine.sandbox.PoolExecutor`.
        cache: Result cache; defaults to a new :class:`~aicls.mathengine.cache.LruCache`.
            Pass ``LruCache(maxsize=1)`` or a shared instance to control the policy; caching
            cannot be switched off, because a deterministic CAS makes stale entries impossible.
        timeout_s: Per-command budget in seconds; defaults to ``settings.compute_timeout_s``.

    Example:
        >>> from aicls.mathengine import Diff, MathEngine
        >>> MathEngine().execute(Diff(op="diff", expr="x^{2}")).plain
        '2*x'
    """

    def __init__(
        self,
        executor: Executor | None = None,
        cache: LruCache | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self._executor: Executor = executor if executor is not None else InProcessExecutor()
        self._cache = cache if cache is not None else LruCache()
        self._timeout_s = timeout_s if timeout_s is not None else get_settings().compute_timeout_s

    @property
    def executor(self) -> Executor:
        """The executor commands are run on."""
        return self._executor

    @property
    def cache(self) -> LruCache:
        """The result cache."""
        return self._cache

    @property
    def timeout_s(self) -> float:
        """Per-command budget in seconds."""
        return self._timeout_s

    def clear_cache(self) -> None:
        """Drop every cached result (used by tests and by ``aicls eval``)."""
        self._cache.clear()

    def validate(self, cmd: CasCommand | Mapping[str, Any]) -> CasCommand:
        """Coerce ``cmd`` into a typed command and enforce the input-size contract.

        Args:
            cmd: A command instance, or a raw ``{"op": ...}`` mapping from the API layer.

        Returns:
            The validated command.

        Raises:
            aicls.core.errors.InputValidationError: If the mapping does not describe a command
                of the union, if the object is not a member of :data:`CasCommand` at all, or if
                the total LaTeX payload exceeds
                :data:`~aicls.mathengine.models.MAX_INPUT_CHARS`.
        """
        if isinstance(cmd, Mapping):
            try:
                command: CasCommand = COMMAND_ADAPTER.validate_python(dict(cmd))
            except ValidationError as exc:
                raise InputValidationError(
                    messages_tr.MSG_VALIDATION_ERROR,
                    details={"errors": exc.errors(include_url=False)[:8]},
                ) from exc
        elif isinstance(cmd, CAS_COMMAND_TYPES):
            command = cmd
        else:
            # ``aicls.nlu.types.Command`` is wider than ``CasCommand``: a ``BookFetch`` or a
            # ``UiAction`` is a perfectly valid parse result that simply belongs to another
            # executor. Without this branch such an object passed straight through to the
            # worker and surfaced as an ``AttributeError`` from inside the pool -- a 500 for
            # what is really "understood, but not a computation".
            raise InputValidationError(
                messages_tr.MSG_VALIDATION_ERROR,
                details={
                    "reason": "not_a_cas_command",
                    "type": type(cmd).__name__,
                    "op": str(getattr(cmd, "op", "")),
                },
            )
        check_input_size(command)
        return command

    def execute(self, cmd: CasCommand | Mapping[str, Any]) -> Result:
        """Execute one command, serving it from the cache when possible.

        Args:
            cmd: A typed command or the raw mapping the API received.

        Returns:
            The :class:`~aicls.mathengine.models.Result`. ``cache_hit`` says whether SymPy ran;
            ``timings`` carries ``cas`` (or ``cache``) and ``total`` in milliseconds and
            ``elapsed_ms`` repeats the total.

        Raises:
            aicls.core.errors.InputValidationError: On an invalid or oversized command.
            aicls.mathengine.errors.CasTimeout: If the executor's budget is exceeded.
            aicls.mathengine.errors.UnsupportedProblem: If SymPy cannot answer the problem.
            aicls.latex2sympy.ParseError: If a LaTeX field is not in the grammar.
        """
        timer = StageTimer()
        command = self.validate(cmd)
        key = canonical_key(command)

        cached = self._cache.get(key)
        if cached is not None:
            total = timer.total_ms
            _logger.debug("mathengine.execute.cache_hit", op=command.op, total_ms=total)
            return cached.model_copy(
                update={
                    "cache_hit": True,
                    "timings": {
                        **cached.timings,
                        "cache": round(total, 3),
                        "total": round(total, 3),
                    },
                    "elapsed_ms": round(total, 3),
                }
            )

        with timer.stage("cas"):
            result = self._executor.run(command, self._timeout_s)
        total = timer.total_ms
        # ``cas`` is what the engine waited for (SymPy plus, in the sandbox, the pipe round
        # trip); ``sympy`` is what the computation itself cost inside the worker.
        timings = {"sympy": result.timings.get("cas", 0.0), **timer.stages}
        timings["total"] = round(total, 3)
        finished = result.model_copy(
            update={"cache_hit": False, "timings": timings, "elapsed_ms": round(total, 3)}
        )
        self._cache.put(key, finished)
        _logger.debug("mathengine.execute.done", op=command.op, kind=finished.kind, ms=total)
        return finished


__all__ = ["MathEngine"]
