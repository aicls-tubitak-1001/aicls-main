"""Stage timer for latency accounting (ADR-0014, ADR-0016).

``PipelineService`` wraps ``nlu`` / ``parse`` / ``cas`` / ``render`` in
``with timer.stage(name):`` blocks; the API layer emits :meth:`StageTimer.server_timing_header`
(RFC 9211 ``Server-Timing``) and stores :attr:`StageTimer.stages` in ``operations.timings``.
Stages may nest (the outer stage includes the inner ones) and repeat (durations accumulate).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import re
import time

from aicls.core import messages_tr
from aicls.core.errors import InputValidationError

_STAGE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")


class StageTimer:
    """Accumulates per-stage wall-clock durations in milliseconds.

    Args:
        clock: Monotonic seconds source (default :func:`time.perf_counter`); injectable so tests
            are deterministic.

    Example:
        >>> timer = StageTimer()
        >>> with timer.stage("nlu"):
        ...     pass
        >>> sorted(timer.stages) == ["nlu"]
        True
    """

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock: Callable[[], float] = clock if clock is not None else time.perf_counter
        self._stages: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._stack: list[str] = []
        self._start = self._clock()

    @staticmethod
    def _check_name(name: str) -> str:
        if not isinstance(name, str) or not _STAGE_NAME_RE.match(name):
            raise InputValidationError(
                messages_tr.msg("INVALID_STAGE_NAME", name=name), details={"name": name}
            )
        return name

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the enclosed block under ``name`` (nestable; repeated names accumulate)."""
        self._check_name(name)
        self._stages.setdefault(name, 0.0)  # reserve first-seen order (outer before inner)
        self._stack.append(name)
        started = self._clock()
        try:
            yield
        finally:
            elapsed_ms = (self._clock() - started) * 1000.0
            self._stack.pop()
            self._record(name, elapsed_ms)

    def add(self, name: str, ms: float) -> None:
        """Record an externally measured duration (e.g. client ``render_ms``).

        Raises:
            InputValidationError: On an invalid name or a negative duration.
        """
        self._check_name(name)
        if ms < 0:
            raise InputValidationError(
                messages_tr.msg("INVALID_DURATION", value=ms), details={"name": name, "ms": ms}
            )
        self._record(name, float(ms))

    def _record(self, name: str, ms: float) -> None:
        self._stages[name] = self._stages.get(name, 0.0) + ms
        self._counts[name] = self._counts.get(name, 0) + 1

    @property
    def stages(self) -> dict[str, float]:
        """Copy of ``{stage: milliseconds}`` in first-seen order."""
        return dict(self._stages)

    @property
    def counts(self) -> dict[str, int]:
        """Copy of ``{stage: number of recordings}``."""
        return dict(self._counts)

    @property
    def total_ms(self) -> float:
        """Wall-clock milliseconds since the timer was created (or last reset)."""
        return (self._clock() - self._start) * 1000.0

    @property
    def depth(self) -> int:
        """Current nesting depth (0 outside any stage)."""
        return len(self._stack)

    @property
    def current_stage(self) -> str | None:
        """Name of the innermost open stage, or None."""
        return self._stack[-1] if self._stack else None

    def as_dict(self) -> dict[str, float]:
        """Alias of :attr:`stages` for JSON serialisation (``operations.timings``)."""
        return self.stages

    def reset(self) -> None:
        """Drop all recorded stages and restart the total clock."""
        self._stages.clear()
        self._counts.clear()
        self._stack.clear()
        self._start = self._clock()

    def server_timing_header(self, *, total_name: str | None = "app") -> str:
        """Format the ``Server-Timing`` header value (RFC 9211).

        Args:
            total_name: Name of the trailing metric carrying :attr:`total_ms`; None omits it.

        Returns:
            ``"nlu;dur=12.3, parse;dur=1.1, cas;dur=41.0, render;dur=0.4, app;dur=58.2"``.
        """
        parts = [f"{name};dur={ms:.1f}" for name, ms in self._stages.items()]
        if total_name is not None:
            self._check_name(total_name)
            parts.append(f"{total_name};dur={self.total_ms:.1f}")
        return ", ".join(parts)


__all__ = ["StageTimer"]
