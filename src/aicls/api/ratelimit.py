"""In-memory sliding-window rate limiter (docs/guvenlik-gizlilik.md §7, api.md §1).

One process, one limiter: the platform runs a single uvicorn worker per host (ADR-0018), so a
per-process counter is the honest implementation. Redis is the documented next step and needs an
ADR; the interface here (:meth:`RateLimiter.check`) does not change when that happens.

Windows are *sliding*: each key keeps the timestamps of its recent hits and a hit is admitted
only when fewer than ``limit`` of them fall inside ``window_s``. A fixed window would let a
client send ``2 * limit`` requests across a window boundary, which is exactly what the İP-9
security checklist tests for.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import time

#: Keys evicted in one sweep once the table grows past :data:`MAX_TRACKED_KEYS`.
_SWEEP_BATCH = 256

#: Upper bound on distinct keys kept in memory (a bounded table cannot be a memory DoS).
MAX_TRACKED_KEYS = 20_000


@dataclass(frozen=True, slots=True)
class RateRule:
    """One rate-limit rule.

    Attributes:
        name: Stable rule name, used as part of the bucket key and reported in ``details``.
        limit: Number of requests allowed inside the window.
        window_s: Length of the sliding window in seconds.
        scope: ``"ip"`` counts per client address, ``"user"`` per authenticated account
            (falling back to the address for anonymous callers).
    """

    name: str
    limit: int
    window_s: float
    scope: str = "user"


#: Rules of api.md §1, keyed by the path prefix they apply to (longest prefix wins).
DEFAULT_RULES: dict[str, tuple[RateRule, ...]] = {
    "/auth": (
        RateRule("auth_ip", 5, 60.0, "ip"),
        RateRule("auth_account", 20, 3600.0, "user"),
    ),
    "/compute": (RateRule("compute", 60, 60.0, "user"),),
    "/ocr": (RateRule("upload", 20, 60.0, "user"),),
    "/asr": (RateRule("upload", 20, 60.0, "user"),),
    "/contribute/recordings": (RateRule("recordings", 60, 3600.0, "user"),),
}


class RateLimiter:
    """Sliding-window counters keyed by ``(rule, identity)``.

    Args:
        clock: Monotonic seconds source; injectable so tests do not sleep.
    """

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._hits: dict[str, deque[float]] = {}

    def check(self, rule: RateRule, identity: str) -> float | None:
        """Record one hit and report whether it exceeded ``rule``.

        Args:
            rule: The rule to apply.
            identity: Client address or user id, depending on ``rule.scope``.

        Returns:
            ``None`` when the request is admitted, otherwise the number of seconds after which
            the oldest hit leaves the window (the ``Retry-After`` value).
        """
        now = self._clock()
        key = f"{rule.name}:{identity}"
        window = self._hits.get(key)
        if window is None:
            window = deque()
            self._hits[key] = window
            if len(self._hits) > MAX_TRACKED_KEYS:
                self._sweep(now)
        cutoff = now - rule.window_s
        while window and window[0] <= cutoff:
            window.popleft()
        if len(window) >= rule.limit:
            return max(0.0, window[0] + rule.window_s - now)
        window.append(now)
        return None

    def _sweep(self, now: float) -> None:
        """Drop the oldest empty or stale buckets so the table stays bounded."""
        stale = [key for key, window in self._hits.items() if not window or window[-1] < now - 3600]
        for key in stale[:_SWEEP_BATCH]:
            self._hits.pop(key, None)
        if len(self._hits) > MAX_TRACKED_KEYS:
            for key in list(self._hits)[:_SWEEP_BATCH]:
                self._hits.pop(key, None)

    def reset(self) -> None:
        """Forget every counter (test teardown, ``aicls api reset-limits``)."""
        self._hits.clear()

    @property
    def tracked_keys(self) -> int:
        """Number of buckets currently held in memory."""
        return len(self._hits)


def rules_for(
    path: str, rules: dict[str, tuple[RateRule, ...]] | None = None
) -> tuple[RateRule, ...]:
    """Return the rules that apply to ``path`` (longest matching prefix wins).

    Args:
        path: Request path *without* the ``/api/v1`` prefix (``"/compute/text"``).
        rules: Rule table; :data:`DEFAULT_RULES` when omitted.

    Returns:
        The matching rules, or an empty tuple when the path is unlimited.
    """
    table = DEFAULT_RULES if rules is None else rules
    best: tuple[RateRule, ...] = ()
    best_len = -1
    for prefix, entries in table.items():
        if (path == prefix or path.startswith(prefix + "/")) and len(prefix) > best_len:
            best, best_len = entries, len(prefix)
    return best


__all__ = [
    "DEFAULT_RULES",
    "MAX_TRACKED_KEYS",
    "RateLimiter",
    "RateRule",
    "rules_for",
]
