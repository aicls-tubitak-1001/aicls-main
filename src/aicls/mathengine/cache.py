"""Bounded LRU cache for CAS results (İP-4 risk plan, ADR-0016).

The cache is keyed by :func:`aicls.mathengine.models.canonical_key` -- the canonical JSON of the
typed command -- so two textually different but semantically identical requests
(``diff(x^2)`` typed twice, or the same command reached through speech and through handwriting)
share one entry. It is deliberately a plain in-process dictionary: the pilot runs one API
process per node, and a shared cache would have to serialise :class:`~aicls.mathengine.Result`
objects anyway.
"""

from __future__ import annotations

from collections import OrderedDict
import threading

from aicls.core import messages_tr
from aicls.core.errors import InputValidationError
from aicls.mathengine.models import Result

#: Default number of cached results (ADR-0016: "Sonuç önbelleği ... LRU 4096").
DEFAULT_CACHE_SIZE = 4096


class LruCache:
    """Thread-safe, size-bounded least-recently-used cache of :class:`Result` values.

    Args:
        maxsize: Maximum number of entries; the least recently *used* entry is evicted first.

    Raises:
        aicls.core.errors.InputValidationError: If ``maxsize`` is not positive.

    Example:
        >>> cache = LruCache(maxsize=2)
        >>> cache.get("missing") is None
        True
    """

    def __init__(self, maxsize: int = DEFAULT_CACHE_SIZE) -> None:
        if maxsize < 1:
            raise InputValidationError(
                messages_tr.msg("VALIDATION_ERROR"),
                details={"field": "maxsize", "value": maxsize, "reason": "maxsize must be >= 1"},
            )
        self._maxsize = maxsize
        self._entries: OrderedDict[str, Result] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def maxsize(self) -> int:
        """Maximum number of entries."""
        return self._maxsize

    @property
    def size(self) -> int:
        """Number of entries currently held."""
        with self._lock:
            return len(self._entries)

    @property
    def hits(self) -> int:
        """Number of :meth:`get` calls that found an entry."""
        return self._hits

    @property
    def misses(self) -> int:
        """Number of :meth:`get` calls that found nothing."""
        return self._misses

    def get(self, key: str) -> Result | None:
        """Return the cached result for ``key`` and mark it most recently used."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return entry

    def put(self, key: str, value: Result) -> None:
        """Store ``value`` under ``key``, evicting the least recently used entry if needed."""
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
            self._entries[key] = value
            while len(self._entries) > self._maxsize:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        """Drop every entry and reset the hit/miss counters."""
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, int]:
        """Return ``{"hits", "misses", "size", "maxsize"}`` for ``GET /health/ready``."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._entries),
                "maxsize": self._maxsize,
            }

    def __len__(self) -> int:
        return self.size

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._entries


__all__ = ["DEFAULT_CACHE_SIZE", "LruCache"]
