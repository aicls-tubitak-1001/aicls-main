"""Latency budget of the book resolver (docs/mimari/veri-modeli.md §5.4: p50 < 20 ms).

Marked ``perf`` so it runs nightly and before every report; it is cheap enough to keep in the
default run as well. Measured on SQLite with the sample book loaded, which is the configuration
the pilot's development instance uses.
"""

from __future__ import annotations

from statistics import median
import time
from typing import Any

import pytest

from aicls.book import BookResolver, UnitType

pytestmark = [pytest.mark.integration, pytest.mark.perf]

#: Contractual budget for one resolve on the sample book (ADR-0019 "Sonuçlar").
P50_BUDGET_MS = 20.0
#: Number of measured resolutions.
REPEATS = 3


def test_book_resolve_p50_under_20ms(
    resolver: BookResolver, fetch_rows: list[dict[str, Any]]
) -> None:
    """Median resolution time over the whole fetch set stays inside the 20 ms budget.

    The cache is cleared before each pass, so what is measured is the alias-index path, not a
    memoised answer.
    """
    durations: list[float] = []
    for _ in range(REPEATS):
        resolver.cache_clear()
        for row in fetch_rows:
            unit_type = UnitType(row["unit_type"]) if row["unit_type"] else None
            started = time.perf_counter()
            resolver.resolve(row["query"], unit_type=unit_type)
            durations.append((time.perf_counter() - started) * 1000.0)
    p50 = median(durations)
    assert len(durations) == REPEATS * len(fetch_rows)
    assert p50 < P50_BUDGET_MS, f"p50={p50:.2f} ms over {len(durations)} resolutions"


def test_cached_resolution_is_faster_than_the_first_one(resolver: BookResolver) -> None:
    """The LRU is what keeps a repeated command off the database (İP-8 risk plan)."""
    resolver.cache_clear()
    query = "türevin tanımını ekrana getir"
    resolver.resolve(query)
    started = time.perf_counter()
    for _ in range(50):
        resolver.resolve(query)
    cached_ms = (time.perf_counter() - started) * 1000.0 / 50
    assert resolver.cache_info()["hits"] == 50
    assert cached_ms < P50_BUDGET_MS
