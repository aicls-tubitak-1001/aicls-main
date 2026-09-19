"""İP-8 contractual metric: command-fetch accuracy over ``datasets/book/fetch_test.jsonl``.

The target in the proposal is **> 95 %** top-1 accuracy. The set is authored by the project team,
not sampled from an independent corpus — the progress report must say so (see
``datasets/book/README.md``, "Ölçüm notu").
"""

from __future__ import annotations

from typing import Any

import pytest

from aicls.book import BookResolver, UnitType

pytestmark = pytest.mark.integration

#: Contractual İP-8 threshold for "komutla getirme doğruluğu".
FETCH_ACCURACY_TARGET = 0.95
#: Smallest set size that makes the ratio meaningful.
MIN_FETCH_ROWS = 60


def _accuracy(resolver: BookResolver, rows: list[dict[str, Any]]) -> tuple[float, list[str]]:
    """Return top-1 accuracy over ``rows`` and the ids of the queries that missed."""
    misses: list[str] = []
    for row in rows:
        unit_type = UnitType(row["unit_type"]) if row["unit_type"] else None
        hits = resolver.resolve(row["query"], unit_type=unit_type)
        got = hits[0].unit.id if hits else None
        if got != row["expect_unit"]:
            misses.append(f"{row['id']}: {row['query']!r} -> {got} (beklenen {row['expect_unit']})")
    return (len(rows) - len(misses)) / len(rows), misses


def test_fetch_test_set_is_large_and_well_formed(fetch_rows: list[dict[str, Any]]) -> None:
    """The set holds at least 60 distinct Turkish phrasings with unique ids."""
    assert len(fetch_rows) >= MIN_FETCH_ROWS
    ids = [row["id"] for row in fetch_rows]
    queries = [row["query"] for row in fetch_rows]
    assert len(set(ids)) == len(ids)
    assert len(set(queries)) == len(queries)
    assert all(row["query"].strip() for row in fetch_rows)
    assert all(
        row["unit_type"] in {None, *(member.value for member in UnitType)} for row in fetch_rows
    )


def test_book_fetch_test_set_accuracy_above_0_95(
    resolver: BookResolver, fetch_rows: list[dict[str, Any]]
) -> None:
    """Top-1 fetch accuracy clears the İP-8 target of 95 %."""
    accuracy, misses = _accuracy(resolver, fetch_rows)
    assert accuracy > FETCH_ACCURACY_TARGET, f"accuracy={accuracy:.3f}, misses={misses}"


def test_fetch_set_covers_every_unit_kind(fetch_rows: list[dict[str, Any]]) -> None:
    """Suffixed, ASCII, typo, verb-carrying and type-free phrasings are all represented."""
    kinds = {row["kind"] for row in fetch_rows}
    assert {"command", "question", "suffix_gen", "suffix_acc", "ascii", "typo"} <= kinds
    assert sum(1 for row in fetch_rows if row["unit_type"] is None) >= 10


def test_every_expected_unit_exists_in_the_loaded_book(
    resolver: BookResolver, fetch_rows: list[dict[str, Any]]
) -> None:
    """No row of the metric set points at a unit the sample chapters do not contain."""
    stored = {unit.id for unit in resolver.repo.units()}
    expected = {row["expect_unit"] for row in fetch_rows}
    assert expected <= stored
