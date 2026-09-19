"""The sample chapters must keep matching ``datasets/book/expected_units.json`` (golden file).

A change in the taggers that silently renumbers, retitles or reorders a unit would break the
alias index and the fetch test set; this test makes it visible in the diff instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aicls.book import coverage, generate_alias_entries
from aicls.book.ingest.validators import UntaggedHeadingError, require_full_coverage
from tests.unit.book.sample_set import SAMPLE_SOURCES, tag_source


@pytest.mark.parametrize(("relative", "chapter", "fmt"), SAMPLE_SOURCES)
def test_sample_chapter_matches_expected_units(
    book_datasets: Path,
    expected_units: dict[str, Any],
    relative: str,
    chapter: str,
    fmt: str,
) -> None:
    """Every unit, number, title, page and edge of the sample set is reproduced exactly."""
    expected = next(item for item in expected_units["chapters"] if item["source"] == relative)
    result = tag_source(book_datasets, relative, chapter, fmt)
    assert coverage(result) == expected["coverage"]
    actual = [
        {
            "id": node.id,
            "type": node.type.value,
            "number": node.number,
            "title_tr": node.title_tr,
            "latex_blocks": len(node.latex_blocks),
            "source_page": node.source_page,
            "aliases": node.aliases,
        }
        for node in result.nodes
    ]
    assert actual == expected["units"]
    assert [[t.subject, t.predicate.value, t.object] for t in result.triples] == expected["triples"]


def test_invalid_chapter_matches_expected_untagged_headings(
    book_datasets: Path, expected_units: dict[str, Any]
) -> None:
    """The deliberately broken chapter keeps failing with the same two headings."""
    expected = expected_units["invalid"]
    result = tag_source(book_datasets, expected["source"], expected["chapter"], "markdown")
    assert coverage(result) == expected["coverage"]
    assert [span.text for span in result.untagged_spans] == expected["untagged_headings"]
    with pytest.raises(UntaggedHeadingError) as excinfo:
        require_full_coverage(result)
    assert excinfo.value.headings == expected["untagged_headings"]


def test_sample_chapter_aliases_have_no_tie_at_the_top(book_datasets: Path) -> None:
    """No alias is claimed by two units at the same weight — fetch accuracy must be decidable.

    Aliases may overlap (a proof of a theorem shares its name); what must never happen is a tie
    at the highest weight, because then the indexed lookup would order by identifier and the
    answer to "türevlenebilirlik teoremi" would be arbitrary.
    """
    owners: dict[str, list[tuple[str, float]]] = {}
    for relative, chapter, fmt in SAMPLE_SOURCES:
        for node in tag_source(book_datasets, relative, chapter, fmt).nodes:
            for entry in generate_alias_entries(node.title_tr, node.type):
                owners.setdefault(entry.alias, []).append((node.id, entry.weight))
    ties = {
        alias: rows
        for alias, rows in owners.items()
        if len({unit for unit, weight in rows if weight == max(w for _, w in rows)}) > 1
    }
    assert ties == {}
