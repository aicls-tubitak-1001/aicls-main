"""Fixtures for the ``aicls.book`` unit tests: the sample chapters under ``datasets/book``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from aicls.book import TagResult
from tests.unit.book.sample_set import tag_source


@pytest.fixture(scope="session")
def book_datasets(repo_root: Path) -> Path:
    """Path of ``datasets/book``."""
    return repo_root / "datasets" / "book"


@pytest.fixture(scope="session")
def expected_units(book_datasets: Path) -> dict[str, Any]:
    """Parsed ``datasets/book/expected_units.json`` (the golden tagging result)."""
    document: dict[str, Any] = json.loads(
        (book_datasets / "expected_units.json").read_text(encoding="utf-8")
    )
    return document


@pytest.fixture
def chapter_one(book_datasets: Path) -> TagResult:
    """Tagged ``sample/bolum-01-limit.md`` (plain ``## Tanım 1.1:`` headings)."""
    return tag_source(book_datasets, "sample/bolum-01-limit.md", "bolum-1", "markdown")


@pytest.fixture
def chapter_two(book_datasets: Path) -> TagResult:
    """Tagged ``sample/bolum-02-turev.md`` (``:::`` blocks with aliases and relations)."""
    return tag_source(book_datasets, "sample/bolum-02-turev.md", "bolum-2", "markdown")


@pytest.fixture
def chapter_four(book_datasets: Path) -> TagResult:
    """Tagged ``sample/bolum-04-integral.tex`` (LaTeX environments)."""
    return tag_source(book_datasets, "sample/bolum-04-integral.tex", "bolum-4", "latex")


@pytest.fixture
def untagged_chapter(book_datasets: Path) -> TagResult:
    """Tagged ``invalid/bolum-03-etiketsiz.md`` — two headings deliberately carry no number."""
    return tag_source(book_datasets, "invalid/bolum-03-etiketsiz.md", "bolum-3", "markdown")
