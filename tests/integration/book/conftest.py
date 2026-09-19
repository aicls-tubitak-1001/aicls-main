"""Fixtures for the book integration tests: a migrated in-memory database with the sample book.

The schema always comes from ``alembic upgrade head`` (never ``metadata.create_all``), so these
tests exercise the real ``book_units`` / ``book_aliases`` / ``book_triples`` tables, their
foreign keys and their indexes — the point of testing the loader at all.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from aicls.book import BookResolver, ContentUnit, TagResult, Triple, load_units
from aicls.book.ingest.validators import require_full_coverage
from aicls.db import BookRepository, make_engine, session_scope, upgrade_head
from tests.unit.book.sample_set import tag_markdown_samples, tag_source

MEMORY_URL = "sqlite+pysqlite:///:memory:"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def book_datasets(repo_root: Path) -> Path:
    """Path of ``datasets/book``."""
    return repo_root / "datasets" / "book"


@pytest.fixture(scope="session")
def fetch_rows(book_datasets: Path) -> list[dict[str, Any]]:
    """Rows of ``datasets/book/fetch_test.jsonl`` (the İP-8 fetch-accuracy set)."""
    lines = (book_datasets / "fetch_test.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@pytest.fixture
def engine() -> Iterator[Engine]:
    """A migrated in-memory SQLite engine (``StaticPool``: one shared connection)."""
    created = make_engine(MEMORY_URL)
    upgrade_head(created)
    try:
        yield created
    finally:
        created.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """A session on the migrated database, committed at the end of the test."""
    with session_scope(engine) as db_session:
        yield db_session


@pytest.fixture
def repo(session: Session) -> BookRepository:
    """An empty book repository."""
    return BookRepository(session)


@pytest.fixture
def tagged_book(book_datasets: Path) -> list[TagResult]:
    """Both Markdown sample chapters, tagged."""
    return tag_markdown_samples(book_datasets)


@pytest.fixture
def sample_units(tagged_book: list[TagResult]) -> list[ContentUnit]:
    """Every unit of the sample book, in load order."""
    return [node for result in tagged_book for node in result.nodes]


@pytest.fixture
def sample_triples(tagged_book: list[TagResult]) -> list[Triple]:
    """Every edge of the sample book, in load order."""
    return [triple for result in tagged_book for triple in result.triples]


@pytest.fixture
def loaded_repo(
    repo: BookRepository,
    tagged_book: list[TagResult],
    sample_units: list[ContentUnit],
    sample_triples: list[Triple],
) -> BookRepository:
    """A repository holding the whole sample book (units, triples and generated aliases)."""
    ratio = require_full_coverage(*tagged_book)
    load_units(sample_units, sample_triples, repo, tagged_ratio=ratio)
    return repo


@pytest.fixture
def resolver(loaded_repo: BookRepository) -> BookResolver:
    """A resolver over the loaded sample book."""
    return BookResolver(loaded_repo)


@pytest.fixture
def untagged_chapter(book_datasets: Path) -> TagResult:
    """The deliberately broken chapter (two headings without a number)."""
    return tag_source(book_datasets, "invalid/bolum-03-etiketsiz.md", "bolum-3", "markdown")
