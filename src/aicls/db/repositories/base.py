"""Repository foundations: session binding, cursor pagination and mapping interop.

Repositories are the only place in the platform that builds a query (ADR-0003); everything above
them receives ORM objects or the plain records of :mod:`aicls.db.results`. Because neighbouring
packages (``aicls.api``, ``aicls.book``, ``aicls.collect``) hand over pydantic models, dataclasses
or plain dictionaries, :func:`field_of` reads a value from any of those shapes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from sqlalchemy import Select
from sqlalchemy.orm import InstrumentedAttribute, Session

from aicls.db.base import Base
from aicls.db.results import Page

#: Largest page a caller may request (docs/mimari/api.md §1: ``limit=50`` default).
MAX_PAGE_SIZE = 200
#: Page size used when the caller does not pass one.
DEFAULT_PAGE_SIZE = 50

T = TypeVar("T", bound=Base)

_MISSING = object()


def field_of(source: Any, *names: str, default: Any = None) -> Any:
    """Return the first present value of ``names`` from a mapping, dataclass or model.

    Args:
        source: Mapping, pydantic model, dataclass or any object with attributes.
        *names: Candidate field names, tried in order; ``None`` values are skipped so an alias
            can fill in for an explicitly-null primary name.
        default: Value returned when no name yields a non-``None`` value.

    Returns:
        The first non-``None`` value found, else ``default``.
    """
    for name in names:
        value = _MISSING
        if isinstance(source, Mapping):
            value = source.get(name, _MISSING)
        elif hasattr(source, name):
            value = getattr(source, name)
        if value is not _MISSING and value is not None:
            return value
    return default


def clamp_limit(limit: int | None) -> int:
    """Clamp a caller-supplied page size into ``1..MAX_PAGE_SIZE``."""
    if limit is None:
        return DEFAULT_PAGE_SIZE
    return max(1, min(int(limit), MAX_PAGE_SIZE))


class Repository:
    """Base class holding the :class:`~sqlalchemy.orm.Session` a repository works on."""

    def __init__(self, session: Session) -> None:
        """Bind the repository to ``session`` (never commits on its own; see
        :func:`aicls.db.engine.session_scope`)."""
        self.session = session

    @property
    def dialect(self) -> str:
        """Name of the bound dialect (``"sqlite"`` or ``"mysql"``)."""
        return self.session.get_bind().dialect.name

    def flush(self) -> None:
        """Flush pending changes so generated ids are visible without committing."""
        self.session.flush()

    def cursor_page(
        self,
        stmt: Select[tuple[T]],
        id_column: InstrumentedAttribute[str],
        *,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> Page[T]:
        """Return one descending page of ``stmt`` keyed on a ULID column.

        ULIDs sort by creation time, so ``id < cursor`` is a stable keyset pagination that never
        skips or repeats a row when new rows arrive (docs/mimari/api.md §1).

        Args:
            stmt: Select of a single ORM entity, already filtered.
            id_column: The ULID column to page on (usually ``Model.id``).
            cursor: ``next_cursor`` of the previous page, or ``None`` for the first page.
            limit: Page size, clamped to :data:`MAX_PAGE_SIZE`.

        Returns:
            A :class:`~aicls.db.results.Page` whose ``next_cursor`` is ``None`` on the last page.
        """
        size = clamp_limit(limit)
        if cursor:
            stmt = stmt.where(id_column < cursor)
        rows = list(self.session.scalars(stmt.order_by(id_column.desc()).limit(size + 1)))
        has_more = len(rows) > size
        items = rows[:size]
        next_cursor = str(getattr(items[-1], id_column.key)) if has_more and items else None
        return Page(items=items, next_cursor=next_cursor)


__all__ = [
    "DEFAULT_PAGE_SIZE",
    "MAX_PAGE_SIZE",
    "Repository",
    "clamp_limit",
    "field_of",
]
