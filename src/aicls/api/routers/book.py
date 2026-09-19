"""Textbook knowledge-graph endpoints (İP-8, api.md §3.7).

Every read goes through :class:`aicls.db.repositories.book.BookRepository`, so the user's search
string is always a bound parameter — the SQL-injection guarantee of docs/guvenlik-gizlilik.md §5,
pinned by ``test_api_sql_injection_string_in_search_is_parameterized``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Response

from aicls.api.deps import AppSettings, CurrentUser, DbSession, StaffUser
from aicls.api.schemas import ChapterOut, ContentUnitOut, HitOut, ResolveIn, unit_payload
from aicls.book.export import export_jsonld, export_turtle
from aicls.book.loader import triples_from_repo, units_from_repo
from aicls.book.query import get_unit, related, search
from aicls.book.resolver import BookResolver
from aicls.book.schema import UnitType
from aicls.core import messages_tr
from aicls.core.errors import InputValidationError, NotFoundError
from aicls.db.repositories.book import BookRepository
from aicls.db.repositories.compute import AssetRepository

router = APIRouter(prefix="/book", tags=["book"])

#: Longest accepted search string (docs/guvenlik-gizlilik.md §5).
MAX_QUERY = 200

#: Default number of hits.
DEFAULT_LIMIT = 5


def _unit_type(value: str | None) -> UnitType | None:
    """Coerce a ``type`` query parameter into a :class:`UnitType`.

    Raises:
        InputValidationError: If the value names no unit type.
    """
    if not value:
        return None
    try:
        return UnitType(value)
    except ValueError as exc:
        raise InputValidationError(
            messages_tr.MSG_VALIDATION_ERROR,
            details={"field": "type", "allowed": [item.value for item in UnitType]},
        ) from exc


@router.get("/chapters", response_model=list[ChapterOut])
def chapters(_user: CurrentUser, session: DbSession) -> list[ChapterOut]:
    """The chapter tree of the loaded book."""
    repo = BookRepository(session)
    counts: dict[str, int] = {}
    for row in repo.units():
        counts[row.chapter] = counts.get(row.chapter, 0) + 1
    return [ChapterOut(id=chapter, n_units=counts[chapter]) for chapter in sorted(counts)]


@router.get("/export.jsonld")
def export_graph_jsonld(_user: StaffUser, session: DbSession) -> Response:
    """Export the whole graph as JSON-LD (instructor/admin)."""
    import json

    repo = BookRepository(session)
    document = export_jsonld(units_from_repo(repo), triples_from_repo(repo))
    return Response(
        content=json.dumps(document, ensure_ascii=False, indent=2),
        media_type="application/ld+json",
    )


@router.get("/export.ttl")
def export_graph_turtle(_user: StaffUser, session: DbSession) -> Response:
    """Export the whole graph as Turtle (instructor/admin)."""
    repo = BookRepository(session)
    return Response(
        content=export_turtle(units_from_repo(repo), triples_from_repo(repo)),
        media_type="text/turtle; charset=utf-8",
    )


@router.get("/search", response_model=list[HitOut])
def search_units(
    _user: CurrentUser,
    session: DbSession,
    q: Annotated[str, Query(min_length=1, max_length=MAX_QUERY)],
    type_: Annotated[str | None, Query(alias="type", max_length=32)] = None,
    limit: Annotated[int, Query(ge=1, le=20)] = DEFAULT_LIMIT,
) -> list[HitOut]:
    """Search unit titles and aliases (alias → full text → fuzzy)."""
    hits = search(BookRepository(session), q, unit_type=_unit_type(type_), limit=limit)
    return [
        HitOut(
            unit=unit_payload(hit.unit),
            score=hit.score,
            matched_alias=hit.matched_alias,
            stage=hit.stage,
        )
        for hit in hits
    ]


@router.post("/resolve", response_model=list[HitOut])
def resolve_command(body: ResolveIn, _user: CurrentUser, session: DbSession) -> list[HitOut]:
    """Resolve a Turkish fetch command (``"türev tanımını getir"``) to units (İP-8)."""
    resolved = BookResolver(BookRepository(session)).resolve(body.text, k=DEFAULT_LIMIT)
    return [
        HitOut(
            unit=unit_payload(item.unit),
            score=item.score,
            matched_alias=item.matched_alias,
            stage=item.stage,
        )
        for item in resolved
    ]


@router.get("/units/{unit_id}", response_model=ContentUnitOut)
def get_book_unit(unit_id: str, _user: CurrentUser, session: DbSession) -> ContentUnitOut:
    """One textbook unit by identifier (``tanim-2.4``)."""
    unit = get_unit(BookRepository(session), unit_id)
    if unit is None:
        raise NotFoundError(
            messages_tr.MSG_NOT_FOUND, details={"table": "book_units", "id": unit_id}
        )
    return unit_payload(unit)


@router.get("/units/{unit_id}/related", response_model=list[ContentUnitOut])
def get_related(
    unit_id: str,
    _user: CurrentUser,
    session: DbSession,
    predicate: Annotated[str | None, Query(max_length=32)] = None,
) -> list[ContentUnitOut]:
    """Units one hop away from ``unit_id``."""
    return [
        unit_payload(unit)
        for unit in related(BookRepository(session), unit_id, predicate=predicate)
    ]


@router.get("/assets/{asset_id}")
def get_book_asset(
    asset_id: str, _user: CurrentUser, session: DbSession, settings: AppSettings
) -> Response:
    """Serve a figure attached to a textbook unit."""
    asset = AssetRepository(session).get(asset_id)
    missing = NotFoundError(messages_tr.MSG_NOT_FOUND, details={"table": "assets", "id": asset_id})
    if asset is None or asset.kind.value != "book_figure":
        raise missing
    path = settings.data_root / asset.path
    if not path.is_file():
        raise missing
    return Response(content=path.read_bytes(), media_type=asset.mime)


__all__ = ["DEFAULT_LIMIT", "MAX_QUERY", "router"]
