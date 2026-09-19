"""Textbook knowledge-graph tables (docs/mimari/veri-modeli.md §3.4, ADR-0019).

The full JSON-LD document of a unit is kept in ``book_units.jsonld``; frequently queried fields
are denormalised into columns, RDF relations are flattened into ``book_triples`` and Turkish
alias forms into ``book_aliases``. The MySQL-only ``FULLTEXT(search_text)`` index is created by
the migration under a dialect guard and therefore is deliberately **not** part of the metadata —
SQLite falls back to ``LIKE`` + ``difflib`` in :mod:`aicls.book`.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects import mysql
from sqlalchemy.orm import Mapped, mapped_column

from aicls.db.base import (
    MYSQL_TABLE_KWARGS,
    Base,
    TimestampMixin,
    UlidPk,
    enum_column,
    ulid_fk,
)
from aicls.db.enums import TagStatus, UnitType

#: Name of the MySQL-only ``FULLTEXT(search_text)`` index. Created by migration 0001 behind a
#: dialect guard and deliberately absent from this metadata (module docstring), so the
#: autogenerate comparison in :func:`aicls.db.migrate.diff_metadata` has to know the name to
#: leave it out; otherwise a real MySQL reports it as an index to remove.
FULLTEXT_INDEX_NAME = "ftx_book_units_search_text"

#: Body text: ``TEXT`` on SQLite, ``MEDIUMTEXT`` on MySQL (a chapter body can exceed 64 KB).
BODY_TEXT = Text().with_variant(mysql.MEDIUMTEXT(), "mysql")

#: Length of a book unit identifier such as ``"tanim-2.4"``.
UNIT_ID_LENGTH = 64


class BookUnit(Base, TimestampMixin):
    """A tagged textbook unit (definition, theorem, example, figure, ...)."""

    __tablename__ = "book_units"
    __table_args__ = (
        Index("ix_book_units_unit_type_chapter", "unit_type", "chapter"),
        MYSQL_TABLE_KWARGS,
    )

    id: Mapped[str] = mapped_column(String(UNIT_ID_LENGTH), primary_key=True)
    unit_type: Mapped[UnitType] = mapped_column(enum_column(UnitType, "unit_type"))
    chapter: Mapped[str] = mapped_column(String(64), default="")
    number: Mapped[str] = mapped_column(String(16), default="")
    order_in_chapter: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    title_tr: Mapped[str] = mapped_column(String(255), default="")
    title_normalized: Mapped[str] = mapped_column(String(255), default="", index=True)
    body_md: Mapped[str] = mapped_column(BODY_TEXT, default="")
    latex_json: Mapped[dict[str, Any]] = mapped_column(default=dict)
    search_text: Mapped[str] = mapped_column(Text, default="")
    jsonld: Mapped[dict[str, Any]] = mapped_column(default=dict)
    source_page: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    tag_status: Mapped[TagStatus] = mapped_column(
        enum_column(TagStatus, "tag_status"), default=TagStatus.AUTO
    )
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)


class BookAlias(Base, TimestampMixin):
    """A normalised Turkish alias pointing at a unit ("türevin tanımı" → ``tanim-2.4``)."""

    __tablename__ = "book_aliases"
    __table_args__ = (
        UniqueConstraint(
            "unit_id", "alias_normalized", name="uq_book_aliases_unit_id_alias_normalized"
        ),
        MYSQL_TABLE_KWARGS,
    )

    id: Mapped[UlidPk]
    unit_id: Mapped[str] = mapped_column(
        String(UNIT_ID_LENGTH), ForeignKey("book_units.id", ondelete="CASCADE"), nullable=False
    )
    alias_normalized: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)


class BookTriple(Base, TimestampMixin):
    """One RDF edge between two units (``aicls:requires``, ``skos:related``, ...)."""

    __tablename__ = "book_triples"
    __table_args__ = (
        UniqueConstraint(
            "subject_id", "predicate", "object_id", name="uq_book_triples_subject_predicate_object"
        ),
        Index("ix_book_triples_subject_id_predicate", "subject_id", "predicate"),
        Index("ix_book_triples_object_id_predicate", "object_id", "predicate"),
        MYSQL_TABLE_KWARGS,
    )

    id: Mapped[UlidPk]
    subject_id: Mapped[str] = mapped_column(
        String(UNIT_ID_LENGTH), ForeignKey("book_units.id", ondelete="CASCADE"), nullable=False
    )
    predicate: Mapped[str] = mapped_column(String(32), nullable=False)
    object_id: Mapped[str] = mapped_column(
        String(UNIT_ID_LENGTH), ForeignKey("book_units.id", ondelete="CASCADE"), nullable=False
    )
    provenance: Mapped[str] = mapped_column(String(64), default="")


class BookAsset(Base, TimestampMixin):
    """A figure or table image attached to a unit, with Turkish caption and alt text."""

    __tablename__ = "book_assets"
    __table_args__ = (MYSQL_TABLE_KWARGS,)

    id: Mapped[UlidPk]
    unit_id: Mapped[str] = mapped_column(
        String(UNIT_ID_LENGTH),
        ForeignKey("book_units.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    asset_id: Mapped[str] = ulid_fk("assets.id", ondelete="CASCADE")
    caption_tr: Mapped[str] = mapped_column(String(500), default="")
    alt_tr: Mapped[str] = mapped_column(String(500), default="")


__all__ = ["BODY_TEXT", "BookAlias", "BookAsset", "BookTriple", "BookUnit"]
