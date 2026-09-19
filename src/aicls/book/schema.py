"""Content-unit vocabulary and records of the İP-8 knowledge graph (ADR-0019).

The vocabulary mirrors the JSON-LD context in :mod:`aicls.book.vocab`: :class:`UnitType` members
carry the ``@type`` values (``"Definition"`` ...) and :class:`Predicate` members carry the CURIEs
that end up in ``book_triples.predicate`` (``"aicls:requires"``, ``"skos:related"``). Both are
:class:`~enum.StrEnum` subclasses, so they interoperate with :class:`aicls.db.enums.UnitType` and
with :data:`aicls.db.enums.BOOK_PREDICATES` without an explicit conversion.

:class:`ContentUnit` and :class:`Triple` are the pydantic models every producer (taggers, CLI,
API) hands to :func:`aicls.book.loader.load_units`; :class:`TagResult`, :class:`UntaggedSpan`,
:class:`Hit` and :class:`ResolvedUnit` are plain frozen records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Identifier prefix (slug) used when a tagger mints a unit id such as ``"tanim-2.4"``.
UNIT_ID_SEPARATOR = "-"


class UnitType(StrEnum):
    """Type of a textbook content unit; the JSON-LD ``@type`` and ``book_units.unit_type``."""

    DEFINITION = "Definition"
    THEOREM = "Theorem"
    LEMMA = "Lemma"
    COROLLARY = "Corollary"
    PROOF = "Proof"
    EXAMPLE = "Example"
    EXERCISE = "Exercise"
    FIGURE = "Figure"
    TABLE = "Table"
    REMARK = "Remark"
    SECTION = "Section"


class Predicate(StrEnum):
    """RDF predicate of a :class:`Triple`; the CURIE stored in ``book_triples.predicate``."""

    PART_OF = "aicls:partOf"
    REQUIRES = "aicls:requires"
    ILLUSTRATES = "aicls:illustrates"
    PROVES = "aicls:proves"
    NEXT = "aicls:next"
    HAS_FIGURE = "aicls:hasFigure"
    RELATED = "skos:related"


#: Slug used to mint an identifier for each unit type (``Definition`` → ``tanim-2.4``).
SLUG_BY_TYPE: dict[UnitType, str] = {
    UnitType.DEFINITION: "tanim",
    UnitType.THEOREM: "teorem",
    UnitType.LEMMA: "lemma",
    UnitType.COROLLARY: "sonuc",
    UnitType.PROOF: "kanit",
    UnitType.EXAMPLE: "ornek",
    UnitType.EXERCISE: "alistirma",
    UnitType.FIGURE: "sekil",
    UnitType.TABLE: "tablo",
    UnitType.REMARK: "not",
    UnitType.SECTION: "bolum",
}

#: Reverse of :data:`SLUG_BY_TYPE`; used when a block header names the type (``::: tanim 2.4``).
TYPE_BY_SLUG: dict[str, UnitType] = {slug: unit for unit, slug in SLUG_BY_TYPE.items()}


def unit_id(unit_type: UnitType, number: str) -> str:
    """Mint the canonical identifier of a unit (``Definition`` + ``"2.4"`` → ``"tanim-2.4"``)."""
    return f"{SLUG_BY_TYPE[unit_type]}{UNIT_ID_SEPARATOR}{number}"


class AssetRef(BaseModel):
    """A figure or table image attached to a unit (``book_assets`` row, ``aicls:hasFigure``)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    path: str = ""
    caption_tr: str = ""
    alt_tr: str = ""


class ContentUnit(BaseModel):
    """One tagged textbook unit — the JSON-LD document plus its denormalised fields.

    Attributes:
        id: Stable identifier, ``"tanim-2.4"``.
        type: :class:`UnitType` of the unit.
        chapter: Chapter identifier, ``"bolum-2"`` (also the ``aicls:partOf`` target).
        number: Book numbering, ``"2.4"``.
        title_tr: Turkish title as printed in the book.
        aliases: Author-supplied aliases; the loader adds generated Turkish suffix forms.
        body_md: Markdown body of the unit.
        latex_blocks: Display-math blocks extracted from the body, without delimiters.
        assets: Figures/tables attached to the unit.
        source_page: Printed page number, when known.
        jsonld: Full JSON-LD document; filled in by :func:`aicls.book.export.unit_to_jsonld`
            when the producer leaves it empty.
        order_in_chapter: Position of the unit inside its chapter (document order).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64)
    type: UnitType
    chapter: str = ""
    number: str = ""
    title_tr: str = ""
    aliases: list[str] = Field(default_factory=list)
    body_md: str = ""
    latex_blocks: list[str] = Field(default_factory=list)
    assets: list[AssetRef] = Field(default_factory=list)
    source_page: int | None = None
    jsonld: dict[str, Any] = Field(default_factory=dict)
    order_in_chapter: int = 0

    @field_validator("id")
    @classmethod
    def _strip_id(cls, value: str) -> str:
        """Reject an identifier with surrounding whitespace instead of storing it."""
        return value.strip()


class Triple(BaseModel):
    """One RDF edge of the book graph (``subject predicate object``)."""

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(min_length=1)
    predicate: Predicate
    object: str = Field(min_length=1)
    provenance: str = ""

    def key(self) -> tuple[str, str, str]:
        """Deduplication key ``(subject, predicate, object)``."""
        return self.subject, str(self.predicate), self.object


@dataclass(frozen=True, slots=True)
class UntaggedSpan:
    """A heading that looks like a unit heading but produced no unit.

    Attributes:
        line: 1-based line number in the source.
        text: Heading text as written (without the ``#`` markers).
        kind: Why it failed — ``"missing_number"``, ``"missing_label"`` or ``"unrecognised"``.
        source: Source file name, when the caller supplied one.
    """

    line: int
    text: str
    kind: str = "missing_number"
    source: str = ""


@dataclass(frozen=True, slots=True)
class TagResult:
    """Output of :func:`aicls.book.tag_markdown` / :func:`aicls.book.tag_latex`.

    Attributes:
        nodes: Units recognised in the source, in document order.
        triples: Relations found; edges pointing outside the file are kept here and filtered
            against the database by :func:`aicls.book.loader.load_units`.
        untagged_spans: Unit-looking headings that could not be tagged — non-empty means
            :func:`aicls.book.coverage` is below 1.0 and the load is refused.
        chapter: Chapter identifier the units were tagged into.
        source: Source file name, when the caller supplied one.
    """

    nodes: list[ContentUnit] = field(default_factory=list)
    triples: list[Triple] = field(default_factory=list)
    untagged_spans: list[UntaggedSpan] = field(default_factory=list)
    chapter: str = ""
    source: str = ""

    def unit_ids(self) -> set[str]:
        """Identifiers of every node in this result."""
        return {node.id for node in self.nodes}


@dataclass(frozen=True, slots=True)
class Hit:
    """One search result (``HitOut`` of docs/mimari/api.md §3.7).

    Attributes:
        unit: The matched unit.
        score: Confidence in ``[0, 1]``; 1.0 for an exact alias match.
        matched_alias: The alias that matched, when the hit came from the alias index.
        stage: Which stage produced it — ``"alias"``, ``"fulltext"`` or ``"fuzzy"``.
    """

    unit: ContentUnit
    score: float
    matched_alias: str | None = None
    stage: str = "alias"


@dataclass(frozen=True, slots=True)
class ResolvedUnit:
    """A hit produced by :class:`aicls.book.resolver.BookResolver`.

    Adds what the resolver understood from the command phrase, so the API can explain why a
    unit was returned for ``"türev tanımını getir"``.

    Attributes:
        unit: The matched unit.
        score: Confidence in ``[0, 1]``.
        matched_alias: The alias that matched, when any.
        stage: Stage that produced the hit (``"alias"``, ``"fulltext"``, ``"fuzzy"``).
        query_normalized: Query after Turkish normalisation and verb removal.
        unit_type: Unit type inferred from (or given with) the query.
    """

    unit: ContentUnit
    score: float
    matched_alias: str | None = None
    stage: str = "alias"
    query_normalized: str = ""
    unit_type: UnitType | None = None


__all__ = [
    "SLUG_BY_TYPE",
    "TYPE_BY_SLUG",
    "UNIT_ID_SEPARATOR",
    "AssetRef",
    "ContentUnit",
    "Hit",
    "Predicate",
    "ResolvedUnit",
    "TagResult",
    "Triple",
    "UnitType",
    "UntaggedSpan",
    "unit_id",
]
