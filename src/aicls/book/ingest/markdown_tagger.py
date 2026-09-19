"""Markdown tagger: ``:::`` blocks and plain ``"Tanım 2.4:"`` headings (veri-modeli.md §5.3).

The book team writes ordinary Markdown. Two forms become content units:

1. a fenced block carrying explicit metadata::

       ::: tanim 2.4 "Türev" aliases=["türevin tanımı"] requires=[tanim-2.1] page=58
       $f$ fonksiyonu ...
       :::

2. a plain heading following the printed numbering::

       ## Tanım 2.4: Türev
       ## Teorem 3.1 (Ortalama Değer Teoremi):

A heading that *looks* like a unit heading (starts with "Tanım", "Teorem", "Şekil", ...) but
carries no number cannot be given a stable identifier, so it is reported as an
:class:`~aicls.book.schema.UntaggedSpan` instead of being silently dropped. That is what makes
the İP-8 "100 % tagged" criterion measurable: :func:`aicls.book.coverage` falls below 1.0 and
:func:`aicls.book.load_units` refuses the chapter (ADR-0019).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
import re

from aicls.book.ingest.common import (
    RELATION_ATTRIBUTES,
    first_quoted,
    looks_like_unit_heading,
    match_keyword,
    parse_attributes,
    parse_unit_heading,
    strip_markup,
    to_int,
)
from aicls.book.schema import (
    ContentUnit,
    Predicate,
    TagResult,
    Triple,
    UnitType,
    UntaggedSpan,
    unit_id,
)
from aicls.core.logging import get_logger

_LOGGER = get_logger(__name__)

#: ``## Tanım 2.4: Türev`` — an ATX heading of any level.
HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.*?)\s*#*\s*$")
#: ``::: tanim 2.4 "Türev" aliases=[...]`` — opening fence of a structured block.
BLOCK_OPEN_RE = re.compile(r"^:::\s*(?P<rest>\S.*)$")
#: ``:::`` on its own — closing fence.
BLOCK_CLOSE_RE = re.compile(r"^:::\s*$")
#: ``$$ ... $$`` display math, kept as a LaTeX block on the unit.
DISPLAY_MATH_RE = re.compile(r"\$\$(?P<body>.+?)\$\$", re.DOTALL)
#: ``\[ ... \]`` display math.
BRACKET_MATH_RE = re.compile(r"\\\[(?P<body>.+?)\\\]", re.DOTALL)

#: Provenance written on every triple this tagger produces.
PROVENANCE = "markdown_tagger"


@dataclass(slots=True)
class _Draft:
    """A unit under construction while the tagger walks the source."""

    unit_type: UnitType
    number: str
    title: str
    line: int
    body: list[str]
    aliases: list[str]
    relations: list[tuple[Predicate, str]]
    source_page: int | None = None
    identifier: str = ""

    def build(self, chapter: str, order: int) -> ContentUnit:
        """Freeze the draft into a :class:`~aicls.book.schema.ContentUnit`."""
        body = "\n".join(self.body).strip()
        return ContentUnit(
            id=self.identifier or unit_id(self.unit_type, self.number),
            type=self.unit_type,
            chapter=chapter,
            number=self.number,
            title_tr=self.title,
            aliases=list(dict.fromkeys(self.aliases)),
            body_md=body,
            latex_blocks=extract_latex_blocks(body),
            source_page=self.source_page,
            order_in_chapter=order,
        )


def extract_latex_blocks(body: str) -> list[str]:
    """Return the display-math blocks of ``body`` (``$$...$$`` and ``\\[...\\]``), in order."""
    blocks = [match.group("body").strip() for match in DISPLAY_MATH_RE.finditer(body)]
    blocks += [match.group("body").strip() for match in BRACKET_MATH_RE.finditer(body)]
    return [block for block in blocks if block]


def _draft_from_block_header(rest: str, line: int) -> _Draft | None:
    """Build a draft from a ``::: tanim 2.4 "Türev" ...`` fence header."""
    matched = match_keyword(strip_markup(rest))
    if matched is None:
        return None
    _, unit_type, tail = matched
    number_match = re.match(r"\s*(?P<number>\d+(?:\.\d+)*)", tail)
    if number_match is None:
        return None
    attributes = parse_attributes(tail)
    aliases = attributes.get("aliases", [])
    relations: list[tuple[Predicate, str]] = []
    for name, predicate in RELATION_ATTRIBUTES.items():
        value = attributes.get(name, [])
        targets = value if isinstance(value, list) else [value]
        relations.extend((predicate, target) for target in targets if target)
    identifier = attributes.get("id", "")
    return _Draft(
        unit_type=unit_type,
        number=number_match.group("number"),
        title=first_quoted(tail),
        line=line,
        body=[],
        aliases=list(aliases) if isinstance(aliases, list) else [str(aliases)],
        relations=relations,
        source_page=to_int(attributes.get("page")) if "page" in attributes else None,
        identifier=identifier if isinstance(identifier, str) else "",
    )


def tag_markdown(src: str, *, chapter: str, source: str = "") -> TagResult:
    """Tag a Markdown chapter into content units and relations.

    Args:
        src: Markdown source of one chapter.
        chapter: Chapter identifier every unit is filed under (``"bolum-2"``). A
            ``# Bölüm 2: Türev`` heading becomes the ``Section`` unit carrying exactly this id,
            so ``aicls:partOf`` never dangles.
        source: Optional file name, recorded on the result and on untagged spans.

    Returns:
        A :class:`~aicls.book.schema.TagResult`. ``triples`` may reference units defined in other
        files; :func:`aicls.book.load_units` resolves them against the database and drops the
        rest.
    """
    lines = src.splitlines()
    drafts: list[_Draft] = []
    untagged: list[UntaggedSpan] = []
    current: _Draft | None = None
    in_block = False
    index = 0

    while index < len(lines):
        line = lines[index]
        line_no = index + 1
        if in_block:
            if BLOCK_CLOSE_RE.match(line):
                in_block = False
                current = None
            elif current is not None:
                current.body.append(line)
            index += 1
            continue

        block_open = BLOCK_OPEN_RE.match(line)
        if block_open is not None:
            draft = _draft_from_block_header(block_open.group("rest"), line_no)
            if draft is None:
                untagged.append(UntaggedSpan(line_no, strip_markup(line), "unrecognised", source))
                index += 1
                continue
            drafts.append(draft)
            current = draft
            in_block = True
            index += 1
            continue

        heading = HEADING_RE.match(line)
        if heading is not None:
            text = heading.group("text")
            parsed = parse_unit_heading(text)
            if parsed is not None:
                unit_type, number, title = parsed
                current = _Draft(
                    unit_type=unit_type,
                    number=number,
                    title=title,
                    line=line_no,
                    body=[],
                    aliases=[],
                    relations=[],
                )
                drafts.append(current)
            else:
                current = None
                if looks_like_unit_heading(text):
                    untagged.append(
                        UntaggedSpan(line_no, strip_markup(text), "missing_number", source)
                    )
            index += 1
            continue

        if current is not None:
            current.body.append(line)
        index += 1

    return _assemble(drafts, untagged, chapter=chapter, source=source)


def _assemble(
    drafts: list[_Draft],
    untagged: list[UntaggedSpan],
    *,
    chapter: str,
    source: str,
) -> TagResult:
    """Turn drafts into units and derive the ``partOf`` / ``next`` / attribute triples."""
    section_draft = next((draft for draft in drafts if draft.unit_type is UnitType.SECTION), None)
    nodes: list[ContentUnit] = []
    order = 0
    for draft in drafts:
        if draft is section_draft:
            draft.identifier = draft.identifier or chapter
            nodes.append(draft.build(chapter, 0))
            continue
        order += 1
        nodes.append(draft.build(chapter, order))

    triples: list[Triple] = []
    seen: set[tuple[str, str, str]] = set()

    def add(subject: str, predicate: Predicate, obj: str) -> None:
        triple = Triple(subject=subject, predicate=predicate, object=obj, provenance=PROVENANCE)
        if triple.key() not in seen and subject != obj:
            seen.add(triple.key())
            triples.append(triple)

    body_nodes = [node for node in nodes if node.type is not UnitType.SECTION]
    if section_draft is not None:
        for node in body_nodes:
            add(node.id, Predicate.PART_OF, chapter)
    for left, right in pairwise(body_nodes):
        add(left.id, Predicate.NEXT, right.id)
    for draft, node in zip(drafts, nodes, strict=True):
        for predicate, target in draft.relations:
            add(node.id, predicate, target)

    if untagged:
        _LOGGER.warning(
            "book_untagged_headings",
            source=source,
            chapter=chapter,
            count=len(untagged),
            headings=[span.text for span in untagged],
        )
    return TagResult(
        nodes=nodes,
        triples=triples,
        untagged_spans=untagged,
        chapter=chapter,
        source=source,
    )


__all__ = [
    "BLOCK_CLOSE_RE",
    "BLOCK_OPEN_RE",
    "HEADING_RE",
    "PROVENANCE",
    "extract_latex_blocks",
    "tag_markdown",
]
