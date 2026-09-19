r"""LaTeX tagger: theorem-like environments carrying a ``\label`` (veri-modeli.md §5.3).

The printed book is written in LaTeX, so the same units arrive as environments::

    \section{Bölüm 4: Belirli İntegral}

    \begin{tanim}[Riemann Toplamı]\label{tanim:4.1}
    ...
    \[ S_n = \sum_{i=1}^{n} f(x_i^*)\,\Delta x_i \]
    \end{tanim}

The environment name gives the :class:`~aicls.book.schema.UnitType` (Turkish and English names
are both accepted), the optional bracket argument gives the title and the ``\label`` gives the
printed number — ``\label{tanim:4.1}`` and ``\label{tanim-4.1}`` both yield ``"4.1"``. An
environment **without** a label cannot be given a stable identifier and is reported as an
:class:`~aicls.book.schema.UntaggedSpan` (``kind="missing_label"``), exactly like an unnumbered
Markdown heading; so is a sectioning command whose title starts with a unit keyword.

Relations are written as a trailing comment on the ``\begin`` line, reusing the Markdown block
syntax::

    \begin{ornek}[Alan Hesabı]\label{ornek:4.4} % illustrates=[tanim-4.2]
"""

from __future__ import annotations

from itertools import pairwise
import re

from aicls.book.ingest.common import (
    RELATION_ATTRIBUTES,
    looks_like_unit_heading,
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

#: Environment name (casefolded) → unit type. English names ease import of existing sources.
ENVIRONMENT_TYPES: dict[str, UnitType] = {
    "tanim": UnitType.DEFINITION,
    "tanım": UnitType.DEFINITION,
    "definition": UnitType.DEFINITION,
    "teorem": UnitType.THEOREM,
    "theorem": UnitType.THEOREM,
    "lemma": UnitType.LEMMA,
    "onerme": UnitType.LEMMA,
    "sonuc": UnitType.COROLLARY,
    "sonuç": UnitType.COROLLARY,
    "corollary": UnitType.COROLLARY,
    "kanit": UnitType.PROOF,
    "kanıt": UnitType.PROOF,
    "ispat": UnitType.PROOF,
    "proof": UnitType.PROOF,
    "ornek": UnitType.EXAMPLE,
    "örnek": UnitType.EXAMPLE,
    "example": UnitType.EXAMPLE,
    "alistirma": UnitType.EXERCISE,
    "alıştırma": UnitType.EXERCISE,
    "exercise": UnitType.EXERCISE,
    "sekil": UnitType.FIGURE,
    "şekil": UnitType.FIGURE,
    "figure": UnitType.FIGURE,
    "tablo": UnitType.TABLE,
    "table": UnitType.TABLE,
    "not": UnitType.REMARK,
    "uyari": UnitType.REMARK,
    "remark": UnitType.REMARK,
}

#: ``\begin{tanim}[Riemann Toplamı]`` — opening of a unit environment.
BEGIN_RE = re.compile(r"\\begin\{(?P<env>[A-Za-zÇĞİÖŞÜçğıöşü*]+)\}(?:\[(?P<title>[^\]]*)\])?")
#: ``\end{tanim}`` — closing of a unit environment.
END_RE = re.compile(r"\\end\{(?P<env>[A-Za-zÇĞİÖŞÜçğıöşü*]+)\}")
#: ``\label{tanim:4.1}`` / ``\label{tanim-4.1}``.
LABEL_RE = re.compile(r"\\label\{(?P<label>[^}]*)\}")
#: ``\chapter{...}`` / ``\section{...}`` / ``\subsection{...}``.
SECTIONING_RE = re.compile(r"\\(?:chapter|section|subsection|subsubsection)\*?\{(?P<title>[^}]*)\}")
#: ``\[ ... \]`` and ``$$ ... $$`` display math kept as LaTeX blocks.
DISPLAY_MATH_RE = re.compile(r"\\\[(?P<body>.+?)\\\]|\$\$(?P<alt>.+?)\$\$", re.DOTALL)
#: Number carried by a label: everything after the last ``:`` or ``-``.
LABEL_NUMBER_RE = re.compile(r"(?:^|[:\-])(?P<number>\d+(?:\.\d+)*)\s*$")

#: Provenance written on every triple this tagger produces.
PROVENANCE = "latex_tagger"


def extract_latex_blocks(body: str) -> list[str]:
    """Return the display-math blocks of ``body`` (``\\[...\\]`` and ``$$...$$``), in order."""
    blocks = [
        (match.group("body") or match.group("alt") or "").strip()
        for match in DISPLAY_MATH_RE.finditer(body)
    ]
    return [block for block in blocks if block]


def label_number(label: str) -> str:
    """Extract the printed number from a label (``"tanim:4.1"`` → ``"4.1"``)."""
    match = LABEL_NUMBER_RE.search(label.strip())
    return match.group("number") if match else ""


def tag_latex(src: str, *, chapter: str, source: str = "") -> TagResult:
    r"""Tag a LaTeX chapter into content units and relations.

    Args:
        src: LaTeX source of one chapter.
        chapter: Chapter identifier every unit is filed under (``"bolum-4"``); a sectioning
            command whose title reads ``"Bölüm 4: ..."`` becomes the ``Section`` unit with
            exactly this id.
        source: Optional file name, recorded on the result and on untagged spans.

    Returns:
        A :class:`~aicls.book.schema.TagResult` with the same semantics as
        :func:`aicls.book.tag_markdown`.
    """
    lines = src.splitlines()
    nodes: list[ContentUnit] = []
    untagged: list[UntaggedSpan] = []
    relations: list[tuple[str, Predicate, str]] = []
    section_id: str | None = None
    order = 0
    index = 0

    while index < len(lines):
        line = lines[index]
        line_no = index + 1
        begin = BEGIN_RE.search(line)
        if begin is not None and begin.group("env").casefold() in ENVIRONMENT_TYPES:
            unit_type = ENVIRONMENT_TYPES[begin.group("env").casefold()]
            end_index = _find_end(lines, index, begin.group("env"))
            body_lines = lines[index + 1 : end_index]
            header = line[begin.end() :]
            number = label_number(_first_label([header, *body_lines]))
            if not number:
                untagged.append(UntaggedSpan(line_no, strip_markup(line), "missing_label", source))
                index = end_index + 1
                continue
            order += 1
            attributes = parse_attributes(header)
            node = _build_unit(
                unit_type=unit_type,
                number=number,
                title=(begin.group("title") or "").strip(),
                chapter=chapter,
                order=order,
                body_lines=body_lines,
                attributes=attributes,
            )
            nodes.append(node)
            for name, predicate in RELATION_ATTRIBUTES.items():
                value = attributes.get(name, [])
                targets = value if isinstance(value, list) else [value]
                relations.extend((node.id, predicate, target) for target in targets if target)
            index = end_index + 1
            continue

        sectioning = SECTIONING_RE.search(line)
        if sectioning is not None:
            title = sectioning.group("title")
            parsed = parse_unit_heading(title)
            if parsed is not None and parsed[0] is UnitType.SECTION:
                section_id = chapter
                nodes.insert(
                    0,
                    ContentUnit(
                        id=chapter,
                        type=UnitType.SECTION,
                        chapter=chapter,
                        number=parsed[1],
                        title_tr=parsed[2],
                        order_in_chapter=0,
                    ),
                )
            elif looks_like_unit_heading(title):
                untagged.append(UntaggedSpan(line_no, strip_markup(title), "unrecognised", source))
        index += 1

    triples = _relations_to_triples(nodes, relations, chapter=chapter, has_section=bool(section_id))
    if untagged:
        _LOGGER.warning(
            "book_untagged_environments",
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


def _build_unit(
    *,
    unit_type: UnitType,
    number: str,
    title: str,
    chapter: str,
    order: int,
    body_lines: list[str],
    attributes: dict[str, list[str] | str],
) -> ContentUnit:
    """Assemble one unit from a parsed environment."""
    body = "\n".join(LABEL_RE.sub("", line) for line in body_lines).strip()
    aliases = attributes.get("aliases", [])
    identifier = attributes.get("id", "")
    return ContentUnit(
        id=identifier if isinstance(identifier, str) and identifier else unit_id(unit_type, number),
        type=unit_type,
        chapter=chapter,
        number=number,
        title_tr=title,
        aliases=list(aliases) if isinstance(aliases, list) else [str(aliases)],
        body_md=body,
        latex_blocks=extract_latex_blocks(body),
        source_page=to_int(attributes.get("page")) if "page" in attributes else None,
        order_in_chapter=order,
    )


def _first_label(chunks: list[str]) -> str:
    """Return the first ``\\label{...}`` value found in ``chunks``."""
    for chunk in chunks:
        match = LABEL_RE.search(chunk)
        if match is not None:
            return match.group("label")
    return ""


def _find_end(lines: list[str], start: int, env: str) -> int:
    """Index of the ``\\end{env}`` line closing the environment opened at ``start``."""
    depth = 0
    for index in range(start, len(lines)):
        line = lines[index]
        if index > start and BEGIN_RE.search(line) and f"\\begin{{{env}}}" in line:
            depth += 1
        end = END_RE.search(line)
        if end is not None and end.group("env") == env:
            if depth == 0:
                return index
            depth -= 1
    return len(lines)


def _relations_to_triples(
    nodes: list[ContentUnit],
    relations: list[tuple[str, Predicate, str]],
    *,
    chapter: str,
    has_section: bool,
) -> list[Triple]:
    """Derive ``partOf`` / ``next`` edges and merge the explicit relation attributes."""
    triples: list[Triple] = []
    seen: set[tuple[str, str, str]] = set()

    def add(subject: str, predicate: Predicate, obj: str) -> None:
        triple = Triple(subject=subject, predicate=predicate, object=obj, provenance=PROVENANCE)
        if triple.key() not in seen and subject != obj:
            seen.add(triple.key())
            triples.append(triple)

    body_nodes = [node for node in nodes if node.type is not UnitType.SECTION]
    if has_section:
        for node in body_nodes:
            add(node.id, Predicate.PART_OF, chapter)
    for left, right in pairwise(body_nodes):
        add(left.id, Predicate.NEXT, right.id)
    for subject, predicate, target in relations:
        add(subject, predicate, target)
    return triples


__all__ = [
    "BEGIN_RE",
    "END_RE",
    "ENVIRONMENT_TYPES",
    "LABEL_RE",
    "PROVENANCE",
    "extract_latex_blocks",
    "label_number",
    "tag_latex",
]
