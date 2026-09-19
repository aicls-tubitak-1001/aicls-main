"""JSON-LD and Turtle export — the concrete "RDF tripleri" deliverable of İP-8 (ADR-0019).

:func:`export_jsonld` emits a single ``@graph`` document carrying the packaged ``@context``
inline (never a URL, so no consumer ever fetches anything); :func:`export_turtle` builds the same
graph with ``rdflib`` and serialises it, which is what ``GET /book/export.ttl`` returns.

:func:`unit_to_jsonld` and :func:`unit_from_jsonld` are the per-unit round trip the loader uses
to fill ``book_units.jsonld`` and the query layer uses to read it back.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from aicls.book.schema import AssetRef, ContentUnit, Predicate, Triple, UnitType
from aicls.book.vocab import (
    AICLS_NS,
    DCTERMS_NS,
    NAMESPACES,
    SCHEMA_NS,
    SKOS_NS,
    context_document,
    load_context,
    predicate_uri,
    strip_unit_curie,
    unit_curie,
    unit_type_uri,
    unit_uri,
)
from aicls.core.errors import InputValidationError
from aicls.core.messages_tr import msg

#: Language tag of every literal in the book graph.
LANGUAGE = "tr"

#: JSON-LD term used for each predicate (the keys defined in ``context.jsonld``).
TERM_BY_PREDICATE: dict[Predicate, str] = {
    Predicate.PART_OF: "partOf",
    Predicate.REQUIRES: "requires",
    Predicate.ILLUSTRATES: "illustrates",
    Predicate.PROVES: "proves",
    Predicate.NEXT: "next",
    Predicate.HAS_FIGURE: "figure",
    Predicate.RELATED: "related",
}
#: Terms whose JSON-LD container is ``@set`` (a list even with one member).
SET_TERMS = frozenset({"requires", "illustrates", "figure", "related", "altLabel"})

_PREDICATE_BY_TERM = {term: predicate for predicate, term in TERM_BY_PREDICATE.items()}


def unit_to_jsonld(unit: ContentUnit, triples: Iterable[Triple] = ()) -> dict[str, Any]:
    """Render one unit as a JSON-LD node object (docs/mimari/veri-modeli.md §5.2).

    Args:
        unit: The unit to render.
        triples: Edges of the graph; those whose subject is ``unit.id`` become relation terms.

    Returns:
        A node object using the terms of the packaged context (``@id``, ``@type``, ``name``,
        ``altLabel``, ``bodyMarkdown``, ``latex``, ``requires`` ...).
    """
    document: dict[str, Any] = {
        "@id": unit_curie(unit.id),
        "@type": f"aicls:{unit.type.value}",
        "identifier": unit.number or unit.id,
        "chapter": unit.chapter,
        "number": unit.number,
        "name": unit.title_tr,
        "inLanguage": LANGUAGE,
    }
    if unit.aliases:
        document["altLabel"] = list(dict.fromkeys(unit.aliases))
    if unit.body_md:
        document["bodyMarkdown"] = unit.body_md
    if unit.latex_blocks:
        document["latex"] = list(unit.latex_blocks)
    if unit.assets:
        document["figure"] = [unit_curie(asset.id) for asset in unit.assets]
    if unit.source_page is not None:
        document["source"] = f"s. {unit.source_page}"
    for triple in triples:
        if triple.subject != unit.id:
            continue
        term = TERM_BY_PREDICATE[triple.predicate]
        target = unit_curie(triple.object)
        if term in SET_TERMS:
            targets = document.setdefault(term, [])
            if target not in targets:
                targets.append(target)
        else:
            document[term] = target
    return document


def unit_from_jsonld(document: Mapping[str, Any]) -> ContentUnit:
    """Rebuild a :class:`~aicls.book.schema.ContentUnit` from a JSON-LD node object.

    The inverse of :func:`unit_to_jsonld` for everything the model carries; relation terms are
    read back by :func:`triples_from_jsonld` instead.

    Raises:
        InputValidationError: If ``@id`` or ``@type`` is missing or names an unknown unit type.
    """
    identifier = strip_unit_curie(str(document.get("@id", "")))
    raw_type = strip_unit_curie(str(document.get("@type", "")))
    if not identifier or not raw_type:
        raise InputValidationError(
            msg("VALIDATION_ERROR"), details={"field": "jsonld", "value": dict(document)}
        )
    try:
        unit_type = UnitType(raw_type)
    except ValueError as exc:
        raise InputValidationError(
            msg("VALIDATION_ERROR"),
            details={
                "field": "@type",
                "value": raw_type,
                "allowed": [member.value for member in UnitType],
            },
        ) from exc
    source = str(document.get("source", ""))
    page = source.rpartition(" ")[2]
    return ContentUnit(
        id=identifier,
        type=unit_type,
        chapter=str(document.get("chapter", "")),
        number=str(document.get("number", "")),
        title_tr=str(document.get("name", "")),
        aliases=[str(alias) for alias in document.get("altLabel", [])],
        body_md=str(document.get("bodyMarkdown", "")),
        latex_blocks=[str(block) for block in document.get("latex", [])],
        assets=[AssetRef(id=strip_unit_curie(str(ref))) for ref in document.get("figure", [])],
        source_page=int(page) if page.isdigit() else None,
        jsonld=dict(document),
    )


def triples_from_jsonld(document: Mapping[str, Any], *, provenance: str = "jsonld") -> list[Triple]:
    """Read the relation terms of a node object back into triples."""
    subject = strip_unit_curie(str(document.get("@id", "")))
    triples: list[Triple] = []
    for term, predicate in _PREDICATE_BY_TERM.items():
        value = document.get(term)
        if value is None:
            continue
        targets = value if isinstance(value, list) else [value]
        for target in targets:
            triples.append(
                Triple(
                    subject=subject,
                    predicate=predicate,
                    object=strip_unit_curie(str(target)),
                    provenance=provenance,
                )
            )
    return triples


def export_jsonld(units: Iterable[ContentUnit], triples: Iterable[Triple] = ()) -> dict[str, Any]:
    """Export the whole book as one JSON-LD document with an inline ``@context`` and ``@graph``.

    Args:
        units: Units to include, in the order they should appear.
        triples: Edges; each is attached to the node object of its subject.

    Returns:
        ``{"@context": {...}, "@graph": [node, ...]}`` — ``GET /book/export.jsonld``.
    """
    edges = list(triples)
    document = context_document()
    document["@graph"] = [unit_to_jsonld(unit, edges) for unit in units]
    return document


def units_from_jsonld(
    document: Mapping[str, Any],
) -> tuple[list[ContentUnit], list[Triple]]:
    """Inverse of :func:`export_jsonld`: read a ``@graph`` document back."""
    nodes = document.get("@graph", [])
    units = [unit_from_jsonld(node) for node in nodes]
    triples: list[Triple] = []
    for node in nodes:
        triples.extend(triples_from_jsonld(node))
    return units, triples


def build_graph(units: Iterable[ContentUnit], triples: Iterable[Triple] = ()) -> Any:
    """Build the ``rdflib`` graph of the book (units as resources, triples as edges).

    Raises:
        ImportError: If ``rdflib`` is missing (a declared runtime dependency).
    """
    from rdflib import Graph, Literal, URIRef
    from rdflib.namespace import RDF

    graph = Graph()
    for prefix, namespace in NAMESPACES.items():
        graph.bind(prefix, namespace)

    def literal(subject: str, predicate: str, value: str, *, lang: str | None = LANGUAGE) -> None:
        graph.add((URIRef(unit_uri(subject)), URIRef(predicate), Literal(value, lang=lang)))

    for unit in units:
        subject = URIRef(unit_uri(unit.id))
        graph.add((subject, RDF.type, URIRef(unit_type_uri(unit.type))))
        literal(unit.id, f"{DCTERMS_NS}identifier", unit.number or unit.id, lang=None)
        literal(unit.id, f"{AICLS_NS}chapter", unit.chapter, lang=None)
        if unit.number:
            literal(unit.id, f"{AICLS_NS}number", unit.number, lang=None)
        if unit.title_tr:
            literal(unit.id, f"{SCHEMA_NS}name", unit.title_tr)
        for alias in dict.fromkeys(unit.aliases):
            literal(unit.id, f"{SKOS_NS}altLabel", alias)
        if unit.body_md:
            literal(unit.id, f"{AICLS_NS}bodyMarkdown", unit.body_md)
        for block in unit.latex_blocks:
            literal(unit.id, f"{AICLS_NS}latex", block, lang=None)
        if unit.source_page is not None:
            literal(unit.id, f"{DCTERMS_NS}source", f"s. {unit.source_page}", lang=None)

    for triple in triples:
        graph.add(
            (
                URIRef(unit_uri(triple.subject)),
                URIRef(predicate_uri(triple.predicate)),
                URIRef(unit_uri(triple.object)),
            )
        )
    return graph


def export_turtle(units: Iterable[ContentUnit], triples: Iterable[Triple] = ()) -> str:
    """Serialise the book graph as Turtle (``GET /book/export.ttl``).

    The output re-parses with ``rdflib`` into exactly the same triples, which is what
    ``test_book_export_turtle_reparses_with_rdflib_triple_count_equals_edges`` asserts.
    """
    return str(build_graph(units, triples).serialize(format="turtle"))


def edge_count(turtle: str) -> int:
    """Count the relation edges (predicates of :class:`~aicls.book.schema.Predicate`) in Turtle."""
    from rdflib import Graph, URIRef

    graph = Graph()
    graph.parse(data=turtle, format="turtle")
    wanted = {URIRef(predicate_uri(predicate)) for predicate in Predicate}
    return sum(1 for _, predicate, _ in graph if predicate in wanted)


def context_terms() -> Sequence[str]:
    """Term names defined by the packaged JSON-LD context (used by the schema tests)."""
    return sorted(load_context())


__all__ = [
    "LANGUAGE",
    "SET_TERMS",
    "TERM_BY_PREDICATE",
    "build_graph",
    "context_terms",
    "edge_count",
    "export_jsonld",
    "export_turtle",
    "triples_from_jsonld",
    "unit_from_jsonld",
    "unit_to_jsonld",
    "units_from_jsonld",
]
