"""JSON-LD and Turtle export, and the packaged context/vocabulary (validated with rdflib)."""

from __future__ import annotations

import json

import pytest

from aicls.book import (
    AssetRef,
    ContentUnit,
    Predicate,
    TagResult,
    Triple,
    UnitType,
    context_document,
    export_jsonld,
    export_turtle,
    load_context,
    load_vocab_graph,
    predicate_uri,
    unit_from_jsonld,
    unit_to_jsonld,
    unit_type_uri,
    units_from_jsonld,
)
from aicls.book.export import edge_count, triples_from_jsonld
from aicls.book.vocab import AICLS_NS, expand_curie, strip_unit_curie, vocab_path
from aicls.core.errors import InputValidationError

UNIT = ContentUnit(
    id="tanim-2.4",
    type=UnitType.DEFINITION,
    chapter="bolum-2",
    number="2.4",
    title_tr="Türev",
    aliases=["türev", "türevin tanımı"],
    body_md="$f$ fonksiyonu $a$ noktasında ...",
    latex_blocks=["f'(a) = \\lim_{h \\to 0} \\frac{f(a+h)-f(a)}{h}"],
    assets=[AssetRef(id="sekil-2.3")],
    source_page=58,
)
TRIPLES = [
    Triple(subject="tanim-2.4", predicate=Predicate.REQUIRES, object="tanim-2.1"),
    Triple(subject="tanim-2.4", predicate=Predicate.PART_OF, object="bolum-2"),
    Triple(subject="tanim-2.4", predicate=Predicate.NEXT, object="ornek-2.7"),
]


def test_jsonld_roundtrip_unit() -> None:
    """A unit survives ``unit_to_jsonld`` → ``unit_from_jsonld`` unchanged."""
    document = unit_to_jsonld(UNIT)
    assert document["@id"] == "aicls:tanim-2.4"
    assert document["@type"] == "aicls:Definition"
    assert document["name"] == "Türev"
    assert document["altLabel"] == ["türev", "türevin tanımı"]
    assert document["figure"] == ["aicls:sekil-2.3"]
    assert document["source"] == "s. 58"
    restored = unit_from_jsonld(document)
    assert restored.model_copy(update={"jsonld": {}}) == UNIT
    assert json.loads(json.dumps(document, ensure_ascii=False)) == document


def test_jsonld_carries_the_relation_terms() -> None:
    """Edges of the unit become the context's relation terms, sets stay lists.

    The unit's own ``assets`` also render as ``figure``, so reading the document back yields one
    ``hasFigure`` edge on top of the three that were passed in — the graph and the node object
    stay a single source of truth.
    """
    document = unit_to_jsonld(UNIT, TRIPLES)
    assert document["requires"] == ["aicls:tanim-2.1"]
    assert document["partOf"] == "aicls:bolum-2"
    assert document["next"] == "aicls:ornek-2.7"
    assert document["figure"] == ["aicls:sekil-2.3"]
    assert {(t.subject, t.predicate, t.object) for t in triples_from_jsonld(document)} == {
        (t.subject, t.predicate, t.object) for t in TRIPLES
    } | {("tanim-2.4", Predicate.HAS_FIGURE, "sekil-2.3")}


def test_export_jsonld_is_a_single_graph_with_inline_context() -> None:
    """The export embeds the context (never a URL), so no consumer fetches anything."""
    document = export_jsonld([UNIT], TRIPLES)
    assert isinstance(document["@context"], dict)
    assert document["@context"]["aicls"] == AICLS_NS
    assert [node["@id"] for node in document["@graph"]] == ["aicls:tanim-2.4"]
    units, triples = units_from_jsonld(document)
    assert [unit.id for unit in units] == ["tanim-2.4"]
    assert len(triples) == len(TRIPLES) + 1  # + the unit's own figure reference
    assert units[0].assets[0].id == "sekil-2.3"


def test_book_export_turtle_reparses_with_rdflib_triple_count_equals_edges(
    chapter_two: TagResult,
) -> None:
    """Turtle re-parses and holds exactly as many relation edges as the graph has."""
    turtle = export_turtle(chapter_two.nodes, chapter_two.triples)
    assert edge_count(turtle) == len(chapter_two.triples)

    rdflib = pytest.importorskip("rdflib")
    graph = rdflib.Graph()
    graph.parse(data=turtle, format="turtle")
    subject = rdflib.URIRef(f"{AICLS_NS}tanim-2.4")
    assert (
        subject,
        rdflib.RDF.type,
        rdflib.URIRef(unit_type_uri(UnitType.DEFINITION)),
    ) in graph
    assert (
        subject,
        rdflib.URIRef(predicate_uri(Predicate.REQUIRES)),
        rdflib.URIRef(f"{AICLS_NS}tanim-2.1"),
    ) in graph


def test_turtle_export_is_stable_across_calls() -> None:
    """Two exports of the same graph are byte-identical (reports quote a hash of this file)."""
    assert export_turtle([UNIT], TRIPLES) == export_turtle([UNIT], TRIPLES)


def test_packaged_vocab_parses_with_rdflib() -> None:
    """``vocab.ttl`` is valid RDF and declares a class for every unit type."""
    graph = load_vocab_graph()
    rdflib = pytest.importorskip("rdflib")
    assert vocab_path().is_file()
    for unit_type in UnitType:
        assert (rdflib.URIRef(unit_type_uri(unit_type)), None, None) in graph
    for predicate in Predicate:
        assert (rdflib.URIRef(predicate_uri(predicate)), None, None) in graph


def test_packaged_context_defines_every_term_the_exporter_uses() -> None:
    """Every JSON-LD key produced by the exporter is a term of ``context.jsonld``."""
    context = load_context()
    assert context_document() == {"@context": context}
    document = unit_to_jsonld(UNIT, TRIPLES)
    for key in document:
        if key.startswith("@"):
            continue
        assert key in context, f"{key} is not defined in context.jsonld"


def test_curie_helpers() -> None:
    """CURIE expansion accepts prefixes and absolute IRIs and rejects unknown prefixes."""
    assert expand_curie("aicls:requires") == f"{AICLS_NS}requires"
    assert expand_curie("https://example.org/x") == "https://example.org/x"
    assert strip_unit_curie(f"{AICLS_NS}tanim-2.4") == "tanim-2.4"
    assert strip_unit_curie("aicls:tanim-2.4") == "tanim-2.4"
    assert strip_unit_curie("tanim-2.4") == "tanim-2.4"
    with pytest.raises(InputValidationError):
        expand_curie("bilinmeyen:x")


@pytest.mark.parametrize(
    "document",
    [{}, {"@id": "aicls:x"}, {"@id": "aicls:x", "@type": "aicls:Bilinmeyen"}],
)
def test_unit_from_jsonld_rejects_bad_documents(document: dict[str, str]) -> None:
    """A node object without a usable ``@id``/``@type`` is an input validation error."""
    with pytest.raises(InputValidationError):
        unit_from_jsonld(document)
