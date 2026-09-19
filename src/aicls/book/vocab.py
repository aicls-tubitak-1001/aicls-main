"""JSON-LD context and RDF vocabulary of the book graph (ADR-0019, veri-modeli.md §5.1).

``context.jsonld`` and ``vocab.ttl`` ship inside the package (``[tool.setuptools.package-data]``)
and are the concrete deliverables the proposal calls "JSON-LD metadata şemaları" and
"RDF tripleri". Everything here is read from those two files — no namespace is duplicated in
Python — so a change to the vocabulary cannot silently disagree with the exporter.

Nothing in this module reaches the network: the context is embedded in every export and
``rdflib`` only ever parses local strings.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from aicls.book.schema import Predicate, UnitType
from aicls.core.errors import InputValidationError
from aicls.core.messages_tr import msg
from aicls.core.paths import package_resource

#: Base IRI of the AI-CLS vocabulary (``aicls:`` prefix).
AICLS_NS = "https://ai-cls.example/ns#"
#: Base IRI unit identifiers are minted under (``aicls:tanim-2.4``).
UNIT_NS = AICLS_NS
#: Namespaces reused from the wider linked-data world.
SCHEMA_NS = "https://schema.org/"
SKOS_NS = "http://www.w3.org/2004/02/skos/core#"
DCTERMS_NS = "http://purl.org/dc/terms/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

#: Prefix → namespace map used when expanding a CURIE and when binding an ``rdflib`` graph.
NAMESPACES: dict[str, str] = {
    "aicls": AICLS_NS,
    "schema": SCHEMA_NS,
    "skos": SKOS_NS,
    "dcterms": DCTERMS_NS,
    "rdf": RDF_NS,
}

#: Name of the packaged JSON-LD context file.
CONTEXT_RESOURCE = "context.jsonld"
#: Name of the packaged Turtle vocabulary file.
VOCAB_RESOURCE = "vocab.ttl"


def context_path() -> Path:
    """Filesystem path of the packaged ``context.jsonld``."""
    return package_resource("aicls.book", CONTEXT_RESOURCE)


def vocab_path() -> Path:
    """Filesystem path of the packaged ``vocab.ttl``."""
    return package_resource("aicls.book", VOCAB_RESOURCE)


@lru_cache(maxsize=1)
def _context_document() -> dict[str, Any]:
    """Parse ``context.jsonld`` once."""
    raw = context_path().read_text(encoding="utf-8")
    document: dict[str, Any] = json.loads(raw)
    return document


def load_context() -> dict[str, Any]:
    """Return the ``@context`` mapping of the packaged context document (a fresh copy)."""
    return dict(_context_document()["@context"])


def context_document() -> dict[str, Any]:
    """Return the whole packaged context document, ``{"@context": {...}}`` (a fresh copy)."""
    return {"@context": load_context()}


def load_vocab_text() -> str:
    """Return the Turtle source of the packaged vocabulary."""
    return vocab_path().read_text(encoding="utf-8")


def load_vocab_graph() -> Any:
    """Parse ``vocab.ttl`` with ``rdflib`` and return the resulting ``Graph``.

    Raises:
        ImportError: If ``rdflib`` is not installed (it is a runtime dependency; the import is
            local so the rest of :mod:`aicls.book` works without it).
    """
    from rdflib import Graph

    graph = Graph()
    for prefix, namespace in NAMESPACES.items():
        graph.bind(prefix, namespace)
    graph.parse(data=load_vocab_text(), format="turtle")
    return graph


def expand_curie(curie: str) -> str:
    """Expand ``"aicls:requires"`` into its full IRI.

    Args:
        curie: A ``prefix:local`` CURIE using a prefix of :data:`NAMESPACES`, or an absolute IRI.

    Returns:
        The absolute IRI.

    Raises:
        InputValidationError: If the prefix is unknown.
    """
    if curie.startswith(("http://", "https://")):
        return curie
    prefix, _, local = curie.partition(":")
    namespace = NAMESPACES.get(prefix)
    if namespace is None:
        raise InputValidationError(
            msg("VALIDATION_ERROR"),
            details={"field": "curie", "value": curie, "allowed": sorted(NAMESPACES)},
        )
    return f"{namespace}{local}"


def predicate_uri(predicate: Predicate | str) -> str:
    """Absolute IRI of a predicate (``Predicate.REQUIRES`` → ``…/ns#requires``)."""
    return expand_curie(str(getattr(predicate, "value", predicate)))


def unit_type_uri(unit_type: UnitType | str) -> str:
    """Absolute IRI of a unit type (``UnitType.DEFINITION`` → ``…/ns#Definition``)."""
    return f"{AICLS_NS}{getattr(unit_type, 'value', unit_type)!s}"


def unit_uri(identifier: str) -> str:
    """Absolute IRI of a unit identifier (``"tanim-2.4"`` → ``…/ns#tanim-2.4``)."""
    return f"{UNIT_NS}{identifier}"


def unit_curie(identifier: str) -> str:
    """CURIE of a unit identifier (``"tanim-2.4"`` → ``"aicls:tanim-2.4"``)."""
    return f"aicls:{identifier}"


def strip_unit_curie(value: str) -> str:
    """Inverse of :func:`unit_curie`; accepts a bare id, a CURIE or a full IRI."""
    if value.startswith(AICLS_NS):
        return value[len(AICLS_NS) :]
    if value.startswith("aicls:"):
        return value[len("aicls:") :]
    return value


__all__ = [
    "AICLS_NS",
    "CONTEXT_RESOURCE",
    "DCTERMS_NS",
    "NAMESPACES",
    "RDF_NS",
    "SCHEMA_NS",
    "SKOS_NS",
    "UNIT_NS",
    "VOCAB_RESOURCE",
    "context_document",
    "context_path",
    "expand_curie",
    "load_context",
    "load_vocab_graph",
    "load_vocab_text",
    "predicate_uri",
    "strip_unit_curie",
    "unit_curie",
    "unit_type_uri",
    "unit_uri",
    "vocab_path",
]
