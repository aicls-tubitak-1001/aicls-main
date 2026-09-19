"""Textbook knowledge graph (JSON-LD/RDF): taggers, loader, aliases, query and resolver (İP-8).

Public contract (``modules/book.json``, docs/mimari/moduller.md#aiclsbook); the schema lives in
docs/mimari/veri-modeli.md §5 and the storage decision in ADR-0019. The package depends only on
:mod:`aicls.core` and :mod:`aicls.db` (plus ``rdflib`` for the RDF export) and never imports
torch or transformers (ADR-0002).

Typical use::

    from aicls.book import BookResolver, coverage, load_units, tag_markdown

    result = tag_markdown(chapter_md, chapter="bolum-2")
    assert coverage(result) == 1.0  # İP-8: 100 % tagged, else the load is refused
    load_units(result.nodes, result.triples, BookRepository(session))
    BookResolver(BookRepository(session)).resolve("türev tanımını getir")
"""

from __future__ import annotations

from aicls.book.aliases import (
    TYPE_WORDS,
    AliasEntry,
    generate_alias_entries,
    generate_aliases,
)
from aicls.book.export import (
    export_jsonld,
    export_turtle,
    triples_from_jsonld,
    unit_from_jsonld,
    unit_to_jsonld,
    units_from_jsonld,
)
from aicls.book.ingest import (
    UntaggedHeadingError,
    coverage,
    require_full_coverage,
    tag_latex,
    tag_markdown,
    untagged_headings,
)
from aicls.book.loader import (
    load_markdown_dir,
    load_tag_result,
    load_units,
    tag_markdown_dir,
    triples_from_repo,
    units_from_repo,
)
from aicls.book.query import get_unit, normalize_query, related, search, unit_from_row
from aicls.book.resolver import BookResolver, detect_unit_type
from aicls.book.schema import (
    SLUG_BY_TYPE,
    TYPE_BY_SLUG,
    AssetRef,
    ContentUnit,
    Hit,
    Predicate,
    ResolvedUnit,
    TagResult,
    Triple,
    UnitType,
    UntaggedSpan,
    unit_id,
)
from aicls.book.vocab import (
    AICLS_NS,
    NAMESPACES,
    context_document,
    context_path,
    load_context,
    load_vocab_graph,
    predicate_uri,
    unit_type_uri,
    vocab_path,
)

__all__ = [
    "AICLS_NS",
    "NAMESPACES",
    "SLUG_BY_TYPE",
    "TYPE_BY_SLUG",
    "TYPE_WORDS",
    "AliasEntry",
    "AssetRef",
    "BookResolver",
    "ContentUnit",
    "Hit",
    "Predicate",
    "ResolvedUnit",
    "TagResult",
    "Triple",
    "UnitType",
    "UntaggedHeadingError",
    "UntaggedSpan",
    "context_document",
    "context_path",
    "coverage",
    "detect_unit_type",
    "export_jsonld",
    "export_turtle",
    "generate_alias_entries",
    "generate_aliases",
    "get_unit",
    "load_context",
    "load_markdown_dir",
    "load_tag_result",
    "load_units",
    "load_vocab_graph",
    "normalize_query",
    "predicate_uri",
    "related",
    "require_full_coverage",
    "search",
    "tag_latex",
    "tag_markdown",
    "tag_markdown_dir",
    "triples_from_jsonld",
    "triples_from_repo",
    "unit_from_jsonld",
    "unit_from_row",
    "unit_id",
    "unit_to_jsonld",
    "unit_type_uri",
    "units_from_jsonld",
    "units_from_repo",
    "untagged_headings",
    "vocab_path",
]
