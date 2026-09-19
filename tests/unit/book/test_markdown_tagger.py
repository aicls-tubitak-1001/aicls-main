"""Markdown tagger: the ``"Tanım 1.1:"`` pattern, ``:::`` blocks and untagged headings."""

from __future__ import annotations

from itertools import pairwise

import pytest

from aicls.book import ContentUnit, Predicate, TagResult, UnitType, coverage, tag_markdown
from aicls.book.ingest.markdown_tagger import extract_latex_blocks

CHAPTER = "bolum-1"


def unit_by_id(result: TagResult, unit_id: str) -> ContentUnit:
    """Return the node with ``unit_id`` (fails the test when it is missing)."""
    matches = [node for node in result.nodes if node.id == unit_id]
    assert matches, f"{unit_id} not tagged; got {[node.id for node in result.nodes]}"
    return matches[0]


def test_markdown_tagger_recognises_tanim_1_1(chapter_one: TagResult) -> None:
    """The plain "Tanım 1.1:" pattern named in the İP-8 risk plan becomes a Definition unit."""
    unit = unit_by_id(chapter_one, "tanim-1.1")
    assert unit.type is UnitType.DEFINITION
    assert unit.number == "1.1"
    assert unit.title_tr == "Fonksiyonun Limiti"
    assert unit.chapter == CHAPTER
    assert "\\lim_{x \\to a} f(x) = L" in unit.latex_blocks
    assert "delinmiş" in unit.body_md


def test_markdown_tagger_recognises_block_syntax_with_aliases(chapter_two: TagResult) -> None:
    """A ``::: tanim 2.4 "Türev" ...`` block yields the unit, its aliases and its relations."""
    unit = unit_by_id(chapter_two, "tanim-2.4")
    assert unit.type is UnitType.DEFINITION
    assert unit.title_tr == "Türev"
    assert unit.aliases == ["türevin tanımı", "diferansiyel katsayı"]
    assert unit.source_page == 58
    assert unit.latex_blocks == [
        "f'(a) = \\lim_{h \\to 0} \\frac{f(a+h) - f(a)}{h}",
    ]
    edges = {(triple.subject, triple.predicate, triple.object) for triple in chapter_two.triples}
    assert ("tanim-2.4", Predicate.REQUIRES, "tanim-2.1") in edges
    assert ("tanim-2.4", Predicate.HAS_FIGURE, "sekil-2.3") in edges
    assert ("kanit-2.6", Predicate.PROVES, "teorem-2.5") in edges
    assert ("ornek-2.7", Predicate.ILLUSTRATES, "tanim-2.4") in edges


def test_coverage_is_one_for_sample_chapter(chapter_one: TagResult, chapter_two: TagResult) -> None:
    """Both sample chapters are fully tagged — the İP-8 "%100 etiketleme" criterion."""
    for result in (chapter_one, chapter_two):
        assert result.untagged_spans == []
        assert coverage(result) == 1.0


def test_parenthesised_title_is_used(chapter_one: TagResult) -> None:
    """``## Teorem 1.2 (Sıkıştırma Teoremi):`` takes its title from the parentheses."""
    unit = unit_by_id(chapter_one, "teorem-1.2")
    assert unit.title_tr == "Sıkıştırma Teoremi"
    assert unit.type is UnitType.THEOREM


def test_section_heading_becomes_the_chapter_unit(chapter_two: TagResult) -> None:
    """``# Bölüm 2: Türev`` becomes the Section unit whose id is the chapter id."""
    section = unit_by_id(chapter_two, "bolum-2")
    assert section.type is UnitType.SECTION
    assert section.number == "2"
    assert section.order_in_chapter == 0
    body = [node for node in chapter_two.nodes if node.type is not UnitType.SECTION]
    assert [node.order_in_chapter for node in body] == list(range(1, len(body) + 1))
    part_of = {t.subject for t in chapter_two.triples if t.predicate is Predicate.PART_OF}
    assert part_of == {node.id for node in body}


def test_next_chain_follows_document_order(chapter_one: TagResult) -> None:
    """Consecutive body units are linked with ``aicls:next`` in document order."""
    body = [node.id for node in chapter_one.nodes if node.type is not UnitType.SECTION]
    chain = [(t.subject, t.object) for t in chapter_one.triples if t.predicate is Predicate.NEXT]
    assert chain == list(pairwise(body))


def test_untagged_heading_is_reported_with_line_and_text(untagged_chapter: TagResult) -> None:
    """A keyword heading without a number is reported, not silently dropped."""
    assert coverage(untagged_chapter) == 0.5
    texts = [span.text for span in untagged_chapter.untagged_spans]
    assert texts == ["Tanım: Belirsiz İntegral", "Teorem: Analizin Temel Teoremi"]
    assert all(span.kind == "missing_number" for span in untagged_chapter.untagged_spans)
    assert all(span.line > 0 for span in untagged_chapter.untagged_spans)
    assert all(span.source == "bolum-03-etiketsiz.md" for span in untagged_chapter.untagged_spans)


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("## Tanım 2.4: Türev", ("tanim-2.4", "Türev")),
        ("### Teorem 3.1 (Ortalama Değer Teoremi):", ("teorem-3.1", "Ortalama Değer Teoremi")),
        ("## Örnek 1.3 - Belirsiz Oran", ("ornek-1.3", "Belirsiz Oran")),
        ("#### Şekil 4.2. Teğet", ("sekil-4.2", "Teğet")),
        ("## İSPAT 5.1: Tümevarım", ("kanit-5.1", "Tümevarım")),
        ("## Alıştırma 6.10: Zincir", ("alistirma-6.10", "Zincir")),
    ],
)
def test_heading_variants_are_tagged(heading: str, expected: tuple[str, str]) -> None:
    """Colon, dash, dot and parenthesis forms all resolve to the same identifier scheme."""
    result = tag_markdown(f"{heading}\n\ngövde\n", chapter="bolum-9")
    assert result.untagged_spans == []
    assert (result.nodes[0].id, result.nodes[0].title_tr) == expected


@pytest.mark.parametrize(
    "heading",
    ["## Tanım: Süreklilik", "## Teorem — Ara Değer", "## Şekil: Grafik"],
)
def test_numberless_headings_are_untagged(heading: str) -> None:
    """Without a number there is no stable identifier, so the heading stays untagged."""
    result = tag_markdown(f"{heading}\n\ngövde\n", chapter="bolum-9")
    assert result.nodes == []
    assert [span.text for span in result.untagged_spans] == [heading.lstrip("# ")]


def test_non_unit_heading_is_ignored() -> None:
    """A heading that is not a unit heading is neither tagged nor reported."""
    result = tag_markdown("## Giriş\n\nmetin\n", chapter="bolum-9")
    assert result.nodes == []
    assert result.untagged_spans == []
    assert coverage(result) == 1.0


def test_keyword_prefix_is_not_a_keyword() -> None:
    """ "Tanımlayıcı" must not be read as the keyword "Tanım"."""
    result = tag_markdown("## Tanımlayıcı 1.1: Ad\n", chapter="bolum-9")
    assert result.nodes == []
    assert result.untagged_spans == []


def test_block_without_number_is_reported() -> None:
    """A ``:::`` fence that names no number cannot be tagged."""
    result = tag_markdown('::: tanim "Türev"\ngövde\n:::\n', chapter="bolum-9")
    assert result.nodes == []
    assert [span.kind for span in result.untagged_spans] == ["unrecognised"]


def test_block_id_override_wins() -> None:
    """``id=`` on a block overrides the minted identifier."""
    result = tag_markdown('::: tanim 2.4 "Türev" id=turev-ozel\ngövde\n:::\n', chapter="bolum-2")
    assert result.nodes[0].id == "turev-ozel"


def test_extract_latex_blocks_handles_both_delimiters() -> None:
    """``$$...$$`` and ``\\[...\\]`` are both collected, in that order."""
    blocks = extract_latex_blocks("metin $$a=b$$ ve \\[c=d\\] son")
    assert blocks == ["a=b", "c=d"]


def test_empty_source_is_fully_covered() -> None:
    """An empty file has nothing untagged, so its coverage is 1.0."""
    result = tag_markdown("", chapter="bolum-9")
    assert result.nodes == []
    assert coverage(result) == 1.0
