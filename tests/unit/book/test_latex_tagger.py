"""LaTeX tagger: labelled theorem-like environments and the missing-label refusal."""

from __future__ import annotations

from aicls.book import Predicate, TagResult, UnitType, coverage, tag_latex
from aicls.book.ingest.latex_tagger import label_number


def test_latex_tagger_recognises_labelled_environments(chapter_four: TagResult) -> None:
    """Every ``\\begin{...}[Başlık]\\label{...}`` environment becomes a unit."""
    assert coverage(chapter_four) == 1.0
    assert [node.id for node in chapter_four.nodes] == [
        "bolum-4",
        "tanim-4.1",
        "tanim-4.2",
        "teorem-4.3",
        "ornek-4.4",
    ]
    riemann = chapter_four.nodes[1]
    assert riemann.type is UnitType.DEFINITION
    assert riemann.title_tr == "Riemann Toplamı"
    assert riemann.number == "4.1"
    assert riemann.latex_blocks == ["S_n = \\sum_{i=1}^{n} f(x_i^*) \\, \\Delta x_i"]
    assert "\\label" not in riemann.body_md


def test_latex_section_becomes_the_chapter_unit(chapter_four: TagResult) -> None:
    """``\\section{Bölüm 4: ...}`` becomes the Section unit and every unit is ``partOf`` it."""
    section = chapter_four.nodes[0]
    assert section.type is UnitType.SECTION
    assert section.id == "bolum-4"
    assert section.title_tr == "Belirli İntegral"
    part_of = {t.subject for t in chapter_four.triples if t.predicate is Predicate.PART_OF}
    assert part_of == {"tanim-4.1", "tanim-4.2", "teorem-4.3", "ornek-4.4"}


def test_environment_without_label_is_untagged() -> None:
    """No ``\\label`` means no stable identifier, so the environment is reported."""
    result = tag_latex(
        "\\begin{tanim}[Türev]\nGövde.\n\\end{tanim}\n", chapter="bolum-9", source="x.tex"
    )
    assert result.nodes == []
    assert [span.kind for span in result.untagged_spans] == ["missing_label"]
    assert coverage(result) == 0.0


def test_keyword_section_without_number_is_untagged() -> None:
    """A sectioning command whose title starts with a keyword but carries no number fails."""
    result = tag_latex("\\subsection{Tanım: Süreklilik}\n", chapter="bolum-9")
    assert result.nodes == []
    assert [span.text for span in result.untagged_spans] == ["Tanım: Süreklilik"]


def test_english_environment_names_are_accepted() -> None:
    """``theorem``/``definition`` map to the same vocabulary as their Turkish names."""
    result = tag_latex(
        "\\begin{theorem}[Ara Değer]\\label{teorem:7.2}\nGövde.\n\\end{theorem}\n",
        chapter="bolum-7",
    )
    assert result.nodes[0].id == "teorem-7.2"
    assert result.nodes[0].type is UnitType.THEOREM


def test_relation_attributes_on_the_begin_line() -> None:
    """Relations written as a comment on the ``\\begin`` line become triples."""
    source = (
        "\\begin{ornek}[Alan]\\label{ornek:4.4} % illustrates=[tanim-4.2] page=91\n"
        "Gövde.\n\\end{ornek}\n"
    )
    result = tag_latex(source, chapter="bolum-4")
    assert result.nodes[0].source_page == 91
    assert (result.triples[0].subject, result.triples[0].predicate, result.triples[0].object) == (
        "ornek-4.4",
        Predicate.ILLUSTRATES,
        "tanim-4.2",
    )


def test_label_number_reads_both_separators() -> None:
    """``tanim:4.1`` and ``tanim-4.1`` both yield ``4.1``; a label without digits yields ''."""
    assert label_number("tanim:4.1") == "4.1"
    assert label_number("tanim-4.1") == "4.1"
    assert label_number("sec:giris") == ""


def test_unknown_environment_is_ignored() -> None:
    """A non-unit environment is neither tagged nor reported as untagged."""
    result = tag_latex("\\begin{itemize}\n\\item a\n\\end{itemize}\n", chapter="bolum-9")
    assert result.nodes == []
    assert result.untagged_spans == []
    assert coverage(result) == 1.0
