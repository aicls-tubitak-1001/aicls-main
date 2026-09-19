"""Helpers for reading the ``datasets/book`` sample set from the tests."""

from __future__ import annotations

from pathlib import Path

from aicls.book import TagResult, tag_latex, tag_markdown

#: ``(relative path, chapter id, format)`` of every source in the sample set.
SAMPLE_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("sample/bolum-01-limit.md", "bolum-1", "markdown"),
    ("sample/bolum-02-turev.md", "bolum-2", "markdown"),
    ("sample/bolum-04-integral.tex", "bolum-4", "latex"),
)

#: The two Markdown chapters the loader and resolver tests work on.
MARKDOWN_SOURCES: tuple[tuple[str, str], ...] = tuple(
    (relative, chapter) for relative, chapter, fmt in SAMPLE_SOURCES if fmt == "markdown"
)


def tag_source(book_datasets: Path, relative: str, chapter: str, fmt: str) -> TagResult:
    """Tag one file of the sample set with the tagger its format calls for."""
    path = book_datasets / relative
    text = path.read_text(encoding="utf-8")
    tagger = tag_markdown if fmt == "markdown" else tag_latex
    return tagger(text, chapter=chapter, source=path.name)


def tag_markdown_samples(book_datasets: Path) -> list[TagResult]:
    """Tag both Markdown sample chapters, in chapter order."""
    return [
        tag_source(book_datasets, relative, chapter, "markdown")
        for relative, chapter in MARKDOWN_SOURCES
    ]
