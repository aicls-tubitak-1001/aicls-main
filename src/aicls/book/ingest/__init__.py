"""Automatic tagging of textbook sources (İP-8 risk plan: "otomatik etiketleme betikleri").

* :mod:`aicls.book.ingest.markdown_tagger` — ``:::`` blocks and plain ``"Tanım 2.4:"`` headings.
* :mod:`aicls.book.ingest.latex_tagger` — ``\\begin{tanim}[...]\\label{tanim:2.4}`` environments.
* :mod:`aicls.book.ingest.validators` — :func:`~aicls.book.ingest.validators.coverage` and the
  refusal that turns an untagged heading into a failed load (İP-8 "100 % tagged" criterion).
"""

from __future__ import annotations

from aicls.book.ingest.latex_tagger import tag_latex
from aicls.book.ingest.markdown_tagger import tag_markdown
from aicls.book.ingest.validators import (
    UntaggedHeadingError,
    coverage,
    require_full_coverage,
    untagged_headings,
)

__all__ = [
    "UntaggedHeadingError",
    "coverage",
    "require_full_coverage",
    "tag_latex",
    "tag_markdown",
    "untagged_headings",
]
