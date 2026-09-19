"""Tagging coverage — the measurable form of the İP-8 "%100 etiketleme" criterion (ADR-0019).

``coverage(result)`` is the share of unit-looking headings that actually became content units.
It is 1.0 exactly when :attr:`~aicls.book.schema.TagResult.untagged_spans` is empty; anything
below that makes :func:`aicls.book.load_units` refuse the chapter and list the offending
headings, so a partially tagged book can never reach the database unnoticed.

The Turkish user message comes from :mod:`aicls.core.messages_tr`; the heading list travels in
``details`` (RFC 9457 problem document, docs/mimari/api.md §2 and §3.7: a load whose
``tagged_ratio < 1.0`` is a 422 that names the untagged headings).
"""

from __future__ import annotations

from collections.abc import Sequence

from aicls.book.schema import TagResult, UntaggedSpan
from aicls.core.errors import InputValidationError
from aicls.core.messages_tr import msg

#: Coverage a chapter must reach before it may be loaded (İP-8 criterion).
REQUIRED_COVERAGE = 1.0


class UntaggedHeadingError(InputValidationError):
    """A chapter carries headings that could not be tagged; the load is refused.

    ``details`` holds ``untagged_headings`` (the heading texts, the list the operator needs),
    ``untagged`` (full ``line``/``text``/``kind``/``source`` records) and ``tagged_ratio``.
    """

    def __init__(self, spans: Sequence[UntaggedSpan], *, tagged_ratio: float) -> None:
        """Build the error from the untagged spans of one or more tag results."""
        super().__init__(
            msg("VALIDATION_ERROR"),
            details={
                "field": "book_chapter",
                "tagged_ratio": tagged_ratio,
                "untagged_headings": [span.text for span in spans],
                "untagged": [
                    {
                        "line": span.line,
                        "text": span.text,
                        "kind": span.kind,
                        "source": span.source,
                    }
                    for span in spans
                ],
            },
        )
        self.spans = list(spans)
        self.tagged_ratio = tagged_ratio

    @property
    def headings(self) -> list[str]:
        """Texts of the headings that could not be tagged."""
        return [span.text for span in self.spans]


def coverage(result: TagResult) -> float:
    """Return the share of unit-looking headings that became content units.

    ``1.0`` when nothing was left untagged (including an empty source); otherwise
    ``len(nodes) / (len(nodes) + len(untagged_spans))``.

    Args:
        result: Output of :func:`aicls.book.tag_markdown` or :func:`aicls.book.tag_latex`.

    Returns:
        A ratio in ``[0, 1]``.
    """
    tagged = len(result.nodes)
    untagged = len(result.untagged_spans)
    total = tagged + untagged
    if total == 0:
        return REQUIRED_COVERAGE
    return tagged / total


def untagged_headings(*results: TagResult) -> list[UntaggedSpan]:
    """Collect the untagged spans of several tag results, in the order they were produced."""
    spans: list[UntaggedSpan] = []
    for result in results:
        spans.extend(result.untagged_spans)
    return spans


def require_full_coverage(*results: TagResult) -> float:
    """Return the combined coverage of ``results``, refusing anything below 1.0.

    Args:
        *results: Tag results of every file being loaded together.

    Returns:
        The combined ratio (always 1.0 when it returns).

    Raises:
        UntaggedHeadingError: If any result carries an untagged heading; the exception lists
            them (İP-8 100 % criterion).
    """
    tagged = sum(len(result.nodes) for result in results)
    spans = untagged_headings(*results)
    total = tagged + len(spans)
    ratio = REQUIRED_COVERAGE if total == 0 else tagged / total
    if spans:
        raise UntaggedHeadingError(spans, tagged_ratio=ratio)
    return ratio


__all__ = [
    "REQUIRED_COVERAGE",
    "UntaggedHeadingError",
    "coverage",
    "require_full_coverage",
    "untagged_headings",
]
