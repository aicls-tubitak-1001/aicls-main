"""Export one operation as LaTeX source or a one-page PDF (api.md §3.3, proposal A.2.2).

The PDF is produced with Matplotlib's ``mathtext`` renderer on the Agg backend, which the
platform already depends on for plots (ADR-0016): no TeX distribution is installed on the server
and none is required. Mathtext understands a smaller language than KaTeX, so the formula is drawn
inside a ``try`` and a plain-text fallback is written when it cannot be typeset — an export must
never fail because of a glyph.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from aicls.core.latex_safety import sanitize_latex

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aicls.db.models import Operation

#: Preamble of the exported ``.tex`` document.
TEX_PREAMBLE = "\n".join(
    (
        r"\documentclass[12pt]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{amsmath,amssymb}",
        r"\begin{document}",
    )
)

#: Page size of the exported PDF in inches (A4).
PDF_PAGE_INCHES = (8.27, 11.69)

#: Characters of a LaTeX field kept in the export.
MAX_EXPORT_CHARS = 2048


def _clean(value: str | None) -> str:
    """Sanitise a stored LaTeX fragment before it is written into a document."""
    if not value:
        return ""
    return sanitize_latex(value, max_len=MAX_EXPORT_CHARS)


def operation_to_tex(row: Operation) -> str:
    """Render one operation as a standalone LaTeX document.

    Args:
        row: The ``operations`` row to export.

    Returns:
        UTF-8 LaTeX source with the input, the recognised command and the result.
    """
    lines = [TEX_PREAMBLE, r"\section*{AI-CLS}"]
    if row.source_text:
        lines.append(rf"\textbf{{Komut:}} {row.source_text}")
        lines.append("")
    if row.input_latex:
        lines.append(r"\textbf{Girdi:}")
        lines.append(rf"\[{_clean(row.input_latex)}\]")
    if row.result_latex:
        lines.append(r"\textbf{Sonuç:}")
        lines.append(rf"\[{_clean(row.result_latex)}\]")
    if row.error_code:
        lines.append(rf"\textbf{{Hata:}} {row.error_code}")
    lines.append(r"\end{document}")
    return "\n".join(lines) + "\n"


def operation_to_pdf(row: Operation) -> bytes:
    """Render one operation as a single-page PDF using Matplotlib mathtext.

    Args:
        row: The ``operations`` row to export.

    Returns:
        The PDF bytes (they start with ``%PDF``).
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure

    figure = Figure(figsize=PDF_PAGE_INCHES)
    axes = figure.add_subplot(111)
    axes.axis("off")
    axes.text(0.05, 0.94, "AI-CLS", fontsize=18, va="top")
    cursor = 0.86
    if row.source_text:
        axes.text(0.05, cursor, f"Komut: {row.source_text[:200]}", fontsize=11, va="top")
        cursor -= 0.06
    for label, latex in (("Girdi", row.input_latex), ("Sonuç", row.result_latex)):
        if not latex:
            continue
        axes.text(0.05, cursor, f"{label}:", fontsize=12, va="top")
        cursor -= 0.05
        axes.text(0.08, cursor, _mathtext(latex), fontsize=14, va="top")
        cursor -= 0.08
    if row.error_code:
        axes.text(0.05, cursor, f"Hata: {row.error_code}", fontsize=11, va="top")
    buffer = io.BytesIO()
    figure.savefig(buffer, format="pdf")
    return buffer.getvalue()


def _mathtext(latex: str) -> str:
    """Return ``latex`` wrapped for mathtext, or a plain fallback when it cannot be typeset."""
    cleaned = _clean(latex)
    candidate = f"${cleaned}$"
    try:
        from matplotlib import mathtext

        mathtext.MathTextParser("path").parse(candidate)
    except Exception:
        return cleaned
    return candidate


__all__ = [
    "MAX_EXPORT_CHARS",
    "PDF_PAGE_INCHES",
    "TEX_PREAMBLE",
    "operation_to_pdf",
    "operation_to_tex",
]
