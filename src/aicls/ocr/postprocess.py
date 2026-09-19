"""Postprocessing of a decoder's raw output: brace repair, canonical spacing, grammar precheck.

A sequence model happily emits ``\\frac{x}{2`` — one missing brace makes the whole expression
unparseable for KaTeX and for `aicls.latex2sympy`. The functions here are the last deterministic
stage of the OCR pipeline: they repair what can be repaired, canonicalise whitespace exactly like
the tokenizer does, run the core sanitiser (KaTeX allow-list) and, for diagnostics, report why an
expression is unlikely to parse.

Nothing here guesses semantics: an expression that cannot be repaired is returned as-is and the
precheck simply reports the reasons, which the API turns into ``PARSE_ERROR`` details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aicls.core.latex_safety import find_disallowed_commands, sanitize_latex
from aicls.ocr.tokenizer import join_tokens, tokenize_latex

#: Operators that may not end an expression (a decoder that stopped too early).
TRAILING_OPERATORS: frozenset[str] = frozenset({"+", "-", "*", "/", "=", "^", "_", "<", ">", ","})


def canonicalize_whitespace(latex: str) -> str:
    """Re-space ``latex`` the way the tokenizer would write it (idempotent)."""
    return join_tokens(tokenize_latex(latex))


def _brace_positions(latex: str) -> tuple[list[int], list[int]]:
    """Return the indices of unescaped ``{`` and unmatched ``}``."""
    open_stack: list[int] = []
    unmatched_close: list[int] = []
    index = 0
    while index < len(latex):
        char = latex[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            open_stack.append(index)
        elif char == "}":
            if open_stack:
                open_stack.pop()
            else:
                unmatched_close.append(index)
        index += 1
    return open_stack, unmatched_close


def _strip_dangling_backslash(latex: str) -> str:
    """Drop a trailing backslash that starts an escape with nothing to escape."""
    index = 0
    while index < len(latex):
        if latex[index] == "\\":
            if index == len(latex) - 1:
                return latex[:-1]
            index += 2
            continue
        index += 1
    return latex


def repair_braces(latex: str) -> str:
    """Balance curly braces: drop unmatched ``}``, append the missing ``}``.

    Escaped braces (``\\{``, ``\\}``) are literal characters and are never touched. A dangling
    trailing backslash is removed first, otherwise it would escape the brace we append.

    Args:
        latex: Raw model output.

    Returns:
        A string with balanced (unescaped) braces; idempotent.
    """
    latex = _strip_dangling_backslash(latex)
    open_stack, unmatched_close = _brace_positions(latex)
    if unmatched_close:
        drop = set(unmatched_close)
        latex = "".join(char for i, char in enumerate(latex) if i not in drop)
        open_stack, _ = _brace_positions(latex)
    return latex + "}" * len(open_stack)


@dataclass(frozen=True, slots=True)
class PrecheckReport:
    """Result of :func:`grammar_precheck`.

    Attributes:
        ok: True when no problem was found.
        reasons: Machine-readable reason slugs, in a stable order.
    """

    ok: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (goes into the ``PARSE_ERROR`` problem details)."""
        return {"ok": self.ok, "reasons": list(self.reasons)}


def grammar_precheck(latex: str) -> PrecheckReport:
    """Cheap structural check before handing LaTeX to the parser.

    Detected problems: ``empty``, ``unbalanced_braces``, ``unbalanced_delimiters``
    (``\\left`` without ``\\right``), ``trailing_operator`` and ``disallowed_commands``.

    Args:
        latex: Candidate LaTeX.

    Returns:
        A :class:`PrecheckReport`; ``ok`` is True only when no reason was found.
    """
    reasons: list[str] = []
    stripped = latex.strip()
    if not stripped:
        return PrecheckReport(ok=False, reasons=["empty"])

    open_stack, unmatched_close = _brace_positions(stripped)
    if open_stack or unmatched_close:
        reasons.append("unbalanced_braces")

    tokens = tokenize_latex(stripped)
    if tokens.count("\\left") != tokens.count("\\right"):
        reasons.append("unbalanced_delimiters")
    if tokens and tokens[-1] in TRAILING_OPERATORS:
        reasons.append("trailing_operator")
    if find_disallowed_commands(stripped):
        reasons.append("disallowed_commands")
    return PrecheckReport(ok=not reasons, reasons=reasons)


def postprocess_latex(latex: str, *, sanitize: bool = True) -> str:
    """Full postprocessing pipeline applied to every recogniser output.

    Steps: canonical whitespace -> brace repair -> (optional) core sanitiser -> brace repair.

    Args:
        latex: Raw decoded string.
        sanitize: Apply :func:`aicls.core.latex_safety.sanitize_latex` (KaTeX allow-list). Turn it
            off only for tests and for tokenizer round-trips.

    Returns:
        The cleaned LaTeX; an empty string stays empty.
    """
    cleaned = repair_braces(canonicalize_whitespace(latex))
    if sanitize:
        cleaned = repair_braces(canonicalize_whitespace(sanitize_latex(cleaned)))
    return cleaned


__all__ = [
    "TRAILING_OPERATORS",
    "PrecheckReport",
    "canonicalize_whitespace",
    "grammar_precheck",
    "postprocess_latex",
    "repair_braces",
]
