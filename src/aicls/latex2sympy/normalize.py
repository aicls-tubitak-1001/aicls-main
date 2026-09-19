"""Input normalisation with tracked repairs (docs/mimari/latex-ayristirici.md 3.1).

``normalize_latex`` turns the many equivalent spellings that reach us (hand-written OCR
output, ASR transcripts, textbook copy-paste, ``sympy.latex()`` output) into the single
spelling the grammar accepts, and **records every change it makes**: the API returns the list
in ``ParseOut.repairs`` so a correction is never hidden from the user.

Pipeline (repeated to a fixed point, at most :data:`_MAX_PASSES` times, which is what makes
the function idempotent):

#. NFC, control characters and math-mode delimiters (``$...$``, ``\\[...\\]``);
#. ``\\text{...}`` / ``\\mathrm{...}`` unwrapping (``\\mathrm{d}x`` -> ``dx``);
#. ``\\dfrac`` / ``\\tfrac`` / ``\\cfrac`` -> ``\\frac``;
#. unicode operators and symbols (``×``, ``·``, ``−``, ``²``, ``√``, ``∞``, ``π``, ``≤``,
   Greek letters) -> the corresponding command;
#. ``\\left`` / ``\\right`` (and ``\\bigl``...) removal, ``\\left|`` -> ``\\lvert``;
#. bare ``|...|`` -> ``\\lvert ... \\rvert`` (only when the bars balance);
#. spacing macros (``\\,``, ``\\quad``, ``~``, ``\\displaystyle``, ``\\limits``) -> space;
#. Turkish decimal comma between two digits -> ``.`` (never inside ``[a,b]`` or ``f(x,y)``);
#. empty ``{}`` groups;
#. whitespace squeezing;
#. exactly one missing closing brace at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata

#: Repair kinds (stable machine-readable strings, safe to show in an API response).
REPAIR_MATH_DELIMITERS = "math_delimiters"
REPAIR_TEXT_DROPPED = "text_dropped"
REPAIR_MATHRM = "mathrm_unwrapped"
REPAIR_FRAC_VARIANT = "frac_variant"
REPAIR_UNICODE = "unicode_symbol"
REPAIR_LEFT_RIGHT = "left_right"
REPAIR_ABS_BARS = "abs_bars"
REPAIR_FUNCTION_POWER = "function_power_moved"
REPAIR_SPACING = "spacing"
REPAIR_DECIMAL_COMMA = "decimal_comma"
REPAIR_EMPTY_GROUP = "empty_group"
REPAIR_MISSING_BRACE = "missing_closing_brace"

_MAX_PASSES = 6


@dataclass(frozen=True, slots=True)
class NormalizeResult:
    """Normalised text plus the ordered, de-duplicated list of repairs applied."""

    text: str
    repairs: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------------
_GREEK_UNICODE = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega",
    "Γ": "Gamma",
    "Δ": "Delta",
    "Θ": "Theta",
    "Λ": "Lambda",
    "Ξ": "Xi",
    "Π": "Pi",
    "Σ": "Sigma",
    "Φ": "Phi",
    "Ψ": "Psi",
    "Ω": "Omega",
}

#: Unicode character -> replacement text.
UNICODE_REPLACEMENTS: dict[str, str] = {
    "×": "\\times ",
    "·": "\\cdot ",
    "⋅": "\\cdot ",
    "∙": "\\cdot ",
    "÷": "/",
    "−": "-",
    "–": "-",
    "—": "-",
    "∞": "\\infty ",
    "≤": "\\le ",
    "⩽": "\\le ",
    "≥": "\\ge ",
    "⩾": "\\ge ",
    "≠": "\\neq ",
    "≈": "\\approx ",
    "→": "\\to ",
    "⟶": "\\to ",
    "±": "\\pm ",
    "∂": "\\partial ",
    "∈": "\\in ",
    "∫": "\\int ",
    "∑": "\\sum ",
    "∏": "\\prod ",
    "ℝ": "\\mathbb{R}",
    "ℕ": "\\mathbb{N}",
    "ℤ": "\\mathbb{Z}",
    "ℚ": "\\mathbb{Q}",
    "ℂ": "\\mathbb{C}",
    "⁰": "^{0}",
    "¹": "^{1}",
    "²": "^{2}",
    "³": "^{3}",
    "⁴": "^{4}",
    "⁵": "^{5}",
    "⁶": "^{6}",
    "⁷": "^{7}",
    "⁸": "^{8}",
    "⁹": "^{9}",
    "½": "\\frac{1}{2}",
    "⅓": "\\frac{1}{3}",
    "¼": "\\frac{1}{4}",
}
UNICODE_REPLACEMENTS.update({ch: f"\\{name} " for ch, name in _GREEK_UNICODE.items()})

_DELIMITERS: tuple[tuple[str, str], ...] = (
    ("$$", "$$"),
    ("\\[", "\\]"),
    ("\\(", "\\)"),
    ("$", "$"),
)

_TEXT_RE = re.compile(r"\\(?:text|textrm|textbf|textit|textnormal|mbox)\s*\{[^{}]*\}")
_MATHRM_RE = re.compile(r"\\mathrm\s*\{\s*([A-Za-z]*)\s*\}")
_FRAC_VARIANT_RE = re.compile(r"\\[dtc]frac(?![A-Za-z])")
_SQRT_UNICODE_RE = re.compile(r"√\s*(\{[^{}]*\}|\\[A-Za-z]+|[A-Za-z0-9])")
_LEFT_RIGHT_RE = re.compile(
    r"\\(?:left|right)\s*\.|"
    r"\\(?:left|right)(?![A-Za-z])|"
    r"\\(?:bigl|bigr|Bigl|Bigr|biggl|biggr|Biggl|Biggr|big|Big|bigg|Bigg)(?![A-Za-z])"
)
_LEFT_BAR_RE = re.compile(r"\\left\s*\|")
_RIGHT_BAR_RE = re.compile(r"\\right\s*\|")
_SPACING_RE = re.compile(
    r"(?<!\\)\\[,;:!>]|"
    r"\\(?:quad|qquad|enspace|thinspace|medspace|thickspace|negthinspace"
    r"|displaystyle|textstyle|scriptstyle|limits|nolimits)(?![A-Za-z])|"
    r"(?<!\\)\\ |~"
)
_FUNC_GROUP_RE = re.compile(
    r"(\\(?:arcsin|arccos|arctan|arccot|sinh|cosh|tanh|coth|sech|csch"
    r"|sin|cos|tan|cot|sec|csc|ln|lg|log|exp))\s*\{"
)
#: The exponent that follows a braced function argument: a brace group, a single alphanumeric, or
#: a bare command that takes no argument of its own. The last alternative is why ``\sin{x}^\alpha``
#: is repaired like ``\sin{x}^{\alpha}``; without it the two spellings parsed *differently*
#: (``sin(x^alpha)`` against ``sin(x)^alpha``, verified 2026-09-08) and ``latex_v1`` was not
#: idempotent, because its own spaced output was the spelling that matched. ``(?!\s*\{)`` keeps
#: an argument-taking command such as ``\frac`` out: moving only its name would strip the
#: exponent from its arguments -- and ``(?![A-Za-z])`` is what stops ``+`` from backtracking to a
#: prefix of the name (without it ``\frac{1}{2}`` matched as ``\fra`` followed by ``c``).
_POWER_RE = re.compile(r"\s*\^\s*(?:\{([^{}]*)\}|(\\[A-Za-z]+(?![A-Za-z])(?!\s*\{)|[A-Za-z0-9]))")
_APPLIED_FUNC_RE = re.compile(r"[fghFGH]'*\s*\{")
_INTERVAL_RE = re.compile(r"\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*")
_EMPTY_GROUP_RE = re.compile(r"(?<!\\)\{\s*\}")
_WS_RE = re.compile(r"\s+")
_BAR_RE = re.compile(r"(?<!\\)\||\\vert(?![A-Za-z])")


# ---------------------------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------------------------
def _clean_characters(text: str) -> str:
    """NFC-normalise and drop control/format characters (newlines and tabs become spaces)."""
    normalised = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    kept: list[str] = []
    for char in normalised:
        if char in "\n\t":
            kept.append(" ")
        elif unicodedata.category(char) in {"Cc", "Cf", "Cs", "Co", "Cn"}:
            continue
        else:
            kept.append(char)
    return "".join(kept)


def _strip_delimiters(text: str) -> str:
    stripped = text.strip()
    for start, end in _DELIMITERS:
        if (
            len(stripped) >= len(start) + len(end)
            and stripped.startswith(start)
            and stripped.endswith(end)
        ):
            return stripped[len(start) : -len(end)]
    return text


def _apply_unicode(text: str) -> str:
    out = _SQRT_UNICODE_RE.sub(lambda m: _wrap_sqrt(m.group(1)), text)
    out = out.replace("√", "\\sqrt")
    for char, replacement in UNICODE_REPLACEMENTS.items():
        if char in out:
            out = out.replace(char, replacement)
    return out


def _wrap_sqrt(operand: str) -> str:
    if operand.startswith("{"):
        return f"\\sqrt{operand}"
    return f"\\sqrt{{{operand}}}"


def _find_group_end(text: str, start: int) -> int:
    """Index just past the brace group that opens at ``start`` (or ``-1`` if unterminated)."""
    depth = 0
    index = start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return -1


def _unwrap_applied_function(text: str) -> str:
    """``f{(x + h)}`` -> ``f(x + h)``.

    ``sympy.latex`` prints an applied undefined function as ``f{\\left(x \\right)}``; without
    this the braces would turn the application into an implicit product. Only a group whose
    content is a parenthesised expression is unwrapped, so a hand-written ``f{x}`` (which TeX
    renders as the product ``fx``) keeps its meaning.
    """
    result = text
    position = 0
    while True:
        match = _APPLIED_FUNC_RE.search(result, position)
        if match is None:
            return result
        brace = match.end() - 1
        end = _find_group_end(result, brace)
        if end < 0:
            return result
        content = result[brace + 1 : end - 1].strip()
        if content.startswith("(") and content.endswith(")"):
            result = result[:brace] + content + result[end:]
            position = brace
        else:
            position = match.end()


def _move_function_power(text: str) -> str:
    """``\\ln{x}^{2}`` -> ``\\ln^{2}{x}``.

    ``sympy.latex`` prints a powered function that way, and a writer who braces the argument
    means the same thing; without this the grammar would read the exponent as part of the
    argument (``\\ln x^{2}``), which is a *silent* misparse.
    """
    result = text
    position = 0
    while True:
        match = _FUNC_GROUP_RE.search(result, position)
        if match is None:
            return result
        brace = match.end() - 1
        end = _find_group_end(result, brace)
        power = _POWER_RE.match(result, end) if end >= 0 else None
        if power is None:
            position = match.end()
            continue
        exponent = power.group(1) if power.group(1) is not None else power.group(2)
        result = (
            result[: match.end(1)] + f"^{{{exponent}}}" + result[brace:end] + result[power.end() :]
        )
        position = match.end(1)


def _convert_bars(text: str) -> str:
    """Turn balanced bare ``|`` pairs into ``\\lvert``/``\\rvert``; leave odd counts alone."""
    matches = list(_BAR_RE.finditer(text))
    if not matches or len(matches) % 2 == 1:
        return text
    pieces: list[str] = []
    cursor = 0
    for index, match in enumerate(matches):
        pieces.append(text[cursor : match.start()])
        pieces.append("\\lvert " if index % 2 == 0 else " \\rvert")
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces)


def _interval_commas(text: str) -> set[int]:
    """Indices of commas that separate the two bounds of an interval such as ``[0,1]``.

    Only a bracket whose whole content is *two plain numbers* counts; ``(x + 1,5)`` is a
    Turkish decimal, ``f(x, y)`` has no digit on both sides and is never touched.
    """
    separators: set[int] = set()
    stack: list[int] = []
    for index, char in enumerate(text):
        if char in "([":
            stack.append(index)
        elif char in ")]" and stack:
            start = stack.pop()
            inner = text[start + 1 : index]
            match = _INTERVAL_RE.fullmatch(inner)
            if match is not None:
                separators.add(start + 1 + inner.index(","))
    return separators


def _convert_decimal_comma(text: str) -> str:
    """``3,5`` -> ``3.5``, except where the comma separates interval bounds."""
    separators = _interval_commas(text)
    out: list[str] = []
    for index, char in enumerate(text):
        if (
            char == ","
            and index not in separators
            and 0 < index < len(text) - 1
            and text[index - 1].isdigit()
            and text[index + 1].isdigit()
        ):
            out.append(".")
            continue
        out.append(char)
    return "".join(out)


def _count_braces(text: str) -> tuple[int, int]:
    opens = closes = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            opens += 1
        elif char == "}":
            closes += 1
        index += 1
    return opens, closes


def _pass(text: str, repairs: list[str]) -> str:
    """One full normalisation pass; appends the kinds of the repairs it performed."""

    def record(kind: str) -> None:
        if kind not in repairs:
            repairs.append(kind)

    out = _clean_characters(text)
    stripped = _strip_delimiters(out)
    if stripped != out or "$" in stripped:
        record(REPAIR_MATH_DELIMITERS)
    out = stripped.replace("$", "")

    replaced = _TEXT_RE.sub(" ", out)
    if replaced != out:
        record(REPAIR_TEXT_DROPPED)
    out = replaced

    replaced = _MATHRM_RE.sub(lambda m: m.group(1), out)
    if replaced != out:
        record(REPAIR_MATHRM)
    out = replaced

    replaced = _FRAC_VARIANT_RE.sub("\\\\frac", out)
    if replaced != out:
        record(REPAIR_FRAC_VARIANT)
    out = replaced

    replaced = _apply_unicode(out)
    if replaced != out:
        record(REPAIR_UNICODE)
    out = replaced

    replaced = _RIGHT_BAR_RE.sub(" \\\\rvert", _LEFT_BAR_RE.sub("\\\\lvert ", out))
    replaced = _LEFT_RIGHT_RE.sub("", replaced)
    if replaced != out:
        record(REPAIR_LEFT_RIGHT)
    out = replaced

    replaced = _convert_bars(out)
    if replaced != out:
        record(REPAIR_ABS_BARS)
    out = replaced

    replaced = _move_function_power(_unwrap_applied_function(out))
    if replaced != out:
        record(REPAIR_FUNCTION_POWER)
    out = replaced

    replaced = _SPACING_RE.sub(" ", out)
    if replaced != out:
        record(REPAIR_SPACING)
    out = replaced

    replaced = _convert_decimal_comma(out)
    if replaced != out:
        record(REPAIR_DECIMAL_COMMA)
    out = replaced

    replaced = _EMPTY_GROUP_RE.sub("", out)
    if replaced != out:
        record(REPAIR_EMPTY_GROUP)
    out = replaced

    out = _WS_RE.sub(" ", out).strip()

    opens, closes = _count_braces(out)
    if opens - closes == 1:
        out += "}"
        record(REPAIR_MISSING_BRACE)
    return out


def normalize_latex(src: str) -> NormalizeResult:
    """Normalise ``src`` and report the repairs that were applied.

    Args:
        src: Raw LaTeX from OCR, ASR, the UI or a dataset row.

    Returns:
        A :class:`NormalizeResult` whose ``text`` is safe to hand to the grammar and whose
        ``repairs`` lists every applied repair kind once, in the order they first occurred.
        The function is idempotent: ``normalize_latex(normalize_latex(s).text).text`` equals
        ``normalize_latex(s).text``.
    """
    repairs: list[str] = []
    text = src
    for _ in range(_MAX_PASSES):
        nxt = _pass(text, repairs)
        if nxt == text:
            break
        text = nxt
    return NormalizeResult(text=text, repairs=repairs)


__all__ = [
    "REPAIR_ABS_BARS",
    "REPAIR_DECIMAL_COMMA",
    "REPAIR_EMPTY_GROUP",
    "REPAIR_FRAC_VARIANT",
    "REPAIR_FUNCTION_POWER",
    "REPAIR_LEFT_RIGHT",
    "REPAIR_MATHRM",
    "REPAIR_MATH_DELIMITERS",
    "REPAIR_MISSING_BRACE",
    "REPAIR_SPACING",
    "REPAIR_TEXT_DROPPED",
    "REPAIR_UNICODE",
    "UNICODE_REPLACEMENTS",
    "NormalizeResult",
    "normalize_latex",
]
