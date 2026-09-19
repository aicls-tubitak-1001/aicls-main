"""Pieces shared by the Markdown and LaTeX taggers: keyword table, numbering, math extraction.

Both taggers recognise the same Turkish unit keywords ("Tanım", "Teorem", "Şekil", ...) and mint
the same identifiers (``tanim-2.4``), so the vocabulary lives here once. Keyword matching happens
on the ``tr_casefold``-ed text because ``"İSPAT".casefold()`` would otherwise produce ``i`` plus a
combining dot (see :mod:`aicls.core.turkish`).
"""

from __future__ import annotations

import re

from aicls.book.schema import Predicate, UnitType
from aicls.core.turkish import tr_casefold

#: Casefolded keyword → unit type. Matching tries the longest keyword first.
KEYWORD_TYPES: dict[str, UnitType] = {
    "tanım": UnitType.DEFINITION,
    "tanim": UnitType.DEFINITION,
    "teorem": UnitType.THEOREM,
    "lemma": UnitType.LEMMA,
    "önerme": UnitType.LEMMA,
    "onerme": UnitType.LEMMA,
    "sonuç": UnitType.COROLLARY,
    "sonuc": UnitType.COROLLARY,
    "kanıt": UnitType.PROOF,
    "kanit": UnitType.PROOF,
    "ispat": UnitType.PROOF,
    "örnek": UnitType.EXAMPLE,
    "ornek": UnitType.EXAMPLE,
    "alıştırma": UnitType.EXERCISE,
    "alistirma": UnitType.EXERCISE,
    "şekil": UnitType.FIGURE,
    "sekil": UnitType.FIGURE,
    "tablo": UnitType.TABLE,
    "not": UnitType.REMARK,
    "uyarı": UnitType.REMARK,
    "uyari": UnitType.REMARK,
    "bölüm": UnitType.SECTION,
    "bolum": UnitType.SECTION,
}

#: Keywords ordered longest-first so ``"alıştırma"`` is not shadowed by a shorter prefix.
KEYWORDS_BY_LENGTH: tuple[str, ...] = tuple(sorted(KEYWORD_TYPES, key=len, reverse=True))

#: Block/environment attribute name → predicate written for each listed target.
RELATION_ATTRIBUTES: dict[str, Predicate] = {
    "requires": Predicate.REQUIRES,
    "illustrates": Predicate.ILLUSTRATES,
    "proves": Predicate.PROVES,
    "figure": Predicate.HAS_FIGURE,
    "related": Predicate.RELATED,
}

#: ``2.4`` / ``12`` / ``2.4.1`` — the numbering that turns a heading into a taggable unit.
NUMBER_RE = re.compile(r"\d+(?:\.\d+)*")

#: What follows the keyword in a unit heading: number, optional parenthesised title, then title.
_REST_RE = re.compile(
    r"^[\s.]*(?P<number>\d+(?:\.\d+)*)[\s.]*"
    r"(?:\((?P<paren>[^)]*)\))?"
    r"\s*[:.–—-]?\s*(?P<title>.*?)\s*$"
)

#: Emphasis, bullet and quote markers stripped before a line is examined.
_LEADING_MARKUP_RE = re.compile(r"^[\s>*_+-]+")
_TRAILING_MARKUP_RE = re.compile(r"[\s*_]+$")


def strip_markup(text: str) -> str:
    """Remove Markdown emphasis, list bullets and quote markers around a candidate line."""
    return _TRAILING_MARKUP_RE.sub("", _LEADING_MARKUP_RE.sub("", text)).strip()


def match_keyword(text: str) -> tuple[str, UnitType, str] | None:
    """Split ``text`` into ``(keyword, unit type, remainder)`` when it opens with a keyword.

    The comparison is done on the Turkish-casefolded text; the remainder is sliced out of the
    original so the title keeps its capitalisation. Inputs whose casefold changes length (never
    the case for the Turkish keywords) are rejected rather than mis-sliced.
    """
    folded = tr_casefold(text)
    if len(folded) != len(text):
        return None
    for keyword in KEYWORDS_BY_LENGTH:
        if folded.startswith(keyword):
            tail = folded[len(keyword) :]
            if tail and (tail[0].isalpha() or tail[0].isdigit()):
                continue  # "Tanımlayıcı" is not the keyword "Tanım"
            return keyword, KEYWORD_TYPES[keyword], text[len(keyword) :]
    return None


def parse_unit_heading(text: str) -> tuple[UnitType, str, str] | None:
    """Parse ``"Teorem 3.1 (Ortalama Değer Teoremi):"`` into type, number and title.

    Returns:
        ``(unit_type, number, title_tr)``, or ``None`` when the text does not start with a unit
        keyword followed by a number — which is exactly what makes a heading *untagged*.
    """
    matched = match_keyword(strip_markup(text))
    if matched is None:
        return None
    _, unit_type, rest = matched
    rest_match = _REST_RE.match(rest)
    if rest_match is None:
        return None
    title = (rest_match.group("paren") or rest_match.group("title") or "").strip()
    return unit_type, rest_match.group("number"), strip_markup(title)


def looks_like_unit_heading(text: str) -> bool:
    """True when a heading starts with a unit keyword, whether or not it can be tagged."""
    return match_keyword(strip_markup(text)) is not None


def parse_attributes(text: str) -> dict[str, list[str] | str]:
    """Parse ``aliases=["a","b"] requires=[tanim-2.1] page=58`` into a mapping.

    Bracketed values become lists (quotes and whitespace stripped); everything else stays a
    string. Unknown attribute names are kept so the caller can decide what to do with them.
    """
    attributes: dict[str, list[str] | str] = {}
    for match in re.finditer(
        r'(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>\[[^\]]*\]|"[^"]*"|\S+)', text
    ):
        raw = match.group("value")
        if raw.startswith("["):
            items = [item.strip().strip('"').strip("'") for item in raw[1:-1].split(",")]
            attributes[match.group("key")] = [item for item in items if item]
        else:
            attributes[match.group("key")] = raw.strip('"').strip("'")
    return attributes


def first_quoted(text: str) -> str:
    """Return the first double-quoted run of ``text`` (the title of a ``:::`` block)."""
    match = re.search(r'"([^"]*)"', text)
    return match.group(1) if match else ""


def to_int(value: object) -> int | None:
    """Best-effort integer conversion for an attribute value (``page=58``)."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


__all__ = [
    "KEYWORDS_BY_LENGTH",
    "KEYWORD_TYPES",
    "NUMBER_RE",
    "RELATION_ATTRIBUTES",
    "first_quoted",
    "looks_like_unit_heading",
    "match_keyword",
    "parse_attributes",
    "parse_unit_heading",
    "strip_markup",
    "to_int",
]
