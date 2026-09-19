"""Turkish text helpers shared by the NLU, book resolver and evaluation normalisers.

Python's ``str.casefold``/``str.lower`` follow the default Unicode mapping, which turns
``İ`` into ``i̇`` (``i`` + combining dot) and ``I`` into ``i`` — wrong for Turkish, where
``İ→i`` and ``I→ı``. :func:`tr_casefold` fixes both before delegating to Unicode casefolding and
is the ``tr_v1`` normaliser's lower-casing step (ADR-0015), so its behaviour is frozen.
"""

from __future__ import annotations

from collections.abc import Iterable
import unicodedata

_I_MAP = str.maketrans({"İ": "i", "I": "ı"})
_DIACRITIC_MAP = str.maketrans(
    {
        "ç": "c",
        "Ç": "C",
        "ğ": "g",
        "Ğ": "G",
        "ı": "i",
        "İ": "I",
        "ö": "o",
        "Ö": "O",
        "ş": "s",
        "Ş": "S",
        "ü": "u",
        "Ü": "U",
    }
)

#: Common inflectional suffixes by family, all vowel-harmony variants (casefolded).
#: Used for alias matching ("türevini" → "türev") in the book resolver and the NLU slot filler.
TR_SUFFIXES: dict[str, list[str]] = {
    "acc": ["ı", "i", "u", "ü", "yı", "yi", "yu", "yü"],
    "gen": ["ın", "in", "un", "ün", "nın", "nin", "nun", "nün"],
    "dat": ["a", "e", "ya", "ye"],
    "loc": ["da", "de", "ta", "te"],
    "abl": ["dan", "den", "tan", "ten"],
    "poss3": ["ı", "i", "u", "ü", "sı", "si", "su", "sü"],
    "poss3_acc": ["ını", "ini", "unu", "ünü", "sını", "sini", "sunu", "sünü"],
    "poss3_gen": ["ının", "inin", "unun", "ünün", "sının", "sinin", "sunun", "sünün"],
    "poss3_dat": ["ına", "ine", "una", "üne", "sına", "sine", "suna", "süne"],
    "plural": ["lar", "ler"],
    "plural_acc": ["ları", "leri"],
    "plural_gen": ["ların", "lerin"],
}

_DEFAULT_MIN_STEM = 3


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def tr_casefold(s: str) -> str:
    """Turkish-aware casefold: NFC, ``İ→i``, ``I→ı``, then Unicode casefold.

    Idempotent; never yields ``İ``, ``I`` or ``i`` + combining dot.
    """
    return _nfc(_nfc(s).translate(_I_MAP).casefold())


def tr_lower(s: str) -> str:
    """Turkish-aware lower-casing (``İ→i``, ``I→ı``); unlike casefold keeps ``ß`` as is."""
    return _nfc(_nfc(s).translate(_I_MAP).lower())


def strip_diacritics(s: str) -> str:
    """Replace Turkish letters by their ASCII base (``ş→s``, ``ı→i`` ...) and drop combining marks.

    Case is preserved. Useful for fuzzy alias matching and ASCII-only identifiers.
    """
    mapped = _nfc(s).translate(_DIACRITIC_MAP)
    decomposed = unicodedata.normalize("NFD", mapped)
    return unicodedata.normalize(
        "NFC", "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    )


def strip_suffix(
    word: str,
    *,
    families: Iterable[str] | None = None,
    min_stem: int = _DEFAULT_MIN_STEM,
) -> str:
    """Remove one (longest matching) inflectional suffix from a casefolded word.

    Args:
        word: A single token; it is casefolded first.
        families: Keys of :data:`TR_SUFFIXES` to consider (default: all).
        min_stem: Never shorten the stem below this many characters.

    Returns:
        The stem, or the casefolded word unchanged when nothing matches.
    """
    token = tr_casefold(word)
    keys = list(TR_SUFFIXES) if families is None else list(families)
    candidates = sorted(
        {suffix for key in keys for suffix in TR_SUFFIXES[key]}, key=len, reverse=True
    )
    for suffix in candidates:
        if token.endswith(suffix) and len(token) - len(suffix) >= min_stem:
            return token[: -len(suffix)]
    return token


__all__ = ["TR_SUFFIXES", "strip_diacritics", "strip_suffix", "tr_casefold", "tr_lower"]
