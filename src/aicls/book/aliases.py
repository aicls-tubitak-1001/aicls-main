"""Turkish suffix-table alias generation for content units (İP-8 risk plan, ADR-0019).

A student asks for ``"türevin tanımını getir"``, ``"türev tanımı"`` or simply ``"türev"``; the
book graph must answer with the same unit for all three. Rather than stem the *query* (lossy) we
expand the *title* once at load time into every inflected form the resolver may meet, store them
in the indexed ``book_aliases`` table and match exactly. That keeps the hot path a single indexed
lookup (docs/mimari/veri-modeli.md §5.4, "getirme p50 < 20 ms").

The generator applies Turkish vowel harmony, the buffer consonants ``y``/``n``/``s`` after a
vowel-final stem, ``d``/``t`` assimilation after a voiceless consonant and final-consonant
softening (``örnek`` → ``örneği``). Forms produced for the stem ``türev``:

    türev, türevi, türevin, türeve, türevde, türevden, türevini, türevinin, türevine,
    türevler, türevleri, türevlerin

and, combined with the unit-type word ``tanım``:

    türev tanımı, türevin tanımı, türev tanımını, türevin tanımının, ...

An ASCII-folded variant of every alias is emitted as well, so a query typed without Turkish
characters (``"turev tanimi"``) still hits the index. Coverage of the suffix table is deliberately
finite; misses are meant to be found by ``datasets/book/fetch_test.jsonl`` and fixed here
(ADR-0019, "Sonuçlar").
"""

from __future__ import annotations

from dataclasses import dataclass

from aicls.book.schema import UnitType
from aicls.core.turkish import strip_diacritics, strip_suffix, tr_casefold

#: Turkish vowels, ordered back-unrounded, back-rounded, front-unrounded, front-rounded.
VOWELS = "aeıioöuü"
_BACK_VOWELS = "aıou"
_HIGH_BY_LAST_VOWEL = {
    "a": "ı",
    "ı": "ı",
    "e": "i",
    "i": "i",
    "o": "u",
    "u": "u",
    "ö": "ü",
    "ü": "ü",
}
#: Consonants after which ``d`` becomes ``t`` (``örnekte``, ``kitaptan``).
VOICELESS = "fstkçşhp"
#: Final-consonant softening before a vowel-initial suffix (``örnek`` → ``örneğ`` + ``i``).
SOFTENING = {"k": "ğ", "p": "b", "ç": "c"}
#: Words whose stem loses a vowel before a possessive suffix (``şekil`` → ``şekli``).
IRREGULAR_STEMS = {
    "şekil": "şekl",
    "resim": "resm",
    "isim": "ism",
    "burun": "burn",
    "oğul": "oğl",
}

#: The Turkish noun naming each unit type; the second entry of a pair is a synonym.
TYPE_WORDS: dict[UnitType, tuple[str, ...]] = {
    UnitType.DEFINITION: ("tanım",),
    UnitType.THEOREM: ("teorem",),
    UnitType.LEMMA: ("lemma", "önerme"),
    UnitType.COROLLARY: ("sonuç",),
    UnitType.PROOF: ("kanıt", "ispat"),
    UnitType.EXAMPLE: ("örnek",),
    UnitType.EXERCISE: ("alıştırma", "soru"),
    UnitType.FIGURE: ("şekil", "grafik"),
    UnitType.TABLE: ("tablo",),
    UnitType.REMARK: ("not", "uyarı"),
    UnitType.SECTION: ("bölüm", "kesim"),
}

#: Weight written to ``book_aliases.weight`` per alias kind; higher wins a tie in the index.
WEIGHT_BY_KIND: dict[str, float] = {
    "title": 1.0,
    "title_inflected": 0.9,
    "core": 0.85,
    "combo": 0.8,
    "head": 0.7,
    "section": 0.4,
}

#: An ASCII-folded alias keeps the *relative* order of its source, at half its weight, so a
#: diacritic-free query resolves the same way a properly typed one does.
ASCII_WEIGHT_FACTOR = 0.5


@dataclass(frozen=True, slots=True)
class AliasEntry:
    """One generated alias with the weight and provenance the loader stores.

    Attributes:
        alias: Normalised alias text (already ``tr_casefold``-ed).
        weight: ``book_aliases.weight``; ties in the alias index are broken by it.
        kind: Which rule produced it (``"title"``, ``"combo"``, ``"ascii"`` ...).
    """

    alias: str
    weight: float
    kind: str


def _last_vowel(word: str) -> str:
    """Return the last Turkish vowel of ``word`` (empty string when there is none)."""
    for char in reversed(word):
        if char in VOWELS:
            return char
    return ""


def _high(word: str) -> str:
    """The harmonised high vowel (``ı``/``i``/``u``/``ü``) for a suffix attached to ``word``."""
    return _HIGH_BY_LAST_VOWEL.get(_last_vowel(word), "i")


def _low(word: str) -> str:
    """The harmonised low vowel (``a``/``e``) for a suffix attached to ``word``."""
    return "a" if _last_vowel(word) in _BACK_VOWELS else "e"


def _vowel_count(word: str) -> int:
    """Number of vowels in ``word`` (a proxy for its syllable count)."""
    return sum(1 for char in word if char in VOWELS)


def soften(word: str) -> str:
    """Apply final-consonant softening before a vowel-initial suffix.

    ``örnek`` → ``örneğ``, ``kitap`` → ``kitab``, ``şekil`` → ``şekl`` (irregular table).
    Monosyllables keep their consonant (``ek`` → ``ek``), and ``t`` is never softened because
    Turkish does so only in a small closed class (``kanıt`` stays ``kanıt``).
    """
    if word in IRREGULAR_STEMS:
        return IRREGULAR_STEMS[word]
    if not word or _vowel_count(word) < 2:
        return word
    replacement = SOFTENING.get(word[-1])
    return word[:-1] + replacement if replacement else word


def harden(word: str) -> str:
    """Inverse of :func:`soften` for a stem read back from an inflected form.

    ``örneğ`` → ``örnek``, ``kitab`` → ``kitap``, ``şekl`` → ``şekil``.
    """
    for plain, softened in IRREGULAR_STEMS.items():
        if word == softened:
            return plain
    reverse = {value: key for key, value in SOFTENING.items()}
    replacement = reverse.get(word[-1:]) if word else None
    return word[:-1] + replacement if replacement else word


def inflect(phrase: str) -> list[str]:
    """Return the inflected forms of ``phrase``, declining only its last word.

    The nominative comes first; the remaining twelve forms are accusative, genitive, dative,
    locative, ablative, 3rd-person possessive and its accusative/genitive/dative, plural and the
    plural accusative/genitive. Order is stable, so alias weights are reproducible.
    """
    phrase = phrase.strip()
    if not phrase:
        return []
    prefix, _, stem = phrase.rpartition(" ")
    lead = f"{prefix} " if prefix else ""
    high = _high(stem)
    low = _low(stem)
    vowel_final = bool(stem) and stem[-1] in VOWELS
    voiced = "t" if stem and stem[-1] in VOICELESS else "d"
    soft = stem if vowel_final else soften(stem)
    buf_y = "y" if vowel_final else ""
    buf_n = "n" if vowel_final else ""
    buf_s = "s" if vowel_final else ""
    forms = [
        stem,
        f"{soft}{buf_y}{high}",
        f"{soft}{buf_n}{high}n",
        f"{soft}{buf_y}{low}",
        f"{stem}{voiced}{low}",
        f"{stem}{voiced}{low}n",
        f"{soft}{buf_s}{high}",
        f"{soft}{buf_s}{high}n{high}",
        f"{soft}{buf_s}{high}n{high}n",
        f"{soft}{buf_s}{high}n{low}",
        f"{stem}l{low}r",
        f"{stem}l{low}r{high}",
        f"{stem}l{low}r{high}n",
    ]
    return [f"{lead}{form}" for form in _unique(forms)]


def genitive(phrase: str) -> str:
    """Return the genitive of ``phrase`` (``türev`` → ``türevin``, ``masa`` → ``masanın``)."""
    phrase = phrase.strip()
    if not phrase:
        return ""
    prefix, _, stem = phrase.rpartition(" ")
    lead = f"{prefix} " if prefix else ""
    vowel_final = stem[-1] in VOWELS
    soft = stem if vowel_final else soften(stem)
    buffer = "n" if vowel_final else ""
    return f"{lead}{soft}{buffer}{_high(stem)}n"


def possessive_forms(word: str) -> list[str]:
    """Return the 3rd-person possessive forms of a type word (``tanımı``, ``tanımını`` ...)."""
    high = _high(word)
    low = _low(word)
    vowel_final = bool(word) and word[-1] in VOWELS
    soft = word if vowel_final else soften(word)
    buf_s = "s" if vowel_final else ""
    base = f"{soft}{buf_s}{high}"
    return _unique([base, f"{base}n{high}", f"{base}n{high}n", f"{base}n{low}"])


def _unique(values: list[str]) -> list[str]:
    """Deduplicate while keeping the first occurrence's position."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def head_stem(phrase: str) -> str:
    """Return the bare stem of the phrase's last word (``"limiti"`` → ``"limit"``)."""
    last = phrase.rpartition(" ")[2]
    return harden(strip_suffix(last)) if last else ""


def _core_words(title: str, type_words: tuple[str, ...]) -> list[str]:
    """Drop a trailing word that merely repeats the unit type (``"… Teoremi"`` → ``"…"``)."""
    words = title.split()
    if len(words) > 1 and head_stem(words[-1]) in type_words:
        return words[:-1]
    return words


def generate_alias_entries(title_tr: str, unit_type: UnitType) -> list[AliasEntry]:
    """Generate the weighted aliases of a unit title (see :func:`generate_aliases`).

    Args:
        title_tr: Turkish title as printed in the book (``"Türev"``).
        unit_type: Type of the unit; picks the type word (``Definition`` → ``tanım``) and
            decides whether bare forms are emitted.

    Returns:
        Deterministically ordered, deduplicated entries. The first entry is always the
        normalised title itself.
    """
    title = " ".join(tr_casefold(title_tr).split())
    if not title:
        return []
    type_words = TYPE_WORDS[unit_type]
    core_words = _core_words(title, type_words)
    core = " ".join(core_words)
    head = head_stem(core) if core_words else ""

    is_section = unit_type is UnitType.SECTION
    # A section is normally addressed as "<başlık> bölümü"; its bare title collides with the
    # definition carrying the same name ("Bölüm 2: Türev" vs. "tanim-2.4: Türev"), so a section
    # gets no inflected forms and its title is weighted below every other kind.
    kind = "section" if is_section else "title"
    entries: list[AliasEntry] = [AliasEntry(title, WEIGHT_BY_KIND[kind], kind)]
    if not is_section:
        for form in inflect(title):
            entries.append(AliasEntry(form, WEIGHT_BY_KIND["title_inflected"], "title_inflected"))
        if core and core != title:
            for form in inflect(core):
                entries.append(AliasEntry(form, WEIGHT_BY_KIND["core"], "core"))
        # A title that already carries a possessive ("Ortalama Değişim Hızı") must be declined
        # from its bare stem, otherwise "hızını" — the form a student actually types — is missed.
        stem_phrase = " ".join([*core_words[:-1], head]) if core_words and head else ""
        if stem_phrase and stem_phrase != core:
            for form in inflect(stem_phrase):
                entries.append(AliasEntry(form, WEIGHT_BY_KIND["core"], "core"))
        if head and len(core_words) == 1:
            for form in inflect(head):
                entries.append(AliasEntry(form, WEIGHT_BY_KIND["head"], "head"))

    left_forms = _unique([core, genitive(core), head, genitive(head)])
    for type_word in type_words:
        for right in possessive_forms(type_word):
            for left in left_forms:
                entries.append(AliasEntry(f"{left} {right}", WEIGHT_BY_KIND["combo"], "combo"))

    ascii_entries = [
        AliasEntry(folded, entry.weight * ASCII_WEIGHT_FACTOR, "ascii")
        for entry in entries
        if (folded := strip_diacritics(entry.alias)) != entry.alias
    ]
    return _dedupe_entries(entries + ascii_entries)


def _dedupe_entries(entries: list[AliasEntry]) -> list[AliasEntry]:
    """Keep the highest-weight occurrence of each alias, preserving first-seen order."""
    best: dict[str, AliasEntry] = {}
    order: list[str] = []
    for entry in entries:
        alias = " ".join(entry.alias.split())
        if not alias:
            continue
        current = best.get(alias)
        if current is None:
            best[alias] = AliasEntry(alias, entry.weight, entry.kind)
            order.append(alias)
        elif entry.weight > current.weight:
            best[alias] = AliasEntry(alias, entry.weight, entry.kind)
    return [best[alias] for alias in order]


def generate_aliases(title_tr: str, unit_type: UnitType) -> list[str]:
    """Return the Turkish alias forms of a unit title, most specific first.

    ``generate_aliases("Türev", UnitType.DEFINITION)`` yields ``türev``, ``türevi``, ``türevin``,
    ``türevini``, ``türevinin``, … together with ``türev tanımı``, ``türevin tanımı``,
    ``türev tanımını`` and the ASCII-folded variant of each.

    Args:
        title_tr: Turkish title as printed in the book.
        unit_type: Type of the unit (selects the type word: ``Definition`` → ``tanım``).

    Returns:
        Deduplicated, deterministically ordered aliases; the normalised title comes first.
    """
    return [entry.alias for entry in generate_alias_entries(title_tr, unit_type)]


__all__ = [
    "ASCII_WEIGHT_FACTOR",
    "IRREGULAR_STEMS",
    "SOFTENING",
    "TYPE_WORDS",
    "VOICELESS",
    "VOWELS",
    "WEIGHT_BY_KIND",
    "AliasEntry",
    "generate_alias_entries",
    "generate_aliases",
    "genitive",
    "harden",
    "head_stem",
    "inflect",
    "possessive_forms",
    "soften",
]
