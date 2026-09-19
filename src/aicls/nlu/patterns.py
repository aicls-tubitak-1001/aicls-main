"""Compilation of the inventory's Turkish utterance patterns into regular expressions.

A pattern is written against the *canonical token stream*
(:class:`aicls.nlu.normalize.NormalizedText`), one unit per whitespace-separated item:

======================  =========================================================================
``{expr}``              a slot: a lazy capture group named after the slot
``{expr}?``             an optional slot
``kelime``              a literal token, tolerating one inflectional suffix token after it
``kelime?``             an optional literal
``(al|bul|hesapla)``    a required alternation (also tolerates a following suffix token)
``(al|bul)?``           an optional alternation
``<GEN>``               one suffix of the ``gen`` family of :data:`aicls.core.turkish.TR_SUFFIXES`
``<ABL>?``              optional ablative suffix; ``<ANY>`` matches any suffix at all
``<OBJ>?``              a word class: ``ifade``/``fonksiyon``/``denklem`` and friends
``<NUM>``               a bare numeral token as the normaliser writes it (``2``, ``2.5``)
======================  =========================================================================

Two design points matter:

* **suffix tolerance is automatic.** ``türev`` matches ``türev``, ``türev i`` and ``türev ini``
  because the normaliser has already split the suffix into its own token and the compiler
  appends an optional suffix group after every literal. Patterns therefore stay readable and the
  inventory never enumerates vowel-harmony variants by hand (ADR-0005).
* **slots swallow the suffix that belongs to them.** ``{expr} türev`` captures ``x kare nin``
  from ``x kare nin türev ini al``; the extractor drops suffix tokens when it slices the token
  list, so the spoken-math grammar still sees ``x kare``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re

from aicls.core.turkish import TR_SUFFIXES
from aicls.nlu.errors import InventoryError

#: ``<NAME>`` macro -> the suffix strings it may match. ``ANY`` is every suffix of every family.
SUFFIX_MACROS: dict[str, tuple[str, ...]] = {
    name.upper(): tuple(values) for name, values in TR_SUFFIXES.items()
}
SUFFIX_MACROS["ANY"] = tuple(
    sorted({suffix for family in TR_SUFFIXES.values() for suffix in family})
)

#: ``<NAME>`` word-class macros: closed sets of carrier words that may sit between a slot and the
#: keyword that identifies the intent ("x kare **ifadesinin** türevini al"). Unlike suffix macros
#: these are ordinary literals, so they also tolerate a following suffix token.
WORD_CLASSES: dict[str, tuple[str, ...]] = {
    "OBJ": ("ifade", "fonksiyon", "denklem", "eşitlik", "bağıntı", "sayı"),
    "DIR": ("sağdan", "soldan", "sağ", "sol", "pozitif", "negatif"),
    "VERB": ("al", "bul", "hesapla", "yap", "çıkar", "göster", "getir"),
}


def _ordinal_words() -> tuple[str, ...]:
    """Ordinal words from ``inventory/numbers.yaml`` (``<ORD>``)."""
    from aicls.nlu.numbers import load_numbers

    return tuple(load_numbers().ordinals)


#: ``<NUM>``: a numeral token exactly as the normaliser writes it. It exists so a pattern can
#: *place* a numeral -- ``"{expr} <NUM> (mertebe|derece) türev"`` -- instead of leaving the lazy
#: ``{expr}`` capture to swallow it, which is how "x üzeri üçün iki mertebeden türevini al" put
#: the order numeral inside the expression and made it unreadable.
NUMERAL_PATTERN = r"\d+(?:\.\d+)?"


#: Capture width per slot type. A one-token slot (``VAR``) must not swallow the words that
#: follow it, and a bounded slot (``POINT``, ``BOUNDS``) takes at most two tokens ("eksi sonsuz").
#: Free slots (``EXPR``, ``QUERY``) may run over any number of tokens.
SLOT_CAPTURE: dict[str, str] = {
    "EXPR": r"[^\s].*?",
    "QUERY": r"[^\s].*?",
    "POINT": r"[^\s]+(?:\s+[^\s]+)??",
    "BOUNDS": r"[^\s]+(?:\s+[^\s]+)??",
    "VAR": r"[^\s]+",
    "ORDER": r"[^\s]+",
    "DIRECTION": r"[^\s]+",
    "UNIT_TYPE": r"[^\s]+",
}

#: Capture used when the slot's type is unknown to the compiler.
DEFAULT_CAPTURE = SLOT_CAPTURE["EXPR"]

#: How many stacked suffix tokens a literal tolerates. Turkish stacks endings ("ifade-si-nin"),
#: and a slow dictation can put a space before each one, so two is the realistic maximum.
MAX_TOLERATED_SUFFIXES = 2

_SLOT_RE = re.compile(r"^\{(?P<name>[a-z_][a-z0-9_]*)\}(?P<opt>\?)?$")
_MACRO_RE = re.compile(r"^<(?P<name>[A-Z_]+)>(?P<opt>\?)?$")
_GROUP_RE = re.compile(r"^\((?P<body>[^()]+)\)(?P<opt>\?)?$")
_WORD_RE = re.compile(r"^(?P<word>[^\s(){}<>?]+)(?P<opt>\?)?$")


def _alternation(values: Sequence[str]) -> str:
    """Regex alternation of literal ``values``, longest first so the longest wins."""
    ordered = sorted({value for value in values if value}, key=lambda item: (-len(item), item))
    return "|".join(re.escape(value) for value in ordered)


@dataclass(frozen=True, slots=True)
class CompiledPattern:
    """One inventory pattern, ready to match.

    Attributes:
        source: The pattern exactly as written in ``commands.yaml``.
        regex: The compiled regular expression (anchored matching is the caller's choice).
        slots: Slot names the pattern captures, in order of appearance.
        literals: Number of literal units, used for the token-coverage score.
    """

    source: str
    regex: re.Pattern[str]
    slots: tuple[str, ...]
    literals: int

    def fullmatch(self, text: str) -> re.Match[str] | None:
        """Match the whole normalised text."""
        return self.regex.fullmatch(text)

    def search(self, text: str) -> re.Match[str] | None:
        """Match anywhere inside the normalised text."""
        return self.regex.search(text)


def compile_pattern(
    source: str,
    *,
    suffixes: Sequence[str],
    slot_types: Mapping[str, str] | None = None,
    intent: str = "",
) -> CompiledPattern:
    """Compile one inventory pattern into a :class:`CompiledPattern`.

    Args:
        source: The pattern text.
        suffixes: Every suffix token the normaliser can emit (for the tolerance groups).
        slot_types: Slot name -> slot type name, used to pick the capture width
            (:data:`SLOT_CAPTURE`). Unknown slots get :data:`DEFAULT_CAPTURE`.
        intent: Intent id, used only in error details.

    Returns:
        The compiled pattern.

    Raises:
        aicls.nlu.errors.InventoryError: If a unit is malformed, a slot name repeats, or the
            resulting regular expression does not compile.
    """
    units = source.split()
    if not units:
        raise InventoryError("empty_pattern", details={"intent": intent, "pattern": source})
    # The tolerance group is *lazy*: a literal only swallows the following suffix tokens when the
    # rest of the pattern cannot match otherwise. Greedy tolerance would let "için" eat the "e"
    # of "e üzeri x" (a dative suffix and the constant e are spelled alike).
    suffix_group = (
        f"(?:\\s+(?:{_alternation(suffixes)})){{0,{MAX_TOLERATED_SUFFIXES}}}?" if suffixes else ""
    )
    parts: list[str] = []
    slots: list[str] = []
    literals = 0
    captures = {
        name: SLOT_CAPTURE.get(type_name, DEFAULT_CAPTURE)
        for name, type_name in (slot_types or {}).items()
    }
    for index, unit in enumerate(units):
        body, optional = _unit_regex(
            unit, suffix_group, captures=captures, source=source, intent=intent
        )
        if optional and index == 0:
            raise InventoryError(
                "optional_first_unit", details={"intent": intent, "pattern": source}
            )
        if unit.startswith("{"):
            name = unit[1:].split("}")[0]
            if name in slots:
                raise InventoryError(
                    "duplicate_slot", details={"intent": intent, "pattern": source, "slot": name}
                )
            slots.append(name)
        elif not unit.startswith("<"):
            literals += 1
        separator = "" if index == 0 else "\\s+"
        if optional:
            parts.append(f"(?:{separator}{body})?")
        else:
            parts.append(f"{separator}{body}")
    try:
        regex = re.compile("".join(parts))
    except re.error as exc:  # pragma: no cover - defensive: units are validated above
        raise InventoryError(
            "bad_pattern", details={"intent": intent, "pattern": source, "error": str(exc)}
        ) from exc
    return CompiledPattern(source=source, regex=regex, slots=tuple(slots), literals=literals)


def _unit_regex(
    unit: str,
    suffix_group: str,
    *,
    captures: Mapping[str, str],
    source: str,
    intent: str,
) -> tuple[str, bool]:
    """Return ``(regex_body, optional)`` for one pattern unit."""
    slot = _SLOT_RE.match(unit)
    if slot is not None:
        name = slot["name"]
        return f"(?P<{name}>{captures.get(name, DEFAULT_CAPTURE)})", bool(slot["opt"])
    macro = _MACRO_RE.match(unit)
    if macro is not None:
        name = macro["name"]
        suffix_values = SUFFIX_MACROS.get(name)
        if suffix_values is not None:
            return f"(?:{_alternation(suffix_values)})", bool(macro["opt"])
        if name == "NUM":
            return f"(?:{NUMERAL_PATTERN}){suffix_group}", bool(macro["opt"])
        words = WORD_CLASSES.get(name) if name != "ORD" else _ordinal_words()
        if words is None:
            raise InventoryError(
                "unknown_macro", details={"intent": intent, "pattern": source, "macro": name}
            )
        return f"(?:{_alternation(words)}){suffix_group}", bool(macro["opt"])
    group = _GROUP_RE.match(unit)
    if group is not None:
        options = [option for option in group["body"].split("|") if option]
        if not options:
            raise InventoryError("empty_group", details={"intent": intent, "pattern": source})
        return f"(?:{_alternation(options)}){suffix_group}", bool(group["opt"])
    word = _WORD_RE.match(unit)
    if word is not None:
        return f"{re.escape(word['word'])}{suffix_group}", bool(word["opt"])
    raise InventoryError("bad_unit", details={"intent": intent, "pattern": source, "unit": unit})


def compile_patterns(
    sources: Sequence[str],
    *,
    suffixes: Sequence[str],
    slot_types: Mapping[str, str] | None = None,
    intent: str = "",
) -> tuple[CompiledPattern, ...]:
    """Compile every pattern of one intent, keeping the inventory order."""
    return tuple(
        compile_pattern(item, suffixes=suffixes, slot_types=slot_types, intent=intent)
        for item in sources
    )


__all__ = [
    "DEFAULT_CAPTURE",
    "MAX_TOLERATED_SUFFIXES",
    "NUMERAL_PATTERN",
    "SLOT_CAPTURE",
    "SUFFIX_MACROS",
    "WORD_CLASSES",
    "CompiledPattern",
    "compile_pattern",
    "compile_patterns",
]
