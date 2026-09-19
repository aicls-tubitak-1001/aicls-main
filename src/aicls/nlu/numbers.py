"""Turkish number words to digits (docs/mimari/metin-komut-katmani.md §3).

Two things happen here:

* :func:`to_number` reads a Turkish numeral phrase -- ``"yüz yirmi üç"`` → ``123``,
  ``"iki buçuk"`` → ``2.5``, ``"iki virgül beş"`` → ``2.5``, ``"üç bin dört yüz"`` → ``3400``.
  Turkish numerals are strictly additive inside a scale group and multiplicative across scale
  words, and a bare scale word means one of it (``"bin"`` = 1000, ``"yüz yirmi"`` = 120), which
  is exactly what the accumulator below implements.
* :func:`ordinal_value` reads an ordinal (``"ikinci"`` → 2), which is how the ``ORDER`` slot
  understands "ikinci türevi".

**A phrase that is not a numeral is refused, not summed** (:data:`ORDER_OF_MAGNITUDE`). Turkish
spells a numeral in strictly decreasing magnitude classes inside a scale group -- hundreds, then
tens, then units -- so ``"dört sıfır"`` and ``"üç beş"`` are not numerals at all. The accumulator
used to add them up regardless, and that is how ``"x kare eksi dört sıfır denklemini çöz"``
silently lost its last content word and solved ``x² − 4`` instead of asking what was meant
(KNOWN_ISSUES PRIORITY 1). :func:`aicls.nlu.normalize.normalize_text` reacts to the refusal by
shortening the run it folds, so the tokens survive into the token stream.

The tables live in ``inventory/numbers.yaml`` so a new spoken form is a data change, not a code
change; the module keeps no Turkish string of its own.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any, Never

from aicls.core.turkish import tr_casefold
from aicls.nlu.errors import InventoryError
from aicls.nlu.resources import NUMBERS_RESOURCE, load_yaml, resource_path

#: A digit token as the normaliser emits it (``"12"``, ``"2.5"``).
DIGIT_RE = re.compile(r"^\d+(?:\.\d+)?$")

#: Magnitude class of each numeral word, coarsest first. A well-formed Turkish numeral names the
#: classes of one scale group in strictly decreasing order ("iki yüz otuz dört"), so the
#: accumulator only accepts a word whose class is *below* the class it last consumed. ``OPEN`` is
#: the state of an empty group and ``CLOSED`` the state after a fraction, which ends the phrase.
OPEN = 4
HUNDREDS = 3
TENS = 2
UNITS = 1
CLOSED = 0

#: Documentation alias for the class ladder above (imported by the tests and the docs page).
ORDER_OF_MAGNITUDE: dict[str, int] = {
    "open": OPEN,
    "hundreds": HUNDREDS,
    "tens": TENS,
    "units": UNITS,
    "closed": CLOSED,
}


@dataclass(frozen=True, slots=True)
class NumberTable:
    """The spoken-numeral vocabulary loaded from ``inventory/numbers.yaml``.

    Attributes:
        units: ``sıfır``..``dokuz`` → 0..9.
        tens: ``on``..``doksan`` → 10..90.
        scales: ``yüz``/``bin``/``milyon``/``milyar`` → their multiplier.
        fractions: ``buçuk``/``yarım``/``çeyrek`` → the fractional part they add.
        decimal_markers: Words that introduce a decimal fraction (``virgül``).
        ordinals: ``birinci``..``onikinci`` → 1..12.
    """

    units: Mapping[str, int]
    tens: Mapping[str, int]
    scales: Mapping[str, int]
    fractions: Mapping[str, float]
    decimal_markers: frozenset[str]
    ordinals: Mapping[str, int]

    @property
    def words(self) -> frozenset[str]:
        """Every word that may take part in a cardinal numeral phrase."""
        return frozenset(
            {
                *self.units,
                *self.tens,
                *self.scales,
                *self.fractions,
                *self.decimal_markers,
            }
        )

    def is_number_word(self, token: str) -> bool:
        """Whether ``token`` is a cardinal numeral word or a digit literal."""
        return token in self.words or bool(DIGIT_RE.match(token))


def _int_map(data: Mapping[str, Any], key: str, path: str) -> dict[str, int]:
    section = data.get(key, {})
    if not isinstance(section, dict):
        raise InventoryError("numbers_section_invalid", details={"section": key, "path": path})
    out: dict[str, int] = {}
    for word, value in section.items():
        if not isinstance(value, int):
            raise InventoryError(
                "numbers_value_invalid", details={"section": key, "word": str(word)}
            )
        out[tr_casefold(str(word))] = value
    return out


def _float_map(data: Mapping[str, Any], key: str, path: str) -> dict[str, float]:
    section = data.get(key, {})
    if not isinstance(section, dict):
        raise InventoryError("numbers_section_invalid", details={"section": key, "path": path})
    out: dict[str, float] = {}
    for word, value in section.items():
        if not isinstance(value, int | float):
            raise InventoryError(
                "numbers_value_invalid", details={"section": key, "word": str(word)}
            )
        out[tr_casefold(str(word))] = float(value)
    return out


@lru_cache(maxsize=2)
def load_numbers(path: str | None = None) -> NumberTable:
    """Load (and cache) the numeral table.

    Args:
        path: Optional override; the packaged ``inventory/numbers.yaml`` when omitted.

    Returns:
        The parsed :class:`NumberTable`.

    Raises:
        aicls.nlu.errors.InventoryError: If a section is missing or malformed.
    """
    from pathlib import Path

    resolved = Path(path) if path else resource_path(NUMBERS_RESOURCE)
    data = load_yaml(resolved)
    markers = data.get("decimal_markers", [])
    if not isinstance(markers, list):
        raise InventoryError("numbers_section_invalid", details={"section": "decimal_markers"})
    return NumberTable(
        units=_int_map(data, "units", str(resolved)),
        tens=_int_map(data, "tens", str(resolved)),
        scales=_int_map(data, "scales", str(resolved)),
        fractions=_float_map(data, "fractions", str(resolved)),
        decimal_markers=frozenset(tr_casefold(str(item)) for item in markers),
        ordinals=_int_map(data, "ordinals", str(resolved)),
    )


def _tokenise(words: str | Iterable[str]) -> list[str]:
    if isinstance(words, str):
        return [tr_casefold(part) for part in words.split() if part]
    return [tr_casefold(str(part)) for part in words if str(part).strip()]


def _reject(reason: str, **extra: Any) -> Never:
    """Raise the layer's validation error for a phrase that is not a numeral."""
    from aicls.core import messages_tr
    from aicls.core.errors import InputValidationError

    details: dict[str, Any] = {"field": "words", "reason": reason}
    details.update(extra)
    raise InputValidationError(messages_tr.MSG_VALIDATION_ERROR, details=details)


class _NumeralAccumulator:
    """Reads a Turkish numeral phrase token by token, refusing ill-formed sequences.

    Turkish spells one scale group in strictly decreasing magnitude classes, so the accumulator
    keeps the class it last consumed (:data:`ORDER_OF_MAGNITUDE`) and refuses any word that does
    not go *below* it. ``"iki yüz otuz dört"`` walks hundreds → tens → units; ``"dört sıfır"``
    asks for units twice and is refused.

    Args:
        table: The numeral vocabulary.
    """

    def __init__(self, table: NumberTable) -> None:
        self._table = table
        self._total = 0.0
        self._group = 0.0
        self._fraction = 0.0
        self._rank = OPEN
        self._last_big_scale: int | None = None
        self._seen_value = False

    @property
    def seen_value(self) -> bool:
        """Whether any numeral word has been consumed."""
        return self._seen_value

    def value(self) -> float:
        """The value accumulated so far."""
        return self._total + self._group + self._fraction

    def feed(self, token: str) -> None:
        """Consume one cardinal token.

        Raises:
            aicls.core.errors.InputValidationError: If ``token`` is not a numeral word, or is one
                that cannot follow what has already been read.
        """
        if DIGIT_RE.match(token):
            self._digit(token)
        elif token in self._table.units:
            self._additive(float(self._table.units[token]), UNITS, token)
        elif token in self._table.tens:
            self._additive(float(self._table.tens[token]), TENS, token)
        elif token in self._table.scales:
            self._scale(token)
        elif token in self._table.fractions:
            self._fractional(token)
        else:
            _reject("not_a_number_word", token=token)
        self._seen_value = True

    def _digit(self, token: str) -> None:
        # A digit literal already carries every class it needs, so nothing may precede it in the
        # same group: "4 0" is two numerals a decoder ran together, never the numeral forty.
        if self._rank != OPEN:
            _reject("numeral_out_of_order", token=token)
        self._group += float(token)
        self._rank = UNITS

    def _additive(self, value: float, rank: int, token: str) -> None:
        if self._rank <= rank:
            _reject("numeral_out_of_order", token=token)
        self._group += value
        self._rank = rank

    def _scale(self, token: str) -> None:
        scale = self._table.scales[token]
        if scale >= 1000:
            if self._rank == CLOSED or (
                self._last_big_scale is not None and scale >= self._last_big_scale
            ):
                _reject("numeral_out_of_order", token=token)
            self._total += (self._group or 1.0) * scale
            self._group = 0.0
            self._rank = OPEN
            self._last_big_scale = scale
            return
        # "yüz" multiplies its group, so at most a single unit may precede it ("dört yüz").
        if self._rank not in (OPEN, UNITS):
            _reject("numeral_out_of_order", token=token)
        self._group = (self._group or 1.0) * scale
        self._rank = HUNDREDS

    def _fractional(self, token: str) -> None:
        if self._rank == CLOSED:
            _reject("numeral_out_of_order", token=token)
        self._fraction += self._table.fractions[token]
        self._rank = CLOSED


def to_number(words: str | Sequence[str], *, table: NumberTable | None = None) -> int | float:
    """Convert a Turkish numeral phrase to a number.

    Args:
        words: The phrase (``"yüz yirmi üç"``) or its tokens (``["yüz", "yirmi", "üç"]``).
            Digit literals are accepted and pass through (``"12"`` → 12).
        table: Numeral table to use; the packaged one when omitted.

    Returns:
        An ``int`` when the value is integral, otherwise a ``float``.

    Raises:
        aicls.core.errors.InputValidationError: If the phrase is empty, contains a word that is
            not part of the numeral vocabulary, or names its magnitude classes in an order
            Turkish does not use (``"dört sıfır"``, ``"üç beş"``).
    """
    numbers = table or load_numbers()
    tokens = _tokenise(words)
    if not tokens:
        _reject("empty")

    accumulator = _NumeralAccumulator(numbers)
    decimal_tokens: list[str] = []
    in_decimal = False
    for token in tokens:
        if in_decimal:
            decimal_tokens.append(token)
            continue
        if token in numbers.decimal_markers:
            in_decimal = True
            continue
        accumulator.feed(token)

    if not accumulator.seen_value and not decimal_tokens:
        _reject("empty")

    value = accumulator.value()
    if decimal_tokens:
        digits = "".join(str(int(to_number(token, table=numbers))) for token in decimal_tokens)
        value += float(f"0.{digits}") if digits else 0.0
    if value.is_integer():
        return int(value)
    return value


def ordinal_value(word: str, *, table: NumberTable | None = None) -> int | None:
    """Read an ordinal word or a ``"2"`` / ``"2."`` literal.

    Args:
        word: A single token (``"ikinci"``, ``"2."``, ``"2"``).
        table: Numeral table to use; the packaged one when omitted.

    Returns:
        The order as an ``int``, or ``None`` when the token is not an ordinal.
    """
    numbers = table or load_numbers()
    token = tr_casefold(word).rstrip(".")
    if token in numbers.ordinals:
        return numbers.ordinals[token]
    if token.isdigit():
        return int(token)
    return None


def format_number(value: int | float) -> str:
    """Render a number the way the normaliser writes it into the token stream."""
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return repr(float(value))


__all__ = [
    "CLOSED",
    "DIGIT_RE",
    "HUNDREDS",
    "OPEN",
    "ORDER_OF_MAGNITUDE",
    "TENS",
    "UNITS",
    "NumberTable",
    "format_number",
    "load_numbers",
    "ordinal_value",
    "to_number",
]
