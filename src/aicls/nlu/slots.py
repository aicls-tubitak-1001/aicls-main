"""Slot extraction: from a pattern match to typed, validated values.

A slot is filled from one of four sources, in this order:

1. **pattern** -- the named capture group of the matched utterance pattern;
2. **scan** -- a lexical sweep of the whole utterance, for slots that have no fixed position
   (``DIRECTION``: "sağdan"; ``UNIT_TYPE``: "teoremini"; ``ORDER``: "ikinci");
3. **context** -- :class:`aicls.nlu.types.ParseContext` (``last_latex`` fills ``expr`` for
   "türevini al", ``active_var`` fills ``var``);
4. **default** -- what the inventory declares.

``EXPR``, ``POINT`` and ``BOUNDS`` values are validated by conversion, not by inspection: the
captured tokens go through the spoken-math grammar and the resulting LaTeX must be accepted by
:func:`aicls.latex2sympy.parse_latex`. A slot that fails that round trip is *unresolved*, which
demotes its candidate intent (:mod:`aicls.nlu.matcher`) and, when the slot is required, produces
a :class:`~aicls.nlu.types.Clarification` instead of a wrong command. That is the mechanism
behind the proposal's "out-of-scope requests are filtered" requirement.

``VAR`` is inferred from the expression's free symbols when the utterance does not name it, and
stays unresolved when the expression has several symbols and none of them is the default -- the
case ``test_nlu_missing_variable_returns_clarification_tr`` pins down.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import dataclasses
from dataclasses import dataclass, field
from functools import lru_cache
import re
from typing import Any

from aicls.core.errors import AiclsError
from aicls.latex2sympy import parse_latex
from aicls.latex2sympy.ast import Node, Sym
from aicls.latex2sympy.symbols import SymbolTable
from aicls.latex2sympy.to_latex import to_latex
from aicls.mathengine.models import is_valid_var
from aicls.nlu.errors import SpokenMathError
from aicls.nlu.inventory.schema import IntentSpec, Inventory, SlotSpec
from aicls.nlu.lexicon import Lexicon, ScanRule, load_lexicon
from aicls.nlu.matcher import PatternMatch
from aicls.nlu.normalize import NormalizedText
from aicls.nlu.numbers import ordinal_value
from aicls.nlu.spoken_math.parser import spoken_to_latex
from aicls.nlu.types import ParseContext, SlotSource, UnitType

#: Slot types whose value is a LaTeX expression produced by the spoken-math grammar.
LATEX_SLOT_TYPES = frozenset({"EXPR", "POINT", "BOUNDS"})

#: A variable is a single latin letter (optionally subscripted) or a greek name. The mathengine's
#: ``VAR_PATTERN`` is deliberately wider -- it accepts any identifier -- so without this narrower
#: check a Turkish word captured by a pattern ("x küpün x **kare** ye göre türevini al") would be
#: accepted as a variable name and silently differentiate with respect to nothing.
VARIABLE_RE = re.compile(r"^[A-Za-z](?:_\{[A-Za-z0-9]{1,8}\})?$")


def is_variable_name(token: str) -> bool:
    """Whether ``token`` may name a variable in a spoken command."""
    if not is_valid_var(token):
        return False
    return bool(VARIABLE_RE.match(token)) or token in _greek_names()


@lru_cache(maxsize=1)
def _greek_names() -> frozenset[str]:
    """Greek letter names the LaTeX front end knows (``alpha``, ``theta``, ...)."""
    from aicls.latex2sympy.commands import command_table

    return frozenset(
        entry.latex.lstrip("\\") for entry in command_table().entries if entry.kind == "greek"
    )


#: Words that make a bare digit an order rather than a numeral inside the expression
#: ("2 mertebeden türev", "3 dereceden türev").
#:
#: ``türev`` used to be in this set, and that is exactly how ``"x üzeri üçün türevini al"`` read
#: its own exponent as a third-order derivative and answered ``6`` instead of ``3x²``
#: (KNOWN_ISSUES PRIORITY 1). A bare digit in front of ``türev`` is genuinely ambiguous between
#: an exponent and an order, so the layer now requires the order to be *said*: an ordinal word
#: (``ikinci``) or an explicit ``mertebe``/``derece`` phrase.
ORDER_ANCHORS = frozenset({"mertebe", "derece"})

#: Canonical words that introduce an exponent. A numeral directly after one of them is the
#: *exponent of the expression* and can never be the order of a derivative, however the rest of
#: the sentence reads -- the second half of the PRIORITY 1 fix, and the reason
#: ``"x üzeri iki mertebeden türevini al"`` takes the safe reading instead of squaring the
#: numeral into both slots.
POWER_WORDS = frozenset({"üzeri", "üssü"})

#: A captured fragment is only retried as raw LaTeX when it *looks* like LaTeX. Without this
#: guard ``parse_latex("logaritma x")`` happily reads eleven implicit factors and the slot fills
#: with nonsense instead of failing honestly.
LATEX_HINT_RE = re.compile(r"[\\{}_^]")


@dataclass(frozen=True, slots=True)
class SlotResolution:
    """The outcome of resolving every slot of one intent against one pattern match.

    Attributes:
        values: Resolved slot values by name.
        sources: Where each resolved value came from.
        missing: Required slots that could not be resolved.
        unresolved: Every slot that could not be resolved, required or not.
        trace: Machine-readable trace lines for the "how I understood you" panel.
        match: The pattern match this resolution used, if any.
    """

    values: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, SlotSource] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    trace: tuple[str, ...] = ()
    match: PatternMatch | None = None

    @property
    def complete(self) -> bool:
        """Whether every required slot resolved."""
        return not self.missing


def iter_nodes(node: Node) -> Iterator[Node]:
    """Yield ``node`` and every AST node beneath it (generic dataclass walk)."""
    yield node
    for info in dataclasses.fields(node):  # type: ignore[arg-type]  # nodes are dataclasses
        value = getattr(node, info.name, None)
        if isinstance(value, Node):
            yield from iter_nodes(value)
        elif isinstance(value, tuple):
            for item in value:
                if isinstance(item, Node):
                    yield from iter_nodes(item)


def free_symbols(node: Node) -> tuple[str, ...]:
    """Names of the free symbols in ``node``, sorted.

    The reserved names of :class:`aicls.latex2sympy.symbols.SymbolTable` (``e``, ``pi``, ``i``,
    ``infty``) are excluded: the LaTeX front end parses a bare ``e`` as a symbol node but
    ``to_sympy`` turns it into Euler's number, so counting it as a variable would make
    ``e^{x}`` look like a two-variable expression and trigger a needless clarification.
    """
    names = {
        item.name
        for item in iter_nodes(node)
        if isinstance(item, Sym) and item.name not in SymbolTable.RESERVED
    }
    return tuple(sorted(names))


def canonical_latex(source: str) -> str | None:
    """Re-print ``source`` through the LaTeX front end, or ``None`` when it does not parse."""
    try:
        return to_latex(parse_latex(source, strict=False))
    except AiclsError:
        return None


def _latex_from_tokens(analysed: NormalizedText, first: int, last: int) -> str | None:
    """Convert a token range to LaTeX: spoken-math grammar first, raw LaTeX second."""
    fragment = analysed.slice_math(first, last)
    if fragment:
        try:
            return spoken_to_latex(fragment, normalized=True)
        except SpokenMathError:
            pass
    start, end = analysed.source_span(first, last)
    raw = analysed.original[start:end].strip()
    if raw and LATEX_HINT_RE.search(raw):
        return canonical_latex(raw)
    return None


def _scan_words(analysed: NormalizedText, rules: Sequence[ScanRule]) -> str | None:
    """Return the value of the first scan rule whose words occur consecutively."""
    words = [
        token
        for token, kind in zip(analysed.tokens, analysed.kinds, strict=True)
        if kind != "suffix"
    ]
    for rule in rules:
        size = len(rule.words)
        for index in range(len(words) - size + 1):
            if tuple(words[index : index + size]) == rule.words:
                return rule.value
    return None


def consumed_indices(
    analysed: NormalizedText, match: PatternMatch | None, *, exclude: str = ""
) -> frozenset[int]:
    """Token indices the capture groups of ``match`` already claimed.

    Args:
        analysed: The normalised utterance the match was taken over.
        match: The adopted pattern match, or ``None`` when no pattern was adopted.
        exclude: Slot name to leave out (a slot never blocks itself).

    Returns:
        Every token index covered by another slot of the adopted reading.
    """
    if match is None:
        return frozenset()
    claimed: set[int] = set()
    groups = match.match.groupdict()
    for name in match.pattern.slots:
        if name == exclude or groups.get(name) is None:
            continue
        start, end = match.match.span(name)
        if start < 0:
            continue
        first, last = analysed.token_range(start, end)
        claimed.update(range(first, last))
    return frozenset(claimed)


def scan_order(analysed: NormalizedText, *, skip: frozenset[int] = frozenset()) -> int | None:
    """Find a derivative/series order in the utterance.

    An order has to be *said*, in one of exactly two ways:

    * an **ordinal word** -- ``"ikinci türevi"``, ``"üçüncü mertebeden"``. It is ignored when it
      sits inside a span another slot already claimed (``skip``);
    * a **digit with an explicit order noun** after it -- ``"2 mertebeden"``, ``"3 dereceden"``.
      The noun is decisive evidence, so such a digit counts even where another slot's lazy
      capture happened to swallow it; but a digit right after ``üzeri``/``üssü``
      (:data:`POWER_WORDS`) is an exponent and never an order.

    Everything else -- a bare numeral, a numeral in front of ``türev`` -- leaves the order at the
    inventory default. Reading ``"x üzeri üçün türevini al"`` as a *third* derivative is what
    made the platform answer ``6`` instead of ``3x²`` (KNOWN_ISSUES PRIORITY 1).

    Args:
        analysed: The normalised utterance.
        skip: Token indices another slot of the adopted reading already consumed.

    Returns:
        The order, or ``None`` when the utterance does not name one.
    """
    indices = [index for index, kind in enumerate(analysed.kinds) if kind != "suffix"]
    for position, index in enumerate(indices):
        token = analysed.tokens[index]
        value = ordinal_value(token)
        if value is None:
            continue
        if not token.rstrip(".").isdigit():
            if index in skip:
                continue
            return value
        previous = analysed.tokens[indices[position - 1]] if position else ""
        following = analysed.tokens[indices[position + 1]] if position + 1 < len(indices) else ""
        if previous in POWER_WORDS or following not in ORDER_ANCHORS:
            continue
        return value
    return None


class SlotExtractor:
    """Resolves the slots of one intent against a normalised utterance.

    Args:
        inventory: The inventory the intent belongs to (for defaults).
        lexicon: Lexicon holding the scan rules; the packaged one when omitted.
    """

    def __init__(self, inventory: Inventory, *, lexicon: Lexicon | None = None) -> None:
        self._inventory = inventory
        self._lexicon = lexicon or load_lexicon()

    def extract(
        self,
        intent: IntentSpec,
        analysed: NormalizedText,
        *,
        match: PatternMatch | None = None,
        context: ParseContext | None = None,
    ) -> SlotResolution:
        """Resolve every slot of ``intent``.

        Args:
            intent: The candidate intent.
            analysed: The normalised utterance.
            match: The pattern match to read capture groups from, if any.
            context: Session context filling ``from_context`` slots.

        Returns:
            The :class:`SlotResolution`; it is never an error, only possibly incomplete.
        """
        values: dict[str, Any] = {}
        sources: dict[str, SlotSource] = {}
        trace: list[str] = []
        unresolved: list[str] = []
        ordered = sorted(intent.slots.items(), key=lambda item: item[1].type == "VAR")
        for name, spec in ordered:
            resolved, source = self._resolve(
                name, spec, intent, analysed, match=match, context=context, values=values
            )
            if resolved is None:
                unresolved.append(name)
                trace.append(f"slot {name}: unresolved")
                continue
            values[name] = resolved
            sources[name] = source
            trace.append(f"slot {name}={resolved!r} ({source})")
        missing = tuple(name for name in unresolved if intent.slots[name].required)
        return SlotResolution(
            values=values,
            sources=sources,
            missing=missing,
            unresolved=tuple(unresolved),
            trace=tuple(trace),
            match=match,
        )

    def best(
        self,
        intent: IntentSpec,
        analysed: NormalizedText,
        matches: Sequence[PatternMatch],
        *,
        context: ParseContext | None = None,
    ) -> SlotResolution:
        """Try each pattern match and keep the resolution that fills the most required slots.

        Args:
            intent: The candidate intent.
            analysed: The normalised utterance.
            matches: Pattern matches for this intent, best first.
            context: Session context.

        Returns:
            The best :class:`SlotResolution`; the first complete one wins, otherwise the one with
            the fewest missing slots.
        """
        if not matches:
            return self.extract(intent, analysed, match=None, context=context)
        best: SlotResolution | None = None
        for candidate in matches:
            resolution = self.extract(intent, analysed, match=candidate, context=context)
            if resolution.complete:
                return resolution
            if best is None or len(resolution.missing) < len(best.missing):
                best = resolution
        assert best is not None
        return best

    # -- per-slot resolution ------------------------------------------------------------
    def _resolve(
        self,
        name: str,
        spec: SlotSpec,
        intent: IntentSpec,
        analysed: NormalizedText,
        *,
        match: PatternMatch | None,
        context: ParseContext | None,
        values: Mapping[str, Any],
    ) -> tuple[Any, SlotSource]:
        """Resolve one slot; returns ``(value, source)`` with ``value=None`` when unresolved."""
        if spec.fill == "scan":
            scanned = self._scan(name, spec, analysed, match)
            if scanned is not None:
                return scanned, "scan"
        else:
            found, captured = self._from_pattern(name, spec, analysed, match)
            if found:
                # The utterance *did* name this slot. If the value does not validate the slot
                # stays unresolved: silently falling back to the context would answer a question
                # the user did not ask ("seri açılımını yap" is not "expand the last board").
                return captured, "pattern"
        contextual = self._from_context(spec, context)
        if contextual is not None:
            return contextual, "context"
        if spec.type == "VAR":
            inferred = self._infer_var(spec, intent, values, context)
            if inferred is not None:
                return inferred, "inferred"
            return None, "default"
        if spec.default is not None:
            return spec.default, "default"
        return None, "default"

    def _scan(
        self,
        name: str,
        spec: SlotSpec,
        analysed: NormalizedText,
        match: PatternMatch | None,
    ) -> Any:
        if spec.type == "ORDER":
            # A token another slot already consumed cannot also be the order: the exponent of
            # "x üzeri üç" belongs to ``expr`` and only to ``expr`` (KNOWN_ISSUES PRIORITY 1).
            return scan_order(analysed, skip=consumed_indices(analysed, match, exclude=name))
        rules = self._lexicon.scan.get(spec.scan_key or "", ())
        value = _scan_words(analysed, rules)
        if value is None:
            return None
        if spec.type == "UNIT_TYPE":
            return UnitType(value).value
        return value

    def _from_pattern(
        self,
        name: str,
        spec: SlotSpec,
        analysed: NormalizedText,
        match: PatternMatch | None,
    ) -> tuple[bool, Any]:
        """Read one slot from the pattern match.

        Returns:
            ``(captured, value)``. ``captured`` says whether the utterance filled this slot at
            all; ``value`` is ``None`` when it did but the text is not a valid value.
        """
        if match is None or name not in match.pattern.slots:
            return False, None
        raw = match.match.groupdict().get(name)
        if raw is None:
            return False, None
        span = match.match.span(name)
        first, last = analysed.token_range(span[0], span[1])
        if first >= last:
            return False, None
        if all(kind == "suffix" for kind in analysed.kinds[first:last]):
            # An optional slot that only caught inflectional suffixes caught nothing.
            return False, None
        if spec.type in LATEX_SLOT_TYPES:
            return True, _latex_from_tokens(analysed, first, last)
        if spec.type == "QUERY":
            return True, analysed.slice_math(first, last).strip() or None
        if spec.type == "VAR":
            token = analysed.slice_math(first, last).strip()
            return True, (token if is_variable_name(token) else None)
        if spec.type == "ORDER":
            return True, ordinal_value(analysed.slice_math(first, last).strip())
        if spec.type == "DIRECTION":
            return True, (raw.strip() or None)
        return True, self._unit_type(analysed.slice_math(first, last).strip())

    def _unit_type(self, token: str) -> str | None:
        """Read a unit type from a captured token.

        The token comes out of the normaliser, so it is a *Turkish* word ("tanım"): the lexicon's
        ``unit_type`` scan rules are the mapping, and the enum value itself is accepted as a
        fallback for an inventory that spells the type out.
        """
        if not token:
            return None
        for rule in self._lexicon.scan.get("unit_type", ()):
            if len(rule.words) == 1 and rule.words[0] == token:
                return UnitType(rule.value).value
        for member in UnitType:
            if member.value.casefold() == token.casefold():
                return member.value
        return None

    def _from_context(self, spec: SlotSpec, context: ParseContext | None) -> Any:
        if context is None or spec.from_context is None:
            return None
        if spec.from_context == "last_latex":
            if not context.last_latex:
                return None
            return canonical_latex(context.last_latex)
        if not context.active_var:
            return None
        return context.active_var if is_variable_name(context.active_var) else None

    def _infer_var(
        self,
        spec: SlotSpec,
        intent: IntentSpec,
        values: Mapping[str, Any],
        context: ParseContext | None,
    ) -> str | None:
        """Pick the variable from the expression's free symbols."""
        default = spec.default or self._inventory.defaults.var
        symbols: tuple[str, ...] = ()
        for other, other_spec in intent.slots.items():
            if other_spec.type != "EXPR" or other not in values:
                continue
            latex = values[other]
            if not isinstance(latex, str):
                continue
            try:
                symbols = free_symbols(parse_latex(latex, strict=False))
            except AiclsError:  # pragma: no cover - the value already parsed once
                symbols = ()
            break
        if context is not None and context.active_var and context.active_var in symbols:
            return context.active_var
        if len(symbols) == 1:
            return symbols[0]
        if not symbols:
            return default if is_variable_name(default) else None
        if default in symbols:
            return default
        return None


__all__ = [
    "LATEX_SLOT_TYPES",
    "ORDER_ANCHORS",
    "VARIABLE_RE",
    "SlotExtractor",
    "SlotResolution",
    "canonical_latex",
    "consumed_indices",
    "free_symbols",
    "is_variable_name",
    "iter_nodes",
    "scan_order",
]
