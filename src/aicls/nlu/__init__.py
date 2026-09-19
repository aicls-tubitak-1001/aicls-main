"""Turkish text -> intent/slots -> Command (rule-based text-to-command layer, İP-4).

The proposal never names this layer, yet nothing works by voice without it: ASR produces
``"x karenin türevini al"`` and the CAS needs ``Diff(expr="x^{2}", var="x")``. ADR-0005 fixes the
design as **rule-based first, and an LLM may never produce a result** -- it may at most name an
intent that already exists in the packaged inventory.

Public entry points (docs/mimari/moduller.md, docs/mimari/metin-komut-katmani.md)::

    load_inventory(path=None) -> Inventory          # 17 v1 intents, YAML package resource
    normalize_text(text) -> NormalizedText          # NFC, İ/ı casefold, suffixes, numbers, spans
    spoken_to_latex(text) -> str                    # "x kare artı iki x" -> "x^{2} + 2 x"
    to_number(words) -> int | float                 # "yüz yirmi üç" -> 123
    parse_command(text, *, context=None, inventory=None, fallback=None) -> ParseOutcome
    ParseOutcome = Matched | Clarification | NoMatch

Everything that leaves the package is a subclass of :class:`aicls.core.errors.AiclsError`; an
utterance the layer does not understand is a :class:`NoMatch` with Turkish suggestions, not an
exception. ``aicls.db`` and ``aicls.book`` are never imported (ADR-0002): a textbook lookup is
returned as a :class:`BookFetch` command for the layer above to execute.
"""

from __future__ import annotations

from aicls.nlu.errors import CommandBuildError, InventoryError, NluError, SpokenMathError
from aicls.nlu.fallback import (
    IntentFallback,
    LlmIntentSuggester,
    fallback_enabled,
    sanitize_suggestion,
)
from aicls.nlu.inventory import (
    MIN_EXAMPLES_PER_INTENT,
    Defaults,
    Example,
    IntentSpec,
    Inventory,
    SlotSpec,
    SlotType,
    Weights,
    load_inventory,
)
from aicls.nlu.lexicon import Lexicon, load_lexicon
from aicls.nlu.mapper import CAS_BUILDERS, build_command
from aicls.nlu.matcher import IntentCandidate, IntentMatcher, PatternMatch
from aicls.nlu.normalize import NormalizedText, Token, normalize_text
from aicls.nlu.numbers import NumberTable, load_numbers, ordinal_value, to_number
from aicls.nlu.patterns import CompiledPattern, compile_pattern
from aicls.nlu.resolver import explain, parse_command
from aicls.nlu.slots import SlotExtractor, SlotResolution, free_symbols
from aicls.nlu.spoken_math import spoken_to_ast, spoken_to_latex
from aicls.nlu.types import (
    Alternative,
    BookFetch,
    Clarification,
    Command,
    Matched,
    Modality,
    NoMatch,
    ParseContext,
    ParseOutcome,
    UiAction,
    UiActionName,
    UnitType,
    context_from_mapping,
)

__all__ = [
    "CAS_BUILDERS",
    "MIN_EXAMPLES_PER_INTENT",
    "Alternative",
    "BookFetch",
    "Clarification",
    "Command",
    "CommandBuildError",
    "CompiledPattern",
    "Defaults",
    "Example",
    "IntentCandidate",
    "IntentFallback",
    "IntentMatcher",
    "IntentSpec",
    "Inventory",
    "InventoryError",
    "Lexicon",
    "LlmIntentSuggester",
    "Matched",
    "Modality",
    "NluError",
    "NoMatch",
    "NormalizedText",
    "NumberTable",
    "ParseContext",
    "ParseOutcome",
    "PatternMatch",
    "SlotExtractor",
    "SlotResolution",
    "SlotSpec",
    "SlotType",
    "SpokenMathError",
    "Token",
    "UiAction",
    "UiActionName",
    "UnitType",
    "Weights",
    "build_command",
    "compile_pattern",
    "context_from_mapping",
    "explain",
    "fallback_enabled",
    "free_symbols",
    "load_inventory",
    "load_lexicon",
    "load_numbers",
    "normalize_text",
    "ordinal_value",
    "parse_command",
    "sanitize_suggestion",
    "spoken_to_ast",
    "spoken_to_latex",
    "to_number",
]
