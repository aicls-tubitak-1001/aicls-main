"""The intent-only fallback protocol (ADR-0005 §7).

The proposal's "no hallucination" claim (A.2.2) means a language model may never produce a
*result*. This module is where that boundary is drawn in code:

* :class:`IntentFallback` may return **only an intent id that already exists in the inventory**.
  Anything else -- a LaTeX string, a command, free text, an unknown id -- is discarded by
  :func:`sanitize_suggestion`, which is what
  ``test_llm_fallback_can_only_return_inventory_intent_id`` pins down.
* the returned intent is then re-run through the *rule-based* slot extractor; the model never
  fills a slot, never sees a result and never writes LaTeX.
* the fallback is **off by default** (``AICLS_LLM_FALLBACK=false``):
  :func:`fallback_enabled` reads :class:`aicls.core.settings.Settings`, and
  ``test_llm_fallback_disabled_by_default`` asserts that a fallback object passed to
  ``parse_command`` is not even called.

No LLM SDK is imported here, and none may be: :class:`LlmIntentSuggester` takes an injected
callable so the transport (and the decision to have one at all) stays outside this package.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from aicls.core.logging import get_logger

_logger = get_logger(__name__)


@runtime_checkable
class IntentFallback(Protocol):
    """Suggests an inventory intent id for text the rules did not understand."""

    def suggest(self, text: str, intents: Sequence[str]) -> str | None:
        """Return one id from ``intents``, or ``None``.

        Args:
            text: The user's utterance.
            intents: Every intent id in the inventory -- the only allowed return values.

        Returns:
            An intent id, or ``None`` when the fallback has no opinion.
        """
        ...


def fallback_enabled() -> bool:
    """Whether ``AICLS_LLM_FALLBACK`` turned the intent fallback on (default: ``False``)."""
    from aicls.core.settings import get_settings

    return bool(get_settings().llm_fallback_enabled)


def sanitize_suggestion(suggestion: object, intents: Sequence[str]) -> str | None:
    """Reduce whatever the fallback returned to a valid intent id, or ``None``.

    Args:
        suggestion: The raw return value of :meth:`IntentFallback.suggest`.
        intents: Allowed intent ids.

    Returns:
        The intent id when ``suggestion`` is exactly one of ``intents``, otherwise ``None``.
    """
    if not isinstance(suggestion, str):
        return None
    candidate = suggestion.strip()
    if candidate in set(intents):
        return candidate
    _logger.warning("nlu.fallback_rejected", suggestion=candidate[:80])
    return None


class LlmIntentSuggester:
    """An :class:`IntentFallback` backed by an injected callable.

    The callable receives the utterance and the allowed intent ids and returns a string. Any
    exception it raises is swallowed and the rule-based outcome stands (ADR-0005 §7, "Ağ/hata").

    Args:
        client: The callable that produces a suggestion.
        enabled: Force the suggester on or off; ``None`` defers to :func:`fallback_enabled`.
    """

    def __init__(
        self,
        client: Callable[[str, Sequence[str]], str | None],
        *,
        enabled: bool | None = None,
    ) -> None:
        self._client = client
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Whether this suggester is allowed to run."""
        return fallback_enabled() if self._enabled is None else self._enabled

    def suggest(self, text: str, intents: Sequence[str]) -> str | None:
        """Ask the client for an intent id and validate it against ``intents``."""
        if not self.enabled:
            return None
        try:
            raw = self._client(text, list(intents))
        except Exception as exc:  # a fallback failure must never break the rule-based path
            _logger.warning("nlu.fallback_failed", error=str(exc)[:200])
            return None
        return sanitize_suggestion(raw, intents)


__all__ = ["IntentFallback", "LlmIntentSuggester", "fallback_enabled", "sanitize_suggestion"]
