"""The intent-only LLM fallback: off by default, and never more than an intent id (ADR-0005 §7)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from aicls.nlu import (
    Clarification,
    IntentFallback,
    LlmIntentSuggester,
    Matched,
    NoMatch,
    ParseContext,
    load_inventory,
    parse_command,
    sanitize_suggestion,
)
from aicls.nlu.fallback import fallback_enabled


class RecordingFallback:
    """An :class:`IntentFallback` that records every call and returns a fixed answer."""

    def __init__(self, answer: str | None = "differentiate") -> None:
        self.answer = answer
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def suggest(self, text: str, intents: Sequence[str]) -> str | None:
        self.calls.append((text, tuple(intents)))
        return self.answer


class TestDisabledByDefault:
    def test_llm_fallback_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the default settings the fallback object is not even called."""
        from aicls.core.settings import get_settings

        monkeypatch.delenv("AICLS_LLM_FALLBACK", raising=False)
        get_settings.cache_clear()
        fallback = RecordingFallback()
        outcome = parse_command("bugün hava nasıl", fallback=fallback)
        get_settings.cache_clear()
        assert fallback.calls == []
        assert isinstance(outcome, NoMatch)

    def test_fallback_enabled_reads_the_setting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aicls.core.settings import get_settings

        monkeypatch.delenv("AICLS_LLM_FALLBACK", raising=False)
        get_settings.cache_clear()
        assert fallback_enabled() is False
        get_settings.cache_clear()

    def test_rule_path_never_consults_the_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AICLS_LLM_FALLBACK", "true")
        from aicls.core.settings import get_settings

        get_settings.cache_clear()
        fallback = RecordingFallback()
        outcome = parse_command("x karenin türevini al", fallback=fallback)
        get_settings.cache_clear()
        assert isinstance(outcome, Matched)
        assert fallback.calls == []


class TestSuggestionSanitising:
    def test_llm_fallback_can_only_return_inventory_intent_id(self) -> None:
        """Anything that is not an inventory intent id is discarded."""
        intents = load_inventory().intent_ids
        assert sanitize_suggestion("differentiate", intents) == "differentiate"
        for rubbish in (
            "diff(x**2, x)",
            "türevini al",
            "DIFFERENTIATE",
            "unknown_intent",
            "",
            None,
            42,
            ["differentiate"],
        ):
            assert sanitize_suggestion(rubbish, intents) is None

    def test_an_invented_command_never_reaches_the_pipeline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AICLS_LLM_FALLBACK", "true")
        from aicls.core.settings import get_settings

        get_settings.cache_clear()
        outcome = parse_command("bugün hava nasıl", fallback=RecordingFallback("diff(x,x)"))
        get_settings.cache_clear()
        assert isinstance(outcome, NoMatch)


class TestEnabledPath:
    def test_a_valid_intent_id_re_runs_the_rule_based_extractor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AICLS_LLM_FALLBACK", "true")
        from aicls.core.settings import get_settings

        get_settings.cache_clear()
        outcome = parse_command(
            "şunun eğimini istiyorum",
            context=ParseContext(last_latex="x^{2}"),
            fallback=RecordingFallback("differentiate"),
        )
        get_settings.cache_clear()
        assert isinstance(outcome, Matched)
        assert outcome.intent == "differentiate"
        assert outcome.source == "llm"
        assert outcome.slots["expr"] == "x^{2}"

    def test_an_unfillable_slot_still_asks_instead_of_guessing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AICLS_LLM_FALLBACK", "true")
        from aicls.core.settings import get_settings

        get_settings.cache_clear()
        outcome = parse_command("şunu istiyorum", fallback=RecordingFallback("differentiate"))
        get_settings.cache_clear()
        assert isinstance(outcome, Clarification)
        assert outcome.missing_slot == "expr"


class TestLlmIntentSuggester:
    def test_is_a_valid_intent_fallback(self) -> None:
        suggester = LlmIntentSuggester(lambda text, intents: "differentiate", enabled=True)
        assert isinstance(suggester, IntentFallback)

    def test_returns_none_when_disabled(self) -> None:
        suggester = LlmIntentSuggester(lambda text, intents: "differentiate", enabled=False)
        assert suggester.suggest("x", ["differentiate"]) is None

    def test_validates_the_client_answer(self) -> None:
        suggester = LlmIntentSuggester(lambda text, intents: "not_an_intent", enabled=True)
        assert suggester.suggest("x", ["differentiate"]) is None

    def test_client_failure_is_swallowed(self) -> None:
        def broken(text: str, intents: Sequence[str]) -> str:
            raise RuntimeError("no network")

        suggester = LlmIntentSuggester(broken, enabled=True)
        assert suggester.suggest("x", ["differentiate"]) is None

    def test_enabled_defers_to_the_setting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aicls.core.settings import get_settings

        monkeypatch.delenv("AICLS_LLM_FALLBACK", raising=False)
        get_settings.cache_clear()
        suggester = LlmIntentSuggester(lambda text, intents: "differentiate")
        assert suggester.enabled is False
        get_settings.cache_clear()

    def test_no_llm_client_is_imported_anywhere_in_the_package(self) -> None:
        """The package must stay offline: not one source file may name a client library."""
        from pathlib import Path
        import re

        import aicls.nlu

        forbidden = re.compile(
            r"^\s*(?:import|from)\s+(openai|anthropic|httpx|requests|urllib|socket)\b",
            re.MULTILINE,
        )
        root = Path(aicls.nlu.__file__).parent
        offenders = [
            path.name
            for path in root.rglob("*.py")
            if forbidden.search(path.read_text(encoding="utf-8"))
        ]
        assert offenders == []
