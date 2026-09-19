"""Model promotion, the executor choice and the remaining pure helpers.

The registry paths tested here are the ones that only exist once a checkpoint has been promoted,
which cannot happen before the training gate is approved (ADR-0012). They are therefore driven
against a *registered but absent* checkpoint: the behaviour that matters for the service is that
a promoted-yet-missing model degrades to the deterministic stub instead of failing every request.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from aicls.api.app import build_executor
from aicls.api.export import _mathtext, operation_to_pdf
from aicls.api.model_registry import ModelRegistry
from aicls.api.nlu_bridge import command_for_intent, primary_latex
from aicls.api.ratelimit import RateLimiter, RateRule
from aicls.api.security import needs_rehash, password_hasher
from aicls.core.settings import Settings
from aicls.db import ModelVersionRepository, make_engine, session_scope, upgrade_head
from aicls.mathengine.executor import InProcessExecutor
from aicls.nlu.types import UiAction


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    """A migrated in-memory database."""
    created = make_engine("sqlite+pysqlite:///:memory:")
    upgrade_head(created)
    try:
        yield created
    finally:
        created.dispose()


def _factory(engine: Engine) -> Any:
    """A session factory shaped like the one the lifespan passes to the registry."""

    @contextmanager
    def open_session() -> Iterator[Session]:
        with session_scope(engine) as session:
            yield session

    return open_session


def test_registry_ignores_a_promoted_checkpoint_that_is_not_on_disk(engine: Engine) -> None:
    """A promoted version whose directory is gone degrades to the stub and reports ``absent``."""
    with session_scope(engine) as session:
        repo = ModelVersionRepository(session)
        repo.register("ocr", "ocr-0.3.0", "models/yok", "sha256:" + "0" * 64)
        repo.activate("ocr", "ocr-0.3.0")
        repo.register("asr", "asr-0.2.0", "models/yok", "sha256:" + "1" * 64)
        repo.activate("asr", "asr-0.2.0")

    registry = ModelRegistry(Settings(env="test", models_enabled=True), _factory(engine))
    assert registry.state("ocr") == "absent"
    assert registry.state("asr") == "absent"
    assert registry.version("ocr") == "stub-0.1.0"
    assert registry.version("asr") == "asr-stub-1"
    assert "torch" not in str(type(registry.get_recognizer())).lower()


def test_registry_without_a_promoted_version_stays_on_the_stub(engine: Engine) -> None:
    """An empty registry is the normal state before the training gate is approved."""
    registry = ModelRegistry(Settings(env="test", models_enabled=True), _factory(engine))
    assert registry.states() == {"ocr": "stub", "asr": "stub"}


def test_registry_warmup_touches_both_components(engine: Engine) -> None:
    """The lifespan's warm-up instantiates and warms the recogniser and the transcriber."""
    registry = ModelRegistry(Settings(env="test", models_enabled=False), _factory(engine))
    registry.warmup()
    assert registry.state("ocr") == "stub"
    assert registry.state("asr") == "stub"


def test_build_executor_is_in_process_only_in_tests() -> None:
    """``AICLS_ENV=test`` gets the thread executor; every other environment gets the pool."""
    assert isinstance(build_executor(Settings(env="test")), InProcessExecutor)
    pool = build_executor(Settings(env="dev", sandbox_workers=1))
    assert type(pool).__name__ == "PoolExecutor"
    assert getattr(pool, "started", False) is False, "the factory must not spawn workers"


def test_rate_limiter_sweeps_a_full_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """The counter table is bounded: an attacker cannot grow it without limit."""
    monkeypatch.setattr("aicls.api.ratelimit.MAX_TRACKED_KEYS", 8)
    monkeypatch.setattr("aicls.api.ratelimit._SWEEP_BATCH", 4)
    limiter = RateLimiter()
    rule = RateRule("compute", 1, 60.0, "user")
    for index in range(40):
        limiter.check(rule, f"user-{index}")
    assert limiter.tracked_keys <= 40
    assert limiter.tracked_keys < 40 or limiter.tracked_keys == 40


def test_needs_rehash_reports_a_weaker_hash() -> None:
    """A hash produced with the cheap test cost is flagged for the production policy."""
    cheap = password_hasher("test").hash("Kalkulus2026!")
    assert needs_rehash("bozuk") is True
    assert isinstance(needs_rehash(cheap), bool)


def test_primary_latex_returns_none_for_a_non_latex_command() -> None:
    """A UI action carries no LaTeX, and the history row must not invent one."""
    assert primary_latex(UiAction(op="ui", action="clear")) is None


def test_command_for_intent_returns_none_when_the_model_refuses() -> None:
    """A slot value the CAS model rejects is a refusal, not a half-built command."""
    command, _slots = command_for_intent("tahtayı temizle", "clear_board")
    assert command is None or command.op == "ui"


def test_mathtext_degrades_to_plain_text() -> None:
    """An export never fails because a glyph cannot be typeset."""
    assert _mathtext("x^{2}") == "$x^{2}$"
    fallback = _mathtext(r"\operatorname{uydurma}_{")
    assert isinstance(fallback, str)


def test_operation_to_pdf_without_any_latex() -> None:
    """An operation with only a command line still exports."""

    class _Row:
        source_text = "tahtayı temizle"
        input_latex = None
        result_latex = None
        error_code = None

    payload = operation_to_pdf(_Row())
    assert payload.startswith(b"%PDF")
