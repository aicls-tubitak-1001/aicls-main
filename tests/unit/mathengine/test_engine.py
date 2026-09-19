"""``MathEngine``: validation, the LRU cache and the timings the latency report is built from."""

from __future__ import annotations

import time
from typing import Any

import pytest

from aicls.core.errors import InputValidationError
from aicls.core.settings import settings_override
from aicls.mathengine import (
    MAX_INPUT_CHARS,
    CasCommand,
    Diff,
    Evaluate,
    InProcessExecutor,
    Integrate,
    LruCache,
    MathEngine,
    Result,
    canonical_key,
)

#: A cache hit must be a dictionary lookup, not a computation (İP-4 risk plan).
CACHE_HIT_BUDGET_MS = 5.0

DIFF = Diff(op="diff", expr="x^{2}")


class CountingExecutor:
    """An executor that records how often it was actually asked to compute."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._inner = InProcessExecutor(enforce_timeout=False)

    def run(self, cmd: CasCommand, timeout_s: float) -> Result:
        self.calls.append(cmd.op)
        return self._inner.run(cmd, timeout_s)


# ---------------------------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------------------------
def test_engine_defaults_to_an_in_process_executor_and_its_own_cache() -> None:
    engine = MathEngine()
    assert isinstance(engine.executor, InProcessExecutor)
    assert isinstance(engine.cache, LruCache)
    assert engine.timeout_s > 0


def test_engine_accepts_positional_and_keyword_construction() -> None:
    executor = InProcessExecutor()
    cache = LruCache(maxsize=8)
    positional = MathEngine(executor, cache, 1.5)
    keyword = MathEngine(executor=executor, cache=cache, timeout_s=1.5)
    assert positional.timeout_s == keyword.timeout_s == 1.5
    assert positional.cache is keyword.cache is cache


def test_timeout_defaults_to_the_configured_budget() -> None:
    with settings_override(compute_timeout_s=1.25):
        assert MathEngine().timeout_s == 1.25


# ---------------------------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------------------------
def test_engine_accepts_the_raw_api_payload() -> None:
    engine = MathEngine()
    payload: dict[str, Any] = {"op": "diff", "expr": "x^{2}", "var": "x", "order": 1}
    assert engine.execute(payload).plain == "2*x"


def test_an_invalid_payload_becomes_a_validation_error() -> None:
    engine = MathEngine()
    with pytest.raises(InputValidationError) as excinfo:
        engine.execute({"op": "nope", "expr": "x"})
    assert excinfo.value.details["errors"]


def test_total_input_over_2048_chars_rejected() -> None:
    """A single field is capped by ``LatexStr``; the engine caps the command as a whole."""
    engine = MathEngine()
    chunk = "1+" * 500 + "1"  # 1001 characters, comfortably under the per-field limit
    command = Evaluate(op="evaluate", expr="x", subs={f"v{i}": chunk for i in range(3)})
    with pytest.raises(InputValidationError) as excinfo:
        engine.execute(command)
    assert excinfo.value.details["max_len"] == MAX_INPUT_CHARS
    assert excinfo.value.details["length"] > MAX_INPUT_CHARS


# ---------------------------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------------------------
def test_cache_hit_on_repeated_command_under_5ms() -> None:
    executor = CountingExecutor()
    engine = MathEngine(executor=executor)

    first = engine.execute(DIFF)
    assert first.cache_hit is False

    started = time.perf_counter()
    second = engine.execute(DIFF)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    assert second.cache_hit is True
    assert second.latex == first.latex
    assert second.srepr == first.srepr
    assert elapsed_ms < CACHE_HIT_BUDGET_MS
    assert second.elapsed_ms < CACHE_HIT_BUDGET_MS
    assert executor.calls == ["diff"]  # SymPy ran exactly once
    assert engine.cache.stats()["hits"] == 1


def test_cache_key_is_the_canonical_command_not_the_text() -> None:
    executor = CountingExecutor()
    engine = MathEngine(executor=executor)
    engine.execute(Diff(op="diff", expr="x^{2}"))
    engine.execute(Diff(op="diff", expr="x^{2}", var="x", order=1))  # same command, spelled out
    assert executor.calls == ["diff"]
    assert canonical_key(DIFF) in engine.cache


def test_different_commands_do_not_share_a_cache_entry() -> None:
    executor = CountingExecutor()
    engine = MathEngine(executor=executor)
    engine.execute(Diff(op="diff", expr="x^{2}"))
    engine.execute(Diff(op="diff", expr="x^{3}"))
    engine.execute(Integrate(op="integrate", expr="x^{2}"))
    assert executor.calls == ["diff", "diff", "integrate"]


def test_clear_cache_forces_recomputation() -> None:
    executor = CountingExecutor()
    engine = MathEngine(executor=executor)
    engine.execute(DIFF)
    engine.clear_cache()
    engine.execute(DIFF)
    assert executor.calls == ["diff", "diff"]


def test_a_bounded_cache_evicts() -> None:
    executor = CountingExecutor()
    engine = MathEngine(executor=executor, cache=LruCache(maxsize=1))
    engine.execute(Diff(op="diff", expr="x^{2}"))
    engine.execute(Diff(op="diff", expr="x^{3}"))
    engine.execute(Diff(op="diff", expr="x^{2}"))
    assert executor.calls == ["diff", "diff", "diff"]


def test_errors_are_not_cached() -> None:
    engine = MathEngine()
    from aicls.mathengine import Solve, UnsupportedProblem

    command = Solve(op="solve", equation=r"\cos(x)=x")
    for _ in range(2):
        with pytest.raises(UnsupportedProblem):
            engine.execute(command)
    assert engine.cache.size == 0


# ---------------------------------------------------------------------------------------------
# timings
# ---------------------------------------------------------------------------------------------
def test_timings_carry_the_stages_the_api_reports() -> None:
    engine = MathEngine()
    result = engine.execute(DIFF)
    assert set(result.timings) >= {"sympy", "cas", "total"}
    assert result.timings["total"] >= result.timings["cas"] >= 0.0
    assert result.elapsed_ms == result.timings["total"]


def test_a_cache_hit_reports_a_cache_stage() -> None:
    engine = MathEngine()
    engine.execute(DIFF)
    hit = engine.execute(DIFF)
    assert "cache" in hit.timings
    assert hit.timings["total"] == hit.timings["cache"]
