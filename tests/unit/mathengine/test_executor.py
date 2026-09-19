"""``InProcessExecutor``: what tests and the CLI run on, and what its timeout can/cannot do."""

from __future__ import annotations

import time

import pytest

from aicls.core.errors import ErrorCode
from aicls.mathengine import CasTimeout, Diff, InProcessExecutor, Result
from aicls.mathengine.executor import Executor
from aicls.mathengine.models import CasCommand

DIFF = Diff(op="diff", expr="x^{2}")


def test_in_process_executor_satisfies_the_protocol() -> None:
    assert isinstance(InProcessExecutor(), Executor)


def test_in_process_executor_computes() -> None:
    with InProcessExecutor() as executor:
        result = executor.run(DIFF, 2.5)
    assert isinstance(result, Result)
    assert result.plain == "2*x"
    assert result.timings["cas"] >= 0.0


def test_inline_mode_skips_the_worker_thread() -> None:
    executor = InProcessExecutor(enforce_timeout=False)
    assert executor.enforce_timeout is False
    assert executor.run(DIFF, 0.000_001).plain == "2*x"  # no budget is enforced inline
    executor.stop()


def test_execute_timeout_returns_cas_timeout_within_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The caller is released on time even though the thread cannot be stopped."""
    budget = 0.2

    def slow(cmd: CasCommand) -> Result:
        time.sleep(5.0)
        raise AssertionError("unreachable")

    monkeypatch.setattr("aicls.mathengine.executor.compute", slow)
    executor = InProcessExecutor()
    started = time.perf_counter()
    with pytest.raises(CasTimeout) as excinfo:
        executor.run(DIFF, budget)
    elapsed = time.perf_counter() - started
    assert elapsed < budget * 5
    assert excinfo.value.code is ErrorCode.CAS_TIMEOUT
    assert excinfo.value.http_status == 504
    assert excinfo.value.details["executor"] == "in_process"
    assert executor.abandoned == 1
    executor.stop()


def test_the_pool_is_rebuilt_after_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A runaway thread must not make the *next* command queue behind it."""
    calls: list[float] = []

    def slow_then_fast(cmd: CasCommand) -> Result:
        calls.append(time.perf_counter())
        if len(calls) == 1:
            time.sleep(3.0)
        return Result(kind="expr", latex="2 x", plain="2*x", srepr="")

    monkeypatch.setattr("aicls.mathengine.executor.compute", slow_then_fast)
    executor = InProcessExecutor()
    with pytest.raises(CasTimeout):
        executor.run(DIFF, 0.15)
    started = time.perf_counter()
    assert executor.run(DIFF, 2.0).plain == "2*x"
    assert time.perf_counter() - started < 1.0
    executor.stop()


def test_start_and_stop_are_idempotent() -> None:
    executor = InProcessExecutor()
    executor.start()
    executor.start()
    executor.run(DIFF, 2.5)
    executor.stop()
    executor.stop()
    assert executor.run(DIFF, 2.5).plain == "2*x"  # a stopped executor rebuilds on demand
    executor.stop()
