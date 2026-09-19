"""Latency budget of the CAS stage (ADR-0016, ADR-0014, proposal metric "< 3 s end to end").

Three numbers the İP-4 latency report is built from, measured against the *real* sandbox:

1. a warm worker answers a median classroom problem well inside the CAS share of the budget;
2. a runaway computation is cut off at ``compute_timeout_s`` and the caller is released
   promptly -- the whole reason ADR-0016 chose processes over threads;
3. the respawn that follows a kill is paid *after* the answer, so the next request is served by
   a warm worker again.

Thresholds are deliberately loose (several times the target) because this runs on a developer
laptop next to a browser; the point is to catch an order-of-magnitude regression -- a lost warm
pool, a timeout that does not fire -- not to certify the contractual figure. The reportable
numbers come from ``aicls eval latency`` on the workstation.
"""

from __future__ import annotations

from collections.abc import Iterator
import statistics
import time

import pytest

from aicls.mathengine import CasTimeout, Diff, Integrate, Limit, MathEngine, PoolExecutor, Solve
from aicls.mathengine.models import CasCommand

pytestmark = [pytest.mark.perf, pytest.mark.slow]

PROBES = "tests.unit.mathengine._worker_probes"

#: ADR-0016: "ısınmış işçi ile CAS aşaması ortanca problemlerde < 150 ms". Ten times that is
#: still two orders of magnitude away from a cold worker (1-2 s of SymPy import).
WARM_MEDIAN_BUDGET_MS = 1500.0

#: The contractual per-command budget (``AICLS_COMPUTE_TIMEOUT_S``).
TIMEOUT_S = 2.5

#: How much longer than the budget the caller may be held before we call it a regression.
TIMEOUT_SLACK = 3.0

#: A median mix of classroom problems (proposal A.2.2).
MEDIAN_PROBLEMS: tuple[CasCommand, ...] = (
    Diff(op="diff", expr="x^{2}"),
    Diff(op="diff", expr=r"x\sin(x)"),
    Integrate(op="integrate", expr="x^{2}", lower="0", upper="1"),
    Integrate(op="integrate", expr=r"\cos(x)"),
    Limit(op="limit", expr=r"\frac{\sin x}{x}", to="0"),
    Solve(op="solve", equation="x^{2}-4=0"),
)


@pytest.fixture(scope="module")
def warm_pool() -> Iterator[PoolExecutor]:
    """A started, warm single-worker pool (the spawn cost is paid once for the module)."""
    pool = PoolExecutor(workers=1, timeout_s=TIMEOUT_S)
    pool.start()
    try:
        yield pool
    finally:
        pool.stop()


def test_warm_worker_serves_median_problems_inside_the_budget(warm_pool: PoolExecutor) -> None:
    engine = MathEngine(executor=warm_pool, timeout_s=TIMEOUT_S)
    durations: list[float] = []
    for command in MEDIAN_PROBLEMS:
        started = time.perf_counter()
        result = engine.execute(command)
        durations.append((time.perf_counter() - started) * 1000.0)
        assert result.cache_hit is False
    median_ms = statistics.median(durations)
    assert median_ms < WARM_MEDIAN_BUDGET_MS, f"median CAS stage {median_ms:.0f} ms"
    assert max(durations) < TIMEOUT_S * 1000.0 * TIMEOUT_SLACK


def test_cached_commands_cost_almost_nothing(warm_pool: PoolExecutor) -> None:
    engine = MathEngine(executor=warm_pool, timeout_s=TIMEOUT_S)
    for command in MEDIAN_PROBLEMS:
        engine.execute(command)
    started = time.perf_counter()
    for command in MEDIAN_PROBLEMS:
        assert engine.execute(command).cache_hit is True
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    assert elapsed_ms < 5.0 * len(MEDIAN_PROBLEMS)


def test_runaway_computation_is_cut_off_at_the_budget() -> None:
    """A worker that never answers must not hold the caller past ``compute_timeout_s``."""
    with PoolExecutor(workers=1, target=f"{PROBES}:never_answers", timeout_s=TIMEOUT_S) as pool:
        engine = MathEngine(executor=pool, timeout_s=TIMEOUT_S)
        started = time.perf_counter()
        with pytest.raises(CasTimeout):
            engine.execute(Diff(op="diff", expr="x^{2}"))
        elapsed = time.perf_counter() - started
        assert elapsed >= TIMEOUT_S
        assert elapsed < TIMEOUT_S * TIMEOUT_SLACK, f"caller held for {elapsed:.2f} s"
        assert pool.restarts == 1


def test_respawn_cost_is_paid_after_the_answer_not_before_it() -> None:
    """The kill is synchronous, the respawn is not: the timeout must not wait for a new worker."""
    from tests.unit.mathengine._worker_probes import SLOW_MARKER

    budget = 0.4
    with PoolExecutor(workers=1, target=f"{PROBES}:slow_on_marker", timeout_s=budget) as pool:
        pool.run(Diff(op="diff", expr="x^{2}"), 30.0)  # warm, and prove the worker answers
        started = time.perf_counter()
        with pytest.raises(CasTimeout):
            pool.run(Diff(op="diff", expr=SLOW_MARKER), budget)
        elapsed = time.perf_counter() - started
        # A synchronous respawn would add the whole spawn + SymPy import here (seconds).
        assert elapsed < budget + 1.0, f"timeout path took {elapsed:.2f} s (respawn is blocking?)"
        assert pool.restarts == 1
