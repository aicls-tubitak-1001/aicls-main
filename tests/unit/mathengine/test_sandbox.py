"""The warm spawn worker pool (ADR-0016): prewarm, kill-on-timeout, respawn, recycle.

These tests spawn real processes. On Windows every worker pays the ``spawn`` + SymPy import
cost, so the pool is kept at one worker and the module reuses a session-scoped pool wherever the
test does not need to break one.

The timeout path is driven through ``PoolExecutor(target=...)`` fault injection
(``tests/unit/mathengine/_worker_probes.py``) rather than a "slow" SymPy expression: which
expressions are slow changes with every SymPy release, and a test that depends on that is a
test that will fail for the wrong reason in 2027.
"""

from __future__ import annotations

from collections.abc import Iterator
import multiprocessing
import threading
import time

import pytest

from aicls.core.errors import AiclsError, ErrorCode, InternalError
from aicls.mathengine import CasTimeout, Diff, PoolExecutor, SandboxPoolExecutor, UnsupportedProblem
from aicls.mathengine.errors import error_to_payload
from aicls.mathengine.executor import Executor
from aicls.mathengine.sandbox import (
    DEFAULT_MAX_TASKS_PER_WORKER,
    DEFAULT_MEMORY_HEADROOM_BYTES,
    DEFAULT_MEMORY_LIMIT_BYTES,
    DEFAULT_TARGET,
    WorkerProbe,
    apply_memory_limit,
    install_memory_cap,
    pool_from_settings,
    process_memory_bytes,
    resolve_target,
    run_command_payload,
    worker_loop,
    worker_probe_fields,
)
from tests.unit.mathengine._worker_probes import SLOW_MARKER

DIFF = Diff(op="diff", expr="x^{2}")
PROBES = "tests.unit.mathengine._worker_probes"

#: Budget for the fault-injection timeout test; the worker sleeps for two minutes.
TIMEOUT_BUDGET_S = 0.4

#: How long a respawn may take before the test gives up (spawn + SymPy import on Windows).
RESPAWN_BUDGET_S = 90.0


@pytest.fixture(scope="module")
def pool() -> Iterator[PoolExecutor]:
    """One warm single-worker pool shared by the tests that do not break it."""
    executor = PoolExecutor(workers=1)
    executor.start()
    try:
        yield executor
    finally:
        executor.stop()


# ---------------------------------------------------------------------------------------------
# worker side, exercised in this process (no spawn)
# ---------------------------------------------------------------------------------------------
def test_run_command_payload_round_trips_a_command() -> None:
    reply = run_command_payload(DIFF.model_dump(mode="json"))
    assert reply["plain"] == "2*x"
    assert reply["kind"] == "expr"


def test_run_command_payload_rejects_an_oversized_payload() -> None:
    from aicls.core.errors import InputValidationError

    payload = {"op": "evaluate", "expr": "x", "subs": {f"v{i}": "1+" * 500 + "1" for i in range(3)}}
    with pytest.raises(InputValidationError):
        run_command_payload(payload)


def _drive_worker(target: str, messages: list[tuple[str, object]]) -> list[tuple[str, object]]:
    """Run :func:`worker_loop` on a thread over a real pipe and collect its replies."""
    parent, child = multiprocessing.get_context("spawn").Pipe(duplex=True)
    thread = threading.Thread(
        target=worker_loop,
        args=(child,),
        kwargs={"memory_headroom_bytes": 0, "target": target},
        daemon=True,
    )
    thread.start()
    replies: list[tuple[str, object]] = []
    try:
        assert parent.poll(30.0), "worker never announced itself"
        replies.append(parent.recv())
        for message in messages:
            parent.send(message)
            if message[0] == "stop":
                continue
            assert parent.poll(30.0), f"no reply to {message[0]}"
            replies.append(parent.recv())
    finally:
        thread.join(timeout=10.0)
        parent.close()
        child.close()
    return replies


def test_worker_loop_announces_a_warm_probe_then_serves_tasks() -> None:
    replies = _drive_worker(
        DEFAULT_TARGET,
        [("probe", None), ("task", DIFF.model_dump(mode="json")), ("stop", None)],
    )
    ready_kind, ready_probe = replies[0]
    assert ready_kind == "ready"
    assert ready_probe["sympy_imported"] is True  # type: ignore[index]
    assert ready_probe["grammar_ready"] is True  # type: ignore[index]
    assert ready_probe["tasks_done"] == 0  # type: ignore[index]
    assert replies[1][0] == "probe"
    kind, body = replies[2]
    assert kind == "ok"
    assert body["plain"] == "2*x"  # type: ignore[index]


def test_worker_loop_reports_our_errors_as_payloads() -> None:
    payload = {"op": "solve", "equation": r"\cos(x)=x", "var": "x", "domain": "reals"}
    replies = _drive_worker(DEFAULT_TARGET, [("task", payload), ("stop", None)])
    kind, body = replies[1]
    assert kind == "error"
    assert body["code"] == ErrorCode.UNSUPPORTED_PROBLEM.value  # type: ignore[index]


def test_worker_loop_reports_foreign_exceptions_as_internal() -> None:
    replies = _drive_worker(f"{PROBES}:always_raises", [("task", {}), ("stop", None)])
    kind, body = replies[1]
    assert kind == "error"
    assert body["code"] == ErrorCode.INTERNAL.value  # type: ignore[index]


def test_worker_loop_refuses_an_unimportable_target() -> None:
    replies = _drive_worker("no.such.module:nope", [])
    kind, body = replies[0]
    assert kind == "error"
    assert body["code"] == ErrorCode.INTERNAL.value  # type: ignore[index]


def test_resolve_target_validates_its_argument() -> None:
    assert resolve_target(DEFAULT_TARGET) is run_command_payload
    for bad in ("no-colon", ":missing", "module:", "aicls.mathengine.sandbox:DEFAULT_TARGET"):
        with pytest.raises(InternalError):
            resolve_target(bad)


def test_memory_limit_is_best_effort_and_says_so() -> None:
    """A 1 GB cap on the *test* process would be hostile, so only the disabled path is asserted."""
    assert apply_memory_limit(0) is False
    assert apply_memory_limit(-1) is False
    assert isinstance(DEFAULT_MEMORY_LIMIT_BYTES, int)
    assert DEFAULT_MEMORY_LIMIT_BYTES == 1 << 30
    assert DEFAULT_MEMORY_HEADROOM_BYTES == DEFAULT_MEMORY_LIMIT_BYTES  # the historic alias


def test_process_memory_bytes_reports_this_process() -> None:
    """The baseline the relative cap is built on must be a real, positive measurement.

    It is what ``RLIMIT_AS`` (POSIX address space) or the Job Object (Windows commit charge)
    counts, so the only portable assertion is "plausible": at least a few megabytes, and far
    below a terabyte.
    """
    usage = process_memory_bytes()

    assert usage is not None, "neither /proc/self/statm nor psapi answered on this host"
    assert 4 << 20 < usage < 1 << 40


def test_install_memory_cap_is_relative_to_the_measured_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR-0016 (2026-09-08): the cap is *baseline + head-room*, never an absolute constant.

    An absolute 1 GB cap is below the floor on Linux, where OpenBLAS reserves 128 MB of address
    space per CPU at import time. Both collaborators are faked so the arithmetic is asserted
    without capping the test process.
    """
    seen: list[int] = []

    def record(cap: int) -> bool:
        seen.append(cap)
        return True

    monkeypatch.setattr("aicls.mathengine.sandbox.process_memory_bytes", lambda: 1_400 << 20)
    monkeypatch.setattr("aicls.mathengine.sandbox.apply_memory_limit", record)

    applied, cap = install_memory_cap(1 << 30)

    assert applied is True
    assert cap == (1_400 << 20) + (1 << 30)
    assert seen == [cap]


def test_install_memory_cap_is_disabled_by_a_zero_headroom() -> None:
    assert install_memory_cap(0) == (False, 0)
    assert install_memory_cap(-1) == (False, 0)


def test_install_memory_cap_declines_when_the_baseline_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No measurement, no cap: a guessed baseline is exactly what broke Linux."""
    monkeypatch.setattr("aicls.mathengine.sandbox.process_memory_bytes", lambda: None)

    assert install_memory_cap(1 << 30) == (False, 0)


def test_install_memory_cap_reports_a_refused_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aicls.mathengine.sandbox.process_memory_bytes", lambda: 1 << 20)
    monkeypatch.setattr("aicls.mathengine.sandbox.apply_memory_limit", lambda cap: False)

    assert install_memory_cap(1 << 30) == (False, 0)


def test_worker_probe_fields_are_stable() -> None:
    assert set(worker_probe_fields()) == {
        "pid",
        "sympy_imported",
        "grammar_ready",
        "memory_limit_applied",
        "memory_limit_bytes",
        "warm_ms",
        "tasks_done",
    }


def test_error_payloads_are_picklable() -> None:
    """Whatever crosses the pipe must survive pickling; our exceptions do not, payloads do."""
    import pickle

    payload = error_to_payload(UnsupportedProblem("condition_set"))
    assert pickle.loads(pickle.dumps(payload)) == payload


# ---------------------------------------------------------------------------------------------
# pool, with real spawned workers
# ---------------------------------------------------------------------------------------------
def test_pool_executor_prewarms_sympy_import(pool: PoolExecutor) -> None:
    """ADR-0016: SymPy and the grammar are imported in the initializer, before any task."""
    diagnostics = pool.prewarm_diagnostics
    assert len(diagnostics) == pool.workers
    for probe in diagnostics:
        assert isinstance(probe, WorkerProbe)
        assert probe.sympy_imported is True
        assert probe.grammar_ready is True
        assert probe.tasks_done == 0  # warm *before* the first task
        assert probe.warm_ms > 0.0
        assert probe.pid > 0

    live = pool.probe()
    assert live.pid == diagnostics[0].pid
    assert live.sympy_imported is True


def test_a_warm_worker_carries_a_memory_cap_above_its_own_baseline(pool: PoolExecutor) -> None:
    """The Linux regression of 2026-09-08, stated as a property rather than as a platform.

    The pool runs with the shipped default head-room. Before the fix an absolute 1 GB cap sat
    *below* the worker's floor on Linux: the warm-up died with ``MemoryError``, ``grammar_ready``
    came back False and every request answered INTERNAL. A cap that is genuinely installed and
    strictly larger than the warm baseline is what makes the sandbox a protection rather than a
    denial of service against itself.
    """
    probe = pool.prewarm_diagnostics[0]

    if not probe.memory_limit_applied:  # pragma: no cover - a host that refuses both mechanisms
        pytest.skip("this host installs neither RLIMIT_AS nor a Job Object limit")
    assert probe.grammar_ready is True
    assert probe.memory_limit_bytes > DEFAULT_MEMORY_HEADROOM_BYTES
    assert pool.run(DIFF, 30.0).plain == "2*x"


def test_the_memory_cap_stops_a_runaway_allocation() -> None:
    """ADR-0016 treats the cap as a security control, so prove it stops something.

    A fault-injection target allocates far past the worker's head-room. The worker must answer
    with an INTERNAL error (its ``MemoryError`` turned into a payload) instead of taking the
    machine's memory with it; the pool stays usable afterwards.

    The head-room is deliberately small, and the test refuses to run at all unless the worker
    reports that a cap really is installed -- without that check a host that silently refuses
    both mechanisms would have this test allocate a gigabyte for real.
    """
    with PoolExecutor(
        workers=1, memory_headroom_bytes=128 << 20, target=f"{PROBES}:allocates_past_the_cap"
    ) as executor:
        probe = executor.prewarm_diagnostics[0]
        if not probe.memory_limit_applied:  # pragma: no cover - host refuses both mechanisms
            pytest.skip("this host installs neither RLIMIT_AS nor a Job Object limit")

        with pytest.raises(AiclsError) as excinfo:
            executor.run(DIFF, 60.0)

        assert excinfo.value.code is ErrorCode.INTERNAL
        assert excinfo.value.details["error"] == "MemoryError"
        assert executor.restarts == 0, "the worker answered; it was not killed"


def test_pool_executor_computes_and_satisfies_the_protocol(pool: PoolExecutor) -> None:
    assert isinstance(pool, Executor)
    result = pool.run(DIFF, 10.0)
    assert result.plain == "2*x"
    assert result.timings["cas"] >= 0.0


def test_pool_executor_carries_our_errors_back(pool: PoolExecutor) -> None:
    from aicls.mathengine import Solve

    with pytest.raises(UnsupportedProblem) as excinfo:
        pool.run(Solve(op="solve", equation=r"\cos(x)=x"), 10.0)
    assert excinfo.value.message_tr == "Bu problem tipi şu an için desteklenmemektedir."
    assert excinfo.value.details["reason"] == "condition_set"


def test_pool_executor_carries_plot_bytes_back(pool: PoolExecutor) -> None:
    from aicls.mathengine import Plot

    result = pool.run(Plot(op="plot", expr="x^{2}", xmin=-2.0, xmax=2.0), 20.0)
    assert result.plot_png is not None
    assert result.plot_png[:8] == b"\x89PNG\r\n\x1a\n"


def test_pool_executor_uses_its_default_budget(pool: PoolExecutor) -> None:
    assert pool.timeout_s > 0
    assert pool.run(DIFF).plain == "2*x"


def test_execute_timeout_kills_worker_and_pool_recovers() -> None:
    """The contract of ADR-0016: kill the worker, count the restart, come back warm."""
    slow = Diff(op="diff", expr=SLOW_MARKER)
    with PoolExecutor(workers=1, target=f"{PROBES}:slow_on_marker") as executor:
        before = executor.run(DIFF, 30.0).plain
        assert executor.restarts == 0
        original_pid = executor.prewarm_diagnostics[0].pid
        assert int(before) == original_pid

        started = time.perf_counter()
        with pytest.raises(CasTimeout):
            executor.run(slow, TIMEOUT_BUDGET_S)
        assert time.perf_counter() - started < TIMEOUT_BUDGET_S * 10
        assert executor.restarts == 1

        deadline = time.perf_counter() + RESPAWN_BUDGET_S
        while not executor.prewarm_diagnostics and time.perf_counter() < deadline:
            time.sleep(0.05)
        assert executor.prewarm_diagnostics, "the pool never respawned a worker"

        after = executor.run(DIFF, 30.0).plain
        assert after != before, "the killed worker was reused instead of being replaced"
        assert int(after) != original_pid
        assert executor.restarts == 1
        assert executor.prewarm_diagnostics[0].sympy_imported is True  # the pool is warm again


def test_execute_timeout_returns_cas_timeout_within_budget() -> None:
    with PoolExecutor(workers=1, target=f"{PROBES}:never_answers") as executor:
        started = time.perf_counter()
        with pytest.raises(CasTimeout) as excinfo:
            executor.run(DIFF, TIMEOUT_BUDGET_S)
        elapsed = time.perf_counter() - started
        assert elapsed < TIMEOUT_BUDGET_S * 10
        assert elapsed >= TIMEOUT_BUDGET_S
        error = excinfo.value
        assert error.code is ErrorCode.CAS_TIMEOUT
        assert error.http_status == 504
        assert error.details["executor"] == "pool"
        assert error.details["restarts"] == 1
        assert str(TIMEOUT_BUDGET_S) in error.message_tr


def test_worker_is_recycled_after_its_task_quota() -> None:
    with PoolExecutor(workers=1, max_tasks_per_worker=1, target=f"{PROBES}:echo_pid") as executor:
        first = executor.run(DIFF, 30.0).plain
        assert executor.recycles == 1
        second = executor.run(DIFF, 30.0).plain
        assert first != second
        assert executor.restarts == 0  # a recycle is not a crash


def test_pool_rejects_an_oversized_command_before_spawning_anything() -> None:
    from aicls.core.errors import InputValidationError
    from aicls.mathengine import Evaluate

    executor = PoolExecutor(workers=1)
    command = Evaluate(op="evaluate", expr="x", subs={f"v{i}": "1+" * 500 + "1" for i in range(3)})
    with pytest.raises(InputValidationError):
        executor.run(command, 1.0)
    assert executor.started is False  # nothing was spawned
    executor.stop()


def test_start_and_stop_are_idempotent() -> None:
    executor = PoolExecutor(workers=1)
    executor.start()
    executor.start()
    assert executor.started is True
    executor.stop()
    executor.stop()
    assert executor.started is False


def test_a_pool_starts_itself_on_first_use() -> None:
    executor = PoolExecutor(workers=1, target=f"{PROBES}:echo_pid")
    try:
        assert executor.started is False
        executor.run(DIFF, 30.0)
        assert executor.started is True
    finally:
        executor.stop()


def test_a_worker_that_dies_mid_task_is_reported_and_replaced() -> None:
    with PoolExecutor(workers=1, target=f"{PROBES}:echo_pid") as executor:
        executor.run(DIFF, 30.0)
        worker = executor._idle.get(timeout=5.0)  # fault injection
        worker.conn.close()
        executor._idle.put(worker)  # hand the broken worker back
        with pytest.raises(AiclsError) as excinfo:
            executor.run(DIFF, 30.0)
        assert excinfo.value.code is ErrorCode.INTERNAL


def test_pool_from_settings_reads_the_configuration() -> None:
    from aicls.core.settings import settings_override

    with settings_override(sandbox_workers=3, compute_timeout_s=1.75):
        executor = pool_from_settings()
        assert executor.workers == 3
        assert executor.timeout_s == 1.75
        override = pool_from_settings(workers=1)
        assert override.workers == 1


def test_sandbox_pool_executor_is_the_documented_alias() -> None:
    assert SandboxPoolExecutor is PoolExecutor
    assert DEFAULT_MAX_TASKS_PER_WORKER == 500
