"""Failure paths of the sandbox pool.

A sandbox is only worth having if it behaves when things go wrong, so the unhappy paths get the
same attention as the happy one: a worker that will not warm up, a target that does not import,
a pipe that breaks, a respawn that fails, a pool with no free worker. Each of these is a real
production failure mode of ADR-0016 and each is asserted to end in a *named* ``AiclsError``.

Where the failure is cheap to stage in this process (the worker loop, ``_Worker.close``) it is
staged here rather than in a spawned child, which keeps the file fast and keeps the code paths
inside the coverage report.
"""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
import importlib
import multiprocessing
import os
import threading
from typing import Any

import pytest

from aicls.core.errors import AiclsError, ErrorCode, InternalError
from aicls.mathengine import CasTimeout, Diff, PoolExecutor
from aicls.mathengine.sandbox import (
    DEFAULT_TARGET,
    WorkerProbe,
    _warm_up,
    _Worker,
    _worker_entry,
    apply_memory_limit,
    worker_loop,
)

DIFF = Diff(op="diff", expr="x^{2}")
PROBES = "tests.unit.mathengine._worker_probes"

#: A cap so far above any real machine that installing it cannot affect this test process,
#: while still exercising the whole Job Object / RLIMIT_AS code path.
HARMLESS_LIMIT_BYTES = 1 << 40  # 1 TiB


@pytest.fixture
def pipe() -> Iterator[tuple[Any, Any]]:
    """A duplex spawn-context pipe, closed afterwards."""
    parent, child = multiprocessing.get_context("spawn").Pipe(duplex=True)
    try:
        yield parent, child
    finally:
        for end in (parent, child):
            with contextlib.suppress(OSError):
                end.close()


# ---------------------------------------------------------------------------------------------
# worker side
# ---------------------------------------------------------------------------------------------
def test_memory_limit_path_runs_end_to_end() -> None:
    """Best effort: the call must return a bool and never raise, whatever the OS decides.

    On POSIX this really does lower the soft ``RLIMIT_AS`` of the *test* process, so the
    previous value is put back: 1 TiB is harmless in absolute terms but leaking process state
    out of a test is not.
    """
    saved: tuple[int, int] | None = None
    if os.name != "nt":
        resource = importlib.import_module("resource")
        saved = resource.getrlimit(resource.RLIMIT_AS)
    try:
        applied = apply_memory_limit(HARMLESS_LIMIT_BYTES)
        assert isinstance(applied, bool)
    finally:
        if saved is not None:
            importlib.import_module("resource").setrlimit(
                importlib.import_module("resource").RLIMIT_AS, saved
            )


def test_memory_limit_degrades_when_resource_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The POSIX branch with no ``resource`` module: best effort means False, never an error.

    This is the Windows host's real situation and a plausible one for a stripped POSIX build, so
    the module is removed by hand rather than by asking which platform the test happens to run
    on -- the assertion is then identical on both.
    """
    import importlib
    import os as os_module

    def no_resource(name: str) -> Any:
        raise ModuleNotFoundError(f"No module named {name!r}")

    monkeypatch.setattr(os_module, "name", "posix")
    monkeypatch.setattr(importlib, "import_module", no_resource)

    assert apply_memory_limit(HARMLESS_LIMIT_BYTES) is False


@pytest.mark.skipif(os.name == "nt", reason="RLIMIT_AS exists only on POSIX")
def test_rlimit_is_really_installed_on_posix() -> None:
    """On Linux the cap must be a fact, not an intention -- and it must be reversible.

    The soft limit is read back to prove ``apply_memory_limit`` did what it says, and the hard
    limit is asserted untouched: lowering it would be irreversible for an unprivileged process
    and would leak out of this test into the rest of the session.
    """
    resource = importlib.import_module("resource")
    soft_before, hard_before = resource.getrlimit(resource.RLIMIT_AS)
    try:
        assert apply_memory_limit(HARMLESS_LIMIT_BYTES) is True
        soft_after, hard_after = resource.getrlimit(resource.RLIMIT_AS)
        assert soft_after == HARMLESS_LIMIT_BYTES
        assert hard_after == hard_before
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft_before, hard_before))


@pytest.mark.skipif(os.name == "nt", reason="RLIMIT_AS exists only on POSIX")
def test_a_stricter_outer_limit_is_left_alone() -> None:
    """A container started with ``--ulimit as=...`` keeps the last word; we do not raise it."""
    resource = importlib.import_module("resource")
    soft_before, hard_before = resource.getrlimit(resource.RLIMIT_AS)
    outer = 512 << 30  # 512 GiB: stricter than the 1 TiB we then ask for, harmless in absolute
    try:
        resource.setrlimit(resource.RLIMIT_AS, (outer, hard_before))

        assert apply_memory_limit(HARMLESS_LIMIT_BYTES) is True
        assert resource.getrlimit(resource.RLIMIT_AS)[0] == outer
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft_before, hard_before))


@pytest.mark.parametrize(
    "failing_call", ["CreateJobObjectW", "SetInformationJobObject", "AssignProcessToJobObject"]
)
def test_a_refused_job_object_degrades_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch, failing_call: str
) -> None:
    """ADR-0016 calls the memory cap best effort; a locked-down host must not break the worker."""
    import ctypes

    if not hasattr(ctypes, "WinDLL"):  # pragma: no cover - POSIX developer machines
        pytest.skip("Job Objects exist only on Windows")

    class _Kernel32:
        def __init__(self) -> None:
            self.closed = 0

        def __getattr__(self, name: str) -> Any:
            def call(*args: object, **kwargs: object) -> int:
                if name == failing_call:
                    return 0
                if name == "CloseHandle":
                    self.closed += 1
                return 1

            return call

    stub = _Kernel32()
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kwargs: stub)
    assert apply_memory_limit(HARMLESS_LIMIT_BYTES) is False
    assert stub.closed == (0 if failing_call == "CreateJobObjectW" else 1)


def test_a_broken_ctypes_layer_degrades_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ctypes

    if not hasattr(ctypes, "WinDLL"):  # pragma: no cover - POSIX developer machines
        pytest.skip("Job Objects exist only on Windows")

    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("kernel32 is unavailable")

    monkeypatch.setattr(ctypes, "WinDLL", explode)
    assert apply_memory_limit(HARMLESS_LIMIT_BYTES) is False


def test_warm_up_reports_failure_instead_of_crashing_the_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A grammar that will not build must still leave a worker that can answer (and say so)."""

    def broken() -> None:
        raise RuntimeError("grammar is broken")

    monkeypatch.setattr("aicls.latex2sympy.parser.lark_parser", broken)
    sympy_imported, grammar_ready, warm_ms = _warm_up()
    assert sympy_imported is True
    assert grammar_ready is False
    assert warm_ms >= 0.0


def test_worker_loop_exits_when_the_parent_disappears(pipe: tuple[Any, Any]) -> None:
    parent, child = pipe
    thread = threading.Thread(
        target=worker_loop, args=(child,), kwargs={"memory_headroom_bytes": 0}, daemon=True
    )
    thread.start()
    assert parent.poll(30.0)
    parent.recv()
    parent.close()
    thread.join(timeout=30.0)
    assert not thread.is_alive(), "the worker did not notice the parent going away"


def test_worker_entry_closes_the_pipe_it_was_given(pipe: tuple[Any, Any]) -> None:
    parent, child = pipe
    thread = threading.Thread(target=_worker_entry, args=(child, 0, DEFAULT_TARGET), daemon=True)
    thread.start()
    assert parent.poll(30.0)
    assert parent.recv()[0] == "ready"
    parent.send(("stop", None))
    thread.join(timeout=30.0)
    assert child.closed is True


class _StubProcess:
    """A process stand-in whose ``kill`` explodes, to exercise ``_Worker.close``'s guard."""

    def __init__(self, *, alive: bool = True) -> None:
        self._alive = alive
        self.joined = 0

    def kill(self) -> None:
        raise OSError("handle already gone")

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout: float | None = None) -> None:
        self.joined += 1
        self._alive = False


def test_worker_close_survives_a_broken_process_handle(pipe: tuple[Any, Any]) -> None:
    parent, _child = pipe
    probe = WorkerProbe(
        pid=1,
        sympy_imported=True,
        grammar_ready=True,
        memory_limit_applied=False,
        warm_ms=1.0,
    )
    process = _StubProcess()
    worker = _Worker(process=process, conn=parent, probe=probe)  # type: ignore[arg-type]
    worker.close(kill=True)
    worker.close(kill=True)  # idempotent
    assert worker.closed is True
    assert process.joined == 1


def test_worker_close_sends_stop_to_a_live_worker(pipe: tuple[Any, Any]) -> None:
    parent, child = pipe
    probe = WorkerProbe(
        pid=1,
        sympy_imported=True,
        grammar_ready=True,
        memory_limit_applied=False,
        warm_ms=1.0,
    )
    worker = _Worker(process=_StubProcess(), conn=parent, probe=probe)  # type: ignore[arg-type]
    worker.close(kill=False)
    assert child.recv() == ("stop", None)


# ---------------------------------------------------------------------------------------------
# pool side
# ---------------------------------------------------------------------------------------------
def test_a_pool_whose_target_does_not_import_fails_to_start() -> None:
    executor = PoolExecutor(workers=1, target="no.such.module:nope")
    with pytest.raises(InternalError) as excinfo:
        executor.start()
    assert excinfo.value.code is ErrorCode.INTERNAL
    executor.stop()


def test_a_worker_that_cannot_warm_up_in_time_fails_to_start() -> None:
    executor = PoolExecutor(workers=1, start_timeout_s=0.001)
    with pytest.raises(InternalError) as excinfo:
        executor.start()
    assert excinfo.value.details["reason"] == "worker_warmup_timeout"
    executor.stop()


def test_a_pool_with_no_free_worker_reports_a_timeout() -> None:
    with PoolExecutor(workers=1, target=f"{PROBES}:echo_pid", acquire_timeout_s=0.05) as executor:
        busy = executor._acquire()  # hold the only worker
        try:
            with pytest.raises(CasTimeout) as excinfo:
                executor.run(DIFF, 5.0)
            assert excinfo.value.details["reason"] == "no_free_worker"
        finally:
            executor._idle.put(busy)  # give it back so stop() can join it


def test_a_failing_respawn_is_logged_and_does_not_wedge_the_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with PoolExecutor(workers=1, target=f"{PROBES}:echo_pid") as executor:
        executor.run(DIFF, 30.0)
        worker = executor._acquire()  # retire it by hand

        def refuse() -> _Worker:
            raise InternalError(details={"reason": "no more processes"})

        monkeypatch.setattr(executor, "_spawn", refuse)
        executor._replace_async(worker, kill=True)  # drive the respawn path
        for thread in executor._threads:  # private, on purpose
            thread.join(timeout=30.0)
        assert executor.prewarm_diagnostics == []
        with pytest.raises(CasTimeout):
            executor._acquire_timeout_s = 0.05  # private, on purpose
            executor.run(DIFF, 5.0)


def test_probe_reports_a_timeout_instead_of_hanging() -> None:
    """A worker busy in a runaway computation cannot answer a probe; the caller is released."""
    with PoolExecutor(workers=1, target=f"{PROBES}:never_answers") as executor:
        worker = executor._acquire()  # make the worker genuinely busy
        worker.conn.send(("task", DIFF.model_dump(mode="json")))
        executor._idle.put(worker)  # private, on purpose
        with pytest.raises(CasTimeout) as excinfo:
            executor.probe(timeout_s=0.3)
        assert excinfo.value.details["reason"] == "probe_timeout"
        assert executor.restarts == 1


def test_probe_on_a_broken_pipe_is_an_internal_error() -> None:
    with PoolExecutor(workers=1, target=f"{PROBES}:echo_pid") as executor:
        worker = executor._acquire()  # break its pipe on purpose
        worker.conn.close()
        executor._idle.put(worker)  # private, on purpose
        with pytest.raises(AiclsError) as excinfo:
            executor.probe(timeout_s=1.0)
        assert excinfo.value.code is ErrorCode.INTERNAL
