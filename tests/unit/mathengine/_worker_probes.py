"""Fault-injection worker entry points for the sandbox pool tests.

``PoolExecutor`` resolves its per-task callable from a dotted ``module:function`` string, which
is what makes the timeout path testable *without* betting on a SymPy expression that happens to
be slow today and fast after the next release. These functions are imported by name in the
spawned child process (``multiprocessing`` hands the parent's ``sys.path`` to the child), so
they must stay at module level and must not depend on any fixture.

The leading underscore keeps pytest from collecting this file as a test module.
"""

from __future__ import annotations

import time
from typing import Any

#: How long :func:`never_answers` sleeps: far beyond any test's budget, but bounded so a leaked
#: worker cannot outlive the test session.
SLEEP_SECONDS = 120.0

#: A command whose ``expr`` is this makes :func:`slow_on_marker` hang. The worker resolves its
#: target once, at startup, so behaviour has to be selected by the *payload*, not by swapping
#: the target after the pool is warm.
SLOW_MARKER = "x^{9999}"


def never_answers(payload: dict[str, Any]) -> dict[str, Any]:
    """Block until the parent kills this worker (simulates a runaway SymPy computation)."""
    time.sleep(SLEEP_SECONDS)
    return {"unreachable": True}


def slow_on_marker(payload: dict[str, Any]) -> dict[str, Any]:
    """Hang on the :data:`SLOW_MARKER` command, answer with the worker's pid otherwise."""
    if payload.get("expr") == SLOW_MARKER:
        time.sleep(SLEEP_SECONDS)
    return echo_pid(payload)


def echo_pid(payload: dict[str, Any]) -> dict[str, Any]:
    """Answer with the worker's process id, so a test can prove the worker was replaced."""
    import os

    return {
        "kind": "expr",
        "latex": str(os.getpid()),
        "plain": str(os.getpid()),
        "srepr": str(os.getpid()),
        "steps": [],
        "plot_png": None,
        "timings": {},
        "cache_hit": False,
        "elapsed_ms": 0.0,
    }


def always_raises(payload: dict[str, Any]) -> dict[str, Any]:
    """Raise a plain (non-``AiclsError``) exception, exercising the worker's INTERNAL path."""
    raise RuntimeError("worker exploded")


#: Ceiling for :func:`allocates_past_the_cap`. Far above the 128 MB head-room its test installs,
#: far below anything that could hurt the host if the cap were (impossibly) absent -- and the
#: test refuses to run unless the worker reported that a cap really is in force.
ALLOCATION_CEILING_BYTES = 1 << 30

#: Allocation step; each block is touched, so a Windows Job Object (which limits *commit*) sees
#: it too, not only a POSIX ``RLIMIT_AS`` (which limits reservations).
ALLOCATION_STEP_BYTES = 32 << 20


def allocates_past_the_cap(payload: dict[str, Any]) -> dict[str, Any]:
    """Allocate beyond the worker's memory head-room, so the cap has something to stop.

    Stands in for a pathological expression whose intermediate result explodes: SymPy inputs
    that do that change with every release, exactly the dependency the ``target=`` hook exists
    to avoid (see the module docstring).
    """
    blocks: list[bytearray] = []
    while len(blocks) * ALLOCATION_STEP_BYTES < ALLOCATION_CEILING_BYTES:
        block = bytearray(ALLOCATION_STEP_BYTES)
        block[::4096] = b"\x01" * len(block[::4096])  # touch it: commit, not just reserve
        blocks.append(block)
    return echo_pid(payload)
