"""Unit tests for :mod:`aicls.ml.queue` (docs/ml/egitim-altyapisi.md §8).

There is exactly one RTX 5090 on the workstation, so a second training process must be refused
rather than silently OOM. ``nvidia-smi`` is monkeypatched in **every** test (an autouse fixture
makes it "absent" by default): a unit test never shells out to a real GPU, and the CPU box the
suite runs on must not behave differently from the workstation.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from aicls.core.errors import AiclsError
from aicls.ml import queue as q
from aicls.ml.manifest import ProcessResult

COMPUTE_APPS_CSV = "12345, 8192, python.exe\n"


@pytest.fixture(autouse=True)
def no_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: ``nvidia-smi`` is not installed (the CPU box), which must not be fatal."""
    monkeypatch.setattr(q, "run_process", lambda *a, **k: None)


def _fake_nvidia_smi(monkeypatch: pytest.MonkeyPatch, stdout: str) -> None:
    monkeypatch.setattr(
        q, "run_process", lambda *a, **k: ProcessResult(("nvidia-smi",), 0, stdout, "")
    )


def _queue(tmp_path: Path, **kwargs: Any) -> q.TrainingQueue:
    return q.TrainingQueue(tmp_path / "queue.lock", **kwargs)


# ------------------------------------------------------------------------------ the lock file
def test_training_queue_refuses_second_job(tmp_path: Path) -> None:
    """Contractual: while one job holds the queue, a second acquire is refused."""
    first, second = _queue(tmp_path), _queue(tmp_path)
    lock = first.acquire()
    assert lock.pid == os.getpid()
    assert first.held is True

    with pytest.raises(q.TrainingQueueBusyError) as excinfo:
        second.acquire()
    assert excinfo.value.details["reason"] == "training_queue_busy"
    assert excinfo.value.details["holder_pid"] == os.getpid()
    assert isinstance(excinfo.value, AiclsError)

    first.release()
    assert second.acquire().pid == os.getpid()
    second.release()


def test_acquire_is_idempotent_for_the_holder(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    first = queue.acquire()
    assert queue.acquire() is first
    queue.release()


def test_queue_is_usable_as_a_context_manager(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    with queue as lock:
        assert lock.pid == os.getpid()
        assert (tmp_path / "queue.lock").is_file()
    assert not (tmp_path / "queue.lock").exists()
    assert queue.held is False


def test_release_without_acquire_is_a_noop(tmp_path: Path) -> None:
    assert _queue(tmp_path).release() is False


def test_release_keeps_a_lock_that_belongs_to_someone_else(tmp_path: Path) -> None:
    holder = _queue(tmp_path)
    holder.acquire()
    other = _queue(tmp_path)
    assert other.release() is False
    assert (tmp_path / "queue.lock").is_file()
    holder.release()


def test_lock_record_roundtrip_is_utf8_lf(tmp_path: Path) -> None:
    queue = _queue(tmp_path, job="ocr_offline_resnet34")
    written = queue.acquire()
    raw = (tmp_path / "queue.lock").read_bytes()
    assert b"\r\n" not in raw
    read_back = queue.read_lock()
    assert read_back is not None
    assert (read_back.pid, read_back.job, read_back.host) == (
        written.pid,
        "ocr_offline_resnet34",
        written.host,
    )
    assert read_back.started_at.endswith("Z")
    queue.release()


def test_read_lock_is_none_when_no_job_runs(tmp_path: Path) -> None:
    queue = _queue(tmp_path)
    assert queue.read_lock() is None
    assert queue.is_busy() is False


def test_a_corrupt_lock_file_is_treated_as_stale(tmp_path: Path) -> None:
    (tmp_path / "queue.lock").write_text("not json at all", encoding="utf-8")
    queue = _queue(tmp_path)
    assert queue.read_lock() is None
    assert queue.acquire().pid == os.getpid()
    queue.release()


def test_stale_lock_from_a_dead_process_is_taken_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = q.QueueLock(pid=999_999, host=q.hostname(), started_at="2026-09-08T00:00:00Z", job="x")
    (tmp_path / "queue.lock").write_text(stale.to_json() + "\n", encoding="utf-8")
    monkeypatch.setattr(q, "process_is_alive", lambda pid: False)
    queue = _queue(tmp_path)
    assert queue.is_busy() is False
    assert queue.acquire().pid == os.getpid()
    queue.release()


def test_a_lock_from_another_host_is_never_assumed_stale(tmp_path: Path) -> None:
    foreign = q.QueueLock(
        pid=999_999, host="other-workstation", started_at="2026-09-08T00:00:00Z", job="x"
    )
    (tmp_path / "queue.lock").write_text(foreign.to_json() + "\n", encoding="utf-8")
    with pytest.raises(q.TrainingQueueBusyError):
        _queue(tmp_path).acquire()


# ------------------------------------------------------- taking over a lock nobody is holding
#
# A container's hostname is a fresh random id, so under ADR-0020 every lock is written by a
# "different host" and the stale-lock takeover above can never fire. On 2026-09-09 the machine
# rebooted mid-run and the next run refused to start behind a lock held by a container that no
# longer existed. Expiring locks automatically would trade a loud, safe failure for a chance at
# the silent, unrecoverable one, so the way out is explicit instead.
def test_take_over_seizes_a_lock_whose_holder_cannot_be_checked(tmp_path: Path) -> None:
    """Exactly the lock a dead container leaves: another "host", pid 1, nothing running."""
    dead = q.QueueLock(pid=1, host="20684ee4158c", started_at="2026-09-09T02:23:57Z", job="ocr")
    (tmp_path / "queue.lock").write_text(dead.to_json() + "\n", encoding="utf-8")
    queue = _queue(tmp_path)
    with pytest.raises(q.TrainingQueueBusyError):
        queue.acquire()

    lock = queue.take_over(reason="rebooted")

    assert queue.held
    assert lock.pid == os.getpid()
    assert queue.read_lock() is not None
    assert queue.read_lock().host == q.hostname()


def test_take_over_refuses_while_the_gpu_is_in_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one check that keeps the flag honest.

    Whoever wrote the lock is unreachable by definition - that is why the flag exists - so the
    lock file cannot answer "is it still running". A training job that really is running holds
    GPU memory, and that is observable. Seizing the queue while it does is the corruption the
    queue exists to prevent, so it is refused however explicitly it was asked for.
    """
    dead = q.QueueLock(pid=1, host="20684ee4158c", started_at="2026-09-09T02:23:57Z", job="ocr")
    (tmp_path / "queue.lock").write_text(dead.to_json() + "\n", encoding="utf-8")
    _fake_nvidia_smi(monkeypatch, "4242, 8192, python.exe\n")
    queue = _queue(tmp_path)

    with pytest.raises(q.TrainingQueueBusyError) as excinfo:
        queue.take_over(reason="rebooted")

    assert excinfo.value.details["reason"] == "gpu_busy"
    assert not queue.held
    assert queue.read_lock().host == "20684ee4158c", "the displaced lock must survive a refusal"


def test_take_over_records_who_it_displaced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A seizure that leaves no trace is indistinguishable from a queue that never worked."""
    dead = q.QueueLock(pid=1, host="20684ee4158c", started_at="2026-09-09T02:23:57Z", job="ocr")
    (tmp_path / "queue.lock").write_text(dead.to_json() + "\n", encoding="utf-8")

    _queue(tmp_path).take_over(reason="windows update rebooted the box")

    out = capsys.readouterr().out
    assert "ml.queue.taken_over" in out
    assert "20684ee4158c" in out
    assert "windows update rebooted the box" in out


def test_take_over_is_idempotent_for_a_holder(tmp_path: Path) -> None:
    """Calling it twice must not displace the caller's own lock."""
    queue = _queue(tmp_path)
    first = queue.acquire()

    assert queue.take_over(reason="again") == first


def test_the_refusal_says_how_to_get_out_of_it(tmp_path: Path) -> None:
    """The message that sent an operator looking for a job that did not exist."""
    dead = q.QueueLock(pid=1, host="20684ee4158c", started_at="2026-09-09T02:23:57Z", job="ocr")
    (tmp_path / "queue.lock").write_text(dead.to_json() + "\n", encoding="utf-8")

    with pytest.raises(q.TrainingQueueBusyError) as excinfo:
        _queue(tmp_path).acquire()

    assert "--force-queue" in excinfo.value.details["hint"]


def test_acquire_can_wait_before_refusing(tmp_path: Path) -> None:
    holder = _queue(tmp_path)
    holder.acquire()
    with pytest.raises(q.TrainingQueueBusyError):
        _queue(tmp_path).acquire(timeout_s=0.05, poll_s=0.01)
    holder.release()


def test_default_queue_path_is_the_untracked_training_lock() -> None:
    path = q.default_queue_path()
    assert path.parent.name == "training"
    assert path.name == q.QUEUE_LOCK_FILENAME == "queue.lock"


# ------------------------------------------------------------- the lock must not dirty the tree
def test_the_lock_file_is_git_ignored(repo_root: Path) -> None:
    """A run that dirties the working tree is recorded ``citable=false`` (ADR-0007).

    The lock is written *while* the run captures its manifest, so a tracked lock file would make
    every queued run non-citable. It must therefore be ignored by git.
    """
    ignored = (repo_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert f"/training/{q.QUEUE_LOCK_FILENAME}" in ignored


def test_the_committed_placeholder_is_never_the_lock_file(repo_root: Path) -> None:
    """``training/queue.txt`` documents the mechanism; nothing writes to it (wave-5 defect 3)."""
    placeholder = repo_root / "training" / "queue.txt"
    assert placeholder.is_file()
    assert q.default_queue_path() != placeholder
    assert placeholder.read_text(encoding="utf-8").lstrip().startswith("#")


def test_an_unparsable_lock_file_is_never_deleted(tmp_path: Path) -> None:
    """Treating an unreadable lock as stale is right; *removing* the file is not.

    Deleting it is what made the committed ``training/queue.txt`` disappear on the first queued
    run. The queue overwrites instead, so a file it did not write survives being inspected.
    """
    lock = tmp_path / "queue.lock"
    lock.write_text("not json at all\n", encoding="utf-8")
    queue = _queue(tmp_path)

    assert queue.is_busy() is False
    assert lock.read_text(encoding="utf-8") == "not json at all\n"


def test_a_stale_lock_is_overwritten_rather_than_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = q.QueueLock(pid=999_999, host=q.hostname(), started_at="2026-09-08T00:00:00Z", job="x")
    lock = tmp_path / "queue.lock"
    lock.write_text(stale.to_json() + "\n", encoding="utf-8")
    monkeypatch.setattr(q, "process_is_alive", lambda pid: False)
    queue = _queue(tmp_path)

    assert queue.is_busy() is False
    assert lock.is_file(), "inspecting a stale lock must not remove it"
    assert queue.acquire().pid == os.getpid()
    read_back = queue.read_lock()
    assert read_back is not None and read_back.pid == os.getpid()
    queue.release()


def test_acquire_writes_over_a_file_it_did_not_create(tmp_path: Path) -> None:
    lock = tmp_path / "queue.lock"
    lock.write_text("# leftover placeholder\n", encoding="utf-8")
    queue = _queue(tmp_path)

    written = queue.acquire()
    assert lock.read_text(encoding="utf-8") == written.to_json() + "\n"
    queue.release()


# ------------------------------------------------------------------------- the operator's error
def test_busy_error_uses_the_training_busy_code(tmp_path: Path) -> None:
    """An operator whose GPU is busy is not an API client that sent too many requests."""
    from aicls.core import messages_tr
    from aicls.core.errors import ErrorCode

    holder = _queue(tmp_path)
    holder.acquire()
    with pytest.raises(q.TrainingQueueBusyError) as excinfo:
        _queue(tmp_path).acquire()
    holder.release()

    error = excinfo.value
    assert error.code is ErrorCode.TRAINING_BUSY
    assert error.message_tr == messages_tr.MSG_TRAINING_QUEUE_BUSY
    assert error.message_tr != messages_tr.MSG_RATE_LIMITED
    assert "GPU" in error.message_tr
    assert error.exit_code == 1
    assert error.http_status == 503


# ------------------------------------------------------------------------------- the GPU probe
def test_missing_nvidia_smi_is_not_fatal(tmp_path: Path) -> None:
    assert q.probe_gpu_processes() is None
    assert _queue(tmp_path).acquire().pid == os.getpid()


def test_probe_gpu_processes_parses_the_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_nvidia_smi(monkeypatch, COMPUTE_APPS_CSV + "77, [N/A], other.exe\n")
    processes = q.probe_gpu_processes()
    assert processes == [
        q.GpuProcess(pid=12345, used_memory_mb=8192.0, name="python.exe"),
        q.GpuProcess(pid=77, used_memory_mb=None, name="other.exe"),
    ]


def test_training_queue_refuses_when_another_process_holds_the_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_nvidia_smi(monkeypatch, COMPUTE_APPS_CSV)
    with pytest.raises(q.TrainingQueueBusyError) as excinfo:
        _queue(tmp_path).acquire()
    assert excinfo.value.details["reason"] == "gpu_busy"
    assert excinfo.value.details["gpu_pids"] == [12345]
    assert not (tmp_path / "queue.lock").exists()


def test_training_queue_ignores_its_own_gpu_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_nvidia_smi(monkeypatch, f"{os.getpid()}, 8192, python.exe\n")
    queue = _queue(tmp_path)
    assert queue.acquire().pid == os.getpid()
    queue.release()


def test_a_small_gpu_allocation_does_not_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_nvidia_smi(monkeypatch, "4242, 64, display.exe\n")
    queue = _queue(tmp_path, gpu_busy_mb=512.0)
    assert queue.acquire().pid == os.getpid()
    queue.release()


def test_the_gpu_check_can_be_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_nvidia_smi(monkeypatch, COMPUTE_APPS_CSV)
    queue = _queue(tmp_path, check_gpu=False)
    assert queue.acquire().pid == os.getpid()
    queue.release()


def test_an_injected_probe_is_used_instead_of_nvidia_smi(tmp_path: Path) -> None:
    calls: list[int] = []

    def probe() -> list[q.GpuProcess]:
        calls.append(1)
        return [q.GpuProcess(pid=4321, used_memory_mb=20000.0, name="train.py")]

    with pytest.raises(q.TrainingQueueBusyError):
        _queue(tmp_path, gpu_probe=probe).acquire()
    assert calls


# ------------------------------------------------------------------------------ pid liveness
def test_process_is_alive_reports_this_process() -> None:
    assert q.process_is_alive(os.getpid()) is True


def test_process_is_alive_reports_a_finished_process() -> None:
    """A process that has exited reads as dead — and is never signalled to find that out."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    # The Popen object keeps the process handle open, so the pid cannot have been reused yet.
    assert q.process_is_alive(child.pid) is False


def test_process_is_alive_rejects_invalid_pids() -> None:
    assert q.process_is_alive(0) is False
    assert q.process_is_alive(-5) is False
