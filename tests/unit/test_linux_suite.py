"""``scripts/run_linux_suite.py``: the standing "does it still work on Linux?" command.

The script itself only builds an argv -- the container work belongs to ``scripts/gpu/aicls-gpu.sh``
(ADR-0020). So what is worth pinning here is exactly that: the argv it builds, the fact that it
delegates instead of growing a second runner, and the fact that it opens no approval gate.

Both host branches are asserted **on both platforms**, by passing the branch in rather than
rewriting the interpreter's ``os.name``. That is not pedantry: the first version of this file did
patch ``os.name`` to "nt", which on Linux corrupts every path and process call made until the
patch is undone and crashed an xdist worker outright (measured 2026-09-08). Running the container
is not attempted from a unit test.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path, PureWindowsPath
import types

import pytest

MODULE = "scripts/run_linux_suite.py"

#: Anything that would turn this task into a gated action (ADR-0012). ``poe test-linux`` runs the
#: ordinary suite; it never downloads, never trains and never carries an approval flag.
FORBIDDEN_IN_COMMAND = (
    "--i-have-approval",
    "AICLS_GATE_",
    "download",
    "train",
    "from_pretrained",
)


@pytest.fixture
def suite(load_script: Callable[[str], types.ModuleType]) -> types.ModuleType:
    """The launcher module, loaded by path (``scripts/`` is not a package)."""
    return load_script(MODULE)


def test_it_delegates_to_the_documented_container_runner(suite: types.ModuleType) -> None:
    """One runner, not two: the docker argv and the image pin live in ADR-0020's script."""
    runner = suite.REPO_ROOT / suite.RUNNER_RELATIVE

    assert suite.RUNNER_RELATIVE.as_posix() == "scripts/gpu/aicls-gpu.sh"
    assert runner.is_file(), f"{runner} is missing; poe test-linux cannot work without it"


def test_posix_command_calls_the_runner_directly(suite: types.ModuleType) -> None:
    command = suite.build_command(["tests/unit/mathengine"], windows=False)

    assert command[:2] == ["bash", "./scripts/gpu/aicls-gpu.sh"]
    assert "--no-gpu" in command  # the suite deselects gpu tests; leave the card alone
    assert command[-5:] == ["--raw", "python", "-m", "pytest", "tests/unit/mathengine"]


def test_windows_command_goes_through_wsl(suite: types.ModuleType) -> None:
    command = suite.build_command(
        ["-k", "sandbox"], windows=True, repo_root=PureWindowsPath("D:/aicls/ai-cls")
    )

    assert command[:4] == ["wsl", "-e", "bash", "-lc"]
    assert command[4].startswith("cd /mnt/d/aicls/ai-cls && bash ./scripts/gpu/aicls-gpu.sh")
    assert command[4].endswith("-m pytest -k sandbox")


def test_the_gpu_flag_hands_the_card_over(suite: types.ModuleType) -> None:
    assert "--no-gpu" not in suite.build_command([], gpu=True, windows=False)


def test_no_arguments_means_the_widest_cpu_selection(suite: types.ModuleType) -> None:
    """The ``poe test-all`` marker set (slow included), with a longer per-test budget."""
    command = suite.build_command([], windows=False)

    assert "not gpu and not data and not mysql and not browser" in command
    assert "--timeout=900" in command  # container CPU torch is several times slower


def test_a_windows_path_becomes_a_wsl_path(suite: types.ModuleType) -> None:
    """Works on Linux too: the argument is re-read as a ``PureWindowsPath``."""
    assert (
        suite.windows_path_to_wsl(PureWindowsPath(r"C:\Users\x\ai-cls")) == "/mnt/c/Users/x/ai-cls"
    )
    assert suite.windows_path_to_wsl("D:/a/b") == "/mnt/d/a/b"
    for rejected in ("/already/posix", "relative/path", r"\\server\share\x"):
        with pytest.raises(ValueError, match="no drive letter"):
            suite.windows_path_to_wsl(rejected)


def test_a_non_ascii_repository_path_is_quoted(suite: types.ModuleType) -> None:
    """The real checkout lives under ``Masaüstü``; an unquoted non-ASCII byte is locale roulette."""
    assert suite._shell_quote("/mnt/c/Users/cagri/Masaüstü/ai-cls").startswith("'")
    assert suite._shell_quote("/mnt/c/plain/path") == "/mnt/c/plain/path"
    assert suite._shell_quote("a b") == "'a b'"


def test_print_runs_nothing(
    suite: types.ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refuse(*args: object, **kwargs: object) -> int:
        raise AssertionError("--print must not start a container")

    monkeypatch.setattr(suite.subprocess, "call", refuse)
    monkeypatch.setattr(suite.shutil, "which", lambda name: "/usr/bin/wsl")

    assert suite.main(["--print"]) == 0
    assert "aicls-gpu.sh" in capsys.readouterr().out


def test_the_command_opens_no_approval_gate(suite: types.ModuleType) -> None:
    """ADR-0012: a convenience task must never become a way around a gate."""
    text = " ".join(suite.build_command([], windows=False))

    for pattern in FORBIDDEN_IN_COMMAND:
        assert pattern not in text, f"{pattern!r} must not appear in the Linux suite command"


def test_poe_exposes_the_task(repo_root: Path) -> None:
    """The documented entry point is ``poe test-linux``; docs and task must not drift apart."""
    import tomllib

    tasks = tomllib.loads((repo_root / "poe_tasks.toml").read_text(encoding="utf-8"))["tasks"]

    assert "test-linux" in tasks
    assert MODULE in tasks["test-linux"]["cmd"]
    strategy = (repo_root / "docs" / "gelistirme" / "test-stratejisi.md").read_text(
        encoding="utf-8"
    )
    assert "poe test-linux" in strategy
