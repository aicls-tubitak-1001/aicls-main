"""Run the test suite on Linux, inside the CUDA container, from either host (ADR-0011, ADR-0016).

Everything in this repository was written and measured on Windows. A classroom pilot would most
plausibly be deployed in Docker, and the first container run (2026-09-08) found defects that no
Windows run could have found: an absolute ``RLIMIT_AS`` below the worker's floor that broke the
whole CAS sandbox, and ``Path("D:/aicls-models")`` -- absolute on Windows, a relative one-component
path on Linux -- which quietly turned two assertions into tautologies.

This launcher exists so that "does it still work on Linux?" is one command instead of a
remembered incantation:

    poe test-linux                 # the widest CPU-only selection, on Linux
    poe test-linux -- tests/unit/mathengine       # any pytest argument, forwarded
    python scripts/run_linux_suite.py --print     # show the command without running it

It is deliberately **not** a second container runner: the docker argv, the image pin, the mounts
and the approval-gate forwarding all come from ``scripts/gpu/aicls-gpu.sh`` (ADR-0020), which is
the single supported entry point. This module only decides how to reach that script -- directly
when it is already running on Linux, through ``wsl -e bash`` when it is on Windows -- and what to
hand it.

No approval gate is involved: the suite deselects ``gpu``, ``data``, ``mysql`` and ``browser``,
runs on CPU (``--no-gpu``) and downloads nothing. ``AICLS_GATE_*`` is forwarded by the runner
exactly as it is outside the container, so a gated test refuses inside it too.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePath, PureWindowsPath
import shutil
import subprocess
import sys

#: Repository root: this file is ``<repo>/scripts/run_linux_suite.py``.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: The container entry point every GPU/Linux command in this project goes through (ADR-0020).
RUNNER_RELATIVE = Path("scripts") / "gpu" / "aicls-gpu.sh"

#: The selection ``poe test-all`` uses -- the widest CPU-only set, ``slow``/``perf``/``e2e``
#: included. A Linux check that skipped the slow half would have missed the two property
#: tests whose Hypothesis health check is version-sensitive (2026-09-08).
DEFAULT_PYTEST_ARGS: tuple[str, ...] = (
    "-q",
    "-p",
    "no:cacheprovider",
    "--no-header",
    "-m",
    "not gpu and not data and not mysql and not browser",
    "-r",
    "f",
    "--timeout=900",
)

#: Wall-clock budget per test inside the container. The image's CPU torch is several times
#: slower than the Windows one (measured: a two-step synthetic OCR run takes ~140 s there
#: against ~20 s here), so the repository's 120 s default would fail honest tests for being slow.
TIMEOUT_NOTE = "--timeout=900 overrides the 120 s default: container CPU torch is much slower."


def windows_path_to_wsl(path: PurePath | str) -> str:
    """Translate ``C:\\Users\\x\\repo`` into ``/mnt/c/Users/x/repo``.

    The argument is re-read as a :class:`~pathlib.PureWindowsPath`, so the translation works on a
    POSIX interpreter too -- a plain ``Path("C:/x")`` there is one relative component with no
    drive, which would make this silently produce nonsense instead of raising.

    Args:
        path: An absolute Windows path.

    Returns:
        The ``/mnt/<drive>/...`` form WSL sees.

    Raises:
        ValueError: If the path has no single-letter drive to translate.
    """
    windows = PureWindowsPath(path)
    drive = windows.drive.rstrip(":")
    if len(drive) != 1 or not drive.isalpha():
        raise ValueError(f"{path} has no drive letter; cannot translate to a WSL path")
    tail = windows.as_posix()[len(windows.drive) :].lstrip("/")
    return f"/mnt/{drive.lower()}/{tail}"


def build_command(
    pytest_args: list[str],
    *,
    gpu: bool = False,
    windows: bool | None = None,
    repo_root: Path | None = None,
) -> list[str]:
    """Build the argv that runs pytest inside the container.

    Args:
        pytest_args: Arguments handed to pytest; :data:`DEFAULT_PYTEST_ARGS` when empty.
        gpu: Whether to let the container see the GPU. The suite deselects every ``gpu`` test,
            so the default is off -- it keeps the run usable while a training job holds the card.
        windows: Which host branch to build for; the running platform when omitted. It is a
            parameter rather than a bare ``os.name`` read so that a test can exercise both
            branches on either platform -- patching the interpreter's global ``os.name`` is not
            a test, it is a way to crash the process that is running it.
        repo_root: The checkout to mount; :data:`REPO_ROOT` when omitted.

    Returns:
        The complete argv, ready for :func:`subprocess.call`.
    """
    on_windows = (os.name == "nt") if windows is None else windows
    root = REPO_ROOT if repo_root is None else repo_root
    selection = list(pytest_args) or list(DEFAULT_PYTEST_ARGS)
    runner_posix = f"./{RUNNER_RELATIVE.as_posix()}"
    inner = ["bash", runner_posix]
    if not gpu:
        inner.append("--no-gpu")
    inner += ["--raw", "python", "-m", "pytest", *selection]

    if not on_windows:
        return inner
    # On Windows the whole thing is one `bash -lc` string: `wsl -e bash -lc "cd <repo> && ..."`.
    quoted = " ".join(_shell_quote(part) for part in inner)
    repo = _shell_quote(windows_path_to_wsl(root))
    return ["wsl", "-e", "bash", "-lc", f"cd {repo} && {quoted}"]


def _shell_quote(value: str) -> str:
    """POSIX single-quote ``value`` (``shlex.quote`` semantics, without importing shlex rules)."""
    # `isascii()` matters here: the repository path contains "Masaüstü", and a non-ASCII byte
    # left unquoted is at the mercy of the receiving shell's locale.
    if value and all((char.isascii() and char.isalnum()) or char in "@%+=:,./-_" for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when omitted. Anything that is not a
            recognised option is forwarded to pytest.

    Returns:
        The container's exit code, or 2 when the environment cannot run it.
    """
    parser = argparse.ArgumentParser(
        prog="run_linux_suite",
        description="Run the pytest suite inside the Linux CUDA container.",
        epilog=TIMEOUT_NOTE,
    )
    parser.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="Print the command and exit without running it.",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Give the container the GPU (default: CPU only)."
    )
    known, pytest_args = parser.parse_known_args(argv)

    runner = REPO_ROOT / RUNNER_RELATIVE
    if not runner.is_file():
        print(f"run_linux_suite: {runner} is missing; see docs/adr/0020-*.md", file=sys.stderr)
        return 2
    if os.name == "nt" and shutil.which("wsl") is None:
        print(
            "run_linux_suite: wsl.exe not found; this needs WSL2 + nvidia-docker", file=sys.stderr
        )
        return 2

    command = build_command(pytest_args, gpu=known.gpu)
    print(" ".join(command), flush=True)
    if known.print_only:
        return 0
    return subprocess.call(command, cwd=REPO_ROOT)


if __name__ == "__main__":  # pragma: no cover - console entry point
    raise SystemExit(main())
