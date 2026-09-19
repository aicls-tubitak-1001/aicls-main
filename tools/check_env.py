"""Developer environment check (ADR-0011): venv location, OneDrive paths, data root, git config.

Run as ``poe env-check`` (``python scripts/check_env.py`` delegates here). Exit code 1 when any
finding has level ``error``; warnings never fail the run. ``--ci`` relaxes workstation-only checks
(venv path, OneDrive) so the report can run on GitHub runners.

The functions are importable so ``tests/unit/test_scaffold.py`` can exercise the OneDrive rule
without touching the real environment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

EXPECTED_VENV = Path(r"C:\Users\cagri\.venvs\ai-cls")
MIN_JWT_SECRET_BYTES = 32
MIN_FREE_GB = 50.0
GATE_ENV_KEYS = (
    "AICLS_GATE_DATASET_DOWNLOAD",
    "AICLS_GATE_WEIGHTS_DOWNLOAD",
    "AICLS_GATE_CUDA_INSTALL",
    "AICLS_GATE_TRAINING",
)
EXPECTED_GIT_CONFIG = {
    "core.autocrlf": "false",
    "core.longpaths": "true",
    "core.fsmonitor": "true",
}


@dataclass(frozen=True, slots=True)
class Finding:
    """One check result."""

    level: str  # "ok" | "warn" | "error"
    check: str
    message: str


def is_under_onedrive(path: str | os.PathLike[str]) -> bool:
    """Return True when *path* lies inside a OneDrive-synced folder.

    Detection is by path component (``OneDrive`` or ``OneDrive - <tenant>``), case-insensitive,
    plus the ``ONEDRIVE`` / ``ONEDRIVECOMMERCIAL`` environment roots when present.

    The components are split on **both** separators rather than through ``Path.parts``: a
    Windows path handed to a POSIX interpreter (this check runs inside the Linux container too)
    is a single ``PosixPath`` component with backslashes in it, and the OneDrive folder in the
    middle of it would go unnoticed.
    """
    p = Path(path)
    for part in re.split(r"[\\/]+", str(path)):
        if part.lower().startswith("onedrive"):
            return True
    for key in ("ONEDRIVE", "ONEDRIVECOMMERCIAL", "ONEDRIVECONSUMER"):
        root = os.environ.get(key)
        if root:
            try:
                p.resolve().relative_to(Path(root).resolve())
                return True
            except (ValueError, OSError):
                continue
    return False


def check_onedrive_paths(
    *,
    repo_root: Path,
    venv_prefix: Path,
    data_root: str | None,
    models_root: str | None,
    pycache_prefix: str | None,
) -> list[Finding]:
    """Findings about OneDrive placement of repo, venv, data root and bytecode cache."""
    out: list[Finding] = []
    if is_under_onedrive(repo_root):
        out.append(
            Finding(
                "warn",
                "onedrive.repo",
                f"Repository is under OneDrive ({repo_root}); expect sync noise/locks. "
                "Move the clone to C:\\src\\ai-cls once the remote exists (ADR-0011).",
            )
        )
    else:
        out.append(Finding("ok", "onedrive.repo", f"Repository outside OneDrive: {repo_root}"))

    if is_under_onedrive(venv_prefix):
        out.append(
            Finding(
                "error", "onedrive.venv", f"Virtual environment is under OneDrive: {venv_prefix}"
            )
        )
    else:
        out.append(Finding("ok", "onedrive.venv", f"venv outside OneDrive: {venv_prefix}"))

    if not data_root:
        out.append(
            Finding(
                "warn",
                "data_root",
                "AICLS_DATA_ROOT is not set (expected D:\\aicls-data or C:\\aicls-data).",
            )
        )
    elif is_under_onedrive(data_root):
        out.append(
            Finding(
                "error",
                "onedrive.data_root",
                f"AICLS_DATA_ROOT is under OneDrive: {data_root} — datasets, SQLite and "
                "checkpoints must live outside OneDrive (ADR-0008).",
            )
        )
    else:
        exists = Path(data_root).exists()
        out.append(
            Finding(
                "ok" if exists else "warn",
                "data_root",
                f"AICLS_DATA_ROOT={data_root}"
                + ("" if exists else " (directory does not exist yet)"),
            )
        )

    if models_root and is_under_onedrive(models_root):
        out.append(
            Finding(
                "error",
                "onedrive.models_root",
                f"AICLS_MODELS_ROOT is under OneDrive: {models_root}",
            )
        )

    if not pycache_prefix:
        out.append(
            Finding(
                "warn",
                "pycache",
                "PYTHONPYCACHEPREFIX is not set; __pycache__ will be written inside the repo "
                "(set it to C:\\Users\\cagri\\.cache\\ai-cls\\pycache).",
            )
        )
    elif is_under_onedrive(pycache_prefix):
        out.append(
            Finding("warn", "pycache", f"PYTHONPYCACHEPREFIX is under OneDrive: {pycache_prefix}")
        )
    else:
        out.append(Finding("ok", "pycache", f"PYTHONPYCACHEPREFIX={pycache_prefix}"))
    return out


def check_interpreter(*, ci: bool) -> list[Finding]:
    """Python version and venv identity."""
    out: list[Finding] = []
    ver = sys.version_info
    if (ver.major, ver.minor) != (3, 11):
        out.append(
            Finding("warn", "python", f"Python {ver.major}.{ver.minor} — project targets 3.11")
        )
    else:
        out.append(
            Finding("ok", "python", f"Python {platform.python_version()} ({sys.executable})")
        )
    if not ci:
        prefix = Path(sys.prefix).resolve()
        if prefix != EXPECTED_VENV.resolve():
            out.append(
                Finding(
                    "warn",
                    "venv",
                    f"Interpreter prefix {prefix} != expected venv {EXPECTED_VENV}; "
                    "activate C:\\Users\\cagri\\.venvs\\ai-cls\\Scripts\\Activate.ps1",
                )
            )
        else:
            out.append(Finding("ok", "venv", f"Using expected venv {EXPECTED_VENV}"))
    return out


def check_torch() -> list[Finding]:
    """Report torch build and CUDA availability (CPU build is the expected state before gate 3)."""
    try:
        import torch  # noqa: F401 — optional at this stage
    except ImportError:
        return [Finding("warn", "torch", "torch not importable (fine until ml work starts)")]
    import torch as _torch

    cuda = _torch.cuda.is_available()
    msg = f"torch {_torch.__version__}, cuda_available={cuda}"
    if cuda:
        cap = _torch.cuda.get_device_capability()
        msg += f", capability={cap}, device={_torch.cuda.get_device_name(0)}"
        return [Finding("ok", "torch", msg)]
    return [Finding("ok", "torch", msg + " (expected: CPU build until CUDA_INSTALL gate)")]


def check_secrets_and_gates(env: dict[str, str]) -> list[Finding]:
    """JWT secret strength and accidental gate approvals in the environment."""
    out: list[Finding] = []
    secret = env.get("AICLS_JWT_SECRET", "")
    if not secret:
        out.append(Finding("warn", "jwt", "AICLS_JWT_SECRET is empty (required before `poe dev`)."))
    elif len(secret) < MIN_JWT_SECRET_BYTES * 2 or secret in {"changeme", "secret", "dev"}:
        out.append(
            Finding(
                "error",
                "jwt",
                "AICLS_JWT_SECRET is too short/default "
                f"(need >= {MIN_JWT_SECRET_BYTES} bytes hex).",
            )
        )
    else:
        out.append(Finding("ok", "jwt", "AICLS_JWT_SECRET length OK"))
    for key in GATE_ENV_KEYS:
        if env.get(key, "").strip().lower() == "approved":
            out.append(
                Finding(
                    "warn",
                    "gates",
                    f"{key}=approved is set in this shell — make sure this is intentional and "
                    "recorded in docs/gelistirme/onay-defteri.md (ADR-0012).",
                )
            )
    return out


def check_git_config(repo_root: Path) -> list[Finding]:
    """Read-only check of the git settings recommended by ADR-0011."""
    git = shutil.which("git")
    if git is None:
        return [Finding("warn", "git", "git not found on PATH")]
    out: list[Finding] = []
    for key, expected in EXPECTED_GIT_CONFIG.items():
        try:
            proc = subprocess.run(
                [git, "config", "--get", key],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            out.append(Finding("warn", "git", f"could not read git config {key}: {exc}"))
            continue
        value = proc.stdout.strip()
        if value.lower() == expected:
            out.append(Finding("ok", f"git.{key}", f"{key}={value}"))
        else:
            out.append(
                Finding(
                    "warn",
                    f"git.{key}",
                    f"{key}={value or '<unset>'} "
                    f"(recommended {expected}; tools/onedrive_setup.ps1)",
                )
            )
    return out


def check_disk(path: Path) -> list[Finding]:
    """Free disk space on the drive that holds *path*."""
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return [Finding("warn", "disk", f"cannot stat {path}: {exc}")]
    free_gb = usage.free / 1e9
    level = "ok" if free_gb >= MIN_FREE_GB else "warn"
    return [Finding(level, "disk", f"{free_gb:.0f} GB free on {path.anchor or path}")]


def run_all(
    *, repo_root: Path, ci: bool = False, env: dict[str, str] | None = None
) -> list[Finding]:
    """Run every check and return the findings."""
    env = dict(os.environ if env is None else env)
    findings: list[Finding] = []
    findings += check_interpreter(ci=ci)
    if not ci:
        findings += check_onedrive_paths(
            repo_root=repo_root,
            venv_prefix=Path(sys.prefix),
            data_root=env.get("AICLS_DATA_ROOT"),
            models_root=env.get("AICLS_MODELS_ROOT"),
            pycache_prefix=env.get("PYTHONPYCACHEPREFIX"),
        )
        findings += check_git_config(repo_root)
        findings += check_disk(Path(env.get("AICLS_DATA_ROOT") or repo_root))
    findings += check_torch()
    findings += check_secrets_and_gates(env)
    return findings


def format_findings(findings: list[Finding]) -> str:
    """Render findings as aligned text lines."""
    icons = {"ok": "OK  ", "warn": "WARN", "error": "ERR "}
    return "\n".join(f"[{icons[f.level]}] {f.check:<22} {f.message}" for f in findings)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ci", action="store_true", help="skip workstation-only checks")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: parent of tools/)",
    )
    args = parser.parse_args(argv)
    findings = run_all(repo_root=args.repo_root, ci=args.ci)
    print(format_findings(findings))
    errors = [f for f in findings if f.level == "error"]
    warns = [f for f in findings if f.level == "warn"]
    print(f"\n{len(errors)} error(s), {len(warns)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
