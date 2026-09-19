"""Verify the containerised GPU environment (ADR-0020); run inside the container only.

This is the verification half of ``scripts/gpu/setup-gpu-env.sh``. It answers, in one place and
with an exit code, the questions the operator handbook asks before any training is allowed to
start:

* is this the Blackwell card, and does torch see it as ``sm_120`` (compute capability ``(12, 0)``)?
* does a bfloat16 matmul actually execute on it and agree with the float32 reference?
* does ``aicls`` import from the **mounted repository** rather than from some other copy?
* does ``git rev-parse HEAD`` answer, so :class:`aicls.ml.manifest.RunManifest` can attribute a
  run to a commit (without it every run is ``citable=False`` and cannot enter the report)?
* are ``AICLS_DATA_ROOT`` / ``AICLS_MODELS_ROOT`` absolute POSIX paths? A Windows path leaks in
  as a *relative* ``PosixPath`` and silently resolves weights next to the working directory.

Nothing here downloads anything and nothing here trains anything: it is a hardware and wiring
check, gate-free by construction (ADR-0012).

Usage (through the runner, never bare)::

    scripts/gpu/aicls-gpu.sh --raw python scripts/gpu/verify_gpu.py
    scripts/gpu/aicls-gpu.sh --raw python scripts/gpu/verify_gpu.py --json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

#: Compute capability of the RTX 5090 (Blackwell). Anything else means the wrong wheel or card.
EXPECTED_CAPABILITY = (12, 0)

#: Largest tolerated relative error of the bf16 matmul against its float32 reference. bfloat16
#: keeps 8 mantissa bits, so a few percent on an accumulated 256x256 product is normal; the check
#: exists to prove the kernel ran and produced numbers, not to measure precision.
BF16_RELATIVE_TOLERANCE = 0.05


@dataclass(slots=True)
class Check:
    """One verification step and its outcome."""

    name: str
    ok: bool
    detail: str
    data: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        """One aligned console line."""
        mark = "PASS" if self.ok else "FAIL"
        return f"  [{mark}] {self.name:<24} {self.detail}"


def _run(args: list[str]) -> tuple[int, str]:
    """Run a fixed command and return ``(returncode, stripped stdout+stderr)``."""
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, str(exc)
    return done.returncode, (done.stdout + done.stderr).strip()


def check_interpreter() -> Check:
    """The interpreter must be the runtime venv, not the image's bare conda python."""
    exe = sys.executable
    ok = "aicls-venv" in exe or bool(os.environ.get("VIRTUAL_ENV"))
    return Check(
        name="interpreter",
        ok=ok,
        detail=f"{platform.python_version()} at {exe}",
        data={"executable": exe, "version": platform.python_version()},
    )


def check_torch() -> tuple[Check, Any]:
    """Import torch and report the build; returns the module so later checks can reuse it."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - the image always ships torch
        return Check("torch", False, f"import failed: {exc}"), None
    detail = f"{torch.__version__} (cuda {torch.version.cuda})"
    return (
        Check(
            "torch",
            True,
            detail,
            {"version": torch.__version__, "cuda": torch.version.cuda},
        ),
        torch,
    )


def check_device(torch: Any) -> Check:
    """CUDA must be available and the card must report compute capability (12, 0)."""
    if torch is None:
        return Check("device", False, "torch unavailable")
    if not torch.cuda.is_available():
        return Check("device", False, "torch.cuda.is_available() is False")
    index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    capability = tuple(int(v) for v in torch.cuda.get_device_capability(index))
    total_gb = torch.cuda.get_device_properties(index).total_memory / (1024**3)
    ok = capability == EXPECTED_CAPABILITY
    detail = f"{name}, capability {capability}, {total_gb:.1f} GiB"
    if not ok:
        detail += f" (expected {EXPECTED_CAPABILITY} - wrong wheel or wrong card)"
    return Check(
        "device",
        ok,
        detail,
        {"name": name, "capability": list(capability), "total_memory_gib": round(total_gb, 2)},
    )


def check_bf16(torch: Any) -> Check:
    """Execute a bfloat16 matmul on the GPU and compare it with the float32 reference."""
    if torch is None or not torch.cuda.is_available():
        return Check("bf16_matmul", False, "no CUDA device")
    if not torch.cuda.is_bf16_supported():
        return Check("bf16_matmul", False, "torch.cuda.is_bf16_supported() is False")
    generator = torch.Generator(device="cuda").manual_seed(1234)
    left = torch.randn(256, 256, device="cuda", generator=generator)
    right = torch.randn(256, 256, device="cuda", generator=generator)
    reference = left @ right
    product = (left.bfloat16() @ right.bfloat16()).float()
    error = ((product - reference).norm() / reference.norm()).item()
    ok = error < BF16_RELATIVE_TOLERANCE
    return Check(
        "bf16_matmul",
        ok,
        f"256x256 bf16 matmul executed, relative error {error:.4f} (< {BF16_RELATIVE_TOLERANCE})",
        {"relative_error": error},
    )


def check_aicls() -> Check:
    """``aicls`` must import from the mounted repository, not from another checkout."""
    try:
        import aicls
    except ImportError as exc:
        return Check("aicls_import", False, f"import failed: {exc}")
    location = Path(aicls.__file__ or "?").resolve().parent
    workdir = Path(os.environ.get("AICLS_GPU_WORKDIR", "/workspace/ai-cls"))
    ok = workdir in location.parents or str(location).startswith(str(workdir))
    detail = f"{aicls.__version__} from {location}"
    if not ok:
        detail += f" (NOT under the mounted repo {workdir}: the container is running other code)"
    return Check("aicls_import", ok, detail, {"version": aicls.__version__, "path": str(location)})


def check_console_script() -> Check:
    """``pip install -e .`` must have produced a working ``aicls`` console script."""
    code, output = _run(["aicls", "--version"])
    ok = code == 0 and output.startswith("aicls")
    return Check("console_script", ok, output or f"exit {code}", {"output": output})


def check_git() -> Check:
    """``git`` must answer inside the container or every run is non-citable (ADR-0007)."""
    code, sha = _run(["git", "rev-parse", "HEAD"])
    if code != 0:
        return Check(
            "git",
            False,
            f"git rev-parse HEAD failed ({sha or 'git missing'}); "
            "RunManifest would record citable=false",
        )
    status_code, status = _run(["git", "status", "--porcelain"])
    dirty = bool(status.strip()) if status_code == 0 else True
    detail = f"HEAD {sha[:7]}, working tree {'DIRTY' if dirty else 'clean'}"
    return Check("git", True, detail, {"sha": sha, "dirty": dirty})


def check_roots() -> list[Check]:
    """``AICLS_DATA_ROOT`` / ``AICLS_MODELS_ROOT`` must be absolute POSIX paths that exist."""
    checks: list[Check] = []
    for name in ("AICLS_DATA_ROOT", "AICLS_MODELS_ROOT"):
        raw = os.environ.get(name, "")
        path = Path(raw) if raw else None
        if path is None:
            checks.append(Check(name.lower(), False, f"{name} is not set"))
            continue
        absolute = path.is_absolute()
        detail = f"{path} (absolute={absolute}, exists={path.exists()})"
        if not absolute:
            detail += " - a Windows path becomes a RELATIVE PosixPath here"
        checks.append(
            Check(
                name.lower(),
                absolute and path.exists(),
                detail,
                {"path": str(path), "absolute": absolute},
            )
        )
    return checks


def check_nvidia_smi() -> Check:
    """Record the driver version; informational, never fatal."""
    code, output = _run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    )
    if code != 0:
        return Check("nvidia_smi", True, f"unavailable ({output[:60]}) - informational only")
    return Check("nvidia_smi", True, output.splitlines()[0], {"output": output})


def run_checks() -> list[Check]:
    """Run every verification step in order."""
    torch_check, torch = check_torch()
    checks = [check_interpreter(), torch_check, check_device(torch), check_bf16(torch)]
    checks.extend([check_aicls(), check_console_script(), check_git()])
    checks.extend(check_roots())
    checks.append(check_nvidia_smi())
    return checks


def main(argv: list[str] | None = None) -> int:
    """Print the verification section; exit 1 when any check failed."""
    parser = argparse.ArgumentParser(description="Verify the AI-CLS GPU container (ADR-0020).")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    checks = run_checks()
    failed = [c for c in checks if not c.ok]
    if args.json:
        payload = {
            "ok": not failed,
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail, **c.data} for c in checks],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("GPU environment verification (ADR-0020)")
        for check in checks:
            print(check.render())
        verdict = "OK" if not failed else f"{len(failed)} FAILED"
        print(f"  ---- {verdict}")
        if not failed:
            print(
                "  NOTE: this is a hardware and wiring check. It proves the GPU is reachable,\n"
                "        not that any model works. No contractual metric follows from it."
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
