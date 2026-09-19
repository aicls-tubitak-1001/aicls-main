"""Tests for the containerised GPU entry point (``scripts/gpu/``, ADR-0020).

Three layers, because the thing under test is a shell script that builds a ``docker run`` argv:

* **text contracts** — every invariant that must hold even on a machine with no bash: the image
  pin, the gate variables that must be forwarded, the absence of anything that could open an
  approval gate, and the documentation cross-links;
* **executable contracts** — the runner is actually executed with a *fake* ``docker`` on ``PATH``
  and ``--print``, so the real argv is asserted (mounts, ``--gpus``, ``--ipc=host``, ulimits,
  ``--user``, gate forwarding, the ``aicls`` prefix). Skipped when no bash is available;
* **live contracts** (``gpu`` marker, skipped unless ``AICLS_TEST_GPU=1``) — the container is
  started for real and a *real-data* ``train`` without approval must still exit ``3``. The gate
  must not become weaker by running inside Docker.

Nothing here opens a gate: the gate variable used in the forwarding tests carries a value that is
deliberately **not** ``approved``, so even a mistake in this file cannot approve anything.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import pytest

GPU_DIR = Path("scripts") / "gpu"
RUNNER = GPU_DIR / "aicls-gpu.sh"
WRAPPER = GPU_DIR / "aicls-gpu.ps1"
SETUP = GPU_DIR / "setup-gpu-env.sh"
VERIFIER = GPU_DIR / "verify_gpu.py"
README = GPU_DIR / "README.md"
EXCLUDES = GPU_DIR / "container-excludes"
ADR = Path("docs") / "adr" / "0020-gpu-calisma-ortami-wsl-nvidia-docker.md"

#: The four approval gates plus the two CI switches. Forwarding all six is a safety property:
#: a gate must behave identically inside and outside the container (ADR-0012).
GATE_VARIABLES = (
    "AICLS_GATE_DATASET_DOWNLOAD",
    "AICLS_GATE_WEIGHTS_DOWNLOAD",
    "AICLS_GATE_CUDA_INSTALL",
    "AICLS_GATE_TRAINING",
    "CI",
    "GITHUB_ACTIONS",
)

#: Never "approved". Used only to prove that the *value* travels into the container.
NOT_APPROVED = "not-approved-test-sentinel"

IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}")


def read(repo_root: Path, relative: Path) -> str:
    """Read one repository file as UTF-8."""
    return (repo_root / relative).read_text(encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# text contracts
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("relative", [RUNNER, WRAPPER, SETUP, VERIFIER, README, EXCLUDES, ADR])
def test_gpu_entry_point_files_exist(repo_root: Path, relative: Path) -> None:
    path = repo_root / relative
    assert path.is_file(), f"{relative.as_posix()} is missing"
    assert path.stat().st_size > 0


@pytest.mark.parametrize("relative", [RUNNER, SETUP, WRAPPER, EXCLUDES])
def test_shell_scripts_use_lf_line_endings(repo_root: Path, relative: Path) -> None:
    """A CRLF ``.sh`` file dies inside the container with ``bad interpreter: ^M``."""
    assert b"\r\n" not in (repo_root / relative).read_bytes()


def test_runner_mounts_the_repository_with_its_git(repo_root: Path) -> None:
    """``RunManifest.capture()`` needs a real ``.git``; a stripped copy is not citable."""
    text = read(repo_root, RUNNER)
    assert '[ -e "$REPO/.git" ] || die' in text
    assert 'add_mount "$REPO" "$AICLS_GPU_WORKDIR"' in text
    # git itself is bind-mounted because the image ships none (verified 2026-09-08).
    assert "/usr/bin/git" in text
    assert "safe.directory" in text


def test_runner_validates_that_roots_are_absolute_posix_paths(repo_root: Path) -> None:
    """A Windows root leaks in as a relative ``PosixPath``; the runner must refuse it."""
    text = read(repo_root, RUNNER)
    assert "require_posix_abs" in text
    for guard in ("*\\\\*", "[A-Za-z]:*", "/*"):
        assert guard in text
    for name in ("AICLS_DATA_ROOT", "AICLS_MODELS_ROOT", "AICLS_GPU_VENV", "AICLS_GPU_WORKDIR"):
        assert f"require_posix_abs {name}" in text


def test_runner_asks_for_the_gpu_host_ipc_and_raised_ulimits(repo_root: Path) -> None:
    """``--gpus``, plus the NGC recommendation for dataloader shared memory."""
    text = read(repo_root, RUNNER)
    assert '--gpus "$AICLS_GPU_GPUS"' in text
    assert "AICLS_GPU_GPUS:-all" in text
    assert "--ipc=host" in text
    assert '--ulimit "memlock=-1"' in text
    assert '--ulimit "stack=${AICLS_GPU_STACK_ULIMIT}"' in text


def test_runner_runs_as_the_calling_user(repo_root: Path) -> None:
    """Files written under /mnt/d must stay usable from Windows and from WSL."""
    assert '--user "$(id -u):$(id -g)"' in read(repo_root, RUNNER)


def test_runner_forwards_every_gate_variable(repo_root: Path) -> None:
    text = read(repo_root, RUNNER)
    for name in GATE_VARIABLES:
        assert name in text, f"{name} is not forwarded; the gate would weaken inside Docker"


def test_powershell_wrapper_shares_the_gate_variables_through_wslenv(repo_root: Path) -> None:
    text = read(repo_root, WRAPPER)
    assert "WSLENV" in text
    for name in GATE_VARIABLES:
        assert name in text
    # A Windows data root must be translated, never forwarded as-is.
    assert "ConvertTo-WslPath" in text
    assert "--data-root" in text


@pytest.mark.gated
@pytest.mark.parametrize("relative", [RUNNER, WRAPPER, SETUP, VERIFIER])
def test_gpu_scripts_never_open_an_approval_gate(repo_root: Path, relative: Path) -> None:
    """No executable here may pass the flag, set an approved value or touch the ledger.

    Documentation (``README.md``, the ADR) is excluded on purpose: it has to name the flag in
    order to explain it. Only the files that can *run* are held to this contract.
    """
    lines = read(repo_root, relative).splitlines()
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith(("#", "//")):
            continue
        assert "--i-have-approval" not in stripped, f"{relative.as_posix()}:{number}"
        assert not re.search(r"AICLS_GATE_\w+\s*=\s*[\"']?approved", stripped), (
            f"{relative.as_posix()}:{number} sets a gate variable to approved"
        )
        assert "onay-defteri" not in stripped, (
            f"{relative.as_posix()}:{number} touches the approval ledger"
        )


def test_runner_pins_the_image_and_the_pin_matches_the_adr(repo_root: Path) -> None:
    """The image is not ours and carries no registry digest; the local image ID is the pin."""
    runner = read(repo_root, RUNNER)
    match = re.search(r"AICLS_GPU_IMAGE_ID:-(sha256:[0-9a-f]{64})", runner)
    assert match, "the runner must carry a default image ID pin"
    pinned = match.group(1)
    assert pinned in read(repo_root, ADR), "ADR-0020 must record the pinned image ID"
    assert "AICLS_GPU_ALLOW_IMAGE_DRIFT" in runner, "an explicit override must exist"
    assert "codex-research-cu130" in runner


def test_runner_has_a_print_mode(repo_root: Path) -> None:
    text = read(repo_root, RUNNER)
    assert "--print|--dry-run) PRINT_ONLY=1" in text
    assert "NOT executed" in text
    # The runner's own options stop at the first non-option, so `aicls train --dry-run` survives.
    assert "*) break ;;" in text


def test_setup_script_is_idempotent_and_installs_the_console_script(repo_root: Path) -> None:
    text = read(repo_root, SETUP)
    assert "pyvenv.cfg" in text, "venv presence must be detected by a host-visible marker"
    assert "--system-site-packages" in text
    assert "pip install --no-deps --no-build-isolation" in text
    assert "verify_gpu.py" in text


def test_container_excludes_only_lists_windows_junctions(repo_root: Path) -> None:
    """Linux git sees ``web/node_modules`` (a Windows junction) as an untracked symlink.

    Without this exclude file every containerised run would be recorded ``dirty``, and a dirty
    run is ``citable=False`` (ADR-0007). The file must stay minimal: it changes what "clean"
    means, so anything beyond restoring Windows parity would be hiding real changes.
    """
    entries = [
        line.strip()
        for line in read(repo_root, EXCLUDES).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert entries == ["web/node_modules"]


def test_adr_records_the_honest_consequences(repo_root: Path) -> None:
    text = read(repo_root, ADR)
    assert "ADR-0020" in text
    for heading in ("## Bağlam ve problem", "## Karar", "## Sonuçlar", "## Doğrulama"):
        assert heading in text
    assert "Öneri metniyle uyum" in text
    # The three consequences the decision must not hide.
    assert "2.9.1" in text and "2.11" in text, "the torch pin mismatch must be recorded"
    assert IMAGE_ID_RE.search(text), "the image ID pin must be recorded"
    assert "--no-deps" in text


def test_adr_is_reachable_from_the_index_and_the_navigation(repo_root: Path) -> None:
    assert ADR.name in read(repo_root, Path("mkdocs.yml"))
    assert ADR.name in read(repo_root, Path("docs") / "adr" / "README.md")


def test_documentation_points_at_the_runner(repo_root: Path) -> None:
    """Both operator pages must name the entry point; either path separator is fine."""
    setup_doc = read(repo_root, Path("docs") / "gelistirme" / "kurulum.md")
    handbook = read(repo_root, Path("docs") / "ml" / "egitim-el-kitabi.md")
    for text in (setup_doc, handbook):
        assert "aicls-gpu.ps1" in text or "aicls-gpu.sh" in text
        assert "scripts/gpu" in text or "scripts\\gpu" in text
        assert "setup-gpu-env.sh" in text
        assert ADR.name in text


def test_check_gates_finds_nothing_in_the_gpu_scripts(load_script: Any, repo_root: Path) -> None:
    """The static download scan must stay clean for the new directory."""
    gates = load_script("tools/check_gates.py")
    hits = [
        hit
        for path in sorted((repo_root / GPU_DIR).rglob("*"))
        if path.is_file() and path.suffix in gates.SCAN_SUFFIXES
        for hit in gates.scan_file(path, root=repo_root)
    ]
    assert hits == [], "\n".join(hit.render(repo_root) for hit in hits)


# ----------------------------------------------------------------------------------------------
# executable contracts — the runner is really executed, with a fake docker on PATH
# ----------------------------------------------------------------------------------------------
FAKE_DOCKER = """#!/usr/bin/env bash
# Stand-in for docker: answers `docker image inspect --format {{.Id}} <ref>` with the pinned id.
if [ "$1" = "image" ] && [ "$2" = "inspect" ]; then
  printf '%s\\n' "$AICLS_TEST_FAKE_IMAGE_ID"
  exit 0
fi
exit 0
"""


def _bash() -> str | None:
    """A bash that can run a Windows-path script, or None."""
    found = shutil.which("bash")
    if found is None:
        return None
    probe = subprocess.run(
        [found, "-c", "printf %s $OSTYPE"], capture_output=True, text=True, check=False
    )
    flavour = probe.stdout.strip()
    if sys.platform == "win32" and not flavour.startswith(("msys", "cygwin")):
        return None  # WSL's bash shim cannot take the Windows script path
    return found


BASH = _bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="no msys/POSIX bash on PATH")


def run_runner(
    repo_root: Path, tmp_path: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Execute the runner with a fake ``docker`` first on ``PATH``."""
    assert BASH is not None
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir(exist_ok=True)
    docker = bin_dir / "docker"
    docker.write_text(FAKE_DOCKER, encoding="utf-8", newline="\n")
    docker.chmod(0o755)

    runner = repo_root / RUNNER
    pin = re.search(r"AICLS_GPU_IMAGE_ID:-(sha256:[0-9a-f]{64})", runner.read_text("utf-8"))
    assert pin is not None
    child_env = dict(os.environ)
    child_env.pop("AICLS_DATA_ROOT", None)
    child_env.pop("AICLS_MODELS_ROOT", None)
    for name in GATE_VARIABLES:
        child_env.pop(name, None)
    child_env["AICLS_TEST_FAKE_IMAGE_ID"] = pin.group(1)
    child_env["PATH"] = f"{bin_dir}{os.pathsep}{child_env['PATH']}"
    if env:
        child_env.update(env)
    return subprocess.run(
        [BASH, str(runner), *args],
        capture_output=True,
        text=True,
        check=False,
        env=child_env,
        cwd=str(repo_root),
    )


@pytest.fixture
def data_root_option() -> list[str]:
    """A POSIX-absolute data root the runner accepts on every platform."""
    return ["--data-root", "/tmp/aicls-gpu-runner-test"]


@needs_bash
def test_print_mode_builds_the_expected_docker_argv(
    repo_root: Path, tmp_path: Path, data_root_option: list[str]
) -> None:
    done = run_runner(
        repo_root,
        tmp_path,
        "--print",
        *data_root_option,
        "train",
        "--config",
        "training/configs/ocr/smoke_synthetic.yaml",
        "--synthetic",
    )
    assert done.returncode == 0, done.stderr
    argv = done.stdout
    assert "docker run --rm" in argv
    assert "--gpus all" in argv
    assert "--ipc=host" in argv
    assert "--ulimit memlock=-1" in argv
    assert "--ulimit stack=" in argv
    assert "--user " in argv
    assert ":/workspace/ai-cls" in argv, "the repository must be mounted"
    assert "-e AICLS_DATA_ROOT=/tmp/aicls-gpu-runner-test" in argv
    assert "-e AICLS_MODELS_ROOT=/tmp/aicls-gpu-runner-test/models" in argv
    assert "-e PYTHONPATH=/workspace/ai-cls/src" in argv
    assert argv.rstrip().endswith(
        "aicls train --config training/configs/ocr/smoke_synthetic.yaml --synthetic"
    )


@needs_bash
def test_print_mode_does_not_start_a_container(
    repo_root: Path, tmp_path: Path, data_root_option: list[str]
) -> None:
    done = run_runner(repo_root, tmp_path, "--print", *data_root_option, "train")
    assert "NOT executed" in done.stdout


@needs_bash
def test_gate_variables_reach_the_container(
    repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, data_root_option: list[str]
) -> None:
    """The value travels; it is deliberately not ``approved``, so nothing can be opened here."""
    monkeypatch.setenv("AICLS_GATE_TRAINING", NOT_APPROVED)
    monkeypatch.setenv("CI", "1")
    done = run_runner(
        repo_root,
        tmp_path,
        "--print",
        *data_root_option,
        "train",
        env={"AICLS_GATE_TRAINING": NOT_APPROVED, "CI": "1"},
    )
    assert done.returncode == 0, done.stderr
    assert f"-e AICLS_GATE_TRAINING={NOT_APPROVED}" in done.stdout
    assert "-e CI=1" in done.stdout, "the CI refusal must survive the container boundary"


@needs_bash
def test_raw_mode_does_not_prefix_the_aicls_command(
    repo_root: Path, tmp_path: Path, data_root_option: list[str]
) -> None:
    done = run_runner(repo_root, tmp_path, "--print", "--raw", *data_root_option, "python", "-V")
    assert done.returncode == 0, done.stderr
    assert done.stdout.rstrip().endswith("aicls-gpu python -V")


@needs_bash
def test_no_gpu_drops_the_device_request(
    repo_root: Path, tmp_path: Path, data_root_option: list[str]
) -> None:
    done = run_runner(repo_root, tmp_path, "--print", "--no-gpu", *data_root_option, "train")
    assert done.returncode == 0, done.stderr
    assert "--gpus" not in done.stdout


@needs_bash
def test_runner_refuses_a_windows_data_root(repo_root: Path, tmp_path: Path) -> None:
    done = run_runner(repo_root, tmp_path, "--data-root", "D:\\aicls-data", "--print", "train")
    assert done.returncode == 2
    assert "backslash" in done.stderr


@needs_bash
def test_runner_refuses_a_relative_data_root(repo_root: Path, tmp_path: Path) -> None:
    done = run_runner(repo_root, tmp_path, "--data-root", "aicls-data", "--print", "train")
    assert done.returncode == 2
    assert "relative" in done.stderr


@needs_bash
def test_runner_refuses_a_repository_without_git(
    repo_root: Path, tmp_path: Path, data_root_option: list[str]
) -> None:
    copy = tmp_path / "no-git-copy"
    copy.mkdir()
    done = run_runner(
        repo_root, tmp_path, "--repo", copy.as_posix(), "--print", *data_root_option, "train"
    )
    assert done.returncode == 2
    assert ".git" in done.stderr


@needs_bash
def test_runner_refuses_a_moved_image_tag(
    repo_root: Path, tmp_path: Path, data_root_option: list[str]
) -> None:
    """A moved tag silently changes the torch/CUDA stack under every recorded run."""
    other = "sha256:" + "b" * 64
    done = run_runner(
        repo_root,
        tmp_path,
        "--print",
        *data_root_option,
        "train",
        env={"AICLS_TEST_FAKE_IMAGE_ID": other},
    )
    assert done.returncode == 2
    assert "pin says" in done.stderr


# ----------------------------------------------------------------------------------------------
# live contracts — real container, skipped unless AICLS_TEST_GPU=1
# ----------------------------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.gated
@pytest.mark.slow
def test_real_data_training_inside_the_container_still_exits_3(repo_root: Path) -> None:
    """The approval gate must not become weaker by running the command in Docker.

    Measured by hand on 2026-09-08 through ``scripts/gpu/aicls-gpu.sh``:
    ``missing: ['cli_flag', 'env', 'ledger']`` and exit code 3.
    """
    wsl = shutil.which("wsl") or shutil.which("wsl.exe")
    if wsl is None:  # pragma: no cover - depends on the host
        pytest.skip("wsl.exe not available")
    script = "scripts/gpu/aicls-gpu.sh"
    done = subprocess.run(
        [
            wsl,
            "-e",
            "bash",
            script,
            "train",
            "--config",
            "training/configs/ocr/offline_resnet34_tfm.yaml",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(repo_root),
        timeout=600,
    )
    assert done.returncode == 3, done.stdout + done.stderr
    assert "GATE_BLOCKED" in (done.stdout + done.stderr)
