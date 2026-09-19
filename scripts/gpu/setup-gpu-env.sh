#!/usr/bin/env bash
# setup-gpu-env.sh - prepare and verify the containerised GPU environment (ADR-0020).
#
# Idempotent: run it as often as you like. It
#   1. checks docker, the nvidia runtime and the pinned image;
#   2. creates AICLS_DATA_ROOT / AICLS_MODELS_ROOT on the Windows disk;
#   3. creates the runtime venv at /mnt/d/aicls-venv (python -m venv --system-site-packages,
#      so torch/transformers come from the image and the venv stays ~140 MB);
#   4. runs `pip install --no-deps --no-build-isolation -e <repo>` so the `aicls` console script
#      exists and resolves to the MOUNTED repository (the first command of the operator handbook);
#   5. verifies the hardware: compute capability (12, 0) and a real bf16 matmul.
#
# It installs NO CUDA wheel: the CUDA torch comes from the container image, so nothing here is a
# `CUDA_INSTALL` (ADR-0012 gate 3) installation into a project environment. Read the "Sonuçlar"
# section of docs/adr/0020-gpu-calisma-ortami-wsl-nvidia-docker.md before relying on that
# sentence: the gate was written to control GPU availability, and this script does make the GPU
# available. It opens no gate, sets no AICLS_GATE_* variable and passes no --i-have-approval.
#
#   scripts/gpu/setup-gpu-env.sh            # prepare + verify
#   scripts/gpu/setup-gpu-env.sh --verify   # verify only, change nothing
#   scripts/gpu/setup-gpu-env.sh --print    # show the docker commands, run nothing
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1; then
  SCRIPT_PATH="$(readlink -f "$SCRIPT_PATH" 2>/dev/null || printf '%s' "$SCRIPT_PATH")"
fi
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO="${AICLS_GPU_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUNNER="$SCRIPT_DIR/aicls-gpu.sh"

AICLS_GPU_VENV="${AICLS_GPU_VENV:-/mnt/d/aicls-venv}"
AICLS_GPU_WORKDIR="${AICLS_GPU_WORKDIR:-/workspace/ai-cls}"
AICLS_DATA_ROOT="${AICLS_DATA_ROOT:-/mnt/d/aicls-data}"
AICLS_MODELS_ROOT="${AICLS_MODELS_ROOT:-${AICLS_DATA_ROOT}/models}"
export AICLS_GPU_VENV AICLS_GPU_WORKDIR AICLS_DATA_ROOT AICLS_MODELS_ROOT

VERIFY_ONLY=0
PRINT_ONLY=0
RECREATE=0
VERIFY_ARGS=()

die() {
  printf 'setup-gpu-env.sh: %s\n' "$*" >&2
  exit 2
}

step() {
  printf '\n== %s\n' "$*"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --verify) VERIFY_ONLY=1; shift ;;
    --print|--dry-run) PRINT_ONLY=1; shift ;;
    --recreate) RECREATE=1; shift ;;
    --json) VERIFY_ARGS+=(--json); shift ;;
    -h|--help) sed -n '2,20p' "$SCRIPT_PATH" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option '$1' (try --help)" ;;
  esac
done

[ -x "$RUNNER" ] || [ -f "$RUNNER" ] || die "runner '$RUNNER' not found"

runner() {
  if [ "$PRINT_ONLY" = "1" ]; then
    bash "$RUNNER" --print "$@"
  else
    bash "$RUNNER" "$@"
  fi
}

# --------------------------------------------------------------------------------------------
step "1/5 host prerequisites"
# --------------------------------------------------------------------------------------------
command -v docker >/dev/null 2>&1 || die "docker not found in this WSL distro"
printf '   docker      %s\n' "$(docker --version)"

if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
  die "the 'nvidia' docker runtime is not registered; nvidia-container-toolkit is required"
fi
printf '   runtimes    nvidia runtime registered\n'

IMAGE="${AICLS_GPU_IMAGE:-codex-research-cu130:latest}"
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null || true)"
[ -n "$IMAGE_ID" ] || die "image '$IMAGE' is absent; it is not built by this project"
printf '   image       %s\n' "$IMAGE"
printf '   image id    %s\n' "$IMAGE_ID"
printf '   repodigests %s (built locally, so no registry digest exists to pin)\n' \
  "$(docker image inspect --format '{{.RepoDigests}}' "$IMAGE")"

if command -v git >/dev/null 2>&1; then
  printf '   host git    %s (bind-mounted into the container; the image ships none)\n' \
    "$(git --version)"
else
  printf '   host git    MISSING - runs inside the container will be non-citable\n'
fi

# --------------------------------------------------------------------------------------------
step "2/5 repository and data roots"
# --------------------------------------------------------------------------------------------
[ -e "$REPO/.git" ] || die "'$REPO' has no .git; the runner mounts the repository itself so that \
RunManifest can attribute a run to a commit"
printf '   repo        %s\n' "$REPO"
printf '   HEAD        %s\n' "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"

case "$AICLS_DATA_ROOT" in
  /*) ;;
  *) die "AICLS_DATA_ROOT='$AICLS_DATA_ROOT' must be an absolute POSIX path" ;;
esac
if [ "$PRINT_ONLY" != "1" ]; then
  mkdir -p "$AICLS_DATA_ROOT" "$AICLS_MODELS_ROOT" "$AICLS_DATA_ROOT/runs"
fi
printf '   data root   %s\n' "$AICLS_DATA_ROOT"
printf '   models root %s\n' "$AICLS_MODELS_ROOT"
printf '   free space  %s\n' "$(df -h "$AICLS_DATA_ROOT" 2>/dev/null | awk 'NR==2 {print $4" on "$6}')"

# --------------------------------------------------------------------------------------------
step "3/5 runtime venv ($AICLS_GPU_VENV)"
# --------------------------------------------------------------------------------------------
if [ "$RECREATE" = "1" ] && [ "$PRINT_ONLY" != "1" ] && [ "$VERIFY_ONLY" != "1" ]; then
  printf '   removing the existing venv (--recreate)\n'
  rm -rf "${AICLS_GPU_VENV:?}"
fi

# `-x .../bin/python` is useless here: that symlink targets /opt/conda inside the image and is
# broken when looked at from the host. pyvenv.cfg is the venv marker that survives both views.
if [ "$VERIFY_ONLY" = "1" ]; then
  printf '   skipped (--verify)\n'
elif [ -f "$AICLS_GPU_VENV/pyvenv.cfg" ] && [ "$PRINT_ONLY" != "1" ]; then
  printf '   present, reused\n'
else
  printf '   creating with --system-site-packages (torch stays in the image)\n'
  if [ "$PRINT_ONLY" != "1" ]; then
    mkdir -p "$(dirname "$AICLS_GPU_VENV")"
  fi
  # `--raw` runs the given argv instead of prefixing it with `aicls`, which does not exist yet.
  # --no-gpu: creating a venv needs no device, and this must work on a machine without one.
  runner --raw --no-gpu python -m venv --system-site-packages "$AICLS_GPU_VENV"
fi

# --------------------------------------------------------------------------------------------
step "4/5 editable install (the 'aicls' console script)"
# --------------------------------------------------------------------------------------------
if [ "$VERIFY_ONLY" = "1" ]; then
  printf '   skipped (--verify)\n'
else
  # --no-deps: every runtime dependency is already in the venv or in the image, and resolving
  #            them would try to reach the network.
  # --no-build-isolation: setuptools 79 + wheel are present, so the build needs no download.
  # The target is the MOUNTED path, so the editable finder points at whatever the runner mounts.
  runner --raw --no-gpu python -m pip install --no-deps --no-build-isolation --quiet \
    -e "$AICLS_GPU_WORKDIR"
  printf '   installed: aicls (editable) -> %s\n' "$AICLS_GPU_WORKDIR"
fi

# --------------------------------------------------------------------------------------------
step "5/5 verification"
# --------------------------------------------------------------------------------------------
runner --raw python "$AICLS_GPU_WORKDIR/scripts/gpu/verify_gpu.py" \
  ${VERIFY_ARGS[@]+"${VERIFY_ARGS[@]}"}

if [ "$PRINT_ONLY" != "1" ]; then
  printf '\nReady. Next:\n'
  printf '  scripts/gpu/aicls-gpu.sh train --config training/configs/ocr/smoke_synthetic.yaml \\\n'
  printf '      --synthetic --device cuda --override train.amp=bf16\n'
  printf 'That run is gate-free (synthetic data) and produces a WIRING CHECK, not a result.\n'
fi
