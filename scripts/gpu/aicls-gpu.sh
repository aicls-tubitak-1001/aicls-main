#!/usr/bin/env bash
# aicls-gpu.sh - run an aicls command on the RTX 5090 through WSL2 + nvidia-docker (ADR-0020).
#
# The Windows venv carries a CPU-only torch, so every GPU command runs inside the
# codex-research-cu130 image, which ships a Blackwell-capable torch (2.9.1+cu130, sm_120).
# This script is the single supported entry point: it builds the docker argv, mounts the
# repository *with its .git*, forces POSIX data roots and forwards the approval-gate
# environment so that a gated command behaves exactly as it does outside the container.
#
#   scripts/gpu/aicls-gpu.sh --print train --config training/configs/ocr/smoke_synthetic.yaml
#   scripts/gpu/aicls-gpu.sh train --config training/configs/ocr/smoke_synthetic.yaml --synthetic
#   scripts/gpu/aicls-gpu.sh --raw python -m pytest -q tests/unit/ml
#
# Runner options are read only until the first argument that is not one of them; everything
# from there on is handed to `aicls` untouched, so `aicls train --dry-run` keeps its own flag.
#
# Documentation: scripts/gpu/README.md, docs/adr/0020-gpu-calisma-ortami-wsl-nvidia-docker.md,
# docs/gelistirme/kurulum.md #7, docs/ml/egitim-el-kitabi.md #2.
set -euo pipefail

# --------------------------------------------------------------------------------------------
# defaults (every one of them overridable from the environment or a runner option)
# --------------------------------------------------------------------------------------------
# The image is NOT ours: it was built locally on 2026-04-30 and has no RepoDigests, so it cannot
# be pinned by a registry digest. We pin its content-addressable image ID instead and refuse to
# run when the tag has moved (ADR-0020 "Sonuçlar").
AICLS_GPU_IMAGE="${AICLS_GPU_IMAGE:-codex-research-cu130:latest}"
AICLS_GPU_IMAGE_ID="${AICLS_GPU_IMAGE_ID:-sha256:a3359cc37244fe5cc7b1acb7eca12ccb63e44d9bda26b866cee510e747fd84d4}"
AICLS_GPU_VENV="${AICLS_GPU_VENV:-/mnt/d/aicls-venv}"
AICLS_GPU_WORKDIR="${AICLS_GPU_WORKDIR:-/workspace/ai-cls}"
AICLS_GPU_GPUS="${AICLS_GPU_GPUS:-all}"
AICLS_GPU_SHM_SIZE="${AICLS_GPU_SHM_SIZE:-}"
AICLS_GPU_STACK_ULIMIT="${AICLS_GPU_STACK_ULIMIT:-67108864}"
AICLS_GPU_MOUNT_GIT="${AICLS_GPU_MOUNT_GIT:-1}"
AICLS_GPU_ALLOW_IMAGE_DRIFT="${AICLS_GPU_ALLOW_IMAGE_DRIFT:-0}"

#: Gate variables are forwarded verbatim. A gate must never become weaker inside Docker:
#: AICLS_GATE_* carries the "env" condition of the triple lock and CI/GITHUB_ACTIONS carry the
#: unconditional CI refusal (ADR-0012). Dropping either would turn `docker run` into a bypass.
GATE_ENV_NAMES=(
  AICLS_GATE_DATASET_DOWNLOAD
  AICLS_GATE_WEIGHTS_DOWNLOAD
  AICLS_GATE_CUDA_INSTALL
  AICLS_GATE_TRAINING
  CI
  GITHUB_ACTIONS
)

#: Ordinary settings that may travel with the operator's shell.
PASSTHROUGH_ENV_NAMES=(
  AICLS_ENV
  AICLS_PROFILE
  AICLS_LOG_LEVEL
  AICLS_TEST_GPU
  AICLS_TEST_DATA
  AICLS_ETHICS_APPROVAL_REF
)

PROGRAM_NAME="$(basename "${BASH_SOURCE[0]}")"

die() {
  printf '%s: %s\n' "$PROGRAM_NAME" "$*" >&2
  exit 2
}

usage() {
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'USAGE'

Runner options (before the aicls command):
  --print, --dry-run     print the exact docker command and exit without running it
  --raw                  run the given argv directly instead of prefixing it with `aicls`
  --shell                open an interactive bash in the container (implies --raw bash)
  --repo PATH            repository to mount (default: the repo this script lives in)
  --image REF            container image tag (default: codex-research-cu130:latest)
  --data-root PATH       AICLS_DATA_ROOT inside the container (must be absolute POSIX)
  --models-root PATH     AICLS_MODELS_ROOT inside the container (must be absolute POSIX)
  --venv PATH            runtime venv inside the container (default: /mnt/d/aicls-venv)
  --gpus SPEC            docker --gpus value (default: all)
  --no-gpu               drop --gpus entirely (CPU parity run)
  --name NAME            docker container name
  -h, --help             this text
USAGE
}

# --------------------------------------------------------------------------------------------
# path validation
# --------------------------------------------------------------------------------------------
# A Windows path leaks into the container as a RELATIVE PosixPath: Settings._absolute_data_root
# would resolve "D:\aicls-data" against the working directory and silently write weights to
# <repo>/D:\aicls-data. That corrupts provenance quietly, so it is a hard error here.
require_posix_abs() {
  local name="$1" value="$2"
  [ -n "$value" ] || die "$name is empty; it must be an absolute POSIX path inside the container"
  case "$value" in
    *\\*) die "$name='$value' contains a backslash; use a POSIX path such as /mnt/d/aicls-data" ;;
    [A-Za-z]:*) die "$name='$value' is a Windows path; inside the container use /mnt/<drive>/..." ;;
    /*) ;;
    *) die "$name='$value' is relative; an absolute POSIX path is required" ;;
  esac
}

# --------------------------------------------------------------------------------------------
# option parsing - stops at the first argument that is not a runner option
# --------------------------------------------------------------------------------------------
PRINT_ONLY=0
RAW=0
CONTAINER_NAME=""
USE_GPU=1
REPO="${AICLS_GPU_REPO:-}"
SHELL_MODE=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --print|--dry-run) PRINT_ONLY=1; shift ;;
    --raw) RAW=1; shift ;;
    --shell) SHELL_MODE=1; RAW=1; shift ;;
    --repo) [ "$#" -ge 2 ] || die "--repo needs a value"; REPO="$2"; shift 2 ;;
    --image) [ "$#" -ge 2 ] || die "--image needs a value"; AICLS_GPU_IMAGE="$2"; shift 2 ;;
    --data-root) [ "$#" -ge 2 ] || die "--data-root needs a value"; AICLS_DATA_ROOT="$2"; shift 2 ;;
    --models-root) [ "$#" -ge 2 ] || die "--models-root needs a value"; AICLS_MODELS_ROOT="$2"; shift 2 ;;
    --venv) [ "$#" -ge 2 ] || die "--venv needs a value"; AICLS_GPU_VENV="$2"; shift 2 ;;
    --gpus) [ "$#" -ge 2 ] || die "--gpus needs a value"; AICLS_GPU_GPUS="$2"; shift 2 ;;
    --no-gpu) USE_GPU=0; shift ;;
    --name) [ "$#" -ge 2 ] || die "--name needs a value"; CONTAINER_NAME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) break ;;
  esac
done

# --------------------------------------------------------------------------------------------
# repository (mounted WITH .git - a run that cannot be attributed to a commit is not citable)
# --------------------------------------------------------------------------------------------
if [ -z "$REPO" ]; then
  script_path="${BASH_SOURCE[0]}"
  if command -v readlink >/dev/null 2>&1; then
    script_path="$(readlink -f "$script_path" 2>/dev/null || printf '%s' "$script_path")"
  fi
  REPO="$(cd "$(dirname "$script_path")/../.." && pwd)"
fi
[ -d "$REPO" ] || die "repository '$REPO' does not exist"
REPO="$(cd "$REPO" && pwd)"
# RunManifest.capture() shells out to real `git`; without .git the run records git_available
# false, citable false, and cannot be quoted in the TUBITAK progress report (ADR-0007).
[ -e "$REPO/.git" ] || die "'$REPO' has no .git; mount the repository itself, not a copy of it \
(RunManifest refuses to attribute a run it cannot map to a commit)"

AICLS_DATA_ROOT="${AICLS_DATA_ROOT:-/mnt/d/aicls-data}"
AICLS_MODELS_ROOT="${AICLS_MODELS_ROOT:-${AICLS_DATA_ROOT}/models}"
require_posix_abs AICLS_DATA_ROOT "$AICLS_DATA_ROOT"
require_posix_abs AICLS_MODELS_ROOT "$AICLS_MODELS_ROOT"
require_posix_abs AICLS_GPU_VENV "$AICLS_GPU_VENV"
require_posix_abs AICLS_GPU_WORKDIR "$AICLS_GPU_WORKDIR"

command -v docker >/dev/null 2>&1 || die "docker not found; run this inside the WSL distro \
that has nvidia-docker (see scripts/gpu/README.md)"

# --------------------------------------------------------------------------------------------
# image pin
# --------------------------------------------------------------------------------------------
actual_image_id="$(docker image inspect --format '{{.Id}}' "$AICLS_GPU_IMAGE" 2>/dev/null || true)"
if [ -z "$actual_image_id" ]; then
  die "image '$AICLS_GPU_IMAGE' is not present locally; it is not built by this project and \
there is nothing to pull it from (see docs/adr/0020-gpu-calisma-ortami-wsl-nvidia-docker.md)"
fi
if [ -n "$AICLS_GPU_IMAGE_ID" ] && [ "$actual_image_id" != "$AICLS_GPU_IMAGE_ID" ]; then
  if [ "$AICLS_GPU_ALLOW_IMAGE_DRIFT" = "1" ]; then
    printf '%s: WARNING image drift accepted: %s is %s, pin says %s\n' \
      "$PROGRAM_NAME" "$AICLS_GPU_IMAGE" "$actual_image_id" "$AICLS_GPU_IMAGE_ID" >&2
  else
    die "image '$AICLS_GPU_IMAGE' is $actual_image_id but the pin says $AICLS_GPU_IMAGE_ID; \
a moved tag changes the torch/CUDA stack under every recorded run. Re-pin AICLS_GPU_IMAGE_ID in \
this script and in ADR-0020, or set AICLS_GPU_ALLOW_IMAGE_DRIFT=1 for a one-off"
  fi
fi

# --------------------------------------------------------------------------------------------
# mounts (deduplicated: a path already covered by an earlier mount is skipped)
# --------------------------------------------------------------------------------------------
MOUNT_ARGS=()
IDENTITY_MOUNTS=()

# add_mount SRC DST [ro]
# Identity mounts (DST == SRC) that already sit inside another identity mount are skipped, so
# /mnt/d/aicls-data/models does not get a second bind under /mnt/d/aicls-data.
add_mount() {
  local src="$1" dst="$2" mode="${3:-rw}" existing
  if [ ! -e "$src" ]; then
    return 0
  fi
  if [ "$dst" = "$src" ]; then
    for existing in ${IDENTITY_MOUNTS[@]+"${IDENTITY_MOUNTS[@]}"}; do
      if [ "$src" = "$existing" ] || [ "${src#"${existing}"/}" != "$src" ]; then
        return 0
      fi
    done
    IDENTITY_MOUNTS+=("$src")
  fi
  if [ "$mode" = "ro" ]; then
    MOUNT_ARGS+=(-v "${src}:${dst}:ro")
  else
    MOUNT_ARGS+=(-v "${src}:${dst}")
  fi
  return 0
}

mkdir -p "$AICLS_DATA_ROOT" "$AICLS_MODELS_ROOT"
add_mount "$REPO" "$AICLS_GPU_WORKDIR"
add_mount "$AICLS_GPU_VENV" "$AICLS_GPU_VENV"
add_mount "$AICLS_DATA_ROOT" "$AICLS_DATA_ROOT"
add_mount "$AICLS_MODELS_ROOT" "$AICLS_MODELS_ROOT"

# The image ships no git at all (verified 2026-09-08: /usr/bin/git absent). RunManifest then
# records git_available=false and every run is non-citable. Host and image are both Ubuntu
# 22.04, so the host git binary runs unmodified inside the container; bind-mounting it is the
# cheapest way to keep provenance working without rebuilding an image we do not own.
if [ "$AICLS_GPU_MOUNT_GIT" = "1" ] && [ -x /usr/bin/git ]; then
  add_mount /usr/bin/git /usr/bin/git ro
  add_mount /usr/lib/git-core /usr/lib/git-core ro
  add_mount /usr/share/git-core /usr/share/git-core ro
fi

# --------------------------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------------------------
ENV_ARGS=(
  -e "HOME=/tmp"
  -e "USER=${USER:-aicls}"
  -e "LOGNAME=${LOGNAME:-${USER:-aicls}}"
  -e "PYTHONPATH=${AICLS_GPU_WORKDIR}/src"
  -e "PYTHONPYCACHEPREFIX=/tmp/aicls-pycache"
  -e "PYTHONUTF8=1"
  -e "PYTHONDONTWRITEBYTECODE="
  -e "AICLS_DATA_ROOT=${AICLS_DATA_ROOT}"
  -e "AICLS_MODELS_ROOT=${AICLS_MODELS_ROOT}"
  -e "AICLS_GPU_VENV_BIN=${AICLS_GPU_VENV}/bin"
  -e "MPLCONFIGDIR=/tmp/mplconfig"
  -e "XDG_CACHE_HOME=/tmp/cache"
)

# git configuration is injected through the environment so that nothing is written into the
# mounted tree. safe.directory keeps git from refusing the bind mount; core.excludesfile
# restores Windows parity for the junctions that Linux git sees as untracked symlinks
# (web/node_modules), which would otherwise mark every containerised run dirty.
ENV_ARGS+=(
  -e "GIT_CONFIG_COUNT=2"
  -e "GIT_CONFIG_KEY_0=safe.directory"
  -e "GIT_CONFIG_VALUE_0=${AICLS_GPU_WORKDIR}"
  -e "GIT_CONFIG_KEY_1=core.excludesfile"
  -e "GIT_CONFIG_VALUE_1=${AICLS_GPU_WORKDIR}/scripts/gpu/container-excludes"
)

for name in "${GATE_ENV_NAMES[@]}" "${PASSTHROUGH_ENV_NAMES[@]}"; do
  if [ -n "${!name-}" ]; then
    ENV_ARGS+=(-e "${name}=${!name}")
  fi
done

# --------------------------------------------------------------------------------------------
# docker argv
# --------------------------------------------------------------------------------------------
DOCKER_ARGS=(docker run --rm)
if [ -t 0 ] && [ -t 1 ]; then
  DOCKER_ARGS+=(-it)
elif [ "$SHELL_MODE" = "1" ]; then
  DOCKER_ARGS+=(-i)
fi
if [ -n "$CONTAINER_NAME" ]; then
  DOCKER_ARGS+=(--name "$CONTAINER_NAME")
fi
if [ "$USE_GPU" = "1" ]; then
  DOCKER_ARGS+=(--gpus "$AICLS_GPU_GPUS")
fi
# --ipc=host gives PyTorch dataloader workers the host's /dev/shm; the NGC images warn that the
# 64 MB default starves them. The ulimits are the matching NGC recommendation.
DOCKER_ARGS+=(--ipc=host --ulimit "memlock=-1" --ulimit "stack=${AICLS_GPU_STACK_ULIMIT}")
if [ -n "$AICLS_GPU_SHM_SIZE" ]; then
  DOCKER_ARGS+=(--shm-size "$AICLS_GPU_SHM_SIZE")
fi
# Files written under /mnt/d must stay usable from Windows and from WSL: run as the calling user.
DOCKER_ARGS+=(--user "$(id -u):$(id -g)")
DOCKER_ARGS+=("${MOUNT_ARGS[@]}")
DOCKER_ARGS+=(-w "$AICLS_GPU_WORKDIR")
DOCKER_ARGS+=("${ENV_ARGS[@]}")
DOCKER_ARGS+=("$AICLS_GPU_IMAGE")

# The prelude puts the runtime venv first on PATH and creates the writable scratch directories a
# non-root user needs. Arguments arrive as "$@", so no user argument is ever re-parsed by a shell.
PRELUDE='set -e
export PATH="${AICLS_GPU_VENV_BIN}:${PATH}"
mkdir -p "${PYTHONPYCACHEPREFIX}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" 2>/dev/null || true
exec "$@"'

if [ "$SHELL_MODE" = "1" ] && [ "$#" -eq 0 ]; then
  set -- bash
fi
[ "$#" -gt 0 ] || die "nothing to run; try: $PROGRAM_NAME --help"

if [ "$RAW" = "1" ]; then
  CONTAINER_ARGV=("$@")
else
  CONTAINER_ARGV=(aicls "$@")
fi

DOCKER_ARGS+=(/bin/bash -c "$PRELUDE" aicls-gpu "${CONTAINER_ARGV[@]}")

if [ "$PRINT_ONLY" = "1" ]; then
  printf '# repo      %s\n' "$REPO"
  printf '# image     %s (%s)\n' "$AICLS_GPU_IMAGE" "$actual_image_id"
  printf '# data root %s (created above if it was missing - the one side effect of --print,\n' \
    "$AICLS_DATA_ROOT"
  printf '#           so that the printed mounts are the mounts a real run would get)\n'
  printf '# the docker command below was NOT executed\n'
  printf '%q ' "${DOCKER_ARGS[@]}"
  printf '\n'
  exit 0
fi

exec "${DOCKER_ARGS[@]}"
