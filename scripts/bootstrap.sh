#!/usr/bin/env bash
# Bootstrap the AI-CLS development environment on Linux / WSL2 / the collection VM (ADR-0011).
# - venv outside any synced folder (default: ~/.venvs/ai-cls)
# - pip install -e .[dev]  (pure-Python dev stack; CPU torch wheel from the PyTorch CPU index)
# - pre-commit hooks, data root (default: ~/aicls-data or $AICLS_DATA_ROOT)
# Nothing gated happens here (no CUDA, no datasets, no weights — ADR-0012).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${AICLS_VENV:-$HOME/.venvs/ai-cls}"
DATA_ROOT="${AICLS_DATA_ROOT:-$HOME/aicls-data}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

cd "$REPO_ROOT"

case "$VENV_PATH" in
  *OneDrive*|*onedrive*) echo "Refusing to create the venv under OneDrive: $VENV_PATH" >&2; exit 1 ;;
esac

PY="${PYTHON:-python3.11}"
if ! command -v "$PY" >/dev/null 2>&1; then PY=python3; fi
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version' || { echo "Python 3.11 required" >&2; exit 1; }

if [ ! -x "$VENV_PATH/bin/python" ]; then
  echo "Creating venv at $VENV_PATH"
  "$PY" -m venv --system-site-packages "$VENV_PATH"
fi
VPY="$VENV_PATH/bin/python"

mkdir -p "$DATA_ROOT"/{models,runs,mlruns,audio/raw,datasets}
echo "Data root: $DATA_ROOT"

if [ "$SKIP_INSTALL" != "1" ]; then
  "$VPY" -m pip install --upgrade pip
  if [ -f requirements/base.txt ]; then
    "$VPY" -m pip install --require-hashes -r requirements/base.txt -r requirements/dev.txt
    "$VPY" -m pip install --no-deps -e .
  else
    "$VPY" -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev]"  # gate-ok: CPU wheel index only, no CUDA
  fi
  "$VPY" -m pre_commit install --hook-type pre-commit --hook-type commit-msg --hook-type pre-push
fi

if [ ! -f .env ]; then
  cp .env.example .env
  SECRET="$("$VPY" -c 'import secrets;print(secrets.token_hex(32))')"
  sed -i.bak -e "s|^AICLS_JWT_SECRET=$|AICLS_JWT_SECRET=$SECRET|" -e "s|^AICLS_DATA_ROOT=.*$|AICLS_DATA_ROOT=$DATA_ROOT|" .env && rm -f .env.bak
  chmod 600 .env
  echo "Created .env with a fresh AICLS_JWT_SECRET"
fi

cat <<EOF

Done. Next:
  source "$VENV_PATH/bin/activate"
  poe env-check --ci ; poe test-fast
  (GPU stack is GATED: scripts/install_ml.ps1 / WSL2 equivalent after the ledger row — ADR-0012)
EOF
