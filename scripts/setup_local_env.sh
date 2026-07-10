#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
fi

"${UV_BIN}" python install "${PYTHON_VERSION}"
"${UV_BIN}" venv "${VENV_DIR}" --python "${PYTHON_VERSION}"

"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" -e "${ROOT_DIR}"

INSTALL_EDITABLE=false "${ROOT_DIR}/scripts/bootstrap_fastwam.sh"
"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" -e "${ROOT_DIR}/external/FastWAM" --no-deps

"${UV_BIN}" pip install --python "${VENV_DIR}/bin/python" -r "${ROOT_DIR}/requirements/local-dev.txt"

echo "Local development environment is ready."
echo "Activate it with:"
echo "source ${VENV_DIR}/bin/activate"
