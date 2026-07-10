#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
INSTALL_TORCH_STACK="${INSTALL_TORCH_STACK:-true}"
BOOTSTRAP_FASTWAM="${BOOTSTRAP_FASTWAM:-true}"
INSTALL_EDITABLE_PACKAGES="${INSTALL_EDITABLE_PACKAGES:-true}"

if [[ "${BOOTSTRAP_FASTWAM}" == "true" ]]; then
  INSTALL_EDITABLE=false "${ROOT_DIR}/scripts/bootstrap_fastwam.sh"
fi

if [[ "${INSTALL_TORCH_STACK}" == "true" ]]; then
  "${PYTHON_BIN}" -m pip install -U pip setuptools wheel
  "${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/requirements/cluster-cu128.txt"
fi

if [[ "${INSTALL_EDITABLE_PACKAGES}" == "true" ]]; then
  "${PYTHON_BIN}" -m pip install -e "${ROOT_DIR}/external/FastWAM" --no-deps
  "${PYTHON_BIN}" -m pip install -e "${ROOT_DIR}"
fi

echo "Cluster environment is ready for checks."
echo "Recommended next step:"
echo "${PYTHON_BIN} ${ROOT_DIR}/scripts/check_environment.py --mode cluster"
