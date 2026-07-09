#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTWAM_REPO="${FASTWAM_REPO:-https://github.com/yuantianyuan01/FastWAM.git}"
FASTWAM_COMMIT="${FASTWAM_COMMIT:-45d8e1458921d83f8ad6cf9ce993d371208dabd0}"
FASTWAM_DIR="${FASTWAM_DIR:-${ROOT_DIR}/external/FastWAM}"
PYTHON_BIN="${PYTHON:-python3}"
INSTALL_EDITABLE="${INSTALL_EDITABLE:-true}"
GIT_HTTP_VERSION="${GIT_HTTP_VERSION:-HTTP/1.1}"

mkdir -p "$(dirname "${FASTWAM_DIR}")"

if [[ -d "${FASTWAM_DIR}/.git" ]]; then
  git -C "${FASTWAM_DIR}" -c http.version="${GIT_HTTP_VERSION}" fetch --depth 1 origin "${FASTWAM_COMMIT}"
else
  git -c http.version="${GIT_HTTP_VERSION}" clone --filter=blob:none --sparse "${FASTWAM_REPO}" "${FASTWAM_DIR}"
fi

git -C "${FASTWAM_DIR}" sparse-checkout set src/fastwam configs scripts
git -C "${FASTWAM_DIR}" -c http.version="${GIT_HTTP_VERSION}" fetch --depth 1 origin "${FASTWAM_COMMIT}"
git -C "${FASTWAM_DIR}" checkout --detach "${FASTWAM_COMMIT}"

echo "[bootstrap] FastWAM ready at ${FASTWAM_DIR}"
echo "[bootstrap] FastWAM commit $(git -C "${FASTWAM_DIR}" rev-parse HEAD)"

if [[ "${INSTALL_EDITABLE}" == "true" ]]; then
  "${PYTHON_BIN}" -m pip install -e "${FASTWAM_DIR}"
  "${PYTHON_BIN}" -m pip install -e "${ROOT_DIR}"
fi
