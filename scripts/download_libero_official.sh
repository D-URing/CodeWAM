#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
HF_CLI="${HF_CLI:-hf}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-4}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-30}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-30}"
VERIFY_WORKERS="${VERIFY_WORKERS:-4}"
REPO_ID="yifengzhu-hf/LIBERO-datasets"
REVISION="f13aa24a3da8c43c7225569f28c562979fa0e35a"
MODE="${1:-all}"

if [[ -z "${DATA_ROOT:-}" ]]; then
  echo "DATA_ROOT is required; refusing to choose a dataset location." >&2
  exit 2
fi
if [[ ! "${HF_MAX_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "HF_MAX_WORKERS must be a positive integer." >&2
  exit 2
fi
if [[ ! "${MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_ATTEMPTS must be a positive integer." >&2
  exit 2
fi
if [[ "${MODE}" != "all" &&
      "${MODE}" != "manifest" &&
      "${MODE}" != "download" &&
      "${MODE}" != "verify" ]]; then
  echo "Usage: $0 [all|manifest|download|verify]" >&2
  exit 2
fi
if [[ ("${MODE}" == "all" || "${MODE}" == "download") ]] &&
  ! command -v "${HF_CLI}" >/dev/null 2>&1; then
  echo "The '${HF_CLI}' command is unavailable. Install huggingface_hub first." >&2
  exit 2
fi

DATA_ROOT="${DATA_ROOT%/}"
DATA_PARENT="$(dirname "${DATA_ROOT}")"
LIBERO_ROOT="${LIBERO_ROOT:-${DATA_ROOT}/libero/official}"
MANIFEST_ROOT="${LIBERO_MANIFEST_ROOT:-${DATA_ROOT}/manifests/libero/official-f13aa24}"
export HF_HOME="${HF_HOME:-${DATA_PARENT}/cache/huggingface}"
export HF_ENDPOINT
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-1200}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

mkdir -p "${LIBERO_ROOT}" "${MANIFEST_ROOT}" "${HF_HOME}"

build_manifest() {
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/libero_official.py" manifest \
    --output-dir "${MANIFEST_ROOT}" \
    --endpoint "${HF_ENDPOINT}"
}

download_dataset() {
  local attempt
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    echo "[libero] download attempt ${attempt}/${MAX_ATTEMPTS}"
    if "${HF_CLI}" download "${REPO_ID}" \
      --repo-type dataset \
      --revision "${REVISION}" \
      --local-dir "${LIBERO_ROOT}" \
      --max-workers "${HF_MAX_WORKERS}"; then
      return 0
    fi
    if ((attempt < MAX_ATTEMPTS)); then
      echo "[libero] interrupted; retrying in ${RETRY_DELAY_SECONDS}s" >&2
      sleep "${RETRY_DELAY_SECONDS}"
    fi
  done
  echo "[libero] download failed after ${MAX_ATTEMPTS} attempts" >&2
  return 1
}

verify_dataset() {
  if [[ ! -f "${MANIFEST_ROOT}/expected_hdf5.sha256" ]]; then
    echo "Missing ${MANIFEST_ROOT}/expected_hdf5.sha256; run manifest first." >&2
    return 2
  fi
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/libero_official.py" verify \
    --root "${LIBERO_ROOT}" \
    --sha256-manifest "${MANIFEST_ROOT}/expected_hdf5.sha256" \
    --report "${MANIFEST_ROOT}/verification.json" \
    --workers "${VERIFY_WORKERS}"
}

case "${MODE}" in
  all)
    build_manifest
    download_dataset
    verify_dataset
    ;;
  manifest)
    build_manifest
    ;;
  download)
    download_dataset
    ;;
  verify)
    verify_dataset
    ;;
esac
