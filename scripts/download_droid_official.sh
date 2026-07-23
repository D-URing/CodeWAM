#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
GSUTIL_BIN="${GSUTIL:-gsutil}"
GSUTIL_PROCESSES="${GSUTIL_PROCESSES:-1}"
GSUTIL_THREADS="${GSUTIL_THREADS:-16}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-100}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-60}"
VERIFY_WORKERS="${VERIFY_WORKERS:-4}"
MODE="${1:-all}"

FULL_SOURCE="gs://gresearch/robotics/droid/1.0.1"
DEBUG_SOURCE="gs://gresearch/robotics/droid_100/1.0.0"

if [[ -z "${DATA_ROOT:-}" ]]; then
  echo "DATA_ROOT is required; refusing to choose a dataset location." >&2
  exit 2
fi
if [[ "${MODE}" != "all" &&
      "${MODE}" != "manifest" &&
      "${MODE}" != "full" &&
      "${MODE}" != "debug" &&
      "${MODE}" != "verify" ]]; then
  echo "Usage: $0 [all|manifest|full|debug|verify]" >&2
  exit 2
fi
if [[ ("${MODE}" == "all" || "${MODE}" == "full" || "${MODE}" == "debug") ]] &&
  ! command -v "${GSUTIL_BIN}" >/dev/null 2>&1; then
  echo "The '${GSUTIL_BIN}' command is unavailable." >&2
  exit 2
fi

DATA_ROOT="${DATA_ROOT%/}"
MANIFEST_ROOT="${DROID_MANIFEST_ROOT:-${DATA_ROOT}/manifests}"
FULL_ROOT="${DROID_ROOT:-${DATA_ROOT}/droid/1.0.1}"
DEBUG_ROOT="${DROID_100_ROOT:-${DATA_ROOT}/droid_100/1.0.0}"

mkdir -p "${FULL_ROOT}" "${DEBUG_ROOT}" "${MANIFEST_ROOT}"

build_manifest() {
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/droid_official.py" manifest \
    --manifest-root "${MANIFEST_ROOT}" \
    --dataset all
}

sync_dataset() {
  local source="$1"
  local destination="$2"
  local attempt
  for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
    echo "[droid] sync attempt ${attempt}/${MAX_ATTEMPTS}: ${source}"
    if "${GSUTIL_BIN}" -m \
      -o "GSUtil:parallel_process_count=${GSUTIL_PROCESSES}" \
      -o "GSUtil:parallel_thread_count=${GSUTIL_THREADS}" \
      rsync -r "${source}" "${destination}"; then
      return 0
    fi
    if ((attempt < MAX_ATTEMPTS)); then
      echo "[droid] interrupted; retrying in ${RETRY_DELAY_SECONDS}s" >&2
      sleep "${RETRY_DELAY_SECONDS}"
    fi
  done
  echo "[droid] sync failed after ${MAX_ATTEMPTS} attempts: ${source}" >&2
  return 1
}

verify_dataset() {
  local dataset="$1"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/droid_official.py" verify \
    --data-root "${DATA_ROOT}" \
    --manifest-root "${MANIFEST_ROOT}" \
    --dataset "${dataset}" \
    --workers "${VERIFY_WORKERS}"
}

case "${MODE}" in
  all)
    build_manifest
    sync_dataset "${DEBUG_SOURCE}" "${DEBUG_ROOT}"
    verify_dataset debug
    sync_dataset "${FULL_SOURCE}" "${FULL_ROOT}"
    verify_dataset full
    ;;
  manifest)
    build_manifest
    ;;
  full)
    sync_dataset "${FULL_SOURCE}" "${FULL_ROOT}"
    ;;
  debug)
    sync_dataset "${DEBUG_SOURCE}" "${DEBUG_ROOT}"
    ;;
  verify)
    verify_dataset all
    ;;
esac
