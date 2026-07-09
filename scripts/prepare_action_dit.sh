#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTWAM_DIR="${FASTWAM_DIR:-${ROOT_DIR}/external/FastWAM}"
PYTHON_BIN="${PYTHON:-python3}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
MODEL_CONFIG="${MODEL_CONFIG:-${ROOT_DIR}/configs/model/codewam.yaml}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"

if [[ ! -f "${FASTWAM_DIR}/scripts/preprocess_action_dit_backbone.py" ]]; then
  echo "FastWAM script not found. Run scripts/bootstrap_fastwam.sh first." >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT}")"

"${PYTHON_BIN}" "${FASTWAM_DIR}/scripts/preprocess_action_dit_backbone.py" \
  --model-config "${MODEL_CONFIG}" \
  --output "${OUTPUT}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}"
