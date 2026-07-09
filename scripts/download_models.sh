#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
MODEL_ROOT="${DIFFSYNTH_MODEL_BASE_PATH:-${ROOT_DIR}/checkpoints}"
INSTALL_DOWNLOAD_DEPS="${INSTALL_DOWNLOAD_DEPS:-true}"
DOWNLOAD_TEXT_ENCODER="${DOWNLOAD_TEXT_ENCODER:-false}"
DOWNLOAD_FASTWAM_RELEASE="${DOWNLOAD_FASTWAM_RELEASE:-true}"
HF_CLI="${HF_CLI:-}"

export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_ROOT}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf}"
PYTHON_BIN_DIR="$(cd "$(dirname "${PYTHON_BIN}")" 2>/dev/null && pwd || true)"
if [[ -n "${PYTHON_BIN_DIR}" ]]; then
  export PATH="${PYTHON_BIN_DIR}:${PATH}"
fi

mkdir -p "${MODEL_ROOT}" "${HF_HOME}"

if [[ "${INSTALL_DOWNLOAD_DEPS}" == "true" ]]; then
  if ! "${PYTHON_BIN}" -m pip install -U huggingface_hub; then
    if command -v uv >/dev/null 2>&1; then
      uv pip install --python "${PYTHON_BIN}" -U huggingface_hub
    else
      echo "Could not install huggingface_hub; install pip or uv first." >&2
      exit 1
    fi
  fi
fi

if [[ -z "${HF_CLI}" ]]; then
  if command -v hf >/dev/null 2>&1; then
    HF_CLI="hf"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI="huggingface-cli"
  else
    echo "Neither `hf` nor `huggingface-cli` is available." >&2
    exit 1
  fi
fi

download_hf() {
  local repo="$1"
  local local_dir="$2"
  shift 2

  mkdir -p "${local_dir}"
  echo "[download] ${repo} -> ${local_dir}"
  "${HF_CLI}" download "${repo}" "$@" --local-dir "${local_dir}"
}

download_hf \
  "Wan-AI/Wan2.2-TI2V-5B" \
  "${MODEL_ROOT}/Wan-AI/Wan2.2-TI2V-5B" \
  --include "diffusion_pytorch_model*.safetensors" \
  --include "Wan2.2_VAE.pth"

if [[ "${DOWNLOAD_TEXT_ENCODER}" == "true" ]]; then
  download_hf \
    "Wan-AI/Wan2.2-TI2V-5B" \
    "${MODEL_ROOT}/Wan-AI/Wan2.2-TI2V-5B" \
    --include "models_t5_umt5-xxl-enc-bf16.pth"

  download_hf \
    "Wan-AI/Wan2.1-T2V-1.3B" \
    "${MODEL_ROOT}/Wan-AI/Wan2.1-T2V-1.3B" \
    --include "google/umt5-xxl/**"
fi

if [[ "${DOWNLOAD_FASTWAM_RELEASE}" == "true" ]]; then
  download_hf \
    "yuanty/fastwam" \
    "${MODEL_ROOT}/fastwam_release" \
    --include "libero_uncond_2cam224.pt" \
    --include "libero_uncond_2cam224_dataset_stats.json" \
    --include "robotwin_uncond_3cam_384.pt" \
    --include "robotwin_uncond_3cam_384_dataset_stats.json"
fi

echo "[download] done. Export this before training:"
echo "export DIFFSYNTH_MODEL_BASE_PATH=\"${MODEL_ROOT}\""
