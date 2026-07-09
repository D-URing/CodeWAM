#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
MODEL_ROOT="${DIFFSYNTH_MODEL_BASE_PATH:-${ROOT_DIR}/checkpoints}"
INSTALL_DOWNLOAD_DEPS="${INSTALL_DOWNLOAD_DEPS:-true}"
DOWNLOAD_TEXT_ENCODER="${DOWNLOAD_TEXT_ENCODER:-false}"
DOWNLOAD_FASTWAM_RELEASE="${DOWNLOAD_FASTWAM_RELEASE:-true}"

export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_ROOT}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf}"

mkdir -p "${MODEL_ROOT}" "${HF_HOME}"

if [[ "${INSTALL_DOWNLOAD_DEPS}" == "true" ]]; then
  "${PYTHON_BIN}" -m pip install -U huggingface_hub
fi

download_hf() {
  local repo="$1"
  local local_dir="$2"
  shift 2

  mkdir -p "${local_dir}"
  echo "[download] ${repo} -> ${local_dir}"
  huggingface-cli download "${repo}" "$@" --local-dir "${local_dir}"
}

download_hf \
  "Wan-AI/Wan2.2-TI2V-5B" \
  "${MODEL_ROOT}/Wan-AI/Wan2.2-TI2V-5B" \
  --include "diffusion_pytorch_model*.safetensors"

download_hf \
  "DiffSynth-Studio/Wan-Series-Converted-Safetensors" \
  "${MODEL_ROOT}/DiffSynth-Studio/Wan-Series-Converted-Safetensors" \
  --include "Wan2.2_VAE.safetensors"

if [[ "${DOWNLOAD_TEXT_ENCODER}" == "true" ]]; then
  download_hf \
    "DiffSynth-Studio/Wan-Series-Converted-Safetensors" \
    "${MODEL_ROOT}/DiffSynth-Studio/Wan-Series-Converted-Safetensors" \
    --include "models_t5_umt5-xxl-enc-bf16.safetensors"

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
