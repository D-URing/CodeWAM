#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${1:?Usage: bash scripts/train_zero2.sh <nproc_per_node> [hydra_overrides...]}"
shift

EXTRA_ARGS=("$@")
TASK_BASENAME="train"
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == task=* ]]; then
    TASK_BASENAME="${arg#task=}"
    TASK_BASENAME="${TASK_BASENAME%.yaml}"
  fi
done

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes "${NPROC_PER_NODE}" \
  scripts/train.py \
  "output_dir=./runs/${TASK_BASENAME}/${RUN_ID}" \
  "wandb.name=${TASK_BASENAME}" \
  "${EXTRA_ARGS[@]}"
