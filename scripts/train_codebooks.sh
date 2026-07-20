#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/codebook_eval/public_latent_codebooks.yaml}"

python scripts/codebook_eval.py train --config "${CONFIG}"

