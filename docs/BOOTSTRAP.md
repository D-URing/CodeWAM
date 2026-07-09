# Bootstrap

This project keeps external code and model artifacts out of git. Reproducible
inputs are pinned in `upstreams.yaml`; local materialized files go under
`external/`, `checkpoints/`, `data/`, and `runs/`.

## 1. Environment

Create a Python 3.10 environment with CUDA-compatible PyTorch. FastWAM upstream
currently documents:

```bash
conda create -n codewam python=3.10 -y
conda activate codewam
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```

Use the CUDA build that matches your machine if `cu128` is not appropriate.

On macOS, use the lightweight local development setup instead of the full CUDA
training environment:

```bash
bash scripts/setup_local_env.sh
source .venv/bin/activate
```

This installs CodeWAM, a sparse FastWAM checkout, and Mac-compatible local
dependencies. It is intended for code work and lightweight checks; full 5B
training/inference still belongs on a Linux CUDA machine.

## 2. Pull The Selected FastWAM Subtree

```bash
bash scripts/bootstrap_fastwam.sh
```

The script sparse-checks out:

- `src/fastwam`
- `configs`
- `scripts`

from `https://github.com/yuantianyuan01/FastWAM.git` at commit
`45d8e1458921d83f8ad6cf9ce993d371208dabd0`, then installs both FastWAM and
CodeWAM editable by default.

Useful overrides:

```bash
FASTWAM_DIR=/mnt/work/FastWAM INSTALL_EDITABLE=false bash scripts/bootstrap_fastwam.sh
```

## 3. Download Models

```bash
bash scripts/download_models.sh
```

Default model root:

```bash
checkpoints/
```

The script downloads by default:

- Wan DiT weights from `Wan-AI/Wan2.2-TI2V-5B`
- Wan VAE from `Wan-AI/Wan2.2-TI2V-5B`
- released FastWAM checkpoints and stats from `yuanty/fastwam`

Optional tokenizer/text assets:

```bash
DOWNLOAD_TEXT_ENCODER=true bash scripts/download_models.sh
```

If Hugging Face is unavailable in your environment, set FastWAM's native loader
to use ModelScope instead:

```bash
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
```

## 4. Prepare ActionDiT Backbone

```bash
bash scripts/prepare_action_dit.sh
```

This reuses FastWAM's `preprocess_action_dit_backbone.py` and writes:

```text
checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

## 5. Check

```bash
python3 scripts/check_environment.py
```

## 6. Train

```bash
bash scripts/train_zero1.sh 8 task=libero_codewam_2cam224
bash scripts/train_zero1.sh 8 task=robotwin_codewam_3cam384
```

The train launcher uses this repository's `configs/` and FastWAM's
`fastwam.runtime.run_training`.
