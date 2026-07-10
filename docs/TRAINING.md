# Training And Cluster Handoff

CodeWAM is intended to be developed locally and trained on a Linux CUDA cluster.
The local macOS environment is only for code work, config composition, and light
smoke checks.

## Local Development

```bash
bash scripts/setup_local_env.sh
source .venv/bin/activate
python scripts/check_environment.py --mode local
```

Local mode expects:

- `codewam` import works
- sparse FastWAM dependency is importable
- lightweight Python dependencies are installed

Local mode does not require:

- Deepspeed
- ActionDiT backbone
- complete training datasets
- RoboTwin / 3cam release checkpoint

## Package Scan V6 Local Demo

`package_scan_v6/` is the current local real-robot demo dataset. It is ignored
by git and should stay on the local machine or be mounted on the cluster.

Run a data smoke demo:

```bash
python scripts/demo_package_scan_v6.py
```

This checks the LeRobot v3 parquet metadata, decodes the AV1 top/wrist videos,
builds CodeWAM-style windows, and writes a preview strip under
`runs/package_scan_v6_demo/`.

## Cluster Setup

On the training cluster:

```bash
python -m venv .venv
source .venv/bin/activate
bash scripts/setup_cluster_env.sh
```

Useful overrides:

```bash
PYTHON=/path/to/python \
FASTWAM_DIR=/path/to/FastWAM \
DIFFSYNTH_MODEL_BASE_PATH=/path/to/models \
bash scripts/setup_cluster_env.sh
```

Check readiness:

```bash
python scripts/check_environment.py --mode cluster
```

Cluster mode expects:

- CUDA torch/deepspeed stack
- `fastwam` and `codewam`
- Wan DiT and VAE files
- generated ActionDiT backbone

## Model Files

Default download:

```bash
bash scripts/download_models.sh
```

Stable low-concurrency download:

```bash
HF_MAX_WORKERS=1 HF_DISABLE_XET=true bash scripts/download_models.sh
```

Optional text encoder/tokenizer assets:

```bash
DOWNLOAD_TEXT_ENCODER=true bash scripts/download_models.sh
```

Optional RoboTwin / 3cam release checkpoint:

```bash
DOWNLOAD_ROBOTWIN_RELEASE=true bash scripts/download_models.sh
```

## ActionDiT Backbone

Generate this on a CUDA machine:

```bash
bash scripts/prepare_action_dit.sh
```

Expected output:

```text
checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

## Train

LIBERO example:

```bash
bash scripts/train_zero1.sh 8 task=libero_codewam_2cam224
```

RoboTwin example:

```bash
bash scripts/train_zero1.sh 8 task=robotwin_codewam_3cam384
```

## Probe

Run the RQ state-codebook P1/P2/P4 probe:

```bash
python -m codewam.probe \
  --config-dir configs \
  --task libero_codewam_2cam224 \
  --model-root checkpoints \
  --output runs/probes/libero_codewam_probe.json
```

For a cluster-specific real robot task:

```bash
python -m codewam.probe \
  --config-dir /path/to/FastWAM/configs \
  --task real_robot_joint_2cam224_v6_clean \
  --model-root /path/to/models \
  --output /path/to/runs/probes/real_robot_codewam_probe.json
```
