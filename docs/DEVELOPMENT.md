# Development and Operations

本文件统一说明仓库结构、本机开发、集群环境、外部依赖、模型文件和运行命令。架构规范见
`ARCHITECTURE.md`;码本数据与实验流程见 `CODEBOOK.md`。

## 1. 仓库结构

```text
CodeWAM/
├── codewam/
│   ├── codebook.py            # legacy online-EMA prototype; disabled by default
│   ├── codebook_eval/
│   │   ├── manifest.py        # episode provenance and deterministic split
│   │   ├── shards.py          # pooled episode shard contract
│   │   ├── streaming.py       # causal descriptors and streaming RQ
│   │   └── pipeline.py        # canonical Q2/Q3/Q5 launcher
│   ├── model.py               # current FastWAM-compatible prototype
│   ├── probe.py               # legacy compatibility probe
│   └── runtime.py             # Hydra factory
├── configs/                   # model, data, task, and codebook configs
├── scripts/                   # setup, checks, export, clustering, and training
├── requirements/              # local and CUDA dependency sets
├── docs/                      # three canonical documents
├── external/                  # ignored external checkouts
├── checkpoints/               # ignored model files
├── data/                      # ignored small/local datasets
├── runs/                      # ignored outputs
└── upstreams.yaml             # pinned repositories and revisions
```

大型公开数据集放在独立共享数据根目录,不复制到仓库。下载、校验和训练 artifact 也不进入 git。

## 2. 本机开发

macOS 使用项目内轻量环境:

```bash
bash scripts/setup_local_env.sh
source .venv/bin/activate
python scripts/check_environment.py --mode local
python -m unittest discover -s tests -v
```

该环境用于代码、配置、单元测试和小规模 MPS/CPU smoke,不承担 5B 模型训练。

Package Scan v6 是本机回归数据,目录保持为 `package_scan_v6/` 且不入库:

```bash
python scripts/demo_package_scan_v6.py
```

它检查 LeRobot v3 parquet、AV1 top/wrist 视频和窗口构造,预览写入
`runs/package_scan_v6_demo/`。它不能支撑正式 codebook 研究结论。

## 3. 集群环境

共享训练集群默认复用管理员提供的 Python、CUDA 和 PyTorch,不要求在共享盘创建项目虚拟环境。
先检查现有栈:

```bash
python - <<'PY'
import sys
import torch
print(sys.executable)
print(torch.__version__)
print(torch.cuda.is_available(), torch.cuda.device_count())
PY
```

已有兼容 torch 时,只安装项目和缺失依赖:

```bash
PYTHON=python \
INSTALL_TORCH_STACK=false \
bash scripts/setup_cluster_env.sh
```

只有明确需要由项目安装 CUDA 依赖时才使用默认 `INSTALL_TORCH_STACK=true`。可覆盖路径:

```bash
PYTHON=/path/to/python \
FASTWAM_DIR=/path/to/FastWAM \
DIFFSYNTH_MODEL_BASE_PATH=/path/to/models \
INSTALL_TORCH_STACK=false \
bash scripts/setup_cluster_env.sh
```

随后运行:

```bash
python scripts/check_environment.py --mode cluster
```

## 4. FastWAM 边界

固定上游:

```bash
bash scripts/bootstrap_fastwam.sh
```

脚本从 `https://github.com/yuantianyuan01/FastWAM.git` sparse-checkout
`src/fastwam`、`configs` 和 `scripts`,revision 由 `upstreams.yaml` 固定。默认 materialize 到
`external/FastWAM/` 并 editable install。

已有独立 checkout 时:

```bash
FASTWAM_DIR=/mnt/work/FastWAM \
INSTALL_EDITABLE=false \
bash scripts/bootstrap_fastwam.sh
```

CodeWAM owns:

- frozen Q2/Q3/Q5 artifacts and nine-measurement interface;
- continuous-state plus code aggregation;
- Policy/Forward-Dynamics/Video-Prior mask program;
- future-code objective and codebook evaluation pipeline;
- CodeWAM configs and tests.

FastWAM currently provides Wan-VAE/Video DiT、ActionDiT、MoT、flow-matching scheduler、dataset
processor 和兼容训练 runtime。它是依赖与实验基线,不是 CodeWAM 的结构上限。

## 5. 模型文件

完整 compatible 模型准备:

```bash
bash scripts/download_models.sh
```

默认模型根目录为 `checkpoints/`,可通过 `DIFFSYNTH_MODEL_BASE_PATH` 覆盖。低并发下载:

```bash
HF_MAX_WORKERS=1 \
HF_DISABLE_XET=true \
bash scripts/download_models.sh
```

可选资源:

```bash
DOWNLOAD_TEXT_ENCODER=true bash scripts/download_models.sh
DOWNLOAD_ROBOTWIN_RELEASE=true bash scripts/download_models.sh
```

RoboTwin/3-camera checkpoint 对 codebook Gate 0-2 不是依赖。离线 latent export 只需要固定版本
Wan-VAE、对应 loader 和预处理配置;完整策略训练才需要 Wan DiT 与 ActionDiT。

无法访问 Hugging Face 且 FastWAM loader 支持 ModelScope 时:

```bash
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
```

## 6. Codebook 开发

canonical backend 回归:

```bash
python -m unittest discover -s tests -v
python scripts/train_streaming_codebooks.py smoke \
  --output runs/codebook_eval/streaming_smoke
```

真实 pooled shards 就绪后:

```bash
python scripts/train_streaming_codebooks.py train \
  --config configs/codebook_eval/streaming_rq_template.yaml
```

数据 contract、搜索顺序、评估指标和 8xA100 布局都在 `CODEBOOK.md`。旧
`scripts/codebook_eval.py` 和 `codewam.probe` 只用于历史兼容检查。

## 7. Compatible 模型训练

从 Wan DiT 初始化 ActionDiT backbone:

```bash
bash scripts/prepare_action_dit.sh
```

默认输出:

```text
checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

当前 FastWAM-compatible 对照:

```bash
bash scripts/train_zero1.sh 8 task=libero_codewam_2cam224
bash scripts/train_zero1.sh 8 task=robotwin_codewam_3cam384
```

这些命令不是 canonical v1 的完成实现。`state_codebook.enabled=false` 必须保持默认,直到
`FrozenRQAdapter` 和 Gate 3 mask tests 完成。

## 8. 本地状态边界

以下目录可存在,但不应被 git 跟踪:

```text
.venv/                         # local macOS environment only
external/FastWAM/              # pinned external checkout
checkpoints/                    # model files
data/, datasets/               # local or mounted datasets
runs/, outputs/, logs/, wandb/ # generated artifacts
.hf/                           # legacy local Hugging Face cache
```

`__pycache__/`、`.pytest_cache/`、`.ruff_cache/` 和 `*.egg-info/` 都是可丢弃状态。提交前只保留
CodeWAM-owned source、configs、scripts、tests 和三份 canonical docs。
