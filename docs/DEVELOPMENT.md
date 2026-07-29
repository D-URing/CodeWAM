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
│   │   ├── pipeline.py        # canonical Q2/Q3/Q5 launcher
│   │   ├── evaluation.py      # frozen-artifact val/test evaluator
│   │   ├── association.py     # train-fit held-out single-family probes
│   │   ├── concentration.py   # cross-parent context concentration
│   │   ├── family_association.py # aligned multi-family contribution
│   │   ├── retrieval.py       # exact RGB retrieval and montage provenance
│   │   ├── temporal_sensitivity.py # frozen temporal counterfactuals
│   │   ├── action_events.py    # held-out Cartesian/gripper event semantics
│   │   ├── visual_perturbations.py # RGB-to-Wan nuisance/geometry stress
│   │   ├── seed_stability.py   # independent-seed partition comparison
│   │   ├── functional_readout.py # frozen-seed proprio/H/C/H+C screen
│   │   ├── usability.py        # provenance-checked ten-gate report
│   │   ├── workflow.py        # resumable candidate end-to-end workflow
│   │   └── droid_pooled_export.py # rank-aware Wan pooled exporter
│   ├── data/
│   │   ├── droid_manifest.py  # exact official join and balanced sample
│   │   ├── droid_rlds.py      # exact-position, rank-aware and sparse RGB reader
│   │   ├── roles.py            # trajectory role and objective masks
│   │   └── package_scan_v6.py # local regression adapter
│   ├── models/
│   │   ├── contracts.py        # typed state/code/policy/action batches
│   │   ├── continuous_state.py # causal state encoder and Stage-0 head
│   │   ├── frozen_codebook.py  # chart-local frozen-center adapter
│   │   ├── belief_core.py      # task-free world belief
│   │   ├── action_flow.py      # continuous action chunk flow
│   │   ├── code_dynamics.py    # independent/prefix future-code heads
│   │   └── codewam_v1.py       # C0/C1/C2 assembly
│   ├── model.py               # legacy FastWAM-compatible prototype
│   ├── probe.py               # legacy compatibility probe
│   └── runtime.py             # legacy Hydra factory
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

大型公开数据集放在独立共享数据根目录,不复制到仓库。下载、校验和训练 artifact 也不进入
git。canonical model 已位于 `codewam/models/`;当前缺口是 real-data `JointWindowCache`、
dataloader、distributed trainer 和部署侧 frozen causal assigner,不是模型文件。

## 2. 本机开发

macOS 已有项目内轻量环境时:

```bash
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/smoke_codewam_v1.py \
  --device cpu \
  --output runs/model_smoke/codewam_v1.json
```

该环境用于代码、配置、单元测试和小规模 MPS/CPU smoke,不承担大模型训练。当前
`scripts/setup_local_env.sh` 会额外准备 FastWAM,只在需要 legacy/F0 环境时使用。

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

已有兼容 torch 时,canonical source 的最低安装是:

```bash
python -m pip install -e .
```

码本导出、DROID reader 或训练分别按入口补齐缺失包,不要替换管理员提供的 torch。当前
`scripts/setup_cluster_env.sh` 是 legacy/F0 inclusive 环境脚本,会 bootstrap FastWAM;使用时
必须显式知道这一点:

```bash
PYTHON=python \
INSTALL_TORCH_STACK=false \
bash scripts/setup_cluster_env.sh
```

`scripts/check_environment.py` 当前也检查 legacy FastWAM/Wan-DiT/ActionDiT 完整环境,不代表
canonical five-module model 的最低依赖。独立模型实现时应同时增加分 profile 的环境检查。

## 4. FastWAM 外部对照与 legacy 边界

只有运行外部 `F0`、旧训练入口或当前已验证的 Wan exporter loader 时才固定上游:

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
- `ContinuousStateEncoder`、`FrozenCodebookAdapter` 和 `WorldBeliefCore`;
- `ActionFlowDecoder`、`CodeDynamicsDecoder` 和两个训练目标;
- module-level visibility contracts and codebook evaluation pipeline;
- CodeWAM configs and tests.

FastWAM checkout 当前提供两类兼容能力:旧 Video-DiT/ActionDiT/MoT 训练对照,以及已被
pooled-cache provenance 锁定的 Wan-VAE loader/转换器。前者只属于 `F0`;后者是可替换的
数据适配器,不是 canonical model runtime。独立 v1 不 import FastWAM model 或 trainer。

## 5. 模型文件

下面的脚本准备 legacy/F0 compatible 模型,不是 canonical v1 的最低下载:

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

RoboTwin/3-camera checkpoint 对 codebook 或 canonical C0-C2 都不是依赖。离线 latent export
只需要固定版本 Wan-VAE、对应 loader 和预处理配置。canonical v1 需要 Wan-VAE、一个明确
版本的 language encoder 和 CodeWAM-owned modules,不需要 Wan DiT 或 ActionDiT checkpoint。
具体 language encoder 在实现 ticket 中选择并冻结版本。

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

正式导出前先用真实 DROID 片段验证 Wan-VAE 的时间因果性:

```bash
PYTHONPATH=. python scripts/audit_wan_causality.py \
  --source-manifest "$DROID_MANIFEST" \
  --data-dir "$DROID_RLDS_ROOT" \
  --output "$POOLED_ROOT/causal_prefix_audit.json" \
  --vae-path "$WAN_VAE_PATH" \
  --fastwam-src "$FASTWAM_SRC" \
  --device cuda:0
```

该审计比较同一片段的完整 21 帧编码与 `1/5/9/13/17/21` 帧前缀编码。每个前缀的全部
latent ticks 都必须与完整编码的对应前缀一致;报告会锁定 manifest、VAE 和实现 SHA。

真实 DROID manifest 就绪后,先按完整 TFRecord shard 分配 rank 并导出:

```bash
PYTHONPATH=. python scripts/export_droid_pooled.py export \
  --source-manifest "$DROID_MANIFEST" \
  --data-dir "$DROID_RLDS_ROOT" \
  --output-dir "$POOLED_ROOT" \
  --vae-path "$WAN_VAE_PATH" \
  --fastwam-src "$FASTWAM_SRC" \
  --rank "$RANK" --world-size "$WORLD_SIZE" \
  --device "cuda:$LOCAL_RANK"

PYTHONPATH=. python scripts/export_droid_pooled.py finalize \
  --source-manifest "$DROID_MANIFEST" \
  --output-dir "$POOLED_ROOT"
```

每个 rank 只拥有完整 source shards;不使用 DDP。相同命令续跑会校验 contract、输出 SHA 和
segment ids,跳过已完成 shard。每完成一个 shard 都会原子更新
`rank-XXX-of-YYY-progress.json`,因此进程在最终 report 前被终止时，续跑仍能保留已完成
shard 的首次耗时与显存证据。contract 同时锁定 VAE、FastWAM loader/转换器以及 CodeWAM
reader/preprocess/writer 的实现 SHA。

pooled shards 就绪后训练与 held-out 评估:

```bash
python scripts/prepare_streaming_codebook_run.py \
  --pooled-export-dir "$POOLED_ROOT" \
  --output-dir "$RQ_ROOT" \
  --cameras wrist_image_left \
  --strides 2 3 5 \
  --pool 4 --k 8 --levels 3 --device cuda:0

python scripts/run_streaming_codebook_candidate.py \
  --train-config "$RQ_ROOT/configs/train_g4_k8_l3.yaml" \
  --evaluation-config "$RQ_ROOT/configs/evaluate_g4_k8_l3.yaml"
```

该单进程命令依次续跑 train、frozen held-out evaluation、单族 association 和跨 parent
context concentration,最后写出带四份报告 SHA 的 `candidate_workflow.json`。任何阶段已有
合法报告都会复用;contract 不一致时拒绝混用。

冻结评估完成后，也可单独运行 train-fit/val-test-only 关联探针:

```bash
python scripts/probe_frozen_codebook_associations.py \
  --manifest "$POOLED_ROOT/pooled_manifest.jsonl" \
  --pooled-shards "$POOLED_ROOT/pooled/*.pt" \
  --artifact q3-dual-g4="$RQ_ROOT/Q3/codebook.pt" \
  --output-dir "$RQ_ROOT/association" \
  --device cuda:0
```

该探针的 code 输入仍然只有 `t-2s,t-s,t` 三个视觉状态。`current_action`、
`future_proprio_change` 和 `future_latent_moment_change` 只是监督评估目标;后两项取
`t+s`，绝不加入 descriptor。条件均值和 target normalization 只由 train 统计，
val/test 遇到样本不足的 tuple 会依次回退到更短 RQ prefix 或 train global mean。

三族 artifact 完成后,用严格对齐的 tick 和共同目标测量互补性:

```bash
python scripts/probe_codebook_family_contributions.py \
  --manifest "$POOLED_ROOT/pooled_manifest.jsonl" \
  --pooled-shards "$POOLED_ROOT/pooled/*.pt" \
  --artifact Q2="$RQ_ROOT/Q2/codebook.pt" \
  --artifact Q3="$RQ_ROOT/Q3/codebook.pt" \
  --artifact Q5="$RQ_ROOT/Q5/codebook.pt" \
  --depth-profile policy-hybrid=Q2:3,Q3:3,Q5:2 \
  --output-dir "$RQ_ROOT/family_association" \
  --device cuda:0
```

它只在 train 上拟合加性类别岭回归,在完全相同的 val/test 时间点比较三个 single、三个 pair
和 joint model。报告 `joint - best single` 以及 joint 相对每个 leave-one-family-out model
的增量,避免不同 stride 的有效窗口或 future horizon 混进横向比较。默认
`max_pair_cells=2,000,000` 会阻止不受控的大容量 prefix table。`--depth-profile` 直接拟合
指定的合法 mixed prefix,不会用 leave-one-out 间接猜测 hybrid 性能。

时间反事实不需要原始 RGB 或重训 centers:

```bash
python scripts/probe_codebook_temporal_sensitivity.py \
  --manifest "$POOLED_ROOT/pooled_manifest.jsonl" \
  --pooled-shards "$POOLED_ROOT/pooled/*.pt" \
  --artifact Q2="$Q2_ROOT/Q2/codebook.pt" \
  --artifact Q3="$Q3_ROOT/Q3/codebook.pt" \
  --artifact Q5="$Q5_ROOT/Q5/codebook.pt" \
  --output-dir "$RQ_ROOT/temporal_sensitivity" \
  --splits val test --device cuda:0
```

它分别测试保持当前端点的 history swap、完整 time reversal 和 static-current repetition,
报告逐层与合法 prefix 的 code change、true-descriptor cross-reconstruction penalty。
结果属于 representation sensitivity,不等同于物理环境因果干预。

生成可审查 RGB retrieval 时,先在独立 held-out output 中把
`representatives_per_code` 设为 32,再从不同 scene 选最近 anchor:

```bash
python scripts/render_codebook_retrievals.py \
  --source-manifest "$DROID_MANIFEST" \
  --pooled-manifest "$POOLED_ROOT/pooled_manifest.jsonl" \
  --droid-data-dir "$DROID_RLDS_ROOT" \
  --evaluation-report "$Q2_HELDOUT_32/evaluation_report.json" \
  --evaluation-report "$Q3_HELDOUT_32/evaluation_report.json" \
  --evaluation-report "$Q5_HELDOUT_32/evaluation_report.json" \
  --output-dir "$RQ_ROOT/rgb_retrieval_scene" \
  --splits test --levels 1 --diversity-by scene
```

reader 根据 pooled shard 中的 `absolute_latent_frame_indices` 只解码请求 JPEG。L2/L3 montage
展示 residual center 在不同 earlier prefixes 下的使用,不能单独解释为完整状态类别。

选定最终规格后可让每个 rank 读取独立 pooled shards,共享 rank-0 全局 reservoir 初始化并只
all-reduce `K x D` 统计量:

```bash
torchrun --standalone --nproc-per-node=8 \
  scripts/train_streaming_codebooks.py train \
  --config "$RQ_ROOT/configs/train_g4_k8_l3.yaml"
```

训练默认 `cpu_threads=4`,避免短 segment tensor 操作在 64 线程机器上过度并行;
K-Means++、Lloyd assignment 和 residual quantization 使用 `training.device`。评估只读取
frozen train normalization/centers,不会用 val/test 重估统计量。配置生成器只接受已经
finalize 的 export,并复核 contract、pooled manifest 以及每个 shard 的大小和 SHA-256。
端到端 candidate workflow 本身只接受单进程;多卡训练后在 rank 0 分别运行 evaluation 和
probes。

完整 P1 可用性链路由四个独立入口补齐:

```text
probe_codebook_action_events.py       absolute-code -> held-out event labels
probe_rgb_visual_perturbations.py     RGB -> Wan -> frozen RQ perturbation
probe_rq_seed_stability.py            three seeds -> NMI/ARI/distortion CV
probe_codebook_functional_readout.py  three seeds -> P0/P1/P2/P3 scale curve
build_rq_usability_report.py          provenance audit + ten gated decisions
```

正式命令和结果解释见 `CODEBOOK.md`。每个 probe 都先写 immutable contract,再原子写报告;
同目录续跑只接受完全相同的 contract。统一报告不会复制数据或模型,只验证并引用输入 SHA。
functional readout 额外锁定 nested scene subset、连续状态的七个因果 offsets、code depth
profile、target 和 P1-only alpha selection;输出目录不能在不同 train fraction 间复用。

官方 DROID manifest 的构建命令、数据 contract、搜索顺序、评估指标和 8xA100 布局都在
`CODEBOOK.md`。一次性下载器和官方数据索引放在共享数据根目录,不放进本仓库。旧
`scripts/codebook_eval.py` 和 `codewam.probe` 只用于历史兼容检查。

## 7. Canonical 模型状态

独立 CodeWAM v1 五模块和最低验收已经实现:

```text
codewam/models/ 不 import FastWAM
C0/C1/C2 使用同一 typed batch contract 和 builder
Stage-0 encoder 输入从结构上裁掉未来 target
action/future-code label 无泄漏
L_action/L_code 梯度路由符合 ARCHITECTURE.md
frozen centers 在 optimizer step 前后逐位不变
basic inference 不调用 CodeDynamics
independent/prefix NLL 使用统一的 per-RQ-level 口径
```

本机或开发机工程验收:

```bash
python scripts/smoke_codewam_v1.py \
  --device cuda \
  --output runs/model_smoke/codewam_v1_cuda.json
```

输出 schema 为 `codewam.model-smoke.v1`,会跑 Stage-0、C0、C1、C2-independent 和 C2-prefix
各一个 optimizer step,再检查 mixed chart、availability、gradient routes、frozen centers 和
basic action inference。它只使用合成 tensor/centers,字段
`scientific_evidence=false`;不能写进实验结果。

`configs/model/codewam_v1.yaml` 可实例化 `CodeWAMConfig`,但当前仍没有合法的 real-data
canonical training command。原因是 joint window exporter 尚未完成,而不是继续缺模型模块。
旧 `scripts/train.py`、`train_zero*.sh` 的成功只能记为 legacy/F0,不能记为 CodeWAM v1。

模型 forward 接收 `CodeMeasurements`,不会在内部从 latent 重新计算 RQ IDs。训练 cache 和
部署 runtime 都必须用同一 artifact 的 descriptor、normalization、centers 与 chart identity;
后续 online assigner 只能做冻结赋码,不能流式更新聚类中心。

## 8. Legacy/F0 模型训练

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

这些命令不是 canonical v1。`state_codebook.enabled=false` 必须保持默认,旧 online-EMA
codebook 和新 frozen nine-token interface 不能混用。

## 9. 本地状态边界

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
