# CodeWAM

CodeWAM 是一个连续视觉状态与离散视觉坐标并存的 world-action model。它复用 Wan-VAE、
Video DiT、ActionDiT 和 flow matching 等基座能力,但不把 FastWAM 的对称 MoT 拓扑当作结构
边界。FastWAM 是依赖与外部对照,CodeWAM 是独立方法。

## 核心结构

```text
unquantized Wan latent -> continuous state H -> 精确几何与动作微调
three frozen RQ-3     -> 9 code tokens     -> 多时间尺度状态坐标
H + code + L/P        -> belief B          -> continuous action policy
shared state + action -> world expert      -> training-only future-code objective
```

三套码本使用严格因果窗口:

```text
Q2(t) = RQ2([u(t-4),  u(t-2), u(t)])
Q3(t) = RQ3([u(t-6),  u(t-3), u(t)])
Q5(t) = RQ5([u(t-10), u(t-5), u(t)])
```

它们彼此独立,三级 residual centers 不共享。九个 code measurement 不求和、不替代连续
latent,也不在 policy 训练中更新。Policy、Forward Dynamics 和 Video Prior 使用不同的输入
contract 与 mask,动作分支永远看不到未来 target。

## 文档

仓库只维护三份权威文档:

- [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md):模型结构、信息身份、mask、loss 和实验门。
- [`docs/CODEBOOK.md`](./docs/CODEBOOK.md):数据、pooled latent、streaming RQ、评估和 8xA100。
- [`docs/DEVELOPMENT.md`](./docs/DEVELOPMENT.md):仓库边界、环境、模型文件和运行命令。

## 代码结构

```text
codewam/
├── codebook.py    # legacy online-EMA prototype; disabled by default
├── codebook_eval/ # canonical manifest, pooled shards, streaming RQ and pipeline
├── data/          # DROID official manifest/RLDS and local regression adapters
├── model.py       # current FastWAM-compatible prototype
├── runtime.py     # Hydra factory
└── probe.py       # legacy compatibility probe
configs/           # model, data, task, and codebook configs
scripts/           # setup, checks, export, clustering, and training
tests/             # canonical codebook contracts and numerical equivalence
```

`codewam/codebook.py` 和当前 `codewam/model.py` 中的单 token online-EMA 路径不是 canonical
CodeWAM,配置必须保持默认关闭。

## 开始开发

本机:

```bash
bash scripts/setup_local_env.sh
source .venv/bin/activate
python scripts/check_environment.py --mode local
python -m unittest discover -s tests -v
```

固定 FastWAM 依赖:

```bash
bash scripts/bootstrap_fastwam.sh
```

集群复用现有 Python/CUDA 时:

```bash
PYTHON=python \
INSTALL_TORCH_STACK=false \
bash scripts/setup_cluster_env.sh
python scripts/check_environment.py --mode cluster
```

完整环境和模型说明见 [`docs/DEVELOPMENT.md`](./docs/DEVELOPMENT.md)。

## Codebook 验证

不需要模型权重的 streaming smoke:

```bash
python scripts/train_streaming_codebooks.py smoke \
  --output runs/codebook_eval/streaming_smoke
```

正式 episode-aware pooled shards 就绪后:

```bash
python scripts/prepare_streaming_codebook_run.py \
  --pooled-export-dir "$POOLED_ROOT" \
  --output-dir "$RQ_ROOT" \
  --pool 4 --k 16 --levels 3 --device cuda:0

python scripts/train_streaming_codebooks.py train \
  --config "$RQ_ROOT/configs/train_g4_k16_l3.yaml"

python scripts/evaluate_streaming_codebooks.py \
  --config "$RQ_ROOT/configs/evaluate_g4_k16_l3.yaml"
```

Package Scan v6 只用于本机数据链路和回归:

```bash
python scripts/demo_package_scan_v6.py
```

研究数据角色固定为:

```text
DROID        main codebook training and in-domain evaluation
LIBERO       controlled geometry/task validation
BridgeData V2 frozen-transfer and independent-refit replication
```

## 当前状态

- 已锁定:连续状态路径、三套独立 causal RQ、九个只读 code measurements、belief aggregator、
  mode-specific masks 和可选 MemoryPort。
- 已实现:episode manifest、scene-level split、pooled-feature shard、Q2/Q3/Q5 causal iterator、
  train-only normalization、GPU K-Means++、三级 streaming RQ、可恢复 patience、frozen
  artifact、只读 held-out evaluator,以及 DROID 1.0.1 精确 metadata/RLDS join、rank-aware
  reader、keep-range audit 和 canonical pooled exporter。
- 已验证:49 项单元测试、synthetic Q2/Q3/Q5 train/eval smoke、58,116-episode canonical
  DROID manifest,以及 26-episode/13-institution 真实 Wan latent 与 RQ 工程 pilot。
- 默认关闭:legacy online-EMA single-token codebook。
- 下一步:4 卡导出 canonical DROID-10k pooled cache,在原始 scene-isolated val/test 上完成
  held-out residual/usage、retrieval、camera 与 action probes,再冻结 K、pool 和有效 RQ prefix。

外部代码 revision 和模型来源固定在 [`upstreams.yaml`](./upstreams.yaml)。数据集、模型、
checkpoints 和运行结果始终放在 git 之外。
