# CodeWAM

CodeWAM 是一个独立的 world-action model。冻结的 Wan-VAE 提供连续视觉 latent,三套离线
RQ codebook 提供多时间尺度离散世界坐标;CodeWAM 自己学习结构化世界状态、连续动作和
action-prefix-conditioned code transition。FastWAM 只作为外部对照与历史兼容路径,不是 canonical
模型依赖。

## 核心结构

```text
Wan latent history -> Causal Visual Stem -> H --------------------+
                                                                  |
timed proprio/actions -> WorldAttention(Q=slots,KV=[H,R]) -> G0 --+-> G
                                                                  ^
Q2/Q3/Q5 IDs -> cumulative RQ prefixes -> 3 clock tokens -> gate -+

S={G,H,clock tokens,current proprio}
       | + language                         | + action prefix/delta time
       v                                    v
ActionFlowDecoder                   MultiClockTransition
       |                                    |
continuous action chunk                future RQ paths
```

三套码本使用严格因果窗口:

```text
Q2(t) = RQ2([u(t-4),  u(t-2), u(t)])
Q3(t) = RQ3([u(t-6),  u(t-3), u(t)])
Q5(t) = RQ5([u(t-10), u(t-5), u(t)])
```

它们彼此独立,三级 residual centers 不共享。v2 在每个 family 内使用累积 RQ 前缀
`e1/e1+e2/e1+e2+e3`,再形成一个 clock token;code 不替代连续 latent,centers 也不在联合训练中
更新。语言只进入 ActionFlowDecoder,未来 code 只作为 MultiClockTransition 标签。联合训练
仍只有 action flow 与 future-code classification 两个 loss,基本推理不运行 transition。

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
├── data/          # DROID cache, frozen language and policy normalization
├── experiments/   # controlled Gate 2 and C0/C1/C2 protocols
├── models/        # retained v1 baseline and structured CodeWAM v2
├── model.py       # legacy FastWAM-compatible prototype
├── runtime.py     # legacy Hydra factory
└── probe.py       # legacy compatibility probe
configs/           # model, data, task, and codebook configs
scripts/           # setup, checks, export, clustering, and training
tests/             # canonical codebook contracts and numerical equivalence
```

`codewam/codebook.py` 和当前 `codewam/model.py` 中的单 token online-EMA 路径不是 canonical
CodeWAM,配置必须保持默认关闭。独立模型位于 `codewam/models/`;legacy `CodeWAM` 仍是惰性
兼容导出。`CodeWAMV1/build_codewam_v1` 保留历史基线;当前候选由
`CodeWAMV2/build_codewam_v2` 显式构造,二者都不依赖 FastWAM。

## 开始开发

本机:

```bash
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/smoke_codewam_v2.py \
  --device cpu \
  --output runs/model_smoke/codewam_v2.json
```

集群工程 smoke 把 `--device cpu` 改为 `--device cuda`;该命令使用合成 tensor 和合成 frozen
centers,只验证接口、梯度、冻结和推理,不产生任何研究结论。

只有复现外部 `F0`、运行当前 legacy Wan exporter 或重建 legacy-inclusive 环境时才准备
FastWAM:

```bash
bash scripts/bootstrap_fastwam.sh
# optional legacy-inclusive setup:
bash scripts/setup_local_env.sh
```

集群复用现有 Python/CUDA 时:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
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
  --cameras wrist_image_left \
  --strides 2 3 5 \
  --pool 4 --k 8 --levels 3 --device cuda:0

python scripts/run_streaming_codebook_candidate.py \
  --train-config "$RQ_ROOT/configs/train_g4_k8_l3.yaml" \
  --evaluation-config "$RQ_ROOT/configs/evaluate_g4_k8_l3.yaml"

python scripts/probe_codebook_family_contributions.py \
  --manifest "$POOLED_ROOT/pooled_manifest.jsonl" \
  --pooled-shards "$POOLED_ROOT/pooled/*.pt" \
  --artifact Q2="$RQ_ROOT/Q2/codebook.pt" \
  --artifact Q3="$RQ_ROOT/Q3/codebook.pt" \
  --artifact Q5="$RQ_ROOT/Q5/codebook.pt" \
  --depth-profile all-l3=Q2:3,Q3:3,Q5:3 \
  --output-dir "$RQ_ROOT/family_association" \
  --device cuda:0

python scripts/probe_codebook_temporal_sensitivity.py \
  --manifest "$POOLED_ROOT/pooled_manifest.jsonl" \
  --pooled-shards "$POOLED_ROOT/pooled/*.pt" \
  --artifact Q2="$RQ_ROOT/Q2/codebook.pt" \
  --artifact Q3="$RQ_ROOT/Q3/codebook.pt" \
  --artifact Q5="$RQ_ROOT/Q5/codebook.pt" \
  --output-dir "$RQ_ROOT/temporal_sensitivity" \
  --splits val test --device cuda:0
```

真实数据导出前用 `scripts/audit_wan_causality.py` 做 full-vs-prefix latent 一致性审计,
把 Wan-VAE 不读取未来帧从结构假设变成可复现证据。

冻结 artifacts 就绪后,canonical world-dynamics 链使用:

```bash
python scripts/audit_droid_endpoints.py \
  --data-dir "$DROID_100_ROOT" \
  --output "$ENDPOINT_AUDIT"

torchrun --standalone --nproc-per-node=8 \
  scripts/export_joint_window_cache.py \
  --source-manifest "$DROID_10K_MANIFEST" \
  --data-dir "$DROID_RLDS_ROOT" \
  --output-dir "$JOINT_CACHE_ROOT" \
  --endpoint-audit "$ENDPOINT_AUDIT" \
  --artifact Q2="$Q2_ARTIFACT" \
  --artifact Q3="$Q3_ARTIFACT" \
  --artifact Q5="$Q5_ARTIFACT" \
  --vae-path "$WAN_VAE_PATH" \
  --fastwam-src "$FASTWAM_ROOT"

python scripts/export_joint_window_cache.py \
  --output-dir "$JOINT_CACHE_ROOT" --finalize-only
```

finalize 会拒绝缺 rank、缺 source shard 或限量导出的半成品;只有工程 smoke 可以显式使用
`--allow-partial-finalize`。

Gate 2 的 PERSIST/NOACT/TRUE/SHUFFLE 命令和有效性规则见
[`docs/CODEBOOK.md`](./docs/CODEBOOK.md#144-正式-gate-2)。

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

- 当前候选:structured CodeWAM v2。三套独立 causal RQ 在 family 内形成三级累积 prefix 和
  三个 clock token;WorldAttention 只读取连续视觉与带相对时间的机器人历史,code 通过零初始化
  gate 修正 global belief。Policy 直接读取 `G/H/C/language/proprio`,避免精细信息瓶颈。
- 已实现:`CodeWAMV2/build_codewam_v2`、hierarchical frozen-center adapter、typed relative-time
  contracts、structured world state、continuous action flow,以及按每族 action prefix/delta time
  对齐的轻量 GRU+MLP transition。`CodeWAMV1` 原样保留为历史对照。
- 已实现的数据边界:expert/failure/recovery/unlabeled interaction/action-free video 五种角色分别
  控制 temporal、action imitation 和 dynamics supervision;失败动作不会进入 imitation loss,
  但失败状态仍可服务 world learning。
- 已验证:future latent/label/action 防泄漏、严格梯度路由、missing family、跨域 chart-local
  centers、全 padding action、frozen-center optimizer 完整性、两种 future-code factorization
  的归一化 NLL 口径、state-dict round trip 和 basic inference 不调用 dynamics。
- 已实现:episode manifest、scene-level split、pooled-feature shard、Q2/Q3/Q5 causal iterator、
  train-only normalization、共享初始化的多卡 streaming RQ、可恢复 patience、frozen artifact、
  只读 held-out evaluator、单族关联、跨 parent context concentration、对齐的多族增量探针,
  scene-diverse RGB retrieval、冻结 temporal counterfactual、held-out action events、
  DROID/LIBERO RGB perturbation、independent-seed stability 和 provenance-checked usability
  report、跨 seed `P0/P1/P2/P3` functional readout,以及 DROID 1.0.1 精确
  metadata/RLDS join、稀疏 RGB reader、keep-range audit 和 canonical pooled exporter。
- 已实现:RLDS endpoint audit、冻结 Q2/Q3/Q5 causal assigner、未池化多相机
  `JointWindowCache v1`、source-rate action/proprio、rank-aware Wan exporter、verified
  dataloader/collator,以及等预算 Gate 2 runner 与 episode-block bootstrap。
- 已验证:199 项单元测试、CodeWAM v2 synthetic optimizer-step smoke、未来标签隔离、RQ prefix
  累积等价、per-clock action-prefix 隔离、单卡/双 rank centers 等价、
  synthetic Q2/Q3/Q5 端到端 smoke、
  58,116-episode canonical DROID manifest、10,000-episode/756,225-tick Wan pooled cache、
  causal-prefix 零差异审计、完整 camera/pool/capacity 候选比较、val/test 一致的时间反事实
  敏感性,以及 seed 7/19/31 的九套独立 RQ artifacts。
- 当前候选:wrist-only frozen RQ 使用 `g=4,K=8,L=3`;DROID 域内 quantizer health、
  temporal hierarchy 和三族互补通过,geometry/action/scene 为 conditional,photometric
  robustness 失败。冻结 DROID artifacts 在 LIBERO 发生中心使用与几何响应 collapse,不能
  作为通用 tokenizer。严格 usability 报告的 `not_ready` 限定于“可重复、跨域通用
  tokenizer”主张;绑定版本和 seed 的 DROID 域内 artifact 可以进入 C1/C2 原型。
  minimal additive `P0/P1/P2/P3` 只证明 hard categorical feature 没有在线性动作读出中超过
  `H`,不再作为是否实现 world model 的结构门。
- 已验证真实 joint smoke:32 条 DROID-100/8,892 steps 的 endpoint audit 通过;DROID 1.0.1
  七个 keep-range segments 生成 157 个窗口,split `24/71/62`,Q2/Q3/Q5 overlap
  恒为 `1/0/0`;三学习条件各 2 步的 GPU Gate 2 链完整运行。test 仅两个独立 episode,
  因而报告按 30-episode 下限正确标记 `invalid`,不构成研究结论。
- 默认关闭:legacy online-EMA single-token codebook。
- 已验证:DROID-10k 561,338-window JointWindowCache 上,seed `7/19/31` 的正式 8-GPU
  Gate 2 均独立通过;aligned action 在 995 个共同 test parent episodes 上稳定降低
  changed-family future-code NLL,Q2/Q3/Q5 三族方向一致。该结果解锁原型,不等同于策略增益。
- 已实现:T5-base token-level frozen language sidecar、完整 source-rate DROID `action_dict`
  sidecar、train-only periodic-safe policy normalization、共享初始化/窗口顺序/flow 噪声的
  8-GPU `C0/C1/C2` trainer、可恢复 checkpoints、固定窗口 action-flow/ODE-sample 评估和
  三种子汇总器。
- 已验证:seed `7/19/31` 各 200 步的 DROID-10k policy pilot。`C1-C0` 方向不稳定;
  `C2-C1` 的 val/test flow MSE 在三个 seed 中均变差。该短跑只覆盖每变体 6,400 个窗口,
  且训练参数量为 `9.91M/15.22M/19.20M`,不能裁决最终结构或闭环收益。
- 已验证:`action` 与 `action_dict.cartesian_position + gripper_position` 在 DROID-10k 的
  3,002,148 个 source rows 上逐值相等;六种 position/velocity 分量均已冻结。尚未实现的是
  部署侧 online runtime 和闭环 benchmark。
- 下一步:先锁定 controller-compatible raw target 与 orientation representation,再在真实 joint
  batch 上完成 v2 CUDA optimizer-step/显存/延迟 smoke,随后做足够收敛的 v2 多种子 C0/C1/C2
  learning curve;只有确认不是短预算或梯度干扰后才调整 C2 权重/训练阶段。
  LIBERO 使用独立 chart/refit,不能把 DROID code ID 当成共享语义。

外部代码 revision 和模型来源固定在 [`upstreams.yaml`](./upstreams.yaml)。数据集、模型、
checkpoints 和运行结果始终放在 git 之外。
