# CodeWAM

CodeWAM 是一个独立的 world-action model。冻结的 Wan-VAE 提供连续视觉 latent,三套离线
RQ codebook 提供多时间尺度离散世界坐标;CodeWAM 自己学习共享 world belief、连续动作和
action-conditioned code transition。FastWAM 只作为外部对照与历史兼容路径,不是 canonical
模型依赖。

## 核心结构

```text
Wan latent -> ContinuousStateEncoder -> H --------------------+
                                                             |
causal Q2/Q3/Q5 IDs -> FrozenCodebookAdapter -> 9 tokens E --+-> WorldBeliefCore -> B
proprio + past actions ---------------------------------------+
                                                                  |          |
                                                        language  |          | GT action (train only)
                                                                  v          v
                                                        ActionFlowDecoder  CodeDynamicsDecoder
                                                                  |          |
                                                          action chunk   future code IDs
```

三套码本使用严格因果窗口:

```text
Q2(t) = RQ2([u(t-4),  u(t-2), u(t)])
Q3(t) = RQ3([u(t-6),  u(t-3), u(t)])
Q5(t) = RQ5([u(t-10), u(t-5), u(t)])
```

它们彼此独立,三级 residual centers 不共享。九个 code measurement 不求和、不替代连续
latent,也不在联合训练中更新。所有可用 code 默认进入任务无关的 `B`;语言只进入
ActionFlowDecoder,未来 code 只作为 CodeDynamicsDecoder 的标签。v1 只有 action flow 与
future-code classification 两个 loss,基本推理不运行 future-code decoder。

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
├── data/          # DROID endpoint, frozen assignment and verified joint cache
├── experiments/   # controlled Gate 2 protocols
├── models/        # independent five-module CodeWAM v1
├── model.py       # legacy FastWAM-compatible prototype
├── runtime.py     # legacy Hydra factory
└── probe.py       # legacy compatibility probe
configs/           # model, data, task, and codebook configs
scripts/           # setup, checks, export, clustering, and training
tests/             # canonical codebook contracts and numerical equivalence
```

`codewam/codebook.py` 和当前 `codewam/model.py` 中的单 token online-EMA 路径不是 canonical
CodeWAM,配置必须保持默认关闭。独立模型位于 `codewam/models/`;legacy `CodeWAM` 仍是惰性
兼容导出,`CodeWAMV1`、`CodeWAMConfig` 和 `build_codewam_v1` 不依赖 FastWAM。

## 开始开发

本机:

```bash
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/smoke_codewam_v1.py \
  --device cpu \
  --output runs/model_smoke/codewam_v1.json
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

- 已锁定:五个 CodeWAM-owned 模块、三套独立 causal RQ、九个只读 code measurements、
  任务无关 world belief、连续 action flow、action-conditioned code dynamics 和双 loss。
- 已实现:`codewam/models/` 的 typed contracts、causal spatiotemporal state encoder、
  chart-local frozen-center adapter、task-free belief、continuous action flow、independent/prefix
  future-code decoder、可丢弃 Stage-0 temporal head,以及 `C0/C1/C2` 单一构建接口。
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
- 已验证:167 项单元测试(本机仅 1 项 CUDA 专项跳过)、单卡/双 rank centers 等价、
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
- 尚未实现:full-scale Gate 2、frozen language token cache、canonical C0/C1/C2 policy trainer、
  部署侧 online runtime 和闭环 benchmark。
- 下一步:按 GPU 与 pod 主存共同确定并发度,导出完整 DROID-10k JointWindowCache,再以
  7/19/31 三个 seed 正式运行 Gate 2。
  只有 Gate 2 通过后才启动 `C0/C1/C2`;LIBERO 使用独立 chart/refit,不能把 DROID code ID
  当成共享语义。

外部代码 revision 和模型来源固定在 [`upstreams.yaml`](./upstreams.yaml)。数据集、模型、
checkpoints 和运行结果始终放在 git 之外。
