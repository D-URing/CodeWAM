# CodeWAM

CodeWAM 是一个连续视觉状态与离散视觉坐标并存的 world-action model。它复用 Wan-VAE、
Video DiT、ActionDiT 和 flow matching 等基座能力,但不把 FastWAM 的对称 MoT 拓扑当作结构
边界。FastWAM 是依赖与外部对照,CodeWAM 是独立方法。

## 核心结构

```text
unquantized Wan latent -> continuous state H -> 精确几何与动作微调
three frozen RQ-3     -> 9 code tokens     -> 多时间尺度状态坐标
H + L/P               -> shared belief B0
B0 + role-masked code -> B-policy/B-world  -> action / future-code experts
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
  --depth-profile policy-hybrid=Q2:3,Q3:3,Q5:2 \
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

- 已锁定:连续状态路径、三套独立 causal RQ、九个只读 code measurements、continuous base
  belief、role-specific measurement routers 和可选 MemoryPort。
- 已实现:episode manifest、scene-level split、pooled-feature shard、Q2/Q3/Q5 causal iterator、
  train-only normalization、共享初始化的多卡 streaming RQ、可恢复 patience、frozen artifact、
  只读 held-out evaluator、单族关联、跨 parent context concentration、对齐的多族增量探针,
  scene-diverse RGB retrieval、冻结 temporal counterfactual、held-out action events、
  DROID/LIBERO RGB perturbation、independent-seed stability 和 provenance-checked usability
  report、跨 seed `P0/P1/P2/P3` functional readout,以及 DROID 1.0.1 精确
  metadata/RLDS join、稀疏 RGB reader、keep-range audit 和 canonical pooled exporter。
- 已验证:105 项单元测试、单卡/双 rank centers 等价、synthetic Q2/Q3/Q5 端到端 smoke、
  58,116-episode canonical DROID manifest、10,000-episode/756,225-tick Wan pooled cache、
  causal-prefix 零差异审计、完整 camera/pool/capacity 候选比较、val/test 一致的时间反事实
  敏感性,以及 seed 7/19/31 的九套独立 RQ artifacts。
- 当前候选:wrist-only frozen RQ 使用 `g=4,K=8,L=3`;DROID 域内 quantizer health、
  temporal hierarchy 和三族互补通过,geometry/action/scene 为 conditional,photometric
  robustness 失败。冻结 DROID artifacts 在 LIBERO 发生中心使用与几何响应 collapse,不能
  作为通用 tokenizer。三 seed distortion CV 最高仅 0.55%,但 L2/L3 最低 NMI 只有
  0.143/0.065,完整前缀映射一致率中位数为 7.31%,因此严格结论为 `not_ready`。连续 latent
  路径必须保留 exterior+wrist。minimal additive Gate 2 在 5%/20%/100% train 和三个 seed
  上的 `P3-P1` 全为负;full-data test 仅为 `-0.023` 至 `-0.019 pp`,说明跨 seed 功能近似
  等价,但 hard code 尚未提供 H 之外的动作增量。
- 默认关闭:legacy online-EMA single-token codebook。
- 下一步:先比较 quantization margin、center confidence 与参数受控的
  code-conditioned interaction,并分开 current-action/future-state targets。只有跨 seed 的
  `H+C` 增量价值稳定后才实现完整 role router。deterministic/consensus initialization 仅解决
  artifact 可重复性;LIBERO 另做同规格独立 refit/calibration 和 simulator intervention。

外部代码 revision 和模型来源固定在 [`upstreams.yaml`](./upstreams.yaml)。数据集、模型、
checkpoints 和运行结果始终放在 git 之外。
