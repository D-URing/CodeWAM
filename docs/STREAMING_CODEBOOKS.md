# Streaming Codebook Foundation

Status: implemented local foundation; public-dataset export and distributed orchestration remain pending.

本页描述 canonical Q2/Q3/Q5 离线码本底座的实际代码状态。架构决策仍以
`CODEWAM_V1_PLAN.md` 为准,大规模数据和 8xA100 方案仍以 `DATASET_SCALE_PLAN.md` 为准。

## 1. 已实现

### Episode contract

`codewam/codebook_eval/manifest.py` 提供:

- `EpisodeRecord`:数据集、episode、scene/building/institution、task、camera、source checksum。
- `EpisodeManifest`:唯一性检查、稳定 fingerprint、JSONL 原子写入。
- `SplitConfig`:按 scene/building/institution/episode 分组的稳定 hash split。
- train/val/test group isolation 检查。

默认 splitter 以 scene 为单位,不会随机拆 frame 或 window。task stratification 使用每个 group
的主 task 作为 hash stratum;正式数据 adapter 仍需检查最终任务分布。

### Pooled feature shard

`codewam/codebook_eval/shards.py` 定义
`codewam.pooled-feature-shard.v1`:

```text
one shard
  metadata
  episodes[]
    episode_id
    split
    timestamps[T]
    pooled_g4[T,V,C,4,4] fp16
    camera_ids[V]
    valid_mask[T,V]
    optional action/proprio
```

writer 使用临时文件加原子 rename,并返回 SHA-256。reader 一次只加载一个 shard,逐 episode
yield,不会拼成全量 tensor。`g=2/1` 从 `g=4` 做 nested average pooling。

shard 必须记录:

```text
dataset_revision
wan_model_id
wan_revision
preprocess_revision
source_checksums
```

### Causal descriptors

`CausalDescriptorSource` 对每套码本独立生成:

```text
Q2(t) = [u(t-4),  u(t-2), u(t)]
Q3(t) = [u(t-6),  u(t-3), u(t)]
Q5(t) = [u(t-10), u(t-5), u(t)]
```

它具有这些硬约束:

- current index 之前的两个历史点加 current,没有 future。
- 不跨 episode。
- 三个位置的全部 camera 必须有效。
- timestamp gap 超过 cadence 门槛时丢弃 descriptor。
- 输出 batch 大小固定,descriptor 不落盘。

### Streaming statistics and clustering

`codewam/codebook_eval/streaming.py` 已实现:

- mergeable Welford moments 和 train-only normalization。
- 固定容量 Algorithm-R uniform reservoir 与 sample-blocked deterministic K-Means++。
- center-blocked nearest assignment。
- batch-streamed Lloyd sums/counts/inertia。
- 可选 `torch.distributed` all-reduce。
- global hardest-sample empty-center refill。
- iteration checkpoint/resume。
- 三级 sequential RQ,residual 只存在于当前 batch。
- frozen artifact,只保存 normalization、centers 和 provenance,不保存全量 codes。

底层统计与 Lloyd 已有 distributed collective primitive,但当前 launcher 会主动拒绝
`world_size > 1`;在 rank-aware shard partition 和共享初始化完成前不能启动多 rank 正式任务。

计算约定:

```text
cache/input:       fp16 or bf16
vectors/centers:   fp32
distance/sums:     fp32
normalization:     mergeable moments, frozen before clustering
```

### One-command launcher

模板:

```text
configs/codebook_eval/streaming_rq_template.yaml
```

正式 pooled shards 就绪后:

```bash
.venv/bin/python scripts/train_streaming_codebooks.py train \
  --config configs/codebook_eval/streaming_rq_template.yaml
```

launcher 顺序训练 Q2/Q3/Q5,每套生成:

```text
output/Q2|Q3|Q5/
  contract.json
  normalization.pt
  checkpoints/
    level_1_kmeans.pt
    level_2_kmeans.pt
    level_3_kmeans.pt
    rq_state.pt
  codebook.pt
  train_summary.json
```

同一 output resume 前会校验 descriptor、K/L、source checksums、manifest fingerprint 和运行参数。
contract 不一致时拒绝串用 checkpoint。

## 2. 本机验证

不需要 Wan 权重或公开数据的一键 smoke:

```bash
.venv/bin/python scripts/train_streaming_codebooks.py smoke \
  --output runs/codebook_eval/streaming_smoke
```

当前本机 MPS 实测:

```text
Q2: N=186 D=72 K=8 L=3 reductions=[0.643, 0.508, 0.467]
Q3: N=162 D=72 K=8 L=3 reductions=[0.641, 0.634, 0.464]
Q5: N=114 D=72 K=8 L=3 reductions=[0.680, 0.745, 0.471]
```

这些数字只证明 plumbing 和 residual 方向正确,没有数据集或模型研究含义。

测试:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

当前 17 项测试覆盖:

- manifest deterministic split、scene isolation 和 round-trip。
- pooled shard 原子 round-trip 与 split filter。
- non-finite feature/timestamp 拒绝。
- Q2 的人工可读 causal offsets。
- invalid tick filtering 和 train-only normalization。
- streaming batch partition invariance。
- uniform reservoir chunk partition invariance。
- streaming Lloyd 与 legacy full-batch Lloyd 数值等价。
- checkpoint resume 与 uninterrupted run 一致。
- converged checkpoint 不重复追加迭代。
- 三级 RQ residual 单调下降。
- frozen artifact round-trip。
- Q2/Q3/Q5 one-command train/resume。

## 3. 旧 Package Scan cache 的边界

当前
`runs/codebook_eval/latents/package_scan_v6_dense` 是 legacy window cache:

```text
latents: [N,48,7,7,14]
meta:    only window index and prompt
```

它不能直接转成正式 canonical cache:

- 单 window 只有 7 latent ticks,Q5 的 `[t-10,t-5,t]` 至少需要 11 ticks。
- overlapping windows 没有保存 `episode_id/start_index/timestamps`。
- 无法证明 split isolation,也无法可靠去重重叠 latent ticks。
- 它保存 full window latent,不是逐 episode `pooled_g4`。

因此旧 cache 只保留为 legacy evaluator 对照。Package Scan 正式 smoke 需要重新做 episode-aware
pooled export,不能给旧 window 人造 episode id 后报告结果。

## 4. 尚未完成

当前不应宣称海量训练器已经全部完成。下一步仍需要:

1. Package Scan episode-aware pooled exporter,随后复用同一接口实现 DROID-100 adapter。
2. scene/task/event balanced reservoir;当前 generic backend 是 uniform reservoir。
3. DDP rank-aware shard partition、共享初始化 artifact 和 1-GPU/8-GPU 等价测试。
4. held-out usage/perplexity/retrieval/geometry/action-probe streaming evaluator。
5. representative episode/time id 收集与 artifact report。
6. DROID-100 的真实 Wan cadence、latent shape、吞吐和缓存量实测。

当前代码已经可以独立推进第 1 和第 4 项,不需要先配置开发机或下载完整 DROID。
