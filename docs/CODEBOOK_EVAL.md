# Offline Codebook Evaluation

> 本页描述当前已实现的早期 evaluator。它仍使用 `current/future/delta` variants,尚未满足
> causal split、train-only normalization 和 held-out probe 要求。下一版迁移目标与实验门以
> `docs/CODEWAM_V1_PLAN.md` 为准;在迁移完成前,本页命令只用于链路检查。

本页描述 CodeWAM 当前阶段最重要的离线工作:

```text
先判断 codebook 是否能形成有用的状态度量,
再决定它应该如何接入 FastWAM/ActionDiT/MoT。
```

Scan v6 可以用于本机 smoke test,但主要结论应来自公开数据集,例如 LIBERO、CALVIN、
BridgeData V2、DROID/Open X-Embodiment 子集。

## 目标

第一阶段只评估码本本身,不训练大模型:

```text
public robot dataset
-> Wan-VAE latent cache
-> temporal interleaved descriptors
-> three RQ-3 codebooks, one for each stride
-> usage / reconstruction / temporal / action-relevance metrics
```

当前默认方案是三套互质时间间隔码本:

```text
stride = 2, 3, 5
each stride: 3-level RQ
```

每个当前 latent state 都可得到三组 code 信息:

```text
c_t = {c_2(t), c_3(t), c_5(t)}
```

这一步暂时只证明状态空间是否值得继续使用。code embedding 放在哪里、是否进
ActionDiT cross-attention、是否作为 MoT register token,留到下一阶段。

## Latent Cache Format

训练脚本以 Wan-VAE latent shard 为标准输入。每个 shard 是 `.pt` 文件:

```python
{
    "latents": Tensor[N, C, T, H, W],   # required
    "action": Tensor[N, T_action, D],   # optional
    "proprio": Tensor[N, T_action, D],  # optional
    "meta": list | dict,                # optional
}
```

这样可以把公开数据集读取问题和码本训练问题解耦。只要 LIBERO、CALVIN、BridgeData 或
DROID 子集能先导出成这个格式,后面的比较完全共享。

## Descriptor

当前实现位于 `codewam/codebook_eval/descriptors.py`。默认 descriptor:

```text
desc_s(t) = concat(pool(f_t), pool(f_{t+s}), pool(f_{t+s}) - pool(f_t))
```

默认 `pool=2`,所以如果 Wan-VAE latent 通道是 `48`,单个帧描述是 `48*2*2=192`,
默认 temporal descriptor 维度是:

```text
192 * 3 = 576
```

这比直接把整张 latent map 拉平更稳,也比再训练一个重 encoder 更符合当前“直接度量
Wan latent”的思路。

## Candidate Codebooks

配置在 `configs/codebook_eval/public_latent_codebooks.yaml`:

```yaml
datasets:
  - name: libero
    enabled: true
    latent_paths:
      - runs/codebook_eval/latents/libero/*.pt

descriptors:
  strides: [2, 3, 5]
  pool: 2

training:
  device: auto
  variants:
    - method: rq
      k: 32
      levels: 3
    - method: rq
      k: 64
      levels: 3
    - method: rq
      k: 128
      levels: 3
```

建议第一批主评估只跑:

- `stride=2/3/5`: 三套互质时间间隔码本。
- `RQ-3`: 每套码本三层 residual quantization。
- `K=32/64/128`: 从小码本开始看 usage、perplexity 和 action relevance。

`K=256/512/1024` 暂时不作为默认项。只有当 `K=128` usage 健康、perplexity 接近上限、
retrieval/action relevance 仍有明显欠拟合迹象时,再向上扩。

KMeans 代码仍保留为临时 baseline,但不再是默认主流程。现在的默认问题是:

```text
三套 stride RQ-3 codebook 是否各自健康,且彼此互补?
```

## Data Preparation

码本训练不直接读原始公开数据集,而是读统一的 Wan latent shards。因此需要先准备:

```text
dataset adapter -> normalized video window -> Wan-VAE latent shard
```

每个 dataset adapter 至少要返回:

```python
{
    "video": Tensor[C, T, H, W],  # normalized to [-1, 1]
}
```

如果能同时返回下面字段,评估会更有用:

```python
{
    "action": Tensor[T_action, D],
    "proprio": Tensor[T_action, D],
    "prompt": str,  # or "task"
}
```

公开数据建议顺序:

```text
1. LIBERO / robomimic
   先做小而干净的码本 sanity check。

2. CALVIN / BridgeData V2
   看长时序和真实视频上 temporal code 是否仍稳定。

3. DROID / Open X-Embodiment subset
   最后做多场景压力测试,先用子集。
```

导出的 latent 放到约定目录:

```text
runs/codebook_eval/latents/libero/*.pt
runs/codebook_eval/latents/calvin/*.pt
runs/codebook_eval/latents/bridgedata_v2/*.pt
runs/codebook_eval/latents/droid_subset/*.pt
```

如果你已经有 `.pt` latent shards,只要满足 `Latent Cache Format`,直接放进对应目录即可。
如果还没有,用 `export_latents_template.yaml` 替换里面的 `export.dataset` 为目标公开数据集
adapter,再运行 `export-latents`。

## GPU Execution

码本训练使用 torch 实现,默认走 GPU-first:

```yaml
training:
  device: auto
```

`auto` 的选择顺序是:

```text
cuda -> mps -> cpu
```

在训练集群上可以显式指定:

```yaml
training:
  device: cuda
```

如果 CUDA 不可用,脚本会回退到 CPU。每行 `summary.tsv` 都会记录实际运行的 `device`。

聚类 assignment 是 chunked torch 计算,不会一次性保留完整 `[N, K]` 距离矩阵。显存/内存不够时,
优先调小:

```yaml
training:
  chunk_size: 2048
  max_vectors_per_descriptor: 50000
```

确认指标稳定后再逐步增加:

```yaml
training:
  chunk_size: 8192
  max_vectors_per_descriptor: 200000
```

## Metrics

当前脚本会输出:

- usage: code 使用率、dead code、perplexity、最大簇占比。
- reconstruction: MSE、relative MSE、R2-like、mean cosine。
- RQ residual: 每层残差是否持续下降,以及每层残差下降比例。
- temporal: 相邻 latent time 的 code 是否稳定但不过度粘滞,以及 transition entropy。
- joint code: 完整 `(level1, level2, level3)` tuple 的 unique/perplexity。
- action relevance: 每层 code 和完整 joint code 对 action segment 的解释度。
- cross-stride: `stride=2/3/5` 三套码本之间的 MI/NMI,判断互补还是重复。

下一步可以继续加:

- retrieval sanity: 同 code 最近邻是否处在相似操作阶段。
- cross-dataset: 一个数据集训练的 descriptor/normalization 逻辑在另一个数据集上是否健康。

## Commands

只测本机训练器:

```bash
python scripts/codebook_eval.py synthetic-smoke
```

用本机 Package Scan v6 跑一个小规模端到端检查:

```bash
python scripts/codebook_eval.py all --config configs/codebook_eval/package_scan_v6_local.yaml
python scripts/codebook_eval.py validate-artifacts --config configs/codebook_eval/package_scan_v6_local.yaml
```

这个配置只导出 16 个窗口,使用较小分辨率和小 K,用途是验证:

```text
Package Scan v6 -> Wan-VAE latent -> stride 2/3/5 descriptors
-> RQ-3 artifacts -> metrics summary -> artifact reconstruction check
```

从已有 latent shards 训练所有候选码本:

```bash
python scripts/codebook_eval.py train --config configs/codebook_eval/public_latent_codebooks.yaml
```

等价一键脚本:

```bash
bash scripts/train_codebooks.sh
```

导出 Wan-VAE latent 模板:

```bash
DIFFSYNTH_MODEL_BASE_PATH=./checkpoints \
python scripts/codebook_eval.py export-latents --config configs/codebook_eval/export_latents_template.yaml
```

默认 `export_latents_template.yaml` 使用 Package Scan v6 作为 adapter 示例。公开数据集接入时,
替换 `export.dataset` 即可。

评估结果会自动生成,不需要手动整理。默认输出文件:

```text
runs/codebook_eval/public_latent_codebooks/
├── summary.tsv
├── summary.json
├── cross_stride_metrics.tsv
├── cross_stride_metrics.json
└── <dataset>/<run>/
    ├── codebook.pt
    └── metrics.json
```

其中:

- `summary.tsv`: 最适合直接打开看对比表。
- `summary.json`: 方便后续画图或脚本读取。
- `cross_stride_metrics.tsv/json`: 三套 stride 码本之间的冗余/互补度量。
- `<dataset>/<run>/codebook.pt`: 冻结码本 artifact。
- `<dataset>/<run>/metrics.json`: 单个候选方案的完整指标。

如果要确认保存的 codebook centers/codes 和 summary 指标一致:

```bash
python scripts/codebook_eval.py validate-artifacts --config <config.yaml>
```

第一眼先看 `summary.tsv` 里的这些列:

```text
device
method, stride, k, levels
relative_mse, r2_like, mean_cosine
level1_usage, level1_perplexity_frac, level1_dead_frac
temporal_same_next_frac, temporal_change_next_frac
joint_code_unique_frac, joint_code_perplexity_frac
joint_transition_entropy, residual_total_reduction
action_joint_r2_in_sample
```

## Decision Rule

我们不应只凭重构误差选择码本。一个候选码本至少要满足:

- usage 健康,没有大面积 dead code。
- RQ 每层残差确实下降。
- temporal code 不随机跳,也不能只复制当前状态。
- 最近邻/任务阶段检查不只是按背景、光照或相机视角聚类。
- 加上 action relevance 后,code 对动作或状态变化有解释力。

只有这些离线指标支持后,再进入下一阶段:

```text
code embedding / code register
-> ActionDiT or MoT mid-layer injection
-> policy and dynamics training
```
