# Offline Codebook Evaluation

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
-> KMeans / RQ codebooks
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
  variants:
    - method: kmeans
      k: 256
      levels: 1
    - method: rq
      k: 256
      levels: 3
    - method: rq
      k: 512
      levels: 3
```

建议第一批只比较:

- 单层 KMeans: baseline,看单码本能否覆盖状态。
- `RQ-3 x K=256`: 默认轻量候选。
- `RQ-3 x K=512`: 看更大词表是否提升或产生死码。

## Metrics

当前脚本会输出:

- usage: code 使用率、dead code、perplexity、最大簇占比。
- reconstruction: MSE、relative MSE、R2-like、mean cosine。
- RQ residual: 每层残差是否持续下降。
- temporal: 相邻 latent time 的 code 是否稳定但不过度粘滞。

下一步可以继续加:

- action relevance: code 对 action/proprio delta 的解释度。
- retrieval sanity: 同 code 最近邻是否处在相似操作阶段。
- cross-dataset: 一个数据集训练的 descriptor/normalization 逻辑在另一个数据集上是否健康。

## Commands

只测本机训练器:

```bash
python scripts/codebook_eval.py synthetic-smoke
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

输出文件:

```text
runs/codebook_eval/public_latent_codebooks/
├── summary.tsv
├── summary.json
└── <dataset>/<run>/
    ├── codebook.pt
    └── metrics.json
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
