# CodeWAM Dataset and 8xA100 Scale Plan

Status: canonical data-selection and offline-codebook execution plan.

本文件规定 CodeWAM v1 在公开数据、Wan feature 导出、海量离线聚类和 8xA100 集群上的
推进方式。模型结构与 mask 仍以 `CODEWAM_V1_PLAN.md` 为准;当前 evaluator 的 legacy 行为见
`CODEBOOK_EVAL.md`。

## 1. 决策摘要

```text
main codebook dataset:        DROID
controlled evaluation:       LIBERO
cross-domain replication:    BridgeData V2
later scale/stress datasets:  AgiBot World and RoboMIND
not a first-stage mixture:    full Open X-Embodiment
```

第一阶段不追求“把能下载的数据全部混起来”。我们先在真实、规模足够、但 embodiment 和
action schema 相对一致的数据上回答:

```text
Q1  Wan latent 中是否存在健康、稳定的多尺度 RQ 状态坐标?
Q2  这些坐标是否描述视觉状态,而不是 camera/dataset/robot identity?
Q3  code 是否给 continuous latent 带来可复现的控制增量?
```

三问成立后才扩大到多 embodiment 数据。

## 2. 数据集分工

| Dataset | 官方规模与模态 | CodeWAM 角色 | 主要风险 |
|---|---|---|---|
| DROID | 76k trajectories,350h,564 scenes,统一 Franka,三路相机、action/proprio | 主码本训练与真实场景 Gate 1/2 | 1.7TB RLDS、相机/场景分布很宽 |
| LIBERO | 130 个受控任务,workspace+wrist RGB、proprio、language | 受控几何/任务 probe 和后续闭环 | 仿真且规模较小,标准结果可能饱和 |
| BridgeData V2 | 60,096 trajectories,24 environments,统一 WidowX | 跨机器人复核和同规格独立 refit | 视角与任务分布比 DROID 窄 |
| AgiBot World | Beta 约 1M trajectories/43.8TB;Alpha 约 92k/8.5TB | 双臂、接触和长程压力测试 | 数据极大、CC BY-NC-SA、系统变量多 |
| RoboMIND | 107k trajectories,479 tasks,4 embodiments,含 failure data | 多 embodiment/失败状态扩展 | action/camera schema 更复杂 |
| Open X-Embodiment | 1M+ episodes,22 embodiments,RLDS 聚合 | 最终跨 embodiment scale study | 容易先学到数据集身份和 robot identity |

官方入口:

- DROID: <https://droid-dataset.github.io/>;数据格式与下载:
  <https://droid-dataset.github.io/droid/the-droid-dataset>
- LIBERO: <https://libero-project.github.io/datasets>;dataset license 为 CC BY 4.0。
- BridgeData V2: <https://rail-berkeley.github.io/bridgedata/>;
  loader/code: <https://github.com/rail-berkeley/bridge_data_v2>
- AgiBot World: <https://github.com/OpenDriveLab/Agibot-World>;data/code 标注为
  CC BY-NC-SA 4.0。
- RoboMIND: <https://x-humanoid-robomind.github.io/>
- Open X-Embodiment: <https://github.com/google-deepmind/open_x_embodiment>

每次下载必须把 dataset revision、URL、checksum 和实际 data terms 写入 manifest。代码仓库
license 不能自动当作数据 license。

## 3. 为什么 DROID 是主数据

DROID 同时满足三个需要:

1. **真实多样性**:大量真实场景、物体、光照、操作者和 camera poses。
2. **系统一致性**:统一 Franka 平台和较完整的 action/proprio schema,减少 embodiment 混杂。
3. **世界模型相关性**:它已经被 action-conditioned latent world model 路线采用;例如
   V-JEPA 2-AC 使用不到 62 小时 DROID robot video。

DROID RLDS 版本约 1.7TB,包含 180x320 的 wrist、exterior-1、exterior-2 RGB 和低维轨迹。
第一阶段不下载 5.6TB/8.7TB raw HD/stereo 版本,因为 RQ 研究不需要 SVO、full-HD stereo 或 depth。

## 4. DROID 数据阶梯

### P0: DROID-100

官方 2GB/100 episodes 调试集。只验证:

- RLDS adapter、episode metadata 和三路 camera 读取。
- action/proprio/video timestamp 对齐。
- prefix-only Wan-VAE 编码与真实 latent tick。
- pooled feature shard、resume 和 checksum。
- 单卡与 8 卡 launcher 行为。

不得用 P0 选择正式 K 或报告研究结论。

### P1: DROID-10k

从 full manifest 中按 scene/task/collector 分层抽取 10k episodes。用于:

- camera policy: exterior-only vs exterior+wrist。
- spatial pool: `g in {1,2,4}`。
- capacity: `K in {16,32,64}`。
- 有效 RQ prefix: L1/L1+L2/L1+L2+L3。
- streaming trainer 的百万向量压力测试。

### P2: DROID-Core

优先使用官方具有 improved camera calibration 的约 36k episodes,完成正式的 held-out
geometry/retrieval/action probe。calibration 不作为 RQ 输入,但用于检查 camera pose 是否成为
code 的主导变量。

### P3: DROID-Full

使用完整 76k episodes 训练选定规格的 Q2/Q3/Q5,并报告规模曲线:

```text
100 episodes -> 10k -> Core-36k -> Full-76k
```

只有规格选择完成后才运行 P3;不在 full data 上暴力搜索全部超参数。

## 5. Split 与采样契约

### 5.1 Split unit

禁止随机拆 frame 或 window。优先级为:

```text
institution/building -> scene -> episode -> frame
```

默认建立:

```text
train: 80% scenes
val:   10% scenes
test:  10% scenes
```

使用稳定 hash 和 task-stratification;另保留 leave-one-institution/building-out 压力测试。所有
normalization、reservoir sample 和 centers 只能读取 train split。

### 5.2 Camera policy

第一版主输入:

```text
exterior-1 + wrist
```

两路分别经同一 frozen Wan-VAE,保留 view identity 后再组成 descriptor。`exterior-2` 默认作为
cross-view consistency 和 camera replacement 测试。后续只有实验证明有增益时才进入主输入。

### 5.3 Sampling policy

连续帧高度相关,不能让长 episode 或静止段按帧数支配聚类。fit reservoir 采用:

- episode/task/scene 分层上限。
- episode 内固定时间 thinning。
- gripper transition、较高 visual/action velocity 单独分桶,避免关键交互被静止段淹没。
- held-out metrics 始终在自然分布上报告,不能只报告 balanced sample。

actions/proprio 只用于分层和 downstream probe,不进入视觉 RQ descriptor。

## 6. Wan pooled feature cache

### 6.1 不保存完整空间 latent

码本搜索只需要固定池化后的 Wan latent。VAE forward 后立即保存最大候选 `g=4`:

```text
Z_t -> adaptive_pool(4x4) -> fp16 pooled feature shard
```

`g=2/1` 可从 `g=4` 做确定性 block average 得到。完整空间 latent 不落盘;连续模型阶段按独立
策略重新计算或只缓存选定 windows。

规划估算,若输入 15Hz、Wan temporal compression 为 4、latent channel 为 48:

```text
350h -> about 4.725M latent ticks
2 views x 48 x 4 x 4 x fp16 -> about 14.5GB pooled values
3 views x 48 x 4 x 4 x fp16 -> about 21.8GB pooled values
```

这是启动前估算,实际 cadence、shape、padding 和 VAE receptive field 必须由 DROID-100 实测。

### 6.2 Shard schema

每个 pooled shard 至少保存:

```python
{
    "episode_id": list[str],
    "split": list[str],
    "timestamps": list[Tensor[T]],
    "pooled_g4": Tensor[N, T, V, 48, 4, 4],  # fp16
    "camera_ids": list[str],
    "action": list[Tensor],
    "proprio": list[Tensor],
    "meta": {
        "dataset_revision": str,
        "wan_model_id": str,
        "wan_revision": str,
        "preprocess_revision": str,
        "source_checksums": list[str],
    },
}
```

真实 episode 长度可变,实现可采用 per-episode tensors 或 offsets+flat storage,不应依赖 padding
制造伪时间点。

### 6.3 Descriptor 不落盘

训练时从 pooled episode stream 即时构造:

```text
D_2(t) = [u_{t-4},  u_{t-2}, u_t]
D_3(t) = [u_{t-6},  u_{t-3}, u_t]
D_5(t) = [u_{t-10}, u_{t-5}, u_t]
```

availability、episode boundary 和时间戳在生成时检查。descriptor、residual 和全量 assignment
都不作为默认永久 cache。

## 7. 海量离线 RQ 训练器

当前 `clustering.py` 是小规模 reference Lloyd 实现。正式 backend 必须满足:

### 7.1 Streaming statistics

- 使用 Welford/mergeable moments 在 train stream 上计算 `mu_s/sigma_s`。
- 多卡对 count/sum/squared-sum 做稳定归并。
- normalization stats 冻结后才开始 fit centers。

### 7.2 Balanced reservoir initialization

- 从 P1/P2 train stream 建立 0.5M-1M descriptor 的分层 reservoir。
- 在 reservoir 上用 deterministic K-Means++ 或 K-Means|| 初始化。
- candidate search 只读取 reservoir;不为每个 K 扫描全量 DROID 50 次。

### 7.3 Distributed blocked Lloyd

每个 rank 只保留当前 batch 和小型统计量:

```text
local nearest-center assignment
-> local sums[K,D], counts[K], inertia
-> all_reduce
-> global center update
```

复杂度随 batch 固定,不能随数据总量增长 RAM/VRAM。空中心从全局 hardest-example reservoir
重置。每个 iteration 保存 center、inertia、counts、RNG 和 shard cursor。

### 7.4 Streaming RQ

三级顺序训练,每一级冻结后进入下一级:

```text
r0 = normalized D
r1 = r0 - e1[c1]
r2 = r1 - e2[c2]
r3 = r2 - e3[c3]
```

residual 只在当前 GPU batch 内产生。input/cache 可为 fp16/bf16;distance、centers、sums 和
metrics accumulation 必须为 fp32。

### 7.5 Artifact policy

正式 artifact 默认只保存:

- train-only normalization。
- 3 family x 有效 RQ levels 的 centers。
- split/data/model/config hashes。
- 每个 center 的少量 representative episode/time ids。
- held-out aggregate metrics 和 reports。

全量 codes 仅在明确需要时按 shard 单独导出,不能打包进 `codebook.pt`。policy 训练和部署只读
artifact,绝不继续更新 centers。

## 8. 顺序搜索,不是全因子暴力网格

### Step A: camera/pool

固定 `s=3,K=32`,在 DROID-10k 比较:

```text
exterior-only vs exterior+wrist
g = 1,2,4
```

### Step B: capacity

固定选中的 camera/pool,比较:

```text
K = 16,32,64
```

选择满足 held-out 质量门槛的最小 K。

### Step C: family/depth

固定 camera/pool/K,独立训练:

```text
Q2, Q3, Q5
RQ prefix = L1 / L1+L2 / L1+L2+L3
```

若某一 level 的 held-out residual reduction 小于门槛或某 family 没有独立 probe 增益,删除它。

### Step D: full refinement

只对最终三个 codebooks 在 DROID-Full 做 1-2 次 streaming Lloyd refinement,再冻结 artifact。

## 9. 8xA100 作业布局

### Job A: feature export

```text
8 independent ranks
rank i owns disjoint RLDS shards
CPU decode/prefetch -> GPU Wan-VAE -> pooled_g4 fp16 -> shard writer
```

不需要 DDP gradient synchronization。每个 rank 原子写临时文件,完成后 rename;支持 episode-level
resume。VAE throughput 先在 DROID-100 测得,再用:

```text
wall time ~= total camera-video hours / (8 x measured realtime multiplier)
```

估算 full export。当前不承诺未经测量的小时数。

### Job B: candidate search

reservoir 规模可由单卡容纳时,一张 A100 可运行一个 candidate;8 卡并行不同 camera/pool/K candidates。
固定 initialization sample 与 seed,保证候选公平。

### Job C: final RQ

选定规格后,8 卡 DDP 顺序训练 Q2/Q3/Q5。每卡读取不同 shards,每次只同步 `K x D` center
statistics。A100 40GB/80GB 都足够;batch size 由启动时显存探测确定。

### Job D: held-out evaluation

8 卡并行 val/test shards,流式累计 usage、perplexity、residual、retrieval candidates、geometry
perturbation 和 action probe sufficient statistics。

### 非 GPU 前提

8 卡并不能弥补慢存储。DROID 路线建议:

```text
usable storage: >= 3TB for RLDS source, pooled cache, staging and reports
CPU:            enough parallel video/RLDS decode workers
I/O:            local NVMe staging or high-throughput shared filesystem
network:        resumable access to the official Google Cloud bucket
```

不下载 raw DROID 可显著降低存储和网络压力。

## 10. 评估矩阵

### DROID in-domain

- held-out scene/building/institution usage、dead fraction、perplexity。
- 每层 residual reduction 和 center-distance error。
- translation/scale sensitivity 与 photometric invariance。
- retrieval montage、camera identity concentration、task/event agreement。
- episode-held-out action/gripper/contact probe。

### BridgeData cross-domain

运行两项不同实验,不能混为一个数字:

```text
1. frozen DROID tokenizer -> BridgeData evaluation
2. same spec, independently refit BridgeData tokenizer
```

第一项测试迁移,第二项测试方法规律是否复现。不能要求跨机器人 numeric code id 对齐。

### LIBERO controlled

- 固定物体/任务下的位置、尺度、光照和 camera 扰动。
- `H-only / C-only / H+C` action probe。
- 后续 C0/C1/C2 闭环策略比较。

## 11. 开发机接续点

### 当前仓库状态

- canonical v1 架构已写入 `CODEWAM_V1_PLAN.md`。
- legacy evaluator 仍保留;新的 manifest、pooled shard、causal descriptor、streaming
  K-Means/RQ、checkpoint 和 frozen artifact 已实现,见 `STREAMING_CODEBOOKS.md`。
- Q2/Q3/Q5 synthetic GPU smoke 与 17 项本机测试已通过。
- 当前在线 EMA `state_codebook` 默认关闭。
- 尚未下载 DROID、LIBERO 或 BridgeData。
- 尚未修改训练集群、共享存储或 8xA100 环境。

### 下一张工程单

```text
Complete the first real-data path and held-out evaluator:

1. Package Scan episode-aware pooled exporter
2. Package Scan held-out split and real Wan shape/cadence smoke
3. streaming usage/perplexity/residual evaluator
4. retrieval/geometry/action-probe report
5. DROID-100 manifest and RLDS adapter
6. scene/task/event balanced reservoir
7. rank-aware shard partition and shared initialization artifact
8. 1-GPU/8-GPU equivalence test and launcher
```

验收不以“跑完”为准,而以这些条件为准:

- peak RAM/VRAM 只随 batch/K/D 变化,不随总 vectors 变化。
- 固定 initialization 时,streaming 与 reference centers/metrics 在小数据上数值一致。
- 1-GPU 与 8-GPU 聚合结果在设定 tolerance 内一致。
- 任意 iteration/level 中断后可恢复。
- train/val/test 与 source checksums 可从 artifact 反查。
- future frame、跨 episode frame 和 held-out statistics 无法进入训练。

### 开发机开始命令

```bash
git checkout main
git pull --ff-only origin main
sed -n '1,260p' docs/DATASET_SCALE_PLAN.md
```

继续对话时可直接引用:

```text
继续 CodeWAM 的 DATASET_SCALE_PLAN,从 EpisodeManifest 和 streaming shard contract 开始实现。
```
