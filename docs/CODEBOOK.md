# Codebook Data, Training, and Evaluation

Status: canonical codebook execution specification and implementation status.

本文件统一规定 CodeWAM 的公开数据选择、Wan feature 导出、离线 RQ、评估门和 8xA100
执行方式。模型结构与 mask 以 `ARCHITECTURE.md` 为准;环境和通用运行命令见
`DEVELOPMENT.md`。旧 window evaluator 只保留为兼容代码,不能生成正式 artifact。

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
| BridgeData V2 | 60,064 trajectories,24 environments,统一 WidowX | 跨机器人复核和同规格独立 refit | 视角与任务分布比 DROID 窄 |
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

所有数据集共享三层判断,不能只用重构误差选择码本:

1. **量化健康度**:usage、dead fraction、perplexity、最大簇占比、每层 residual reduction、
   center-distance error 和多 seed 稳定性。
2. **视觉状态语义**:时间稳定性与 transition entropy、translation/scale sensitivity、
   photometric invariance、跨相机一致性、retrieval montage、camera/task/event concentration。
3. **控制增量**:固定 probe 容量比较 `proprio-only`、`H-only`、`C-only` 和 `H+C`,并做
   family shuffle、RQ suffix dropout、nearest/far-center replacement。

三套联合 probe 必须优于最佳单 family;无独立贡献的 family 删除。RQ 某层若不能在 held-out
数据上稳定降低 residual 或改善 probe,就缩短合法 prefix,不因预设为 RQ-3 强行保留。

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

## 11. 已实现的底座

当前 canonical 路径由 `codewam/codebook_eval/` 提供:

- `manifest.py`:episode provenance、稳定 fingerprint、scene/building/institution group split 和
  split-isolation 检查。
- `shards.py`:`codewam.pooled-feature-shard.v1` 原子 writer、SHA-256、逐 shard/episode reader 和
  `g=4 -> g=2/1` nested pooling。
- `streaming.py`:因果 Q2/Q3/Q5 descriptor、train-only Welford normalization、uniform reservoir、
  deterministic K-Means++、blocked Lloyd、三级 RQ、checkpoint/resume 和 frozen artifact。
- `pipeline.py`:一次顺序训练 Q2/Q3/Q5,校验 manifest fingerprint、source checksums、config contract
  和恢复参数。

input/cache 使用 fp16 或 bf16;normalization、distance、centers 和统计累积使用 fp32。descriptor、
residual 和全量 code assignment 都只在当前 batch 中产生,不作为默认永久 cache。

底层已经具备 distributed collective primitive,但 launcher 会主动拒绝 `world_size > 1`。在
rank-aware shard partition 与共享初始化 artifact 完成前,正式任务只能使用一个进程;8 张 GPU
可以先并行不同候选,不能伪装成一个分布式 RQ run。

当前 17 项单元测试覆盖 manifest round-trip、scene isolation、invalid tick、train-only
normalization、batch partition invariance、streaming/reference Lloyd 等价、checkpoint resume、
RQ residual 下降、artifact round-trip 和 Q2/Q3/Q5 一键训练。

## 12. 命令与产物

不需要 Wan 权重或公开数据的回归测试:

```bash
python -m unittest discover -s tests -v
python scripts/train_streaming_codebooks.py smoke \
  --output runs/codebook_eval/streaming_smoke
```

正式 pooled shards 和 manifest 就绪后:

```bash
python scripts/train_streaming_codebooks.py train \
  --config configs/codebook_eval/streaming_rq_template.yaml
```

每个 family 的标准输出:

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

同一 output resume 前会重新核对 descriptor、K/L、source checksums、manifest fingerprint 和
运行参数。contract 不一致时必须使用新目录,不能串用 checkpoint。

`scripts/codebook_eval.py`、`configs/codebook_eval/package_scan_v6_*.yaml` 和
`public_latent_codebooks.yaml` 属于旧 window evaluator。它们可用于本机回归或历史结果复算,
不能生成 canonical artifact。

## 13. 旧 cache 的边界

旧 Package Scan cache 使用匿名重叠 window:

```text
latents: [N,48,7,7,14]
meta:    window index and prompt only
```

它不能转成正式 cache:Q5 至少需要 11 个连续 latent ticks;window 没有可靠的 episode、timestamp
和 split provenance;重叠 tick 无法去重;并且保存的是 full window latent 而不是逐 episode
`pooled_g4`。不得给旧 window 人造 episode id 后报告正式结果。

Package Scan v6 只用于本机链路、可视化和回归测试。研究结论使用 DROID、LIBERO 和
BridgeData V2。

## 14. 下一张工程单

下一阶段只完成真实数据 Gate 0/1,不提前修改完整模型:

```text
1. DROID RLDS 与 LIBERO HDF5 -> EpisodeManifest adapters
2. episode-aware prefix-only Wan-VAE -> pooled_g4 exporter
3. DROID-100 + LIBERO 小规模 cadence/shape/future-leak audit
4. held-out usage/perplexity/residual evaluator
5. retrieval/geometry/camera/action-probe report
6. scene/task/event balanced reservoir
7. DROID-10k 顺序规格搜索
8. rank-aware shard partition、共享初始化与 1-GPU/8-GPU 等价测试
```

验收不以“程序跑完”为准:

- peak RAM/VRAM 只随 batch/K/D 变化,不随总 vectors 变化。
- 固定 initialization 时,streaming 与 reference centers/metrics 在小数据上数值一致。
- 1-GPU 与 8-GPU 聚合结果在设定 tolerance 内一致。
- 任意 iteration/level 中断后可恢复。
- train/val/test 与 source checksums 可从 artifact 反查。
- future frame、跨 episode frame 和 held-out statistics 无法进入训练。
