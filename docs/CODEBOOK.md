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

在进入 DROID-10k 前先运行 Wan latent 小样本探测:

```bash
export DATA_ROOT=/path/to/datasets
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export FASTWAM_SRC=/path/to/FastWAM/src

python scripts/probe_wan_latents.py run \
  --config configs/codebook_eval/droid100_wan_latent_probe.yaml
```

该探测把四件事分开报告:

1. 绝对 `u_t` 的方差、有效秩与跨 episode 距离,检查 Wan latent 是否坍缩。
2. `u_t-u_{t-s}` 与缩略图变化、proprio/action 变化的相关性。残差仅作运动诊断,
   正式 RQ 输入仍是 `[u_{t-2s},u_{t-s},u_t]`。
3. `g in {1,2,4}`、`K in {8,16,32}` 的使用率、held-out distortion 和多 seed ARI。
4. `tol in {1e-3,1e-4,1e-5}`、`patience in {2,3}` 的逐轮 inertia、中心位移、
   assignment change 和 early-stop 建议。

DROID 没有逐物体 mask,因此 P0 只能判断“可见场景运动是否进入 latent residual”,不能把背景、
camera motion 与物体自身运动完全分离。自动报告和 cluster montage 用于发现明显失败,不用于选择
最终 K。

#### P0 实测基线: 2026-07-28

在单张 A800 上用官方 DROID-100 的前 12 个 episode 完成了真实 Wan2.2-VAE 探测。输入包括
三路相机、3,063 个原始视频帧和 771 个 Wan latent tick;训练/验证/测试按 episode 分成
8/2/2。checkpoint SHA-256 为
`20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36`。

运行事实:

- 三相机编码单 episode 为 3.4--10.3 秒,最长 596 帧;峰值显存为 1.81--2.11 GiB。
- 12-episode 增量导出为 101.5 秒;完整复用和逐 shard 契约/SHA 校验为 24.0 秒。
- 135 次 K-Means 和 27 次 RQ-3 的优化后分析为 305.81 秒。缓存优化前为约 25.9 分钟;
  新旧离散结果相同,centers 最大绝对差为 `9.54e-7`。
- 自动产物位于 `runs/codebook_eval/droid100_wan_probe/{pooled,analysis}`;运行目录不提交 Git。

Gate 0 结论是 **Wan latent 没有坍缩,且包含明确的可见运动信号**:

| camera | g=2 effective rank / rank95 | g=2 image-motion rho Q2/Q3/Q5 | g=4 mean image-motion rho |
|---|---:|---:|---:|
| exterior-1 | 6.56 / 22 | 0.424 / 0.540 / 0.581 | 0.668 |
| exterior-2 | 6.53 / 18 | 0.308 / 0.500 / 0.508 | 0.597 |
| wrist | 8.03 / 36 | 0.539 / 0.587 / 0.608 | 0.685 |

`g=2` 和 `g=4` 的 18 个 camera/stride 组合全部满足
`adjacent distance < stride distance < far distance < cross-episode distance`。空间信息越完整,
latent residual 与像素变化/proprio 变化的相关性越强;因此不能仅凭 48 维全局平均向量否定
Wan latent,也不能在 P0 阶段直接丢弃 `g=4`。

Gate 1 在这个样本量上 **没有通过**:

| camera | train RQ-3 reduction | validation reduction | test reduction Q2/Q3/Q5 |
|---|---:|---:|---:|
| exterior-1 | 87.6% 至 88.0% | -10.3% 至 -6.8% | -0.3% / -0.9% / 0.3% |
| exterior-2 | 90.3% 至 90.5% | -1.3% 至 -1.0% | 3.6% / 3.7% / 3.6% |
| wrist | 72.3% 至 72.4% | 25.5% 至 33.4% | 29.5% / 24.5% / 22.9% |

两个 exterior test episode 的 train-normalized initial MSE 为 2.64--2.75,远高于 train 的 1.0;
montage 也显示 cluster 很容易按场景背景和机器人姿态分组。wrist 的 held-out reduction 更好,
但其差分同时含大量相机自运动,不能解释成物体运动原语。RQ 第三层的 seed ARI 只有
0.05--0.22,所以“固定三层 RQ”仍是假设,不是 P0 结论。

容量 sweep 中,K 从 8 增到 16/32 往往没有改善 held-out distortion,反而提高死码率并降低
seed 稳定性。P0 暂时保留 `K=16,g=2,RQ-3` 只为了统一诊断口径,不得据此冻结正式规格。
early-stop 可先用 `tol=1e-3,patience=2`;三相机平均在 5.67--8.00 轮停止,验证误差仍在所测
最佳值的 0.5% 内。

因此下一步不是修改 CodeWAM 模型,而是先在 DROID-10k 用 scene/institution 隔离重做
`g={2,4},K={8,16,32}` 和有效 RQ prefix。只有 exterior held-out residual、retrieval 和
camera-identity probe 同时通过后,才把某个 codebook 规格接入模型。

### P1: DROID-10k

从 clean full manifest 中先按 split 分配,再做容量受限的 institution 配额、institution 内
scene round-robin,最后用 collector/task 的当前最小计数打破 episode 选择平局。shard-aware
候选选择以 `split x institution` 为目标,避免为了 10k episodes 默认读取全部 2,048 个
TFRecord shard。这个 balanced sample 只用于规格搜索和 fit;正式 held-out 指标仍在 full
manifest 的自然分布上报告。P1 用于:

- camera policy: exterior-only vs exterior+wrist。
- spatial pool: `g in {1,2,4}`。
- capacity: `K in {8,16,32}`。
- 有效 RQ prefix: L1/L1+L2/L1+L2+L3。
- streaming trainer 的百万向量压力测试。

#### DROID 1.0.1 canonical manifest: 2026-07-29

已把 2,048 个官方 RLDS shard 的 95,658 条精确 `(shard,record,file_path,num_steps)` 索引,
`keep_ranges_1_0_1.json`、raw `metadata.json`、language annotations 和 GCS CRC32C/bytes
做确定性 join。clean manifest 含 58,116 个成功 episode、16,749,080 个 RLDS steps 和
15,024,230 个 eligible steps。排除项全部显式计数:

```text
missing raw metadata       21,933
failure episode            15,040
ambiguous metadata id/path    202 / 263
empty eligible keep ranges     90
raw metadata quality flag      14
```

RLDS `num_steps` 是后续读取的权威长度。57,099 个 episode 的 RLDS 长度比 raw metadata
少 1,另有 254 个更大差异和 763 个完全一致;代码记录 delta 而不猜测或修补。`keep_ranges`
保持原始半开区间,不能把分离区间拼接后制造跨边界时间窗口。

scene hash split 的 episode/scenes 为:

```text
train  45,830 / 1,516
val     6,481 /   204
test    5,805 /   188
```

这里的 1,908 个 strict scene 是 `(institution,building,raw scene_id)` tuple,粒度比论文或
网站汇总 scene taxonomy 更细,不能与公开的汇总数字直接比较。分组隔离和 episode key
uniqueness 已通过检查。assigned manifest fingerprint 为
`223180b62b65f194c9f14f14f5a6af01d5b68f7fba23641bbc708bb430dd5927`。

对 canonical 10k 比较了 candidate multiplier:

| multiplier | selected shards | source bytes | sampled scenes | max collector |
|---:|---:|---:|---:|---:|
| 1.00 | 990 | 870.2 GiB | 1,775 | 720 |
| 1.10 | 1,080 | 947.5 GiB | 1,792 | 721 |
| 1.25 | 1,187 | 1,036.7 GiB | 1,811 | 722 |

因此默认 `1.0`:再读取 77--167 GiB 只增加 17--36 个 scene,且不改善 collector 去偏。
最终 sample 精确为 8,000/1,000/1,000,覆盖 1,775 scenes;最大 collector 占比从 full 的
27.2% 降到 7.2%。institution 配额受各 split 可用容量约束,绝不移动 scene 来补齐配额。
sample fingerprint 为
`481a6febba3a04f374c6a8e91280cad338c38fb9b50e1536cb660c3bb672be4e`。

整 shard LPT rank assignment 已在 canonical 10k 上验证:4 rank 各读约 217.3 GiB,最大差
485 MiB;8 rank 各读 108.4--108.9 GiB,最大差 498 MiB。990 个 shard 不跨 rank,10,000 个
episode 无遗漏。真实 record 解码也已核对 manifest key、source path、412-step shape 和两个
独立 keep ranges;TensorFlow 显式不可见 GPU。

temporal audit 显示 canonical 10k 的 3,379,214 个 RLDS steps 中有 3,002,148 个 eligible
steps,分成 15,202 个独立 keep-range segments。Q2/Q3/Q5 分别有
14,441/13,934/13,155 个足够长的 segments,可生成 696,092/667,483/612,987 个 causal
descriptor ticks。因此默认规则是短 segment 自然跳过对应 family,绝不 padding,也不跨 gap
拼接。

#### Canonical keep-range pilot: 2026-07-29

从 train split 按 13 个 institution 各取 2 个不同 scene,得到 26 episodes。原始 RGB/action
audit 先验证官方 keep range 确实聚焦运动:inside/outside 的 exterior 与 wrist 图像运动中位数
比分别为 2.10 和 11.74,proprio motion 为 371.72;flat 7D action 因含绝对 gripper position
不适合作 idle 判据。

两路相机经 frozen Wan2.2-VAE 后生成 37 个独立 segments、1,753 ticks 和 5.54 MiB
`pooled_g4`。单 A800 首次核心导出 159.97 秒,峰值显存 1.79--2.00 GiB;完整续跑核心校验
2.06 秒。首次证据在 resume report 中保留,所有产物为 `0644`。v3/v4 的 latent、timestamp、
action、proprio 与 action components 已逐 tensor 相等。

Gate 0 在这批 canonical keep-range 数据上再次通过:

| camera | g=4 effective rank / rank95 | latent-image rho Q2/Q3/Q5 | high/low motion residual |
|---|---:|---:|---:|
| exterior-1 | 12.86 / 53 | 0.567 / 0.645 / 0.663 | 1.61 / 1.85 / 1.68 |
| wrist | 19.78 / 136 | 0.543 / 0.593 / 0.650 | 1.59 / 1.56 / 1.59 |

像素变化仍混合 camera 与物体运动,所以该表只证明“可见运动进入 latent”,不宣称完成物体级
解耦。严格因果 Q2/Q3/Q5 得到 1,605/1,531/1,388 个 4,608 维 descriptors。

train-only RQ 工程 sweep 中,K=8/16/32 均无死码,三级总 residual reduction 约为
49%/64%/73%;但 K=32 某些第三级 perplexity 仅为容量的 27.6%。这不能用于选择 K:
样本全是 train,更大 K 降低训练误差是预期现象。固定 K=16、`patience=2` 时,
`tol=1e-3` 各层约 8--13 轮;收紧到 `1e-4/1e-5` 可增至 35/49 轮,总 residual 只再改善
约 0.14--0.75 个百分点。因此 P1 初始默认 `tol=1e-3,patience=2`,最终由 held-out
distortion、usage 和 probe 决定。

性能诊断还发现 64 个 Torch CPU threads 会让短 segment descriptor pass 从 0.15 秒膨胀到
33.5 秒。canonical launcher 现显式使用 `cpu_threads=4`,K-Means++ 在目标 GPU 上运行;
同一 pilot 的完整三族 RQ 为 K=8 14.2 秒,K=16/32 并行候选各约 24--26 秒。

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

使用不依赖 free-form task 文本的稳定 scene hash。task/collector 只参与 train fit sample
内部的去偏,不参与 split,避免同一场景因标注差异跨 split。另保留
leave-one-institution/building-out 压力测试。所有 normalization、reservoir sample 和 centers
只能读取 train split。

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

固定 `s=3,K=16`,在 DROID-10k 比较:

```text
exterior-only vs exterior+wrist
g = 1,2,4
```

配置生成器用 `--strides 3` 只训练代表性 Q3;选定 camera/pool/K 后再用默认
`--strides 2 3 5` 训练三族,避免搜索阶段重复计算。

### Step B: capacity

固定选中的 camera/pool,比较:

```text
K = 8,16,32
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

不需要 DDP gradient synchronization。每个 rank 原子写临时文件,完成后 rename;支持
source-shard-level resume。VAE throughput 先在小样本 pilot 实测,再用:

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
  camera-subset contract、GPU/CPU deterministic K-Means++、blocked Lloyd、三级 RQ、可恢复
  patience 和 frozen artifact。
- `pipeline.py`:一次顺序训练 Q2/Q3/Q5,校验 manifest fingerprint、source checksums、config contract
  、实现 SHA 和恢复参数。
- `evaluation.py`:只读 frozen train normalization/centers,在 val/test 流式累计逐层 residual、
  usage、dead fraction、perplexity、temporal transition、联合 tuple 指标和每中心 retrieval
  anchors。
- `wan_causality.py`:用真实视频完整编码与逐级前缀编码的 latent 一致性审计,显式检测
  Wan-VAE temporal look-ahead。
- `droid_pooled_export.py`:rank-aware exact reader 到双相机 Wan `pooled_g4` 的原子 shard export、
  完整实现 SHA contract、逐 shard progress、首次性能证据保留、allocator trim、resume 和
  segment manifest finalize。
- `codewam/data/droid_manifest.py`:官方 raw metadata、RLDS position、keep ranges、language 和
  shard checksum 的精确 join,scene-isolated split,以及 institution/scene/collector-aware sample。
- `codewam/data/droid_rlds.py`:按 manifest `(shard,record)` 精确读取,未选相机跳过 JPEG decode,
  整 shard 的确定性 rank assignment、completed-episode resume,以及不跨 gap 的 eligible
  segment interface。

input/cache 使用 fp16 或 bf16;normalization、distance、centers 和统计累积使用 fp32。descriptor、
residual 和全量 code assignment 都只在当前 batch 中产生,不作为默认永久 cache。

canonical launcher 已支持 `torchrun`:按 pooled shard 大小做确定性 LPT 分区,跨 rank 合并
train-only moments,由 rank 0 在完整 train stream 上建立确定性 reservoir/K-Means++ 初始化,
再向所有 rank 广播 `K x D` centers。Lloyd/RQ 阶段各 rank 只读自己的 shards并 all-reduce
`K x D` sums、`K` counts 和 inertia;只有 rank 0 写 contract、checkpoint 与 artifact。

当前 57 项单元测试覆盖 manifest round-trip、scene isolation、DROID join/exclusion、
institution/shard-aware sampling、shared-readable atomic artifact、invalid tick、train-only
normalization、batch partition invariance、streaming/reference Lloyd 等价、checkpoint resume、
patience resume、RQ residual 下降、artifact round-trip、双 rank resume 与单卡 centers 等价、
DROID pooled export evidence、Wan causal-prefix 正反例和 Q2/Q3/Q5 train/held-out 一键流程。

## 12. 命令与产物

不需要 Wan 权重或公开数据的回归测试:

```bash
python -m unittest discover -s tests -v
python scripts/train_streaming_codebooks.py smoke \
  --output runs/codebook_eval/streaming_smoke
```

共享盘上的官方 DROID 索引就绪后,只构建 manifest 和 canonical 10k sample:

```bash
export DROID_META_ROOT=/path/to/manifests/droid/official-1.0.1

PYTHONPATH=. python scripts/build_droid_manifest.py \
  --metadata-index "$DROID_META_ROOT/raw-metadata/droid_raw_metadata_1_0_1.jsonl.gz" \
  --rlds-index "$DROID_META_ROOT/rlds-index/droid_1_0_1_rlds_shard_index.jsonl.gz" \
  --keep-ranges "$DROID_META_ROOT/supplemental-bcb840c3/keep_ranges_1_0_1.json" \
  --language-annotations "$DROID_META_ROOT/supplemental-bcb840c3/droid_language_annotations.json" \
  --gcs-metadata "$DROID_META_ROOT/gcs_metadata.txt" \
  --output-dir "$DROID_META_ROOT/codewam-manifests" \
  --sample-size 10000
```

该命令只读 metadata/index,不解码 JPEG,不需要 GPU。输出和报告原子写入、默认 `0644`,
并记录所有输入 SHA-256、manifest fingerprint、分布与 selected shard bytes。

正式 pooled shards 和 manifest 就绪后:

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

同一 output resume 前会重新核对 descriptor、K/L、tol/patience、source checksums、manifest
fingerprint、实现 SHA 和运行参数。contract 不一致时必须使用新目录,不能串用 checkpoint。
held-out evaluator 同样锁定 manifest、pooled shard、codebook 和实现 SHA,且只接受 val/test。

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

下一阶段完成真实数据 Gate 1/2,不提前把未验证码本接入完整模型:

```text
1. 用当前 4xA800 导出 canonical DROID-10k pooled cache并验证 rank resume/finalize
2. 在原始 scene-isolated val/test 上运行 frozen held-out residual/usage evaluator
3. 加入 retrieval、camera identity、geometry 与 action probe report
4. 只用 train 建立 scene/task/event balanced descriptor reservoir
5. 顺序比较 camera、g、K、RQ prefix,不做全组合暴力搜索
6. 固定 initialization 后完成 1-GPU/多 GPU 聚合等价测试
7. Gate 1/2 通过后实现 FrozenRQAdapter 与模型 mask tests
```

验收不以“程序跑完”为准:

- peak RAM/VRAM 只随 batch/K/D 变化,不随总 vectors 变化。
- 固定 initialization 时,streaming 与 reference centers/metrics 在小数据上数值一致。
- 1-GPU 与 8-GPU 聚合结果在设定 tolerance 内一致。
- 任意 iteration/level 中断后可恢复。
- train/val/test 与 source checksums 可从 artifact 反查。
- future frame、跨 episode frame 和 held-out statistics 无法进入训练。
