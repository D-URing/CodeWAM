# Codebook Data, Training, and Evaluation

Status: canonical codebook execution specification and implementation status.

本文件统一规定 CodeWAM 的公开数据选择、Wan feature 导出、离线 RQ、评估门和 8xA100
执行方式。模型结构与 mask 以 `ARCHITECTURE.md` 为准;环境和通用运行命令见
`DEVELOPMENT.md`。旧 window evaluator 只保留为兼容代码,不能生成正式 artifact。

文中带日期的 P0/P1 段落保留实验发生时的假设、数字和判定,用于审计研究演化;它们不是
当前模型结构规范。若历史解释与 `ARCHITECTURE.md` 或本文件第 14 节冲突,以后两者为准。

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
Q3  给定当前 world belief 和动作,未来 code transition 是否可学习?
Q4  该 world objective 是否改善策略学习、泛化或闭环控制?
```

Q1/Q2 决定 artifact 能声称什么范围,Q3/Q4 决定码本是否值得进入 world-action model。它们
不能由一个 `code -> current action` 线性读出互相替代。

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

#### Canonical DROID-10k frozen baseline: 2026-07-29

正式 export 覆盖 10,000 个 source episodes、990 个 pooled shards、15,202 个独立 keep-range
segments 和 756,225 个 latent ticks,总计 2,478,763,109 bytes。train/val/test 分别含
602,394/69,310/84,521 ticks;Q2/Q3/Q5 可用 descriptor 总数为
696,092/667,483/612,987。source manifest fingerprint 为
`481a6febba3a04f374c6a8e91280cad338c38fb9b50e1536cb660c3bb672be4e`,
pooled manifest fingerprint 为
`726f19040d3d589cfdc82661059f553d170310f1d99b0b2c65633164b05049c2`。

真实 Wan2.2-VAE causal-prefix audit 对同一 segment 分别编码 1/5/9/13/17/21 帧,比较所有
已产生 latent ticks。两路 camera 的每一行 `max_abs_error=0`、`mismatch_fraction=0`,
因此这批 cache 通过 Gate 0;完整序列没有改变任何较早 latent tick。

双视角 `g=4,K=16,L=3,tol=1e-3,patience=2` 的冻结结果:

| family | train N | iterations L1/L2/L3 | train reduction | val/test reduction | val/test min perplexity/K | val/test active codes |
|---|---:|---:|---:|---:|---:|---:|
| Q2 | 554,007 | 18/11/9 | 32.46% | 28.46% / 28.77% | 0.785 / 0.802 | 15/16/16 ; 15/15/16 |
| Q3 | 531,053 | 11/9/9 | 31.70% | 27.55% / 28.06% | 0.829 / 0.806 | 15/15/16 ; 15/15/16 |
| Q5 | 487,354 | 13/9/8 | 31.03% | 27.08% / 27.43% | 0.796 / 0.796 | 15/16/16 ; 15/16/16 |

held-out 第三级仍独立降低 5.1--6.0% residual,且 L3 全部 16 个中心都激活,当前不能删除。
L1/L2/L3 相邻 code 保持率约为 94.4--95.7% / 83.3--85.1% / 71.9--78.1%,形成稳定的
coarse-to-fine 时间层级。每个 family 的 held-out L1 都稳定缺一个中心;Q3 的 val/test L2
也缺同一个中心。这更像 train-domain-specific center,不是随机初始化造成的整体 collapse。

Q2 的 normalized distortion 暂时比 Q3/Q5 好约 0.7--1.7 个百分点,但这不能证明 Q2 更有
控制价值。高 L1 persistence 也可能主要编码 scene/background/static pose。下一步必须用
camera/pool 对照、动作与未来状态关联、scene/camera concentration 和 retrieval montage
区分“健康量化”与“有用视觉状态”。

生产续跑还暴露并修复了一个 provenance 缺陷:旧实现会用数值相同的 tensor 重写
`codebook.pt`,使文件 SHA 改变并让既有 held-out contract 失效。现在 resume 会先逐 tensor
校验现有 frozen artifact 并保持原文件字节不动;跨 train-resume/evaluation-resume 回归测试
已锁定该行为。基线重建 contract 前后的六行 held-out metrics 逐字段完全相同。

#### P1 camera/pool/capacity 选择: 2026-07-29

固定 Q3 后按顺序完成 camera/pool 与 wrist capacity 对照。所有 association 都只在 train
拟合条件均值并在原始 scene-isolated val/test 上评分;context 指标在同一 held-out split 内
再按 parent episode 分成 fit/evaluation 两半,单 parent group 不参与该指标。

camera/pool 的关键信号是:

| candidate | val/test action gain at best prefix | val/test future-moment gain | val/test scene gain at L3 |
|---|---:|---:|---:|
| exterior `g=4,K=16` | -0.05% / 1.73% | 0.02% / 0.01% | 76.72% / 73.81% |
| dual `g=4,K=16` | 2.40% / 2.84% | 0.50% / 0.42% | 54.91% / 53.79% |
| wrist `g=2,K=16` | 4.29% / 4.43% | 4.37% / 4.51% | 14.63% / 19.14% |
| wrist `g=4,K=16` | 6.20% / 6.37% | 3.86% / 3.84% | 15.61% / 19.85% |

exterior-only 虽有 37.1--37.6% held-out residual reduction,但 action/future association 接近
零且 scene identity 极强;它主要是容易量化的背景/场景坐标,不能据此作为控制码本。dual
descriptor 也被 exterior 信号稀释。wrist `g=4` 的 action/proprio 关联最强,所以第一版
**frozen codebook 选择 wrist-only `g=4`**。这不删除连续路径中的 exterior view:
continuous state 仍保留 exterior+wrist 来支持精确几何和遮挡信息。

wrist `g=1` 的 val/test future latent moment gain 达 10.96%/10.63%,但最佳 action gain 只有
1.63%/1.62%;它更像全局慢运动摘要。它可作为后续 world-side multi-scale auxiliary,不替代
当前有空间布局的主码本。

固定 wrist `g=4,Q3,L=3` 后:

| K | val/test held-out reduction | val/test full-L3 action gain | val/test L3 exact coverage | val/test joint tuple active |
|---:|---:|---:|---:|---:|
| 8 | 28.20% / 29.97% | 6.22% / 6.24% | 99.99% / 99.97% | 92.19% / 93.36% |
| 16 | 33.70% / 35.06% | 5.17% / 5.28% | 99.81% / 99.78% | 90.23% / 90.99% |
| 32 | 37.74% / 39.29% | -0.11% / -0.39% | 84.81% / 86.08% | 53.83% / 56.13% |

K=32 的 L2 仍有约 6.1--6.2% action gain,但 L3 精确覆盖与联合使用率明显下降且完整 prefix
过拟合。K=8 的三个 level 对三类 target 都持续改善,覆盖接近 100%,并用 K=32 四分之一的
中心达到相当或更好的控制关联。因此 P1 当前候选冻结为
**wrist `g=4,K=8,L=3,tol=1e-3,patience=2`**;K=16/32 只保留为 ablation。

这个选择尚未证明 Q2/Q3/Q5 三族联合都必要。单族 future target 取各自 `t+s`,不能直接当作
同一 horizon 比较。最终 family gate 使用严格对齐时间点、共同 `t+1` target 和 train-only
加性类别岭探针,同时报告 joint 相对 best single 以及每个 leave-one-family-out 的增量。

#### P1 three-family single-artifact screen: 2026-07-29

在选定的 wrist `g=4,K=8,L=3` 规格上,Q2/Q3/Q5 一键流水线均完整生成 train、held-out、
association、concentration 和带报告 SHA 的 workflow summary:

| family | iterations L1/L2/L3 | val/test residual reduction | val/test full-L3 action gain | val/test own-horizon future-moment gain | val/test scene gain L3 |
|---|---:|---:|---:|---:|---:|
| Q2 | 11/11/8 | 30.00% / 31.45% | 7.21% / 7.17% | 2.81% / 2.80% | 11.88% / 17.66% |
| Q3 | 9/8/8 | 28.20% / 29.97% | 6.22% / 6.24% | 3.63% / 3.63% | 12.39% / 18.54% |
| Q5 | 9/14/8 | 26.82% / 28.31% | 4.06% / 3.90% | 3.74% / 3.57% | 13.12% / 19.45% |

三族的每一级八个中心在 val/test 全部激活,L3 exact prefix coverage 为
99.97--100.00%。Q2 更偏当前动作,Q5 相对更偏慢速视觉演化,与互质时间窗口的设计动机一致;
但表中的 future target 分别是 `t+2/t+3/t+5`,只能作为族内证据,不能把差异解释为互补性。

#### P1 aligned multi-family contribution: 2026-07-29

联合探针只保留 Q5 也可用的共同 ticks,因此 val/test 精确使用 54,557/68,462 个样本。三个
single、三个 pair 和 joint model 共享 target normalization 与样本;共同 future horizon 为
`t+1`。L3 train-code coverage 最低仍为 99.9835%。默认 `ridge=8` 的结果:

| common target | full joint gain val/test | best single val/test | joint - best single | leave-one-out increment Q2 / Q3 / Q5 val | leave-one-out increment Q2 / Q3 / Q5 test |
|---|---:|---:|---:|---:|---:|
| current action | 6.57% / 6.86% | Q2 6.21% / 6.29% | 0.36% / 0.57% | 1.09% / 0.20% / -0.19% | 1.01% / 0.49% / -0.03% |
| future proprio | 2.69% / 3.02% | Q5 1.87% / Q3 2.07% | 0.82% / 0.95% | 0.25% / 0.32% / 0.56% | 0.31% / 0.49% / 0.48% |
| future latent moment | 2.13% / 2.10% | Q2 1.47% / 1.45% | 0.66% / 0.65% | 0.52% / 0.24% / 0.18% | 0.50% / 0.28% / 0.16% |

`ridge={2,8,32}` sensitivity 的方向完全一致:所有 target/depth/split 的 joint 都优于 best
single。L3 future targets 中三个 family 的 leave-one-out increment 全为正;Q5-L3 只在
current action 上稳定为轻微负值,val 为 -0.20% 至 -0.16%,test 为 -0.03% 至 -0.01%。
Q2+Q3 的 L3 action gain 为 6.76%/6.89%,略高于 all-family 的 6.57%/6.86%;但在 L2,Q5
action increment 又为正。

因此实验层面的结论是 **保留三族**。Q5 对 future targets 有独立贡献,而对 current action
的 code-only 线性关联较弱;这描述了 probe 行为,不等于模型应删除或遮蔽 Q5-L3。

mixed-prefix 直接复测中,最后一个 hybrid 的 action gain 为 6.96%/6.96%,高于 all-L3 的
6.57%/6.86%、Q2+Q3-L3 的 6.76%/6.89% 和 all-L2 的 6.82%/6.21%。hybrid 的 future
proprio 为 2.66%/3.06%、future latent moment 为 2.14%/2.09%,与 all-L3 的
2.69%/3.02%、2.13%/2.10% 基本持平。这个 `Q2-L3+Q3-L3+Q5-L2` profile 继续保留为
**历史 readout ablation**,不再定义 Policy visibility。canonical v1 让九个 available
measurement 全部进入任务无关 `B`,由联合训练学习相关性。这些实验仍是 code-only probe,
尚未回答 action-conditioned code transition 或闭环控制价值。

#### P1 cross-scene retrieval 与时间反事实: 2026-07-29

原 held-out evaluator 只保存每个 center 的全局最近 3 个点,其中相邻 tick 或同一 scene 会
重复占位。naive montage 因而不能区分“码字有场景偏置”和“展示样本本身不够多样”。重新冻结
评估数值不变,只把每码 anchor pool 扩为 32;RGB renderer 再按 scene 去重选择最近 3 个
test clips。每个 clip 精确还原 descriptor 实际读取的 `t-2s,t-s,t` 三个原始 wrist RGB,
并额外显示仅供观察的首尾 RGB 差分:

| family | scene-diverse clips / 24 | 满 3 scenes 的 codes / 8 | median / max source anchor rank | selected-anchor RGB motion eta-squared |
|---|---:|---:|---:|---:|
| Q2 | 22 | 6 | 2.5 / 23 | 0.823 |
| Q3 | 24 | 8 | 3.5 / 23 | 0.808 |
| Q5 | 23 | 7 | 2.0 / 24 | 0.598 |

`eta-squared` 只是 2--3 个跨场景近邻上的描述性分解,不能作显著性或总体 effect size。montage
显示 L1 同时按运动量和明显的容器/台面/颜色/纹理组织;Q2 两个码字、Q5 一个码字在前 32 个
anchor 中仍不足 3 个 scene。因此当前不能宣称 L1 已形成对象级运动原语。

为直接检查时间信息是否被使用,冻结 normalization 和全部 centers,在相同 val/test descriptor
上运行三个反事实:

```text
history_swap:  [z(t-s),  z(t-2s), z(t)]   # 当前端点不变,只交换历史顺序
reverse_time:  [z(t),    z(t-s),  z(t-2s)]
static_current:[z(t),    z(t),    z(t)]
```

下面是 test 的 code-change fraction;`full` 表示任一合法 RQ prefix code 改变:

| family | history-swap L1 / full | reverse L1 / full | static-current L1 / full |
|---|---:|---:|---:|
| Q2 | 3.41% / 13.11% | 2.62% / 21.85% | 22.27% / 66.39% |
| Q3 | 1.63% / 27.65% | 2.73% / 39.30% | 25.52% / 72.71% |
| Q5 | 2.12% / 40.50% | 4.12% / 45.82% | 30.25% / 80.13% |

val 方向独立复现:reverse full-prefix 为 22.84%/42.00%/48.53%,static-current 为
68.38%/75.68%/81.98%。test reverse 的逐层独立变化率分别是
Q2 `2.62/6.17/19.10%`,Q3 `2.73/8.78/36.48%`,Q5 `4.12/33.47/24.63%`。
因此较稳妥的解释是:

- L1 主要是当前内容与粗状态坐标,不能单独当成动态类别。
- 时间顺序主要进入 residual levels;Q2/Q3 更集中于 L3,Q5 的方向信号在 L2 最强。
- Q5 的方向信号在 L2 最强,与 aligned action probe 的 mixed-prefix 结果一致;这只说明层级
  时间敏感性,不决定 Policy visibility,canonical v1 仍读取全部 available levels。
- 反事实是 frozen representation sensitivity,不是物理环境因果干预;对象级平移/缩放仍需
  controlled geometry probe。

当前不修改 canonical descriptor。若跨场景几何/光照测试仍显示静态内容压倒动态,下一轮比较
原始三状态与可逆的
`[z(t), z(t)-z(t-s), z(t-s)-z(t-2s)]` basis。它保留完整当前状态且可重建原三帧,不是
`delta-only`;只有 held-out original-space distortion、action/future probe 和跨场景
retrieval 同时改善时才替换当前输入。

#### P1 RGB-to-RQ usability gate: 2026-07-29

冻结 canonical wrist `g=4,K=8,L=3` artifacts 后,直接从原始 RGB 重走
`resize -> Wan-VAE -> pooled_g4 -> Q2/Q3/Q5`。DROID 使用 32 个 val 和 32 个 test
clip,每个 split 优先选择不同 scene;LIBERO 使用 32 个 clip,覆盖
`spatial/object/goal/10` 四套 suite 的 9/9/7/7 个任务。每个 clip 同时编码 17 个条件:
identity、轻度 brightness/contrast、全局平移/缩放,以及只作用于目标 latent tick 对应末四帧
的平移/缩放。后者仍是整帧合成干预,不是带 object mask 的物理位姿干预。
Wan 每个 224x224 latent tick 是 `48x14x14`,canonical cache 只保存 `48x4x4=768` 维
pooled state;每个 family 的三状态 absolute triplet 因而是 2,304 维,不是把一帧压成
“仅 48 维”再聚类。

identity 路径对 DROID cache 的三族 pooled MSE 和 maximum absolute error 均为 0,九个 level
code 与全部 RQ prefix 100% 一致。因此 raw RGB、Wan causal tick、pooled cache 和 frozen
codes 的端到端时间/预处理 contract 已闭合。

下面的 `photo RQ/natural` 是轻度光照扰动造成的完整 RQ center 位移,除以真实相邻 latent
一步的 center 位移;`endpoint` 是强端点平移/缩放使任一完整 prefix code 改变的比例;
`opposite` 是相反方向产生不同 prefix 的比例:

| family | photo L1 change val/test | photo full change val/test | photo RQ/natural val/test | endpoint change val/test | opposite val/test |
|---|---:|---:|---:|---:|---:|
| Q2 | 10.16% / 10.16% | 65.63% / 60.94% | 1.44 / 2.30 | 33.33% / 27.60% | 51.04% / 39.58% |
| Q3 | 15.63% / 15.63% | 50.78% / 46.88% | 0.78 / 2.32 | 29.17% / 13.02% | 39.58% / 23.96% |
| Q5 | 15.63% / 15.63% | 60.16% / 46.09% | 0.83 / 0.68 | 32.29% / 20.31% | 43.75% / 33.33% |

三个 family、两个 split 的 8px center 位移都不小于 4px,说明存在 dose response。但判定按
最差 `family x split` 而不是总平均:geometry 为 **conditional**,因为 Q3 test 的 endpoint
与 opposite 只有 13.02%/23.96%;photometric 为 **fail**,因为 Q2/Q3 test 的量化位移超过
两个自然步。对应 raw descriptor 位移最多只有 0.266 个自然步,L1 change 也最多 15.63%,
所以失败点主要是硬 RQ suffix 边界放大外观扰动,不是 Wan latent 对几何完全失明。强
endpoint geometry 的 raw descriptor 位移达到自然一步的 0.607--0.871,但量化后只有
0.250--0.851,进一步把“Wan 是否感知变化”和“RQ 是否保留变化”分成两个问题。

动作事件探针仍只把绝对视觉 triplet 编码为 code;`proprio[t]-proprio[t-s]` 只定义 held-out
监督标签。完整 prefix 在 unseen val/test scenes 上对平移/旋转幅值的 normalized accuracy
gain 为 10.74--18.50%,方向为 2.83--10.85%,任何 code coverage 为 100%。gripper gain
仅在 -0.895% 到 +0.324% 间摆动,没有稳定信号。Q5 translation-direction 的平均增益从
L1 2.98% 到 L2 8.36%,L3 仅到 8.60%,再次说明 Q5-L2 的描述性信号较强。轨迹 tick 相关,
这些值是描述性 held-out association,不是显著性或对象因果证明。L3 exact-prefix coverage
为 99.97--100%,但 motion balanced accuracy 仅 0.188--0.379,NMI 仅 0.029--0.056;
因此弱语义不是 backoff 造成,也不足以支持 code-only control。

冻结 DROID artifacts 在 LIBERO 上发生明确 domain collapse:

| family | min active/K | min perplexity/K | photo RQ/natural | endpoint change | opposite |
|---|---:|---:|---:|---:|---:|
| Q2 | 0.50 | 0.189 | 1.44 | 11.98% | 17.71% |
| Q3 | 0.25 | 0.158 | 4.13 | 3.13% | 4.17% |
| Q5 | 0.25 | 0.144 | 7.01 | 4.69% | 8.33% |

因此 cross-domain stress 为 **fail**。这不否定 DROID 域内码本,但禁止把 frozen DROID
artifacts 称为通用视觉 tokenizer。LIBERO 下一步必须分开比较同规格独立 refit、显式
normalization/calibration,以及 simulator 中真正的 object-pose intervention;不同数据域不要求
numeric code id 对齐。

seed 7/19/31 使用完全相同的 descriptors、train-only normalization、`K=8,L=3` 和
`tol=1e-3,patience=2`;只有 initialization seed 不同。所有 27 个 level 都由 early stop
结束,实际轮数和 train residual reduction 为:

| seed | Q2 iterations / reduction | Q3 iterations / reduction | Q5 iterations / reduction |
|---|---:|---:|---:|
| 7 | 11/11/8 / 31.81% | 9/8/8 / 30.23% | 9/14/8 / 28.89% |
| 19 | 9/7/10 / 31.66% | 7/9/8 / 29.97% | 9/9/7 / 28.28% |
| 31 | 15/10/8 / 31.51% | 11/8/13 / 30.41% | 10/8/7 / 28.38% |

共享 val/test descriptors 上,full-RQ residual MSE 的最大跨 seed CV 仅 0.55%,最大相对范围
1.35%;这证明训练目标和 early stop 可重复。但 partition 不可重复:

| family | max distortion CV | minimum NMI L1/L2/L3 | minimum full-prefix NMI |
|---|---:|---:|---:|
| Q2 | 0.23% | 0.560 / 0.138 / 0.088 | 0.540 |
| Q3 | 0.55% | 0.501 / 0.143 / 0.065 | 0.531 |
| Q5 | 0.33% | 0.510 / 0.154 / 0.086 | 0.523 |

完整 prefix 的 label-mapped agreement 只有 4.68--9.70%,中位数 7.31%;full-prefix ARI 也只有
0.082--0.165。高基数 tuple 的 NMI 仍在 0.523 以上,不能覆盖低逐层 NMI、低 ARI 和低 exact
agreement。因此这不是简单的 code id permutation,而是后两层存在多个 distortion 近等价的
residual partitions。按运行前锁定的门槛,seed stability 为 **fail**。

严格聚合的最终状态是:

```text
pass:        causal reproduction, quantizer health, RQ hierarchy,
             Q2/Q3/Q5 complementarity
conditional: scene leakage, synthetic geometry, action-event semantics
fail:        photometric robustness, seed stability,
             frozen DROID -> LIBERO transfer
verdict:     not_ready
```

这个 `fail` 不表示 Wan latent 没有视觉信息,也不表示某一份 frozen artifact 不能被读取;
它阻止的是把当前训练方案称为稳定、可重新生成、跨域通用的视觉 tokenizer。光照响应和
跨域 collapse 必须继续完整披露。绑定 dataset/revision/seed/checksum 的 DROID artifact
可以进入域内 C1/C2 原型,但不得据此声称 code-only precision control、对象运动原语或
跨域 numeric ID 语义;centers 也仍禁止在线更新。

#### P1 frozen-code functional-equivalence screen: 2026-07-29

为裁决“不同 seed 的 suffix partition 虽不一致,是否仍提供相同的下游功能”,固定
seed 7/19/31 的九份 artifact,对同一批 scene-isolated 样本运行四个闭式岭回归读出:

```text
P0 = current proprio
P1 = current proprio + continuous H
P2 = current proprio + frozen C
P3 = current proprio + continuous H + frozen C
```

这里没有语言、DiT 或策略训练。`H` 精确使用 Q2/Q3/Q5 所见七个不重复的 wrist
`pooled_g4` 状态:

```text
{u(t-10), u(t-6), u(t-5), u(t-4), u(t-3), u(t-2), u(t)}
```

每个状态为 `48x4x4=768` 维,加 14 维 current proprio 后共 5,390 维。`C` 固定为 Policy
hybrid profile `Q2-L3 + Q3-L3 + Q5-L2`,分别形成容量为 512/512/64 的 categorical
prefix;它们由各自 seed 的 frozen centers 计算。监督目标只是当前 tick 的 7 维 DROID
action。任何 future observation、future action、held-out statistics 或其他 seed 的 code
都不会进入输入。

训练场景按固定 hash 构造嵌套的 `5%/20%/100%` 子集;val/test 始终为相同的
55,814/69,819 个向量。连续特征归一化、code active columns 和全部回归系数只由对应 train
子集拟合。正则强度只根据 `P1` 的 val normalized-MSE reduction 从
`{1e-4,1e-3,1e-2,1e-1,1}` 选择,随后不变地复用于 P0/P2/P3 和三个 seed。闭式求解没有
readout optimization seed,因此跨 seed 差异只来自码本。

下表报告 test normalized-MSE reduction;数值越高越好,区间为三个 codebook seeds:

| train fraction | scenes / pooled records / vectors | alpha | P0 | P1 | P1-P0 | P2-P0 | P3-P1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | 70 / 395 / 16,129 | 0.01 | 92.406% | 90.742% | -1.664 pp | [-1.928,-1.471] pp | [-2.131,-1.880] pp |
| 20% | 280 / 2,301 / 102,978 | 0.001 | 93.052% | 94.008% | +0.957 pp | [-0.077,-0.047] pp | [-0.163,-0.134] pp |
| 100% | 1,397 / 12,241 / 487,354 | 0.001 | 92.905% | 94.244% | +1.339 pp | [+0.103,+0.116] pp | [-0.023,-0.019] pp |

val 的 `P3-P1` 同样全部为负:5% 为 `[-2.126,-1.719] pp`,20% 为
`[-0.151,-0.130] pp`,100% 为 `[-0.020,-0.018] pp`。三份 report 的 SHA-256 为:

```text
5%    59c6836f816c2efe154420fd4464056b5542f09877c1c4e41cfdbb5c0592dd86
20%   2f53f8f24094bf5370f99da2af664216c6d3d57fcccdb4ae4edcf0bff7e279e7
100%  22c410d36ddaf02a56b66414fdbf3f26f6875460e3c478759d15186bba2955a2
```

90% 以上的绝对读出分数不是策略成功率。当前 proprio 与绝对 Cartesian/gripper action
高度耦合,所以 P0 本来就很强。5% 时 5,390 维 P1 甚至比 P0 差,是明确的小样本高维过拟合;
到 100% 时连续视觉产生稳定的 `+1.339 pp`,证明 `H` 有动作相关增量。code-only 的增量只有
约 `+0.11 pp`,且一旦精确的 `H` 已经存在,hard categorical code 没有留下额外线性信息。

因此两件事必须分开:

1. **functional seed equivalence 成立**:随着数据增加,三个 seed 的 `P3-P1` range 从
   0.251 pp 收缩到 0.004 pp。不同 residual partitions 没有产生不同的 full-data 功能结果。
2. **additive action screen 未通过**:`P3-P1` 在所有 fraction/split/seed 都不为正。它拒绝
   “把 hard categorical code 加进线性 current-action readout 就会带来增益”这一接口,不拒绝
   code 作为静态 world vocabulary 或 future-transition label。

下一项模型实验不再继续拟合 `code -> current action` proxy,而是在同一真实轨迹 tuple 上比较
`current-code persistence`、`B without action`、`B + true action` 和
`B + shuffled action` 的 future-code NLL。margin 与跨 seed 指标继续作为 artifact 诊断,
不是进入 `FrozenCodebookAdapter` 的先决条件。

### P2: DROID-Core

优先使用官方具有 improved camera calibration 的约 36k episodes,完成正式的 held-out
geometry/retrieval/action probe。calibration 不作为 RQ 输入,但用于检查 camera pose 是否成为
code 的主导变量。

### P3: DROID-Full

使用完整 76k episodes 训练 P1/P2 family gate 后保留的 codebooks,并报告规模曲线:

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

P1 证据把两条路径分开:

```text
frozen RQ descriptor:  wrist
continuous state H:    exterior-1 + wrist
```

两路仍分别经同一 frozen Wan-VAE。离散码本只用 wrist `g=4`,避免 exterior scene/background
identity 支配 code;连续状态保留 view identity 和双视角空间细节。`exterior-2` 默认作为
cross-view consistency 和 camera replacement 测试,只有后续 H-path ablation 证明有增益时才
进入连续主输入。

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
- 每个保留 family x 有效 RQ levels 的 centers。
- split/data/model/config hashes。
- 每个 center 的少量 representative episode/time ids。
- held-out aggregate metrics 和 reports。

全量 codes 仅在明确需要时按 shard 单独导出,不能打包进 `codebook.pt`。policy 训练和部署只读
artifact,绝不继续更新 centers。

## 8. 顺序搜索,不是全因子暴力网格

### Step A: camera/pool

固定 `s=3,K=16`,已在 DROID-10k 比较:

```text
exterior-only vs wrist-only vs exterior+wrist
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

只对最终保留的 codebooks 在 DROID-Full 做 1-2 次 streaming Lloyd refinement,再冻结
artifact。

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

第一版冻结关联探针不把 delta 当作码本输入。它用 `t` 的 causal RQ prefix 在 train 上拟合
最小容量的条件均值，再在 val/test 预测当前 action、`t+s` proprio change 和 `t+s` Wan
latent spatial-moment change。报告 standardized MSE 相对 train global-mean baseline 的
改善、exact prefix coverage 和逐级 backoff;这只能证明关联与预测价值，不能单独证明物体级
语义或因果控制。

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
- `association.py`:只用 train 拟合 RQ-prefix 条件均值,在 val/test 测量 action、future proprio
  和 future Wan spatial-moment association,包含逐级 backoff 与 exact coverage。
- `concentration.py`:在 held-out split 内按 parent episode 隔离 fit/evaluation,测量 scene、
  institution 和 exact-task concentration;descriptive MI 不冒充泛化指标。
- `family_association.py`:在共同 tick/target 上拟合 train-only additive categorical ridge,
  比较 single/pair/joint family 与 leave-one-family-out 增量。
- `retrieval.py`:把 held-out anchors 经 pooled provenance 精确映射回 DROID RGB,按
  scene/parent 去重并生成三帧 trajectory、差分 montage 和 machine-readable summary。
- `temporal_sensitivity.py`:冻结 codebook 后运行 history-swap、time-reversal 和
  static-current 反事实,报告逐层/prefix code change 与 cross-reconstruction penalty。
- `action_events.py`:train-only event thresholds 和 prefix tables,在 unseen scenes 上测量
  Cartesian magnitude/direction 与 gripper association;delta 只作标签。
- `visual_perturbations.py`:从 DROID/LIBERO RGB 重编码 Wan latent,验证 cache reproduction、
  photometric nuisance、translation/scale、direction、dose response 和 frozen transfer。
- `seed_stability.py`:在共享 held-out descriptors 上比较独立 seed 的 distortion CV、逐层
  NMI/ARI、联合 prefix partition 和 label-mapped agreement。
- `functional_readout.py`:用嵌套 scene-train fractions 和固定 val/test,比较
  proprio/H/C/H+C 的闭式动作读出,并隔离三个 frozen codebook seeds 的功能差异。
- `usability.py`:验证所有报告的 contract/artifact/manifest provenance,按十道最差组 gate
  生成 JSON 与 Markdown;不使用可掩盖失败的单一总分。
- `workflow.py`:可恢复地串联 train、held-out、association、concentration,并锁定各报告 SHA。
- `wan_causality.py`:用真实视频完整编码与逐级前缀编码的 latent 一致性审计,显式检测
  Wan-VAE temporal look-ahead。
- `droid_pooled_export.py`:rank-aware exact reader 到双相机 Wan `pooled_g4` 的原子 shard export、
  完整实现 SHA contract、逐 shard progress、首次性能证据保留、allocator trim、resume 和
  segment manifest finalize。
- `codewam/data/droid_manifest.py`:官方 raw metadata、RLDS position、keep ranges、language 和
  shard checksum 的精确 join,scene-isolated split,以及 institution/scene/collector-aware sample。
- `codewam/data/droid_rlds.py`:按 manifest `(shard,record)` 精确读取,未选相机跳过 JPEG decode,
  整 shard 的确定性 rank assignment、completed-episode resume、不跨 gap 的 eligible
  segment interface、RLDS first/last/terminal flags,以及只解码指定绝对帧的稀疏 RGB reader。
- `codewam/data/droid_endpoint.py`:验证官方 observation/action endpoint、末步 action validity、
  boundary flags 和 current-vs-shifted action alignment。
- `codewam/data/frozen_assignment.py`:从未池化多相机 Wan latent 确定性生成冻结
  Q2/Q3/Q5 RQ IDs 与精确 descriptor source indices,不更新 normalization 或 centers。
- `codewam/data/joint_cache.py`:去重 episode tensors、原子 shard/sidecar、逻辑 window index、
  完整 provenance/hash 校验、lazy random access 和 typed model collator。
- `codewam/data/joint_cache_export.py`:按完整 DROID source shard 分 rank,串联 RLDS、Wan、
  frozen assignment、action endpoint 与 resumable JointWindowCache。
- `codewam/experiments/gate2.py`:等预算 PERSIST/NOACT/TRUE/SHUFFLE、TRUE 模型动作干预、
  all/changed/family/overlap 流式指标和 episode-block paired bootstrap。

input/cache 使用 fp16 或 bf16;normalization、distance、centers 和统计累积使用 fp32。descriptor、
residual 和全量 code assignment 都只在当前 batch 中产生,不作为默认永久 cache。

canonical launcher 已支持 `torchrun`:按 pooled shard 大小做确定性 LPT 分区,跨 rank 合并
train-only moments,由 rank 0 在完整 train stream 上建立确定性 reservoir/K-Means++ 初始化,
再向所有 rank 广播 `K x D` centers。Lloyd/RQ 阶段各 rank 只读自己的 shards并 all-reduce
`K x D` sums、`K` counts 和 inertia;只有 rank 0 写 contract、checkpoint 与 artifact。

当前 159 项单元测试(本机仅 1 项 CUDA 专项跳过)覆盖 manifest round-trip、scene isolation、
DROID join/exclusion、
institution/shard-aware sampling、shared-readable atomic artifact、invalid tick、train-only
normalization、batch partition invariance、streaming/reference Lloyd 等价、checkpoint resume、
patience resume、RQ residual 下降、artifact round-trip、双 rank resume 与单卡 centers 等价、
DROID pooled export evidence、Wan causal-prefix 正反例、candidate workflow、跨 parent
folds、对齐 multi-family contribution、scene-diverse retrieval provenance 和冻结 temporal
counterfactual,以及 RGB-to-cache reproduction、视觉扰动、动作事件、独立 seed
label-permutation、十门可用性决策、functional readout 的精确因果状态 union、nested
scene subsets、非线性 code partition、三 seed end-to-end/resume,以及 RLDS endpoint、
冻结因果赋码、JointWindowCache round-trip、防篡改、生产 export 对齐、固定 action permutation
和完整 Gate 2 CPU 回归。

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
  --cameras wrist_image_left \
  --strides 2 3 5 \
  --pool 4 --k 8 --levels 3 --device cuda:0

python scripts/run_streaming_codebook_candidate.py \
  --train-config "$RQ_ROOT/configs/train_g4_k8_l3.yaml" \
  --evaluation-config "$RQ_ROOT/configs/evaluate_g4_k8_l3.yaml"
```

跨场景 RGB 检索先用单独 held-out config 把 `representatives_per_code` 提高到 32,不得覆盖标准
三样本报告。随后运行:

```bash
python scripts/render_codebook_retrievals.py \
  --source-manifest "$DROID_MANIFEST" \
  --pooled-manifest "$POOLED_ROOT/pooled_manifest.jsonl" \
  --droid-data-dir "$DROID_RLDS_ROOT" \
  --evaluation-report "$Q2_HELDOUT_32/evaluation_report.json" \
  --evaluation-report "$Q3_HELDOUT_32/evaluation_report.json" \
  --evaluation-report "$Q5_HELDOUT_32/evaluation_report.json" \
  --output-dir "$RQ_ROOT/rgb_retrieval_scene" \
  --splits test --levels 1 --diversity-by scene

python scripts/probe_codebook_temporal_sensitivity.py \
  --manifest "$POOLED_ROOT/pooled_manifest.jsonl" \
  --pooled-shards "$POOLED_ROOT/pooled/*.pt" \
  --artifact Q2="$Q2_ROOT/Q2/codebook.pt" \
  --artifact Q3="$Q3_ROOT/Q3/codebook.pt" \
  --artifact Q5="$Q5_ROOT/Q5/codebook.pt" \
  --output-dir "$RQ_ROOT/temporal_sensitivity" \
  --splits val test --device cuda:0
```

动作事件、RGB 扰动和跨域压力测试:

```bash
python scripts/probe_codebook_action_events.py \
  --manifest "$POOLED_ROOT/pooled_manifest.jsonl" \
  --pooled-shards "$POOLED_ROOT/pooled/*.pt" \
  --artifact Q2="$Q2_ROOT/Q2/codebook.pt" \
  --artifact Q3="$Q3_ROOT/Q3/codebook.pt" \
  --artifact Q5="$Q5_ROOT/Q5/codebook.pt" \
  --output-dir "$RQ_ROOT/action_events" --device cuda:0

python scripts/probe_rgb_visual_perturbations.py \
  --source droid \
  --artifact Q2="$Q2_ROOT/Q2/codebook.pt" \
  --artifact Q3="$Q3_ROOT/Q3/codebook.pt" \
  --artifact Q5="$Q5_ROOT/Q5/codebook.pt" \
  --output-dir "$RQ_ROOT/rgb_perturbation_test" \
  --vae-path "$WAN_VAE_PATH" --fastwam-src "$FASTWAM_SRC" \
  --droid-pooled-manifest "$POOLED_ROOT/pooled_manifest.jsonl" \
  --droid-source-manifest "$DROID_MANIFEST" \
  --droid-data-dir "$DROID_RLDS_ROOT" \
  --droid-split test --max-samples 32 --device cuda:0

python scripts/probe_rgb_visual_perturbations.py \
  --source libero \
  --artifact Q2="$Q2_ROOT/Q2/codebook.pt" \
  --artifact Q3="$Q3_ROOT/Q3/codebook.pt" \
  --artifact Q5="$Q5_ROOT/Q5/codebook.pt" \
  --output-dir "$RQ_ROOT/rgb_perturbation_libero" \
  --vae-path "$WAN_VAE_PATH" --fastwam-src "$FASTWAM_SRC" \
  --libero-root "$LIBERO_ROOT" --max-samples 32 --device cuda:0
```

三套完整 codebook 分别用至少三个不同 seed 训练后,在完全相同的 held-out descriptors 上比较:

```bash
python scripts/probe_rq_seed_stability.py \
  --manifest "$POOLED_ROOT/pooled_manifest.jsonl" \
  --pooled-shards "$POOLED_ROOT/pooled/*.pt" \
  --artifact seed7:Q2="$Q2_ROOT/Q2/codebook.pt" \
  --artifact seed7:Q3="$Q3_ROOT/Q3/codebook.pt" \
  --artifact seed7:Q5="$Q5_ROOT/Q5/codebook.pt" \
  --artifact seed19:Q2="$SEED19_Q2" \
  --artifact seed19:Q3="$SEED19_Q3" \
  --artifact seed19:Q5="$SEED19_Q5" \
  --artifact seed31:Q2="$SEED31_Q2" \
  --artifact seed31:Q3="$SEED31_Q3" \
  --artifact seed31:Q5="$SEED31_Q5" \
  --reference-run seed7 \
  --output-dir "$RQ_ROOT/seed_stability" --device cuda:0
```

跨 seed 功能等价与 `H+C` 增量使用同一组九份 artifact:

```bash
python scripts/probe_codebook_functional_readout.py \
  --manifest "$POOLED_ROOT/pooled_manifest.jsonl" \
  --pooled-shards "$POOLED_ROOT/pooled/*.pt" \
  --artifact seed7:Q2="$SEED7_Q2" \
  --artifact seed7:Q3="$SEED7_Q3" \
  --artifact seed7:Q5="$SEED7_Q5" \
  --artifact seed19:Q2="$SEED19_Q2" \
  --artifact seed19:Q3="$SEED19_Q3" \
  --artifact seed19:Q5="$SEED19_Q5" \
  --artifact seed31:Q2="$SEED31_Q2" \
  --artifact seed31:Q3="$SEED31_Q3" \
  --artifact seed31:Q5="$SEED31_Q5" \
  --code-depth Q2=3 --code-depth Q3=3 --code-depth Q5=2 \
  --train-fraction 1.0 \
  --output-dir "$RQ_ROOT/functional_readout_full" \
  --device cuda:0
```

规模曲线把 `--train-fraction/--output-dir` 分别改为
`0.05/functional_readout_05`、`0.20/functional_readout_20` 和
`1.0/functional_readout_full`;三个目录不能混用。输出 contract 锁定 manifest、全部 pooled
shard SHA、九份 artifact、训练 scene subset、特征/target 定义和实现 SHA。report 自动包含
alpha sweep、P0/P1/P2/P3 的 val/test 原始行与 `P3-P1` 跨 seed summary。

最后用 `scripts/build_rq_usability_report.py` 汇总 comparison、family association、retrieval、
val/test temporal、causal audit、DROID/LIBERO visual、action events、seed stability 和
capacity reports。聚合器重算每个 contract hash,并要求所有 canonical 输入共享相同的
Q2/Q3/Q5 artifact SHA 与 pooled manifest fingerprint;不同版本不能拼成一份结论。

renderer 的 L1 是可独立解释的粗中心。若显式请求 L2/L3,输出表示不同 earlier prefix 下对同一
residual center 的使用,不能把单个 suffix code 当成完整状态类别。temporal probe 只改变
held-out descriptor 并重新读取 frozen centers,不更新 artifact。

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

## 13. Cache 边界

### 13.1 旧 Package Scan cache

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

### 13.2 Canonical `pooled_g4` 也不是 Policy cache

DROID pooled exporter 保存:

```text
pooled_g4       [T_latent,V,48,4,4]
action/proprio  [T_latent,D]
```

这里的 action/proprio 由 `relative_latent_frame_indices` 从 RLDS step 序列抽样,目的是把
held-out code 诊断对齐到 latent tick。它没有保存相邻 latent ticks 之间原始控制频率上的完整
action 序列;`pooled_g4` 也丢掉了 `ContinuousStateEncoder` 所需的 unpooled 空间细节。因此:

```text
可以: RQ fit/eval、retrieval、temporal sensitivity、粗粒度 transition probe
不可以: 直接构造 action chunk、训练 canonical C0/C1/C2、声称精细控制链路已对齐
```

不能对两行下采样 action 做插值后伪造 GT action chunk。正式联合样本必须回到原始 RLDS
step/action 序列,并使用同一 Wan revision 导出未池化 latent。

## 14. JointWindowCache 与 Gate 2

### 14.1 已实现的数据 contract

`JointWindowCache v1` 已按 episode/keep-range 去重保存:

```text
unpooled_latent          [T_latent,V,48,H_z,W_z]
latent_source_indices    [T_latent]
source_action/proprio    [T_source,D]
source_action_valid      [T_source]
code_ids/available       [T_latent,3,3] / [T_latent,3]
descriptor_sources       [T_latent,3,3]
```

`windows.jsonl` 只记录 state/history/action/future 的逻辑切片、当前/未来 code、descriptor
overlap 与 artifact hashes。writer 和 reader 都逐窗口复核 split/range、RLDS endpoint、
source-rate action、future latent tick、availability、code label 和 overlap;任何 shard 或
index SHA 改变都会在 tensor 进入模型前失败。language tokens 当前可选:缺失时只关闭 action
imitation supervision,不关闭 Gate 2 dynamics。

finalize 额外生成 `window_actions.pt`:它只复制体积很小的 source-rate action chunks,不复制
视觉 latent。Gate 2 每个 epoch 先随机 shard 顺序、再随机 shard 内窗口;错误动作从该 mmap
action index 读取。这样仍保持固定随机对照,但不会因 donor 位于另一 episode 而反复加载大
latent shard。每个 learned condition 复用同一组 DataLoader workers,epoch 只更新 sampler;
每个 episode shard 的 SHA 因而在每个 condition/reader 进程首次加载时验证一次,不会每轮
重启 worker 后重新散列。评估则为 val/test 各建立一次 loader,由 PERSIST、NOACT、TRUE、
SHUFFLE 和两个 TRUE intervention 复用。
`summary.json` 同时按 split/Q2/Q3/Q5 报告 changed windows、changed parent episodes 和每层
RQ component/prefix 的变化数。同一原始 RLDS episode 的多个 keep-range segments 只计一个
独立 parent episode。

生产 finalize 还会核验同一 `world_size` 的全体 rank report、每个 rank 的计划/完成 shard
数,以及 report 与磁盘 sidecar 的 source-shard 集合。缺任一 rank、少任一 shard 或混入旧
report 都直接失败。仅 `--max-source-shards` 工程 smoke 可显式增加
`--allow-partial-finalize`;该状态会写入 `summary.json`,不能冒充全量 cache。

官方 RLDS 端点固定为:

```text
observation[t] -- action[t:t+h] --> observation[t+h]
```

`is_last` step 的 action 无效。2026-07-30 在 DROID-100 32 条轨迹、8,892 steps 上验证:
全部 boundary flags 合法;current action 相对 `action[t+1]` 的关节/笛卡尔速度 cosine margin
为 `0.02781/0.03217`,高于预注册 `0.005`。

### 14.2 真实工程 smoke

从 canonical DROID-10k manifest 的同一 source shard 选取 train/val/test 各两个 episodes,
七个独立 keep-range segments 共约 977 个保留 source frames。单 A800 使用两路相机、
官方 Wan2.2-VAE 和 wrist `g=4,K=8,L=3` 三份 frozen artifacts:

```text
首次导出                 32.81 s
episode/windows          7 / 157
train/val/test windows   24 / 71 / 62
Q2/Q3/Q5 overlap        1 / 0 / 0,全部 157 个窗口一致
cache contract           671e6a9a4080c18a4a4410488ab57b7fcb0e5e4b226a13383b248da5e580d16b
episode shard SHA        febbba3a9f0938ee3d1b04b35c4182d41c7b78d98534f7977bff5e91de4c379f
```

相同 export 命令续跑逐项校验后返回 `reused`,不重新加载 VAE。contract 除 checkpoint、
source、artifact 和 CodeWAM 实现外,还锁定 FastWAM VAE、loader 与 converter 三个文件 SHA。

真实 cache 上的 Gate 2 engineering smoke 为每个 learned condition 两个 optimizer steps,
三组预算相等且 NOACT action encoder 保持初始化。PERSIST/NOACT/TRUE/SHUFFLE、
TRUE@NOACT 和 TRUE@SHUFFLE 均完成 GPU 前向、流式指标、checkpoint 与报告。test 只有两个
独立 episodes,因此协议按 `minimum_gate_episodes=30` 返回 `invalid`。该结果只证明链路可跑,
不能支持“动作有用或无用”的研究结论。

### 14.3 正式多卡导出

exporter 不使用 DDP 梯度同步;`torchrun` 只提供 rank/world/local-rank。每个 rank 拥有完整
TFRecord source shards,`LOCAL_RANK` 自动选择对应 GPU:

多 rank 不同时反序列化 Wan-VAE checkpoint。每个节点使用 `/tmp` 下按 cache contract
命名的文件锁,依次完成 checkpoint load、模型迁移到各自 GPU 和 CPU allocator 回收;锁释放
后各 rank 的 JPEG decode、VAE encode、冻结赋码和 shard 写入完全并行。这样避免受限 pod
主存被多个约 2.8GB checkpoint load 峰值击穿,不串行化正式导出。

长轨迹也不整段、整视角同时建立 float32 resize 中间量。canonical bilinear/antialias
预处理按 64 帧分块写入同一输出,每个 camera view 随即搬到对应 GPU 并独立完成 Wan encode,
只保留较小的 latent 再处理下一视角。该调度保持预处理数值定义和 view 顺序,同时让 CPU
峰值由“两路完整 float 视频”降为“一路 dtype 视频 + 一个 64 帧 float 工作块”。

```bash
NPROC="${NPROC:-8}"
torchrun --standalone --nproc-per-node="$NPROC" \
  scripts/export_joint_window_cache.py \
  --source-manifest "$DROID_10K_MANIFEST" \
  --data-dir "$DROID_RLDS_ROOT" \
  --output-dir "$JOINT_CACHE_ROOT" \
  --endpoint-audit "$ENDPOINT_AUDIT" \
  --artifact Q2="$Q2_ARTIFACT" \
  --artifact Q3="$Q3_ARTIFACT" \
  --artifact Q5="$Q5_ARTIFACT" \
  --chart-name droid \
  --vae-path "$WAN_VAE_PATH" \
  --fastwam-src "$FASTWAM_ROOT" \
  --camera exterior_image_1_left \
  --camera wrist_image_left \
  --dtype bfloat16

python scripts/export_joint_window_cache.py \
  --output-dir "$JOINT_CACHE_ROOT" \
  --finalize-only
```

`FASTWAM_ROOT` 可指仓库根目录或直接指包含 `fastwam/` 的 `src/`。中断后原命令续跑;contract
不一致必须换 output directory,不能覆盖旧 shard。正式 finalize 默认要求最后一次完整运行的
`NPROC` 份 rank report 覆盖所有计划 shard;小规模 `--max-source-shards` smoke 才可显式使用
`--allow-partial-finalize`。

`NPROC` 同时受 GPU 和 cgroup 主存约束,不能直接等于可见 GPU 数。DROID 长轨迹会使每个
TensorFlow/PyTorch worker 保留自己的 allocator 高水位;当前实测 16 GB pod 中四 rank
可通过隔离 smoke,但连续导出在第 12 个 source shard 触发 OOM,两 rank 则稳定在约
5.7-6.3 GB。该规格下使用 `NPROC=2`;四/八 rank 节点分别建议至少提供约 32/64 GB pod
主存并先运行最坏长轨迹 smoke。rank/world size 不进入 cache contract,所以失败后可用更小
`NPROC` 原地续跑已完成的原子 shard;finalize 只接受某一组完整且覆盖全部 sidecar 的报告。

### 14.4 正式 Gate 2

四条件和判断口径固定:

```text
PERSIST       current IDs 作为 future IDs,无概率 NLL
NOACT         p(C_future | B)
TRUE          p(C_future | B,A_aligned)
SHUFFLE       p(C_future | B,A_fixed-wrong)
TRUE@NOACT    TRUE-trained model 移除 action
TRUE@SHUFFLE  TRUE-trained model 注入固定 wrong action
```

`SHUFFLE` 只在同 split/horizon 内置换并锁定 permutation hash。所有学习条件共享初始化、
数据顺序、optimizer 和 step budget;loss 只有 future-code CE。主指标是 changed-family
normalized NLL,同时报告 all/stable、family、overlap、prefix accuracy、Brier/ECE/entropy
和 chart-local center MSE。PERSIST 不伪造 NLL。

```bash
for SEED in 7 19 31; do
  torchrun --standalone --nproc-per-node=8 \
    scripts/run_gate2.py \
    --config configs/gate2/droid_joint_v1.yaml \
    --cache-dir "$JOINT_CACHE_ROOT" \
    --output-dir "$GATE2_ROOT/seed-${SEED}" \
    --artifact Q2="$Q2_ARTIFACT" \
    --artifact Q3="$Q3_ARTIFACT" \
    --artifact Q5="$Q5_ARTIFACT" \
    --seed "$SEED"
done

python scripts/summarize_gate2_seeds.py \
  --report "$GATE2_ROOT/seed-7/report.json" \
  --report "$GATE2_ROOT/seed-19/report.json" \
  --report "$GATE2_ROOT/seed-31/report.json" \
  --output "$GATE2_ROOT/multi_seed_summary.json"
```

每个 seed 的 test parent-episode-block paired bootstrap 至少需要 30 个共同 changed-code
原始 episodes,正式目标不少于 100 个。`TRUE` 相对 NOACT、SHUFFLE 与 TRUE@SHUFFLE 的
95% CI 上界必须
全部小于零才通过;总判定要求三个预注册 seed 各自通过。跨 seed 均值只作稳定性描述,不把
三个 seed 当额外 episode 伪造 pooled CI。先完成 independent head;通过后才比较 prefix、
Stage-0 和 C0/C1/C2。失败时先检查 endpoint、coverage、overlap 和 action relevance,
不增加辅助 loss 强行调门。
