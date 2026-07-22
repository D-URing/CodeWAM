# CodeWAM v1: Architecture and Experiment Plan

Status: canonical planning specification.

本文件只回答两个问题:

1. CodeWAM v1 到底是什么。
2. 需要哪些最少实验,才能决定它是否值得进入大规模训练。

详细背景和备选路线保留在 `CODEWAM_HYBRID_ARCHITECTURE.md` 与
`CODEWAM_NATIVE_DESIGN.md`;后续实现和实验以本文件为准。

## 1. 核心判断

CodeWAM v1 不是 code-only policy,也不是 FastWAM 加一个 token。它是一个双路径状态模型:

```text
continuous Wan-VAE latent  -> 精确几何、接触和动作微调
frozen multi-scale RQ code -> 状态分区、阶段结构和动态先验
```

两条路径共同产生动作。码本不替代连续视觉,不在 policy 训练中流式更新,也不以压缩 KV cache
为主要目标。较小 memory 只是可能的伴随收益。

需要验证的假设只有三个:

```text
H1  Causal RQ codes 能把视觉状态组织成稳定且有动态意义的离散坐标。
H2  在相同模型容量下,z + code 比 z-only 更省数据、更稳健。
H3  预测未来 code 的辅助目标能改善控制表示,且推理时不需要显式 rollout。
```

任何一个假设失败,都应修改或删除相应模块,而不是继续堆叠结构。

## 2. 唯一推荐结构

```text
past/current video
       |
       v
frozen Wan-VAE
       |
       +---------------- continuous latent tokens Z_t ----------------+
       |                                                               |
       +-> causal descriptors d_2,d_3,d_5 -> frozen RQ-3 -> registers R |
                                                                       v
language L + proprio/history P --------------------------> State Mixer
                                                                       |
                                           +---------------------------+
                                           |                           |
                                           v                           v
                                  Action Flow Head          Masked Code Dynamics
                                  action chunk a_t          training only
```

FastWAM 在这里是对照和预训练部件来源,不是 CodeWAM 的定义。CodeWAM 的因果证据必须来自
同一骨架上的 `z-only` 与 `z+code` 对比。

### 2.1 连续视觉路径

对每个 policy 决策时刻 `t`,冻结 Wan-VAE 并保留其空间 latent:

```text
Z_t = E_vae(x_<=t)
```

`Z_t` 经线性投影成为视觉 tokens。不得先做全局平均再作为唯一视觉输入,因为精细控制需要
位置、尺度、姿态、遮挡和接触边缘。

### 2.2 离线状态码本

先把原视频按统一帧率处理,再在 Wan latent 时间轴上定义 `s in {2,3,5}`。每套 descriptor
只使用当前和历史信息:

```text
u_t      = flatten(spatial_pool(Z_t, g x g))
d_s(t)   = concat(u_t, u_{t-s}, u_{t-2s})
c_s(t)   = RQ_s(d_s(t)),  s in {2,3,5}
c_s(t)   = (c_s1, c_s2, c_s3)
```

关键约束:

- `d_s(t)` 是当前状态在不同历史尺度下的坐标,不是 delta-only descriptor。
- descriptor 不使用 `t` 之后的帧,训练与推理完全一致。
- 每个时间槽分别使用训练集统计量标准化,避免某一槽或通道支配距离。
- RQ 为三层残差量化;第一轮只比较 `K in {16,32,64}`。
- 选择满足质量门槛的最小 `K`,不因重构误差更低而自动选择最大码本。
- 数据划分、标准化统计和聚类中心都只由 train episodes 产生。
- 码本训练完成后冻结并版本化;下游 policy 不更新中心。
- 历史不足时对应 code family 使用 missing mask,不复制未来或伪造历史。

`2/3/5` 是待验证的多尺度假设。若某套码本在 held-out probe 中没有独立贡献,就删除它。

### 2.3 Code registers 的位置

码本位于感知与推理的边界。每套码本产生一个全局 register:

```text
q_s = center_s1[c_s1] + center_s2[c_s2] + center_s3[c_s3]
R_s = W_s(q_s) + family_embedding_s
R   = {R_2, R_3, R_5}
```

RQ centers 保持冻结,只有投影 `W_s` 参与下游训练。这样 register 既保留真实码本几何,又能
适配模型 hidden dimension。

三个 registers 作为每层 State Mixer 都可访问的全局 K/V memory。它们只注入一次,在各层
通过投影被读取,不取代连续视觉 tokens,也不取代普通时序 KV cache。这个位置让 code 提供
稳定的全局状态坐标,让 latent 保留局部精度。

### 2.4 State Mixer

State Mixer 保持两种信息的角色差异:

```text
visual tokens:   局部时空交互,保存连续细节
code registers:  全局可见,提供粗到细的状态坐标
action queries:  读取 visual/code/language/proprio,输出连续动作
```

第一版不引入额外可学习视觉 tokenizer。Wan-VAE latent 直接进入连续路径和离线聚类路径;
State Mixer 只是下游推理模型,不是聚类前的第二个 encoder。

### 2.5 两个输出头

动作头使用 flow matching 或 diffusion action chunking:

```text
ActionHead(Z_<=t, R_<=t, L, P_<=t, noisy_action) -> action velocity/noise
```

世界模型头只在训练时使用:

```text
DynamicsHead(state belief, action chunk, masked future-code queries)
    -> future RQ indices
```

它预测未来 code 序列,而不是重建像素。推理时默认删除该头,直接生成动作;只有后续 planning
实验明确需要时才启用 code rollout。

## 3. 信息可见性

唯一硬原则是:动作分支在训练时不能看到部署时不可用的信息。

| Query | current/past Z | current/past code | language/proprio | action tokens | GT future code |
|---|---:|---:|---:|---:|---:|
| visual | yes | yes | yes | no | no |
| action | yes | yes | yes | self | no |
| future-code | yes | yes | yes | yes | masked target only |

未来 code target 只能作为分类标签。它不能作为 K/V 回流到 action queries。

Masked future-code objective 使用:

```text
mask_ratio ~ Uniform(0.5, 1.0)
L = L_action + lambda_code * mean_s,level CE(future_code)
```

第一轮只比较 `lambda_code=0` 和 `lambda_code=0.1`;若辅助梯度明显压过动作梯度,再降低
权重,而不是增加更多 loss。

结构有效后再加入抗捷径训练:

```text
code-family dropout: 0.15,至少保留一套 code
RQ prefix depth:      随机使用 1/2/3 层,只允许前缀
proprio history mask: 0.20
camera dropout:       0.10,仅多相机数据
```

这些策略必须单独做 on/off 对照。反事实 code shuffle、near/far code replacement 只用于诊断,
不作为主训练增强。

## 4. 实验门

实验按顺序执行。上一道门不通过,下一道门不启动。

### Gate 0: 数据和因果正确性

目的:确保后续结果不是数据泄漏或时间错位。

必做检查:

- 按 episode/task 划分 train/val/test,再做标准化与聚类。
- 固定视频帧率并记录一个 latent tick 对应的真实秒数。
- 可视化随机 32 个 `(t,t-s,t-2s)` 样本及其动作区间。
- 校验 VAE cache、episode id、时间戳、相机和动作严格对齐。
- artifact 保存 split manifest、normalization stats、config、seed 和 git commit。

Package Scan v6 只承担 Gate 0 和本机 smoke test,不用于论文结论。

通过条件:所有自动对齐检查通过,人工抽查没有未来帧、跨 episode 或动作错位。

### Gate 1: 码本是否形成视觉状态坐标

主数据先用 LIBERO;再用 BridgeData V2 或 DROID 子集做一次跨场景复核。采用顺序搜索,
避免无意义的全因子网格:

```text
1. spatial scale: g = 1,2,4         固定 s=3,K=32
2. capacity:      K = 16,32,64      固定选中的 g,s=3
3. time family:   s = 2,3,5         固定选中的 g,K
4. RQ depth:      读取同一码本的 level 1 / 1+2 / 1+2+3
```

这最多是 7 个独立主训练项,不是 `3 x 3 x 3` 的 27 项暴力搜索。

评价分为四组:

| 维度 | 指标 | 回答的问题 |
|---|---|---|
| 健康度 | usage, dead fraction, perplexity | 码本是否坍塌 |
| 表达力 | held-out residual reduction, cosine/R2 | 每层 RQ 是否真的增加信息 |
| 几何性 | translation/scale sensitivity, photometric invariance | code 是否捕捉运动几何而非纹理 |
| 语义性 | retrieval montage, stage/action-event agreement | 同 code 是否对应相似操作状态 |

最低通过条件:

- 每层 held-out dead-code fraction `< 10%`。
- RQ 第 2、3 层各自带来至少 `5%` 的 held-out residual 相对下降;否则删除无效层。
- 在相近像素扰动强度下,粗层 code 对光照/颜色的稳定性至少比对平移/缩放高 10 个百分点。
- code distance 与平移/缩放幅度呈正向单调关系,bootstrap 置信区间不跨 0。
- retrieval 的阶段或动作事件一致率显著高于随机和 `single-frame` baseline。
- 三套 family 的联合 probe 优于最佳单 family;无独立增益的 family 被删除。

`delta-only` 只保留为诊断:检查 code change 是否对应视觉变化,不参与主码本选择。

### Gate 2: code 是否对控制有增量价值

冻结 VAE 和 codebooks,使用相同容量的小型 probe,比较:

```text
P0: proprio + language
P1: Z + proprio + language
P2: code + proprio + language
P3: Z + code + proprio + language
```

训练数据比例使用 `5% / 20% / 100%`,每项 3 个 seeds。划分必须按 episode;至少增加一个
held-out task 或 held-out scene 条件。

主指标:

- 连续动作:归一化 MSE、末端位姿误差。
- 离散事件:gripper F1、接触/阶段变化召回率。
- 鲁棒性:遮挡、相机缺失和轻度光照变化下的性能下降。
- 动力学:future-code CE/top-k,并在 code-changing 子集单独报告。

通过条件:

- `P3` 相对 `P1` 在低数据、held-out 或扰动条件中至少两项有稳定增益,3-seed bootstrap
  置信区间不跨 0。
- `P2` 不要求超过 `P3`;它只需证明 code 承载可用的状态模式。
- action-conditioned future-code predictor 在 code-changing 子集上同时优于 copy baseline 和
  no-action predictor。
- 删除任一有效 family 或 RQ prefix 后应出现可解释退化;否则删除冗余部分。

如果 `P3` 不优于 `P1`,不接大模型。优先回查 descriptor、空间尺度、数据多样性和码本容量。

### Gate 3: 完整策略是否真正受益

只保留四个递进模型:

```text
C0: continuous CodeWAM core
C1: C0 + frozen code registers
C2: C1 + masked future-code objective
C3: C2 + structured dropout
F0: FastWAM reference,只作外部对照
```

`C0-C3` 使用相同 backbone、参数预算、训练步数和数据。code 的因果贡献只由 `C0` 对 `C1`
判断,不能用不同骨架的 FastWAM 对比替代。

最终主指标是闭环 task success,其次才是 action loss、future-code accuracy、推理延迟和显存。
每个主结果至少 3 seeds,报告均值、置信区间和失败类型。

保留规则:

- `C1` 不优于 `C0`:删除 code integration,回到 Gate 1/2。
- `C2` 只降低 code CE,但不改善控制、泛化或鲁棒性:删除 dynamics head。
- `C3` 没有缓解可测得的 shortcut:删除对应 dropout。
- 只有闭环成功率改善,才允许声称 CodeWAM 改善控制。

## 5. 数据角色

```text
Package Scan v6: 本机链路、对齐、可视化和回归测试
LIBERO:          第一轮受控 representation/action probe
Bridge/DROID:    跨场景、相机和背景鲁棒性
训练集群:        Gate 3 完整策略与多 seed 闭环评估
```

公开数据训练的 codebook 不应直接假定可迁移到所有机器人。必须分别报告:

```text
in-domain fit
frozen cross-domain evaluation
optional union-data refit
```

## 6. 当前代码与本规范的差距

现有代码是早期可运行原型,不能直接当作 v1 实现:

| 当前实现 | v1 要求 |
|---|---|
| `current/future/delta` descriptor | causal `current/past/past` state descriptor |
| 全量 vectors 内部标准化 | train-only stats,原样应用到 val/test |
| in-sample action R2 | episode-held-out probe |
| trainable `StateEncoder` + EMA 更新 | direct latent descriptor + offline frozen RQ |
| 首帧单 token | C2/C3/C5 三个 code registers |
| policy 训练时更新 codebook | frozen centers,只训练 register projection |
| 单步 future-code head | masked multi-step future-code objective |

因此下一轮工程顺序固定为:

```text
1. 修正 evaluator 的 causal split、normalization 和 held-out metrics
2. 增加 retrieval montage 与 translation/scale perturbation suite
3. 用 Package Scan v6 重跑 smoke,只确认链路
4. 接入 LIBERO latent cache,完成 Gate 1 和 Gate 2
5. 根据结果冻结唯一 tokenizer spec
6. 再实现 FrozenStateTokenizer、CodeRegisterAdapter 和 State Mixer
7. 最后在训练集群运行 Gate 3
```

在 Gate 2 之前,不继续扩建当前 `model.py` 中的在线码本路径。

## 7. 最终定义

```text
CodeWAM is a hybrid world-action model that keeps continuous Wan-VAE latents
for precise control and injects frozen causal multi-scale RQ coordinates as
global state registers, with masked future-code prediction used only as a
training-time world objective.
```

它的创新点不在“把视觉压成 code”,而在于建立一种可冻结、可度量、可反事实验证的离散状态
坐标,并让它与连续视觉细节在结构上分工,最后用闭环控制证明这套分工是否成立。
