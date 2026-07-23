# CodeWAM v1: Canonical Architecture and Mask Program

Status: canonical planning and implementation specification.

本文件是 CodeWAM v1 的唯一结构规范。它同时约束离线码本、模型接口、训练模式、attention
可见性和实验门。研究背景与备选方案仍保留在 `CODEWAM_HYBRID_ARCHITECTURE.md` 和
`CODEWAM_NATIVE_DESIGN.md`,但若表述冲突,以本文件为准。

## 0. 一句话定义

CodeWAM 是一个连续精度与离散状态坐标并存的 world-action model:

```text
continuous Wan latent -> 保留位置、尺度、接触和动作微调信息
frozen causal RQ      -> 提供可度量、可预测、可干预的多时间尺度状态坐标
```

码本不替代连续视觉,不在 policy 训练中更新,也不把所有历史压成一个万能 token。它首先是
对当前视觉历史的离散测量。较小 KV cache 或 TTT memory 是可能的伴随收益,不是 v1 的中心目标。

## 1. 不可违反的结构约束

以下约束应直接写成单元测试或训练断言:

1. **严格因果**:决策时刻 `t` 的任何状态输入只来自 `<=t` 的观测和 `<t` 的已执行动作。
2. **三套 RQ 独立**:`Q_2`、`Q_3`、`Q_5` 的训练样本、中心和每一级索引语义均独立。
3. **九个 code token 不合并**:三套 family x 三层 RQ 始终保留为 9 个可区分的测量 token。
4. **码本只读**:Wan-VAE、归一化统计和 RQ centers 在下游训练中冻结并版本化。
5. **连续路径常驻**:动作头始终可以访问未量化 Wan latent 的下游连续表示。
6. **状态先于任务分支**:共享状态前缀不能读取当前待生成动作或未来 target。
7. **动作不看未来**:action branch 中不存在 future slots;GT future code 只能是 world loss label。
8. **训练模式显式分离**:Policy、Forward Dynamics 和 Video Prior 使用不同输入契约与 mask。
9. **RQ 层级只用合法前缀**:允许 `{L1}`、`{L1,L2}`、`{L1,L2,L3}`,禁止单独暴露 L2/L3。
10. **动作保持连续**:v1 使用 flow matching/diffusion action chunk,不建立 action codebook。

另外,`2/3/5` 只表示时间间隔假设,不能预先命名为“局部/中期/阶段”语义。语义必须由
held-out 检索、probe 和干预实验得到。

## 2. 已知量、未知量与监督标签

先固定每个决策时刻的信息边界,再讨论模型结构。

### 2.1 部署时已知

```text
x_<=t       当前及历史多相机观测
p_<=t       当前及历史 proprio
l           任务语言
a_<t        已经执行过的动作
```

它们可以因果地产生:

```text
Z_t         frozen Wan-VAE latent
H_t         未量化的连续视觉状态
C_t         9 个当前离散 code 测量
B_t         当前局部 belief
M_t         可选的跨步工作记忆
```

### 2.2 部署时未知

```text
A_t         当前需要生成的 action chunk
C_{t+h}     未来视觉的离散状态
x_{t+h}     未来观测
```

训练集中的 GT action 和 GT future code 是监督标签,不因“数据里已经存在”就自动变成模型输入。
这一区分是全部 mask 设计的起点。

## 3. 三套因果、独立的 RQ 测量

### 3.1 Wan latent 与时间对齐

在统一帧率和统一 latent tick 上,冻结 Wan-VAE:

```text
Z_t = E_vae(x_<=t)
u_t = flatten(spatial_pool(Z_t, g x g))
```

`spatial_pool` 是固定算子,不是聚类前的可学习 encoder。聚类直接发生在 Wan latent 上。
VAE cache 必须保存 episode、camera、原始时间戳、latent tick 和模型 revision;任何 temporal
compression/look-ahead 都要通过 prefix-only 编码对齐测试确认不会读取未来。

### 3.2 三个 descriptor

第一版采用互质间隔 `S={2,3,5}`:

```text
D_2(t) = [u_{t-4},  u_{t-2}, u_t]
D_3(t) = [u_{t-6},  u_{t-3}, u_t]
D_5(t) = [u_{t-10}, u_{t-5}, u_t]
```

这是三个不同的历史观测窗口,不是同一个 descriptor 的三个名字,也不是 delta-only 表示。
平移、缩放、姿态变化和物体交互由三个实际 latent 状态共同表达。历史不足时输出 availability
mask,不得复制最早帧、跨 episode 取帧或使用未来帧补齐。

每个 family 只使用训练集统计量归一化:

```text
\bar D_s(t) = (D_s(t) - mu_s) / sigma_s
```

`mu_s`、`sigma_s` 属于码本 artifact,验证集和测试集只读取它们。

### 3.3 每套三层 residual quantization

对每个 `s` 独立训练一个三层 RQ:

```text
r_s^0 = \bar D_s
c_s^l = argmin_k ||r_s^{l-1} - e_{s,l,k}||^2
r_s^l = r_s^{l-1} - e_{s,l,c_s^l},  l in {1,2,3}
```

因此当前时刻得到:

```text
C_t = {
  c_{2,1}, c_{2,2}, c_{2,3},
  c_{3,1}, c_{3,2}, c_{3,3},
  c_{5,1}, c_{5,2}, c_{5,3}
}
```

`c_{2,1}=7` 与 `c_{3,1}=7` 没有共同语义。即使都叫第一层,其输入分布和中心也不同。
RQ 内部每层确实是对上一层 residual 做 K-Means,但九组中心都不能共享。

候选容量先比较 `K in {16,32,64}`。选择满足 held-out 质量门槛的最小 K;若某层没有带来
稳定 residual reduction,就删掉该层,而不是因为预设为 RQ-3 强行保留。

### 3.4 Artifact 契约

一个可加载的 frozen tokenizer 至少包含:

```text
manifest.json
  dataset/split manifest and episode hashes
  fps, latent tick, camera policy, history offsets
  Wan model id/revision and preprocessing revision
  spatial pool, descriptor dimension, K, RQ depth
  seed, trainer config, source git commit
normalization/{2,3,5}.pt
centers/{2,3,5}/level_{1,2,3}.pt
metrics/heldout_metrics.json
```

模型配置只能引用完整 artifact id,不能在 policy 启动时临时拟合或更新中心。

## 4. 四种信息身份

CodeWAM 不把所有 token 混成同一种“状态”。四种信息具有不同生命周期:

| 身份 | 符号 | 来源 | 是否量化 | 单次 forward 内是否更新 | 作用 |
|---|---|---|---:|---:|---|
| 离散测量 | `C_t/E_t` | frozen RQ | yes | no | 状态坐标与动态模式 |
| 连续状态 | `H_t` | Wan latent + visual backbone | no | 网络层内更新 | 精确几何与局部细节 |
| 当前 belief | `B_t` | state aggregator | no | yes | 汇总当前决策证据 |
| 工作记忆 | `M_t` | causal memory update | no | 跨步更新 | 可选的长期任务上下文 |

动作 `A_t` 和未来 code `F_t` 不属于当前状态,它们是待生成变量。语言 `L`、proprio `P`
和过去动作是已知条件。

这一区分也回答 codebook 应放在哪里:它位于 frozen perception 与可学习 state reasoning 的边界,
作为只读测量进入聚合器;它不是 TTT memory,也不是 action token。

## 5. Canonical v1 架构

### 5.1 总图

```text
                         OFFLINE, train split only
Wan latent cache -> D2/D3/D5 -> independent RQ2/RQ3/RQ5 -> frozen artifact

                         ONLINE, decision step t
past/current images
       |
       v
frozen Wan-VAE
       |
       +-> unquantized latent -> Visual/State Backbone ---------> H_t
       |
       +-> D2/D3/D5 -> frozen RQ2/RQ3/RQ5 -> 9 measurements ---> E_t

H_t + E_t + language + proprio
       |
       v
Per-Step State Aggregator -> B_t
       |
       +-> optional causal Memory(M_{t-1}, a_{t-1}) -> M_t
       |
       v
S_t = [H_t, E_t, B_t, M_t, L, P_t]
       |
       +---------------------------+
       |                           |
       v                           v
Action DiT                    Code Dynamics Expert
continuous action chunk      future code distributions
inference + training         training only by default
```

FastWAM/Wan 提供初始化、VAE、Video DiT 和 Action DiT 设计参考,但不是此图的强制拓扑。
特别是 v1 不依赖对称 MoT 让 action/video token 在每层任意互看。

### 5.2 连续视觉路径

`H_t` 必须保留空间 token,不得只用 global average token:

```text
H_t = VisualBackbone(Project_z(Z_<=t), time/camera embeddings)
```

第一版可复用当前 3072-wide Video DiT 的部分权重,但输出通过显式 adapter 进入 state width。
Action DiT 当前是 1024-wide,因此 code center 不应直接硬塞进两个专家;所有维度差异在接口
adapter 中解决。投影只是模型接口,不是重新训练一套聚类 encoder。

### 5.3 九个只读 code measurement tokens

每个 RQ center 单独投影:

```text
E_{s,l}(t) = Project_{s,l}(e_{s,l,c_{s,l}(t)})
             + ScaleEmbedding_s
             + LevelEmbedding_l
             + AvailabilityEmbedding_{s,t}
```

输出是 9 个 token,不是下面这种求和:

```text
禁止: E_s = e_{s,1} + e_{s,2} + e_{s,3}
```

原因是 residual level 表示由粗到细的不同误差分量。先相加会抹掉层级身份,也无法做 prefix
dropout、逐层贡献分析或层级未来预测。

在 state aggregator 内,`E_t` 只作为固定 K/V side memory。它们不发 query、不吸收 `H/B/M`
信息,也不在层间被改写。可学习部分只有 `Project_{s,l}` 与 family/level embeddings。
若某个 family 因历史不足而 unavailable,对应三个 slot 仍保留稳定位置,但 center lookup 被
missing embedding 代替且不产生伪 code index。

### 5.4 Per-Step State Aggregator

使用少量 learned belief queries 汇总当前决策证据:

```text
B_t = Aggregator(
    query=B_init,
    kv=[H_t, readonly_kv(E_t), L, P_t]
)
```

`H_t` 保存局部连续细节,`E_t` 提供离散坐标,`B_t` 才是允许两者融合后的当前 belief。
`readonly_kv` 表示 token 内容不被 attention layer 改写,不表示对可学习 projection 做
`detach`。这种单向 cross-attention 比把九个 code token 丢进普通 self-attention 更容易审计。

### 5.5 可选工作记忆

v1 预留 MemoryPort,第一轮完整模型可关闭。启用时只允许:

```text
M_t = Update(M_{t-1}, B_t, E_t, a_{t-1})
M_t = M_{t-1} + tanh(g_t) * DeltaM_t
```

门控初始化在接近 0,避免随机 memory 覆盖当前证据。当前待生成动作 `a_t` 和未来状态不得参与
更新。第一版可以使用少量 gated recurrent registers;只有在长时依赖证据成立后再比较 RoboTTT
式 TTT state。无论采用哪种 memory,`E_t` 仍是只读测量,不能与可更新的 `M_t` 共用身份。

## 6. 分阶段计算与硬可见性

v1 用 staged computation 消除含糊的对称 attention。共享状态只计算一次,随后分叉:

```text
Stage A: S_t = State(x_<=t, p_<=t, l, a_<t)
Stage B-policy: ActionExpert(S_t, A_tau)
Stage B-FD:     WorldExpert(S_t, A_clean, F_query/F_context)
Stage B-prior:  WorldExpert(S_t, no_action, F_query/F_context)
```

### 6.1 可见性矩阵

`yes` 表示该 query 可以读取该 K/V;`label` 不是 K/V,只参与 loss。

| Query/producer | `Z/H` | current `E` | `L/P` | `B_t` | memory | `a_prev` | noisy `A_tau` | clean `A_t` | partial future context | GT future target |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `H_t` | yes | no | optional | no | no | no | no | no | no | no |
| `E_t` | source | self | no | no | no | no | no | no | no | no |
| `B_t` | yes | yes | yes | no | no | no | no | no | no | no |
| `M_t` | no | yes | no | yes | previous | yes | no | no | no | no |
| Policy action query | yes | yes | yes | yes | current | yes | self | no | absent | label only |
| FD future query | yes | yes | yes | yes | current | yes | no | yes | world-only | label only |
| Prior future query | yes | yes | yes | yes | current | yes | no | absent | world-only | label only |

实际实现中 action query 读取的是预先构造的 `S_t`,表中展开它的来源是为了审计。

### 6.2 三条必须自动测试的防泄漏规则

1. Policy mode 的 token sequence 和 attention mask 中都没有 future slot,不是仅把未来值设为 0。
2. 同一 `S_t` 分叉出的 world activations不能作为 action branch 的 K/V,也不能写回共享状态。
3. 把 GT future code 随机置换时,固定噪声下的 policy 输出必须逐位不变;否则存在结构泄漏。

## 7. Mask Program: 三种训练模式

一个 batch item 先采样 mode,再由 mode builder 构造 token、timestep、mask 和 loss。不要用一个
“万能 mask”依靠隐式位置关系同时表达所有任务。

### 7.1 Policy mode

目的:从当前状态生成连续 action chunk。

```text
known:    S_t, diffusion/flow timestep tau, noised action A_tau
unknown:  clean action A_t
absent:   all future-code slots and world branch activations
loss:     flow-matching action loss
```

动作 chunk 内部使用与执行语义一致的 group-causal 或 chunk mask。每个序列 chunk 独立采样
action noise/timestep,避免整段轨迹共享噪声而泄露简单时间模板。

### 7.2 Forward Dynamics mode

目的:学习“当前状态 + 已执行动作 -> 未来离散视觉状态”。

```text
known:    S_t, clean executed action chunk A_t
unknown:  future code slots C_{t+h}
loss:     hierarchical future-code loss + short rollout loss
```

该模式的 clean action 只进入 WorldExpert。ActionExpert 不运行,或与其共享同一个只读 `S_t`
并行运行,但二者 activation 永不互通。

### 7.3 Video Prior mode

目的:利用无动作视频学习 action-free 视觉动态先验。

```text
known:    S_t
absent:   action condition
unknown:  future code slots C_{t+h}
loss:     action-free future-code loss
```

它可以使用没有 robot action 的公开视频,但必须带显式 `no_action` type embedding,不能把缺失动作
填成零向量并与真实静止动作混淆。第一轮 `lambda_prior` 可设为 0,待 FD 基线成立后再启用。

### 7.4 Optional IDM mode

`current state + observed future -> action` 只保留为后续 representation probe,不是 v1 policy
训练默认项。若加入,其 branch 与部署 action branch 参数和 mask 必须显式区分,不能让观测未来
形成训练捷径。

## 8. Future-code world objective

### 8.1 Future mask 分布

只在 WorldExpert 内构造 future slots:

```text
50% batches: all future code slots masked
50% batches: mask_ratio ~ Uniform(0.5, 1.0)
```

未遮挡 future code 只用于学习同一 world branch 内的时空条件结构。它们永远不进入 `S_t` 或
ActionExpert。所有 mask 都遵守 family availability 和合法 RQ prefix。

mask 先以 `(h,s)` family slot 为单位采样,再在 slot 内选择合法 prefix。一个待预测的 L2/L3
query 在 teacher forcing 时只能读取同一 slot 的更低层 GT prefix,不能读取自身或更细层 target。
partial-mask 指标与 all-masked rollout 指标分开报告,模型选择以 all-masked 结果为主。

### 8.2 每个 family 独立、每个 RQ 层级自回归

对 `s in {2,3,5}` 和未来 horizon `h`,预测:

```text
p(c_{s,1}^{t+h} | S_t, A_t)
p(c_{s,2}^{t+h} | S_t, A_t, c_{s,1}^{t+h})
p(c_{s,3}^{t+h} | S_t, A_t, c_{s,1}^{t+h}, c_{s,2}^{t+h})
```

family 之间不共享分类器语义;可以共享 WorldExpert trunk,但每个 `(s,l)` 使用独立输出 head。
训练先用 teacher forcing,再以短 scheduled sampling 检查预测误差是否沿 RQ 层级失控。

### 8.3 Loss

```text
L_policy = FlowMatch(ActionExpert(S_t, A_tau), target_velocity)

L_FD = sum_{h,s,l} w_{h,s,l} * CE(p_{h,s,l}, c_{h,s,l})
       + beta_metric * L_center_distance
       + beta_roll * L_2step_rollout

L_prior = action-free future-code loss

L_total = L_policy + lambda_fd * L_FD + lambda_prior * L_prior
```

`L_center_distance` 使用 frozen center distance 给“错到邻近中心”和“错到远端中心”不同代价;
不能代替 CE。两步 rollout 用第一步预测 code 的 embedding 和下一 action 预测第二步,GT 第二步
仍只作为 label。第一轮先做 `lambda_fd in {0, small}` 对照,并记录共享 state trunk 上 policy/world
梯度范数与夹角。

### 8.4 梯度路由

| 模块 | `L_policy` | `L_FD/L_prior` | 是否冻结 |
|---|---:|---:|---:|
| Wan-VAE | no | no | yes |
| RQ normalization/centers | no | no | yes |
| Visual/State Backbone | yes | yes | no |
| code projections | yes | yes | no |
| State Aggregator/Memory | yes | yes | no |
| ActionExpert | yes | no | no |
| WorldExpert/heads | no | yes | no |

future target 必须在 `no_grad` 下由同一版本 frozen tokenizer 计算。world loss 不应更新 ActionExpert。

## 9. 输入可用性与抗捷径 mask

这些 mask 只作用于当前已知信息,并与 Policy/FD/Prior mode mask 正交:

| Mask | 粒度 | 规则 | 目的 |
|---|---|---|---|
| history availability | whole family | 历史不足则 Q2/Q3/Q5 对应 family unavailable | 保持真实因果边界 |
| family dropout | 3 levels together | 整套丢 Q2、Q3 或 Q5 | 检查 family 冗余 |
| RQ prefix | within family | 仅 L1 / L1+L2 / L1+L2+L3 | 保持 residual 语义 |
| all-code dropout | all 9 tokens | 小概率全部丢弃 | 保留 latent-only fallback |
| proprio history | past P only | 保留当前 proprio | 降低轨迹模板捷径 |
| camera dropout | whole camera | 多相机时整路丢弃 | 相机缺失鲁棒性 |
| visual patch mask | spatial patches | 不常规删除全部 H | 逼迫跨区域证据整合 |
| action-loss mask | sample | 无 action 数据不计算 imitation loss | 合法利用公开视频 |

建议在 Gate 3 前只启用 availability mask;其余逐项 on/off,不要一次全部叠加后无法归因。

mask 只能阻断非法信息,不能证明模型正确使用合法信息。必须配套以下反事实诊断:

```text
code family shuffle across episodes
replace code with nearest/far center
drop one family or one valid RQ suffix
shuffle proprio history while keeping current proprio
mask one camera or selected latent patches
reset/freeze optional memory
```

固定 action noise 后测量动作分布、gripper event 和 world prediction 的变化。attention weight 只作
可视化,不作为“模型使用了 code”的证据。

## 10. 研究借鉴与明确取舍

| 工作 | 采用的原则 | 不直接照搬的部分 |
|---|---|---|
| FastWAM | 训练期 world objective 可帮助 policy,推理不必生成未来 | 对称 MoT 不是 canonical topology |
| UWM | 用 modality/timestep/mask 明确区分 Policy、FD、IDM | 不用一个含糊 mask 混合已知未知量 |
| EgoSteer | training-only world expert 可读取 GT action | world activation 不回流 action |
| V-JEPA2 | frozen target encoder、action-conditioned predictor、短 rollout | 不先承担像素级长 rollout |
| MaskViT | 变化的 future mask 与部分上下文 | 部分 future 只留在 world branch |
| RoboTTT | 动态 working state、近零门控、sequence action forcing | codebook 不是可更新 TTT state |
| Genie | state tokenizer、action、dynamics 分工 | 不用离散 state 取代全部连续视觉 |
| VLANeXt | 多视角/proprio 融合和历史输入需要实证 | 不默认越长 raw history 越好 |
| DiT4DiT | 中间连续 video feature 对动作有价值 | 不只保留最终离散 code |

创新不来自把这些模块并排堆叠,而来自一套清楚的信息契约:冻结的多尺度视觉坐标负责模式,
连续 latent 负责精度,belief/memory 负责推理,world expert 只在合法模式下学习未来。

## 11. 实验门

上一道门不通过,下一道门不扩建。

### Gate 0: 数据与因果正确性

- 先按 episode/task 划分,再计算 normalization 和 RQ centers。
- 固定 fps 与 latent tick,抽查 `(t-2s,t-s,t)` 对应原图和动作区间。
- 检查 VAE prefix-only cache 不读取未来,不跨 episode/camera。
- artifact 完整记录 split、revision、seed、config 和 git commit。

Package Scan v6 只用于本机链路、可视化和回归测试,不支撑论文结论。

### Gate 1: 码本是否是健康的视觉状态坐标

公开数据先用 LIBERO,再用 BridgeData V2 或 DROID 子集复核。顺序搜索:

```text
spatial pool g = 1,2,4
capacity K     = 16,32,64
family stride  = 2,3,5
RQ prefix      = L1 / L1+L2 / L1+L2+L3
```

评价 usage/dead fraction/perplexity、held-out residual reduction、translation/scale sensitivity、
photometric invariance、retrieval montage、阶段/动作事件 agreement。三套联合 probe 必须优于最佳
单 family;无独立贡献的 family 删除。`delta-only` 只作 code-change 诊断,不是主输入。

### Gate 2: code 对控制表示是否有增量价值

在相同 probe 容量下比较:

```text
P0: language + proprio
P1: H + language + proprio
P2: C + language + proprio
P3: H + C + language + proprio
```

使用 episode-held-out split、`5%/20%/100%` 数据和至少 3 seeds。只有 `P3` 相对 `P1` 在低数据、
held-out scene/task 或扰动鲁棒性中出现稳定增益,才进入完整模型。P2 不要求胜过 P3。

### Gate 3: 结构和 mask 正确性

- 所有防泄漏单测通过,包括 future-label permutation invariance。
- 九个 code token 的 family/level/availability identity 在所有层保持可追踪。
- frozen artifact hash 在训练前后不变。
- latent-only、code-only、latent+code 三条路径可独立运行。
- Policy、FD、Prior batch 可单独和混合跑一个 optimizer step。
- 每个 loss 的梯度只到达表 8.4 允许的模块。

### Gate 4: 完整策略价值

只保留递进对照:

```text
C0: continuous state core, H only
C1: C0 + 9 frozen code measurements
C2: C1 + Forward Dynamics objective
C3: C2 + individually validated structured masks
C4: C3 + optional working memory, only if long-horizon task needs it
F0: FastWAM reference, external architecture baseline
```

`C0-C4` 保持 backbone、参数预算、训练步数和数据一致。主指标是闭环 task success,其次是 action
error、future-code accuracy、扰动鲁棒性、延迟和显存。若 C1 不胜 C0,返回 Gate 1/2;若 C2
只降低 code CE 而不改善控制或泛化,删除 world objective;若 C4 无长时收益,保持 memory 关闭。

## 12. 当前代码边界与实现顺序

当前 `codewam/codebook.py` 和 `codewam/model.py` 中的 `StateEncoder + online EMA RQ + first-frame
single token` 是早期兼容原型,不是 canonical v1。配置必须默认关闭该路径。

公开数据角色、DROID 数据阶梯、pooled feature cache、distributed streaming RQ 和 8xA100
作业布局以 `DATASET_SCALE_PLAN.md` 为准。
截至当前提交,步骤 1 和步骤 2 的单机底座、GPU smoke 与 reference-equivalence tests 已实现;
真实 episode exporter、held-out evaluator 和 rank-aware DDP orchestration 仍待完成。实现边界见
`STREAMING_CODEBOOKS.md`。

| 早期实现 | canonical v1 |
|---|---|
| `current/future/delta` evaluator | causal `[past,past,current]` descriptors |
| 全量数据内部标准化 | train-only stats applied unchanged to val/test |
| trainable StateEncoder | direct Wan latent + fixed pooling |
| 单个在线 EMA code token | 3 independent frozen RQ x 3 level tokens |
| policy loss 更新码本 | immutable artifact, learned interface projection only |
| action 与 future 混合 | staged state prefix + separate mode-specific branches |
| 单步 flat code head | independent hierarchical heads + short rollout |

工程顺序固定为:

```text
1. 实现 episode manifest、streaming pooled-feature shards 和 deterministic splits
2. 实现 train-only streaming stats、distributed K-Means/RQ 和 checkpoint/resume
3. 迁移 evaluator: causal descriptors、held-out metrics、retrieval 和几何扰动
4. 用 Package Scan v6/DROID-100 做 smoke,再在 DROID/Bridge/LIBERO 完成 Gate 1/2
5. 定义 StateBatch/CodeMeasurement/ModeBatch typed interfaces
6. 实现 mask program 与防泄漏/梯度路由单测
7. 实现 continuous C0 和 FrozenRQAdapter,完成 C0/C1
8. 实现独立 WorldExpert 与 hierarchical heads,完成 C2
9. 逐项验证 structured masks
10. 最后才接入可选 recurrent/TTT MemoryPort
```

在 Gate 2 之前不继续扩建在线码本原型,也不下载 3-camera 权重或在本机启动大模型训练。

## 13. 最终判定

CodeWAM v1 的可证伪主张是:

```text
Frozen causal multi-scale RQ measurements add a stable and dynamically useful
state coordinate system to continuous Wan latent features, improving control
data efficiency or robustness without sacrificing precise action generation or
requiring future-video rollout at inference time.
```

如果离散坐标没有超过连续路径,就删除它;如果 world objective 只会预测 code 而不改善控制,
就删除它;如果 memory 没有解决可测得的长时问题,就不启用它。结构只由实验保留,不由名字保留。
