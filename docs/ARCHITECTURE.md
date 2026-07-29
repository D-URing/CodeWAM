# CodeWAM Architecture

Status: canonical CodeWAM v1 architecture and experiment specification.

本文件是 CodeWAM 的唯一结构规范,约束模型模块、信息身份、训练目标、可见性和实验门。
码本数据、离线训练与评估见 `CODEBOOK.md`;环境和工程边界见 `DEVELOPMENT.md`。当前
`codewam/model.py` 只保留为历史兼容原型,不定义本文件中的 canonical model。

## 0. 一句话定义

CodeWAM 用冻结的多时间尺度码本定义视觉世界状态与局部变化的离散词典,用未量化连续
latent 保留词典单元内部的精确状态;共享世界 belief 在此基础上联合学习连续动作和
action-conditioned code transition。

```text
continuous H  -> 精确位置、尺度、接触、遮挡和微调信息
frozen C      -> 世界状态与局部变化的离散坐标
belief B      -> 对当前世界的可学习联合表示
policy        -> 当前状态与任务条件下的连续 action chunk
dynamics      -> 该动作会把世界带到哪些 future book IDs
```

Codebook 不是动作词典,也不被要求给 latent 增加原始信息。它是 world rules 的静态状态空间;
动作条件下的 code 转移是规则的动态部分。Policy 从联合训练中学习哪些世界状态与任务动作有关。

## 1. 不可违反的结构约束

以下约束必须写成接口、断言或单元测试:

1. **严格因果**:决策时刻 `t` 的状态只来自 `<=t` 的观测和 `<t` 的已执行动作。
2. **三套 RQ 独立**:`Q2/Q3/Q5` 的 normalization、centers 和 code ID 语义互不共享。
3. **九个 code measurement 不合并**:三套 family x 三层 RQ 保留为九个可区分 token。
4. **基座与码本只读**:Wan-VAE、language encoder、RQ normalization 和 centers 在联合训练中
   冻结并版本化。
5. **连续路径常驻**:Policy 始终可读取未量化视觉形成的连续状态 `H_t`。
6. **世界先于任务**:`B_t` 不读取语言、当前待生成动作或未来 target。
7. **动作不看未来**:GT future code 只能是 CodeDynamics 的监督标签。
8. **全 code 可见,相关性后学**:v1 不根据 action probe 预先删除光照、背景或某个 family。
9. **动作保持连续**:v1 使用 flow matching action chunk,不建立 action codebook。
10. **两个主 loss**:v1 只使用 action flow loss 与 future-code classification loss。
11. **模块非对称**:ActionFlowDecoder 与 CodeDynamicsDecoder 不共享中间 activation,
    只共享已经完成的 `B_t`。
12. **架构独立**:canonical model 不 import FastWAM 的 Video DiT、ActionDiT 或 MoT。

`2/3/5` 只表示时间间隔,不能预先命名为局部、中期或阶段语义。具体作用由 held-out
trajectory、code transition 和联合策略实验决定。

## 2. 已知量、未知量与标签

### 2.1 决策时刻已知

```text
x_<=t       当前及历史多相机观测
p_<=t       当前及历史 proprio
l           任务语言
a_<t        已执行动作
```

这些已知量可以因果地产生:

```text
Z_<=t       frozen Wan-VAE latent
H_t         ContinuousStateEncoder 的连续状态
C_t         当前九个 frozen RQ code IDs
E_t         由 frozen centers 形成的九个 measurement tokens
B_t         当前任务无关 world belief
```

### 2.2 决策时刻未知

```text
A_t         当前需要生成的 action chunk
C_{t+h}     action chunk 结束时的未来 code
x_{t+h}     未来观测
```

训练数据中存在 GT `A_t` 和 `C_{t+h}`,不等于部署时已知。`A_t` 是 action loss label,
也可以作为 CodeDynamics 的训练条件;`C_{t+h}` 始终只作为 future-code label。

## 3. Codebook 作为静态世界词典

### 3.1 Wan latent 与三个时间窗口

冻结 Wan-VAE,在统一 latent tick 上计算:

```text
Z_t = E_vae(x_<=t)
u_t = flatten(spatial_pool(Z_t, g x g))
```

第一版使用:

```text
D_2(t) = [u_{t-4},  u_{t-2}, u_t]
D_3(t) = [u_{t-6},  u_{t-3}, u_t]
D_5(t) = [u_{t-10}, u_{t-5}, u_t]
```

每个 descriptor 同时包含当前状态和短时变化,所以 center 可以描述物体出现、消失、移动、
缩放、机械臂运动、遮挡、背景或光照变化。历史不足时对应 family unavailable,不得 padding、
跨 episode 拼接或读取未来。

三个绝对状态也可无损变换为:

```text
[u_t, u_t-u_{t-s}, u_{t-s}-u_{t-2s}]
```

但这只是后续可比较的等价基底,不是 v1 默认输入,更不是 delta-only。标准 RQ residual 表示
未被前序 centers 解释的量化误差,不能在数学上直接等同于时间差。

### 3.2 三套离线 RQ

每个 family 只使用 train statistics:

```text
Dbar_s = (D_s - mu_s) / sigma_s
r_s^0 = Dbar_s
c_s^j = argmin_k ||r_s^{j-1} - e_{s,j,k}||^2
r_s^j = r_s^{j-1} - e_{s,j,c_s^j}
```

当前时刻的离散坐标是:

```text
C_t = {
  c_{2,1}, c_{2,2}, c_{2,3},
  c_{3,1}, c_{3,2}, c_{3,3},
  c_{5,1}, c_{5,2}, c_{5,3}
}
```

同一个数字在不同 family/level 中没有共同含义。Policy 训练不会更新 centers;每次运行必须
引用包含 dataset、Wan revision、normalization、centers、seed 和 checksum 的完整 artifact。

### 3.3 图结构解释

CodeWAM 可以被理解为三个交错的离散图:

```text
book IDs / center tuples      图节点与局部轨迹模板
C_t -> C_{t+h}                世界状态边
A_t                           边的动作条件
language                      Policy 选择行动方向的任务条件
H_t                           节点内部的连续精确坐标
```

冻结 codebook 定义静态节点。CodeDynamics 学习哪些边存在以及动作如何影响转移。Policy 学习
在当前世界、机器人状态和任务条件下应输出什么动作,而不是从 code ID 查一张固定动作表。

## 4. 五种信息身份

| 身份 | 符号 | 来源 | 生命周期 | 作用 |
|---|---|---|---|---|
| 连续测量 | `Z/H` | Wan latent + CodeWAM state encoder | 当前及历史 | 精确视觉状态 |
| 离散测量 | `C/E` | frozen RQ artifact | 当前及历史 | 静态世界坐标 |
| 世界 belief | `B` | CodeWAM belief queries | 单步内更新 | 融合 H/C/proprio/过去动作 |
| 动作 | `A` | ActionFlowDecoder | 当前待生成 | 连续控制 |
| 未来 code | `C_future` | CodeDynamicsDecoder | 当前待预测 | 动作条件下的世界转移 |

语言不是世界测量,它只进入 Policy。Proprio 和过去动作属于当前世界历史。可选 memory 是未来
扩展,不属于 v1 必需模块。

## 5. Canonical v1 五模块骨架

### 5.1 总图

```text
                          OFFLINE, TRAIN SPLIT ONLY
RGB -> frozen Wan-VAE -> pooled latent -> Q2/Q3/Q5 RQ -> frozen artifacts

                          ONLINE / JOINT TRAINING
past and current images
          |
          v
    frozen Wan-VAE
          |
          +------> ContinuousStateEncoder --------------------> H_t
          |
          +------> FrozenCodebookAdapter ---------------------> C_t, E_t

H_t + E_t + proprio + past actions
          |
          v
      WorldBeliefCore
          |
          v
         B_t
       /     \
      /       \
language       GT A (train) / candidate A (optional plan)
   |                    |
   v                    v
ActionFlowDecoder   CodeDynamicsDecoder
   |                    |
   v                    v
  A_t                logits(C_{t+h})
```

五个 CodeWAM-owned 模块是:

```text
ContinuousStateEncoder
FrozenCodebookAdapter
WorldBeliefCore
ActionFlowDecoder
CodeDynamicsDecoder
```

Wan-VAE 是冻结感知前端,不是 FastWAM runtime。FastWAM 只保留为外部对照 `F0`。

### 5.2 ContinuousStateEncoder

```text
H_t = ContinuousStateEncoder(
    ProjectZ(Z_<=t),
    camera embeddings,
    time embeddings
)
```

`H_t` 必须保留多相机空间信息,不能只使用 wrist pooled codebook state 或一个 global average
token。第一版可使用因果 Transformer 或时空 attention,但接口和层拓扑由 CodeWAM 自己定义。
是否从官方 Wan 权重初始化是独立 ablation,不能改变模块边界。

### 5.3 FrozenCodebookAdapter

每个 code ID 查询其真实 frozen center:

```text
E_{s,j,t} =
    Project_{s,j}(e_{s,j,c_{s,j,t}})
    + FamilyEmbedding_s
    + LevelEmbedding_j
    + AvailabilityEmbedding_{s,t}
```

输出始终是九个 token,不把三级 centers 先求和,也不把 prefix 压成一个 one-hot category。
center 内容作为只读 K/V measurement;可学习的是投影与身份 embedding。v1 不把 margin、
quantization residual 或 handcrafted action association 加入模型输入。

### 5.4 WorldBeliefCore

少量 learned world queries 聚合任务无关的当前证据:

```text
B_t = WorldBeliefCore(
    query=WorldQueries,
    kv=[H_t, readonly(E_t), P_<=t, Embed(a_<t)]
)
```

世界是什么不由语言定义,所以 `l` 不进入 `B_t`。所有 available Q2/Q3/Q5 levels 默认可见,
不预设 Policy hybrid mask。`L_action` 和 `L_code` 共同训练 `B_t`;具体哪些状态与动作无关,
由 ActionFlowDecoder 的联合学习决定。

### 5.5 ActionFlowDecoder

```text
v_hat = ActionFlowDecoder(
    noised_action=A_tau,
    flow_time=tau,
    context=[B_t, language, current proprio]
)
```

动作 token 在自己的 decoder 中更新,通过 cross-attention 读取完成后的只读 `B_t` 和语言。
它不能修改世界 belief,也不能读取 GT future code。动作生成使用连续 flow matching,但 decoder
不是从 FastWAM ActionDiT 继承的对称 Video-DiT 副本。

### 5.6 CodeDynamicsDecoder

动作 chunk 的终点定义为 `t+h`;时间对齐必须写入 dataset contract:

```text
future_logits = CodeDynamicsDecoder(
    queries=9 FutureCodeQueries,
    context=[B_t, Embed(A_t)]
)
```

九个 query 分别输出对应 `(family,level)` 的 K-way logits。所有 future codes 在输入中 absent,
不用 partial future mask、teacher forcing 或 scheduled sampling。该 decoder 学习:

```text
p(C_{t+h} | B_t, A_t)
```

基本 Policy 推理不运行 CodeDynamicsDecoder。以后若增加显式规划,可以对 candidate action
调用它,但这不是 v1 的部署依赖。

### 5.7 Tensor 与最小 block 契约

隐藏宽度和层数留给规模实验,但接口形状固定:

```text
Z             [B,T,V,Cz,Hz,Wz]        frozen Wan latent
H             [B,Nh,d]                 continuous visual tokens
code_ids      [B,3,3]                  Q2/Q3/Q5 x RQ-3
available     [B,3]                    whole-family availability
E             [B,9,d]                  readonly measurement tokens
P             [B,Np,d]                 proprio/history tokens
B             [B,Nb,d]                 learned world-belief queries
L             [B,Nl,d]                 frozen language tokens
A             [B,Ha,Da]                continuous action chunk
future_logits 9 x [B,K_{s,j}]          one head per family/level
```

第一版的最小内部拓扑是:

1. `ContinuousStateEncoder`:latent patch projection 加 camera/time identity;先做帧内空间 attention,
   再做带 causal mask 的时间 attention,输出仍保留多视角空间 token。
2. `FrozenCodebookAdapter`:九个 center 各自投影;measurement 只作为下游 K/V,不在 block 间改写。
3. `WorldBeliefCore`:learned world queries 做 self-attention,再 cross-attend
   `[H,E,P,Embed(a_<t)]`;重复少量 block 后输出 `B`。
4. `ActionFlowDecoder`:noised action tokens 带 chunk-position 与 flow-time embedding;chunk 内
   双向 self-attention,再 cross-attend `[B,L,p_t]`,逐 token 输出 velocity。
5. `CodeDynamicsDecoder`:先把完整 action chunk 编成 action tokens;九个 future queries 可彼此
   self-attend,再 cross-attend `[B,ActionTokens]`,由九个分类 head 输出 logits。

`d/Nb/blocks/heads` 可以缩放,但 C0/C1/C2 必须固定 ActionFlowDecoder、state width、训练步数和
主参数预算。两个 decoder 不共享 block 参数;future query 不嵌入任何 GT future code。

### 5.8 Memory 边界

v1 不实现 recurrent/TTT memory。`C_<=t` 的紧凑历史可能在长时任务中形成 MemoryPort,但必须
先证明固定窗口 `B_t` 无法解决的可测问题。Memory 只能读取过去 belief/code/action,不能读取
当前待生成动作或 future target。

## 6. 硬信息流与防泄漏

### 6.1 训练样本

一条 joint-training 样本是:

```text
known state:   x_<=t, p_<=t, a_<t, language
action label:  A_t
world input:   GT A_t
world label:   C_{t+h}
```

`C_{t+h}` 由同一版本 frozen tokenizer 在 `no_grad` 下离线生成。训练时给 CodeDynamics 的
GT action 是已执行转移条件,不是 Policy 输入。以后可单独比较 predicted-action conditioning,
但不能混进 v1 首个对照。

### 6.2 可见性矩阵

`label` 表示只参与 loss,不作为 K/V。

v1 不把所有变量拼成一个 joint sequence 后再依赖复杂方阵消除泄漏。只保留三类 mask:

```text
temporal causal mask   H_t 不能读取 t 之后的视觉或动作
availability mask     历史不足的 Q2/Q3/Q5 family 使用 missing token
module access mask    由函数参数和张量类型限制每个 decoder 能接收什么
```

第三类主要由 API 边界实现,不是一张容易配错的 attention mask。future-code token 根本不进入
Policy graph,语言也根本不进入 WorldBeliefCore。模块内部可以正常使用 self/cross-attention。

| Producer/query | `Z/H` | current `E` | `P/a_<t` | language | noisy action | GT action | GT future code |
|---|---:|---:|---:|---:|---:|---:|---:|
| ContinuousStateEncoder | yes | no | no | no | no | no | no |
| FrozenCodebookAdapter | source | self | no | no | no | no | no |
| WorldBeliefCore | yes | yes | yes | no | no | no | no |
| ActionFlowDecoder | via B | via B | yes | yes | self | label only | no |
| CodeDynamicsDecoder | via B | via B | via B | no | no | yes | label only |

### 6.3 必须自动测试

1. Policy token sequence 不存在 future-code slots。
2. 置换 GT future code 时,固定 action noise 的 Policy 输出逐位不变。
3. 置换 GT action 时,已构造的 `B_t` 与 Policy 输出逐位不变,只有 CodeDynamics 输出可变。
4. Policy/CodeDynamics 的中间 activation 不写回 `B_t`。
5. frozen artifact hashes 在 optimizer step 前后不变。
6. unavailable family 使用 missing token,不产生伪 code ID。
7. 九个 `(family,level)` token 身份在投影和 belief attention 中可追踪。

### 6.4 推理路径

默认部署:

```text
x_<=t -> Z/H/C/E -> B_t -> ActionFlowDecoder -> A_t
```

CodeDynamics 是 training-time world objective,不要求先生成未来视觉或 future code 才能输出
动作。可选规划模式以后才增加:

```text
candidate A -> predicted C_future -> task-conditioned scoring
```

## 7. 两个训练目标

### 7.1 Action flow loss

```text
L_action = FlowMatch(
    ActionFlowDecoder(B_t, l, p_t, A_tau, tau),
    target_velocity(A_t, noise, tau)
)
```

每个 action chunk 独立采样 flow time 与 noise。Action loss 更新 ContinuousStateEncoder、
Codebook projection、WorldBeliefCore 和 ActionFlowDecoder,但不更新 Wan-VAE/RQ centers。

### 7.2 Future-code loss

```text
L_code =
    mean_available_{s,j} CE(
        logits_{s,j}(B_t, A_t),
        c_{s,j}^{t+h}
    )
```

九个 head 等权起步,只用 availability mask 排除历史不足的 label。v1 不加入 center-distance、
photometric consistency、contrastive、continuous-future 或 multi-step rollout loss。

### 7.3 Total loss

```text
L_total = L_action + lambda_code * L_code
```

只有一个可调权重 `lambda_code`。首轮只比较 `0` 与一个预注册非零值;若 future-code accuracy
提高但 Policy 无收益,不能继续叠加辅助 loss 掩盖结论。

### 7.4 梯度路由

| 模块 | `L_action` | `L_code` | 冻结 |
|---|---:|---:|---:|
| Wan-VAE | no | no | yes |
| language encoder | no | no | yes |
| RQ normalization/centers | no | no | yes |
| ContinuousStateEncoder | yes | yes | no |
| Codebook center projections | yes | yes | no |
| WorldBeliefCore | yes | yes | no |
| ActionFlowDecoder | yes | no | no |
| CodeDynamicsDecoder | no | yes | no |

这张表定义“联合学习”:两个目标只在世界表示处相遇,不让两个 decoder 互相泄漏 activation。

## 8. 保持简单的边界

以下能力不是 v1 默认项:

```text
future-video generation
Video Prior mode
IDM future-observation shortcut
partial future masking
hierarchical future teacher forcing
center-distance auxiliary loss
photometric invariance loss
multi-step rollout loss
code-conditioned low-rank adapters
recurrent or TTT memory
```

它们只能针对一个已经被测量的具体失败逐项加入。诊断量如 assignment margin、光照响应和
retrieval 可以继续计算,但不是训练 loss。

## 9. 实验门

### Gate 0: 数据与因果

- episode/scene split 先于 normalization 和 RQ。
- VAE prefix-only audit 不读取未来。
- descriptor 不跨 episode、keep-range gap 或 camera。
- artifact 记录数据、模型、实现和中心 hashes。

### Gate 1: 静态世界词典是否健康

检查:

```text
usage/dead codes/perplexity
held-out RQ distortion
RQ levels 是否持续降低 residual
时间顺序与状态变化是否产生稳定响应
retrieval 是否覆盖多 scene 的相似状态/变化
code transitions 是否平滑而非随机边界跳转
domain/revision 边界是否明确
```

光照、背景或物体外观属于可见世界状态,code 对它们变化不是自动失败。应区分稳定可预测的
状态响应与低 margin 的随机跳转。动作/gripper association 只作描述性诊断,不能用于删除
world-state code。

现有 DROID usability report 的 `not_ready` 仍适用于“可重复、跨域通用 tokenizer”主张。
对于绑定固定 artifact 的 DROID 域内 CodeWAM 原型,低 cross-seed NMI 和 photometric code
change 不单独阻止 joint training;冻结 DROID -> LIBERO collapse 仍禁止通用跨域声称。

### Gate 2: code transition 是否形成 world rule

使用真实 joint-training tuple 测量:

```text
p(C_{t+h} | B_t, A_t)
```

至少比较:

```text
current-code persistence baseline
B_t without action
B_t + true action
shuffled-action control
```

如果真实 action 不能稳定降低 held-out future-code NLL,code graph 尚未形成可学习的
action-conditioned dynamics。当前动作线性读出不再是 Gate 2。

已完成的 P0/P1/P2/P3 ridge screen 保留为历史接口诊断。它说明 hard categorical additive
feature 没有超过连续 H,也说明该 proxy 受 absolute action/proprio shortcut 和高维小样本
过拟合影响;它不裁决 CodeWAM 架构。

### Gate 3: 独立架构正确性

- 五模块可分别构造和保存 state dict。
- canonical model 不 import FastWAM model/MoT。
- 所有第 6.3 节防泄漏测试通过。
- C0/C1/C2 在同一 batch contract 下各跑一个 optimizer step。
- `L_action/L_code` 梯度只到达第 7.4 节允许模块。
- basic inference 完全不构造 CodeDynamics future queries。

### Gate 4: 联合 world-action 价值

只保留三个主对照:

```text
C0: H -> B -> ActionFlowDecoder
    loss = L_action

C1: H + frozen C/E -> B -> ActionFlowDecoder
    loss = L_action

C2: H + frozen C/E -> B -> ActionFlowDecoder + CodeDynamicsDecoder
    loss = L_action + lambda_code * L_code

F0: FastWAM, external baseline only
```

`C0-C2` 保持 state width、action decoder、训练数据、步数和主参数预算可比。主指标是闭环
task success;其次是 action loss、future-code NLL/accuracy、学习速度、延迟和显存。

```text
C1 > C0  当前离散世界坐标直接帮助 Policy
C2 > C1  学习 code transition 反过来帮助动作
code 指标提高而控制不提高  world objective 可学但无控制增益
```

只有在固定窗口模型出现明确长时失败后才增加 C3 memory。

## 10. 研究借鉴与取舍

| 工作 | 吸收的原则 | CodeWAM 的取舍 |
|---|---|---|
| FastWAM | training-time world objective 可辅助 Policy | 只作 F0,不采用双 DiT/MoT |
| Genie | state tokenizer、action、dynamics 分工 | 保留连续 H,code 不替代精确视觉 |
| V-JEPA2 | frozen target 与 action-conditioned prediction | 预测离散 world code,先不做长 rollout |
| UWM | 显式区分已知、未知和监督标签 | 用模块边界而不是万能 attention mask |
| MaskViT | 离散视觉状态可作为预测目标 | v1 不使用 partial future masking |
| RoboTTT | 紧凑历史状态可能替代长 KV | memory 延后,codebook 本身不在线更新 |
| DiT4DiT | 连续中间视觉特征对精细动作有价值 | H 永久保留 |
| tau0-WM | 困难状态可按需验证候选动作 | basic path 不 rollout,规划作为后续可选层 |

这些工作共同支持四个 v1 决策:状态、动作和动态分工;world objective 只在训练期塑造表示;
连续视觉精度常驻;future prediction 必须由动作条件化。它们没有共同证明 mixed attention、
partial-future mask、fast weights 或 test-time rollout 应成为默认结构。

下一轮 research library 复盘优先寻找三类证据:离散状态图与 inverse/forward dynamics 的关系、
task-free world state 如何被 Policy 选择性读取、以及简单双目标联合训练何时优于复杂辅助 loss。
任何新增模块必须先能用本文件的信息身份解释,再对应一个已测得的失败。

预留扩展只通过以下触发条件进入:

| 已测得的失败 | 允许比较的扩展 | 主要参考 |
|---|---|---|
| 相同当前观测对应不同任务阶段,固定窗口无法消歧 | belief/code/action history `MemoryPort` | RoboTTT |
| 多步 future-code rollout 比单步 transition 明确改善规划 | masked/iterative code decoder | MaskViT |
| CodeDynamics entropy 与真实动作失败稳定相关 | 只对高不确定状态做 candidate rerank/rollout | tau0-WM |
| 需要利用无 action 视频扩大 world pretraining | world-only transition data 或独立 latent-action adapter | UWM / Genie |

这些是接口预留,不是 v1 backlog。尤其是 latent-action adapter 不替代真实机器人连续动作头。

## 11. 与 FastWAM 的边界

CodeWAM 在三个层面独立:

1. **方法独立**:五模块、两个 loss 和信息流不以 FastWAM 为模板。
2. **代码独立**:canonical model/runtime 不 import FastWAM 的 model、MoT 或 trainer。
3. **权重开放**:Wan-VAE 使用官方权重;其他兼容权重只能作为可选初始化实验。

FastWAM checkout、ActionDiT conversion 和旧训练脚本只服务 `F0` 对照。它们不能成为 CodeWAM
训练环境的必需依赖。

## 12. 当前代码边界与实现顺序

已完成的是数据、Wan pooled cache、离线 RQ、held-out diagnostics 和大规模 artifact workflow。
独立 CodeWAM v1 model 尚未实现。当前 `codewam/model.py`、`codewam/codebook.py` 和相关
FastWAM-compatible configs 全部是 legacy。

目标 package:

```text
codewam/models/
  continuous_state.py
  frozen_codebook.py
  belief_core.py
  action_flow.py
  code_dynamics.py
  codewam_v1.py
```

下一工程顺序:

```text
1. 定义 StateInputs/CodeMeasurements/WorldBelief/ActionBatch/FutureCodeTargets
2. 实现 FrozenCodebookAdapter 和九 token identity tests
3. 实现 ContinuousStateEncoder 与 WorldBeliefCore
4. 实现独立 ActionFlowDecoder,先完成 C0/C1
5. 从 pooled trajectory 构造 action-chunk-end future-code labels
6. 实现 CodeDynamicsDecoder 与 L_code,完成 C2
7. 加入 gradient-routing、future-permutation 和 inference-path tests
8. 在 DROID/LIBERO 进行 C0/C1/C2,FastWAM 仅作 F0
9. 只在证据要求时增加 memory 或新的 loss
```

## 13. 可证伪主张

```text
A frozen multi-scale codebook provides a compact, static vocabulary of visual
world states and local changes. Jointly learning continuous actions and
action-conditioned transitions between book IDs improves world-structured
policy learning while unquantized visual state preserves precise control.
```

这项主张不要求 `C` 比 `H` 包含更多信息,也不要求 code 直接对应动作。若 `L_code` 只提高
future-code accuracy 而不改善 Policy 学习或闭环控制,删除 CodeDynamics objective;若 `C1`
也不胜 `C0`,码本只保留为分析工具。结构由可归因的 C0/C1/C2 结果决定。
