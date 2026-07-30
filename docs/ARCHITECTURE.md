# CodeWAM Architecture

Status: canonical CodeWAM v1, verified JointWindowCache and Gate 2 runner
implemented; full-scale Gate 2, joint policy trainer and online runtime pending.

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

### 3.4 Domain chart 而不是跨域共享 ID

DROID、LIBERO 或其他域独立 refit 后得到的是不同离散坐标图:

```text
DROID/Q2/L1/code=3  !=  LIBERO/Q2/L1/code=3
```

每个 chart 保留自己的 normalization、centers、revision 和 checksum。
`FrozenCodebookAdapter` 按样本 chart 查询本地 center,再通过 chart-local projection 和
chart identity 映射到共享 belief width;它不对齐或翻译原始 ID。一个联合模型当前要求各
chart 使用相同 family、RQ 深度和每层 K,因为 future heads 必须有固定输出形状;不同 K 的
artifact 应分开实验,不能用 padding 伪造共享 vocabulary。同一 chart 的 Q2/Q3/Q5 还必须
共享 manifest、Wan/preprocess revision 和 source checksums,禁止拼接不同 provenance。
adapter state dict 也携带完整 chart/family metadata,加载到不同 provenance 时必须失败。

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
          +------> ContinuousStateEncoder ------------------------------> H_t
          |
          +------> causal descriptor + frozen RQ assignment -> C_t
                                                                |
                                                                v
                                                   FrozenCodebookAdapter -> E_t

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

`FrozenCodebookAdapter` 消费已经由相同 chart 的 frozen assigner 生成的 `CodeMeasurements`,
不在模型 forward 内从 RGB/latent 重新聚类或改变 center。每个 code ID 查询其真实 frozen
center:

```text
E_{s,j,t} = FamilyEmbedding_s + LevelEmbedding_j + ChartEmbedding_d
              + Project_{d,s,j}(e_{d,s,j,c})  if family available
              + MissingToken_{s,j}            otherwise
```

输出始终是九个 token,不把三级 centers 先求和,也不把 prefix 压成一个 one-hot category。
center 内容作为只读 K/V measurement;可学习的是投影与身份 embedding。v1 不把 margin、
quantization residual 或 handcrafted action association 加入模型输入。

训练时 current/future IDs 由 `JointWindowCache` 离线保存并逐样本复算验真。
`FrozenCausalCodeAssigner` 已实现从未池化 Wan latent 历史构造 Q2/Q3/Q5 descriptor,
再使用对应 chart 的 train-only normalization 和 centers 冻结赋码。它属于确定性 perception
preprocessing,不是可训练第六模块;当前缺口是把同一逻辑封装进 canonical online runtime。

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
    queries=FutureCodeQueries,
    context=[B_t, Embed(A_t)]
)
```

所有 future codes 在输入中 absent,不用 partial future mask、teacher forcing 或 scheduled
sampling。第一轮注册两种输出 factorization:

```text
independent  9 个 query/head,每个 (family,level) 输出 K-way logits
prefix       3 个 query/head,每个 family 输出 product(K_l)-way RQ tuple
```

`independent` 样本效率和逐层诊断更好,但近似
`p(c1,c2,c3)=p(c1)p(c2)p(c3)`;`prefix` 直接预测完整 RQ 路径,保留层间依赖,但类别更稀疏。
对 `K=8,L=3` 每族 prefix 有 512 类,仍可作为 Gate 2 对照。两者只改变 future head,不改变
belief、action decoder 或主信息流。该 decoder 学习:

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
future_logits independent: 9 x [B,K_{s,j}]
              prefix:      3 x [B,product_j K_{s,j}]
```

第一版的最小内部拓扑是:

1. `ContinuousStateEncoder`:latent patch projection 加 camera/time identity;先做帧内空间 attention,
   再做带 causal mask 的时间 attention,输出仍保留多视角空间 token。
2. `FrozenCodebookAdapter`:九个 center 各自投影;measurement 只作为下游 K/V,不在 block 间改写。
3. `WorldBeliefCore`:learned world queries 做 self-attention,再 cross-attend
   `[H,E,P,Embed(a_<t)]`;重复少量 block 后输出 `B`。
4. `ActionFlowDecoder`:noised action tokens 带 chunk-position 与 flow-time embedding;chunk 内
   双向 self-attention,再 cross-attend `[B,L,p_t]`,逐 token 输出 velocity。
5. `CodeDynamicsDecoder`:先把完整 action chunk 编成 action tokens;future queries 可彼此
   self-attend,再 cross-attend `[B,ActionTokens]`,由 independent 或 prefix heads 输出 logits。

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
8. 整段 padded action 不产生 NaN。
9. independent/prefix 对同一 tuple 概率给出相同 normalized NLL。

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

### 6.5 轨迹角色只控制监督,不定义世界是否存在

数据角色通过 per-sample mask 控制哪些 objective 合法:

| role | codebook/temporal | action imitation | code dynamics |
|---|---:|---:|---:|
| expert | yes | yes | yes |
| failure | yes | no | yes |
| recovery | yes | yes | yes |
| unlabeled interaction | yes | no | yes,如果有 action |
| action-free video | yes | no | no |

失败、停滞、碰撞和恢复都是真实世界状态,不应从 codebook 或 temporal learning 中删除;
但失败动作不是专家 policy label。这个分工由 `TrajectoryRole` 和 `SupervisionMasks` 实现,
不通过修改 attention mask 或 code visibility 实现。最终 action/dynamics mask 还必须与
“完整 action chunk 可用”相交;role 不能凭空创造动作监督。任何数据仍须先通过域、质量和
授权检查。

## 7. 分阶段初始化与两个联合目标

### 7.0 可选 Stage-0 temporal initialization

在 joint training 前可单独训练 `ContinuousStateEncoder`,让时刻 `t` 的连续 token 预测冻结
Wan-VAE 的较晚 raw latent patch:

```text
H_t = ContinuousStateEncoder(Z_<=t)
Z_hat_{t+d} = TemporalLatentPredictor(H_t)
L_temporal = masked MSE(Z_hat_{t+d}, stopgrad(Z_{t+d}))
```

实现会从传给 encoder 的 tensor 中直接裁掉 `>t` 的未来,同时保留 temporal causal mask,
并按 target-view availability 排除缺失相机。这一阶段不读取 code、语言或动作;预测 head 在
初始化完成后丢弃。它只提供
“连续状态应包含可预测动态”的初始化候选,不是第三个 joint loss。正式实验必须比较
scratch 与 Stage-0 init,不能预设预训练一定有益。

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

`independent` 对九个 level CE 等权平均。`prefix` 对每族完整 tuple 做 CE,再除以 RQ 深度
`L`,使 `lambda_code` 的单位仍是“每 RQ level NLL”。只用 availability mask 排除历史不足的
label。评估同时报告:

```text
normalized NLL             family-prefix NLL / L,两种 factorization 可比
raw family-prefix NLL      完整三级路径的负对数似然
family-prefix accuracy     完整三级路径全对才算正确
per-head accuracy/ECE      诊断量;不同 factorization 不能直接横比
center reconstruction MSE chart-local RQ center sum 的几何误差
```

v1 不加入 center-distance、photometric consistency、contrastive、continuous-future 或
multi-step rollout loss。

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

正式 Gate 2 固定使用同一 cache、初始化、batch 顺序、优化器和更新步数。`SHUFFLE` 是按
`split x horizon` 预先生成并哈希的全局置换,非 singleton group 不能映射回自身并尽可能跨
episode。主分析只看完整 RQ tuple 确实改变的 family,同时报告 all/stable、Q2/Q3/Q5 和
descriptor overlap `0/1/2/3` 分层。`TRUE` 还要原模型执行 `TRUE@NOACT` 与
`TRUE@SHUFFLE` 干预,排除三个独立训练结果的偶然差异。

门判定使用原始 RLDS parent episode 为 block 的 paired bootstrap;多个 keep-range segments
不能冒充独立样本,点估计和 window 数也不能代替独立 episode。默认至少需要 30 个具有
changed-family label 的共同 test parent episodes,不足时只能是 `invalid`,不能输出科学性
pass/fail。通过要求 `TRUE-NOACT`、`TRUE-SHUFFLE` 和
`TRUE-TRUE@SHUFFLE` 的 changed-family normalized-NLL 95% CI 上界都小于零。

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

## 12. 当前实现与唯一下一张工程单

以下真实训练前边界已经实现:

```text
codewam/data/droid_endpoint.py       RLDS endpoint/flag/alignment audit
codewam/data/frozen_assignment.py    frozen causal Q2/Q3/Q5 assignment
codewam/data/joint_cache.py          deduplicated episode shards + verified windows
codewam/data/joint_cache_export.py   rank-aware DROID -> Wan -> code/cache export
codewam/experiments/gate2.py         fixed-budget four-condition Gate 2
codewam/experiments/gate2_summary.py conservative three-seed decision
```

`JointWindowCache v1` 物理上按 episode/keep-range 去重保存未池化多相机 latent、source-rate
action/proprio、逐步 action validity、冻结 codes 和三个 descriptor source indices;
`windows.jsonl` 只保存状态/history/action/future 的半开区间与 artifact hashes。contract
锁定 source manifest、endpoint audit、Wan checkpoint、FastWAM VAE 实现、预处理、三份
codebook 和 CodeWAM writer 实现。reader 在取样时重新验证 shard SHA、端点、切片、
code label 与 overlap,再构造 typed `CodeWAMBatch`。finalize 同时生成按 window 对齐的紧凑
action index;Gate 2 以 shard-local 随机顺序读取大 latent 文件,错误动作只查询该 action
index,不再随机加载 donor 的视觉 shard。cache summary 在训练前报告每个 split/family 的
changed windows、changed parent episodes 和三级 RQ component/prefix 变化覆盖。

2026-07-30 的真实工程验收使用官方 DROID-100 32 条轨迹和 DROID 1.0.1 单个 TFRecord
source shard。端点审计覆盖 8,892 steps;`action[t]` 相对错位 `action[t+1]` 的关节/笛卡尔
速度 cosine 优势为 `0.02781/0.03217`,且 32 条轨迹的 first/last/terminal 边界全部合法。
六个 manifest episodes 的七个 keep-range segments 经 frozen Wan 与三份 DROID RQ artifact
生成 157 个窗口,split 为 `24/71/62`;Q2/Q3/Q5 overlap 恒为 `1/0/0`。该小 cache 上
PERSIST/NOACT/TRUE/SHUFFLE 和 TRUE 模型两种干预均完成 GPU 前反向、checkpoint、resume
与全指标报告。因为 test 只有两个独立 episode,更新后的协议正确返回 `invalid`;这只是
engineering smoke,不是 Gate 2 结果。

唯一下一张工程单改为 **DROID-10k JointWindowCache 扩展与正式 Gate 2**:

```text
1. 用 8 个独立 rank 导出 canonical 10k 的完整未池化 cache,每 rank 拥有完整 source shards。
2. finalize 后复核所有 shard/index SHA、split episode 数、changed-family coverage 与 overlap。
3. independent dynamics 固定三个 seed,每个 seed 等预算训练 NOACT/TRUE/SHUFFLE。
4. 以 >=30 个共同 changed-code test episodes 为硬有效性下限,目标使用 >=100 个。
5. 三个 seed 均报告 all/changed/family/overlap、paired episode CI 和 TRUE 模型动作干预,
   再由固定汇总器核对协议并要求三个 seed 独立通过。
6. 只有 independent Gate 2 通过,才比较 prefix head 和 scratch/Stage-0 initialization。
7. 之后实现冻结 language cache 与参数预算一致的 DROID C0/C1/C2 trainer。
8. LIBERO 使用独立 chart/refit 重复;FastWAM 仍只作 F0。
```

online assigner/runtime、language token export、正式 joint policy trainer 和闭环 benchmark
仍未完成。当前 `codewam/model.py`、`codewam/codebook.py`、`codewam/runtime.py` 与旧训练配置
仍是 legacy,不能接到 canonical v1 上冒充真实 trainer。

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
