# CodeWAM Native Design

本文件用于规划 CodeWAM 从 "FastWAM-compatible 插件式接入" 走向
"以离散行为状态为核心的 World-Action Model"。它不是当前代码的逐行说明,
而是后续技术路线、实验判断和架构取舍的讨论稿。

## 1. 定位

CodeWAM 当前有两条线:

```text
A. CodeWAM-compatible
   复用 FastWAM 主干,把 state code 作为额外 context token 和 dynamics loss 接进去。
   作用:公平对照、复用预训练、快速验证大模型训练链路。

B. CodeWAM-native
   让离散 state code 成为主状态,围绕 code 做 policy、dynamics、diagnostics 和 planning。
   作用:探索 CodeWAM 真正区别于 FastWAM 的结构价值。
```

当前代码主要处在 A 线。B 线是下一阶段技术方案的核心。

CodeWAM-native 要回答的问题不是 "能不能给 FastWAM 加一个码本",而是:

```text
机器人控制中,离散视觉状态 code 能不能成为比连续 latent 更好的
world-action 中间状态?
```

如果成立,CodeWAM 的优势不是多了一个 regularizer,而是建立了一个可预测、
可诊断、可复用、可规划的离散行为状态空间。

## 2. 为什么需要 Code

真机小数据训练中最核心的问题是 proprio shortcut:

```text
current proprio -> action
```

动作 loss 很容易被当前关节状态解释掉,视觉和语言只提供弱梯度。模型可能在
open-loop loss 上看起来正常,但闭环时忽略物体、目标位置、条码朝向和接触状态。

Code 的目标是把视觉 latent 压成一个行为相关的离散状态:

```text
image/video -> state code
state code + proprio + task -> action
state code + action -> next state code
```

也就是说,CodeWAM-native 中的 code 不应该只是辅助 token,而应该是 policy 和
world model 都围绕它组织的主状态。

## 3. Code 的潜在优势

### 3.1 抗 proprio shortcut

连续视觉 latent 大而冗余,动作专家可以绕过它。离散 code 是强瓶颈,如果训练目标
设计正确,它会迫使视觉表示保留任务相关因素:

- 包裹位置
- 条码朝向
- 夹爪和包裹的相对位置
- 是否抓住
- 是否接近托盘或盒子
- 当前处于任务哪个阶段

关键不是 code 能重构视觉,而是:

```text
code + proprio -> action  明显强于  proprio-only -> action
```

### 3.2 廉价世界模型

像素级或 latent 扩散式未来想象成本高。Code-space dynamics 可以更便宜:

```text
c_t, a_t -> c_{t+1}
c_t, a_{t:t+k} -> c_{t+k}
```

短 horizon 上,这可以支持动作候选评估、rollout、value/success prediction,
而不需要每次生成未来视频。

### 3.3 可诊断

离散 code 可以被统计、聚类和可视化:

- 哪些 code 对应接近物体
- 哪些 code 对应抓取成功前后
- 哪些 code 对应条码朝上/不朝上
- 图像扰动是否改变 code
- code 改变是否导致 action 改变
- 某些 code 是否只编码背景/光照

这比连续 latent 更容易做失败分析。

### 3.4 可复用

如果 code 学到的是行为状态,它可以跨任务复用:

```text
shared tokenizer / code dynamics
task-specific policy / value / goal head
```

新任务未必需要重学视觉表示,可以先复用 code space,再学习 code 到 action 或
goal 的映射。

### 3.5 可接 planning 和 retrieval

离散状态天然适合检索和短 horizon 搜索:

```text
current code -> retrieve similar histories -> reuse action priors
candidate action -> predicted future code -> value/success -> select action
```

这会让 CodeWAM 和普通 imitation policy 明确分开。

## 4. Code 的风险和劣势

### 4.1 信息损失

抓取任务可能依赖细微几何信息。过强的离散瓶颈会丢掉夹爪边缘、包裹角点、
接触距离等连续细节。

应对方向:

- 分层 code: coarse stage code + fine geometry code
- 保留局部/patch code,而不是只保留全局 pooled code
- policy 同时接收 proprio 和必要的低维连续 residual

### 4.2 码本坍塌

少数 code 被频繁使用,大量 code 死掉,状态空间表达力不足。

必须长期监控:

- per-level usage
- perplexity
- dead code ratio
- code transition entropy
- code-action mutual information proxy

### 4.3 错误抽象

code 可能按背景、光照、相机偏差聚类,而不是按动作相关状态聚类。

因此训练目标不能只做 reconstruction 或 next-code prediction,还要引入:

```text
code -> action
code + proprio -> action
code + action -> next code
future code -> success/value/stage
```

### 4.4 时间混淆

单帧 code 可能无法区分:

- 正在接近物体
- 已经抓住物体
- 抓取失败但画面相似

CodeWAM-native 需要 belief state,而不是只用单帧 code:

```text
history of codes + proprio + action history -> belief
```

### 4.5 和 policy 脱钩

这是当前 compatible 方案的最大风险。code 可能预测未来做得不错,但 action expert
仍然忽略 code。

因此最终判据必须是 action-level:

```text
image/code ablation changes action
code + proprio beats proprio-only
closed-loop success improves
```

## 5. 当前 Compatible 架构

当前 CodeWAM-compatible 结构:

```text
video -> Wan-VAE latent
first-frame latent -> StateCodebook -> state token
text + proprio token + state token -> ActionDiT context
ActionDiT + MoT -> action loss
StateCodebook -> next-code dynamics loss
```

优点:

- 最大化复用 FastWAM/Wan 预训练
- 容易和 FastWAM 做 apples-to-apples 对照
- 工程风险小
- 可以验证 code token 是否带来收益

限制:

- code 仍是 context 中的附属 token
- video latent 主干仍然存在,policy 可以绕开 code
- dynamics 是辅助 loss,不是 policy 主结构
- 很难证明模型真的依赖 code

所以 compatible 线适合做 baseline 和大模型训练,但不应限制 CodeWAM 的最终形态。

## 6. Native 架构草案

CodeWAM-native 建议拆成五层。

### 6.1 Perception Tokenizer

输入:

```text
multi-camera image/video window
```

输出:

```text
discrete state codes c_t
optional continuous residual r_t
```

可选实现:

- Wan-VAE latent + RQ/RVQ
- Wan-VAE latent + FSQ/LFQ
- patch/object-centric tokenizer
- temporal tokenizer over short video clips

设计目标:

```text
encode action-relevant visual state, not generic image appearance
```

### 6.2 Belief State

输入:

```text
c_{t-h:t}, proprio_{t-h:t}, action_{t-h:t-1}, task
```

输出:

```text
b_t
```

作用:

- 处理单帧 code 的时间歧义
- 融合 proprio 和视觉 code
- 为 policy 和 dynamics 提供统一状态

可选实现:

- small Transformer
- GRU/state-space model
- MoT-style attention over code/proprio/action tokens

### 6.3 Action Policy

输入:

```text
b_t, task/goal
```

输出:

```text
action chunk
```

训练目标:

```text
L_action = MSE / flow matching / diffusion action loss
```

关键 ablation:

```text
proprio-only
code-only
code + proprio
code + proprio + history
```

### 6.4 Code Dynamics

输入:

```text
c_t or b_t, action
```

输出:

```text
c_{t+1} or c_{t+k}
```

训练目标:

```text
L_dyn = cross entropy over future code levels
```

必要基线:

```text
copy current code
proprio-only dynamics
action-only dynamics
```

### 6.5 Evaluation Heads

输入:

```text
b_t or predicted future codes
```

输出:

```text
success / stage / value / contact / affordance
```

作用:

- 支持 code-space planning
- 提供可解释诊断
- 让 future code 和任务结果对齐

## 7. 可替代和新增模块

### 7.1 RQ/RVQ

当前默认方向。优点是直接、可解释、容易做多级 code。缺点是需要防坍塌,
EMA 和 dead-code reset 要仔细调。

适合:

- 第一阶段延续现有实现
- 做和当前 probe 的连续性对照

### 7.2 FSQ/LFQ

可作为 RQ 的替代 tokenizer。潜在优势是训练更稳、没有传统 embedding codebook
的死码问题。风险是表达形式和 dynamics/head 需要重新适配。

适合:

- 当 RQ usage/perplexity 不稳定时作为替代路线
- 作为 tokenizer ablation

### 7.3 Hierarchical Code

拆成 coarse/fine 两层:

```text
coarse: task stage / object relation / semantic state
fine: geometry / local pose / contact detail
```

优势是兼顾抽象和几何。风险是训练复杂度上升,需要明确每层监督或结构约束。

### 7.4 Object-Centric Code

Package Scan 任务中,关键对象比较明确:

- package
- gripper
- pallet
- box
- barcode

Object-centric code 可以比全图 code 更有效。实现可以从 patch attention、
region pooling 或 lightweight detector/mask prior 开始,不一定一开始就引入复杂检测器。

### 7.5 Retrieval Memory

用 code 做最近邻检索:

```text
current code -> similar trajectories -> future code/action distribution
```

适合小数据真机任务,因为相似状态的历史动作有很强参考价值。

### 7.6 Code-Space Planning

用 dynamics 评估候选动作:

```text
for candidate action:
    rollout future code
    score by value/success/stage progress
select best action
```

这是 CodeWAM-native 最有区别度的方向,但必须先有可靠的 code dynamics 和 value/success head。

## 8. 实验路线

### 8.1 本机 Package Scan v6 小实验

先不训练 5B 主模型,用 `package_scan_v6` 跑小模型验证 code 是否值得成为主状态。

最小实验:

```text
E1: proprio-only -> action
E2: code-only -> action
E3: code + proprio -> action
E4: code + proprio + history -> action
E5: code + action -> next code
E6: future code -> task stage / success proxy
```

通过条件:

```text
E3 > E1
E4 > E3 or more stable than E3
E5 > copy-current-code baseline
image perturbation changes code and action in sensible directions
```

### 8.2 Compatible 大模型对照

保留当前 FastWAM-compatible 训练链路,用于回答:

```text
在同数据/同评测下,加 state code token + dynamics loss 是否提升 FastWAM?
```

必要对照:

- FastWAM baseline
- CodeWAM without dynamics loss
- CodeWAM without state token
- CodeWAM with frozen tokenizer
- CodeWAM with trainable tokenizer

### 8.3 Native 原型

当小实验确认 code 有 action-relevant 信息后,实现小型 native policy:

```text
image -> Wan-VAE latent -> tokenizer -> code
code/proprio/task/history -> policy
code/action -> dynamics
future code -> value/stage
```

第一版不需要复刻 FastWAM 的完整 5B 结构,应优先保证诊断清楚、迭代快。

## 9. 评估标准

CodeWAM 不能只看 reconstruction 或 dynamics top1。真正标准分三层。

### 9.1 Representation

- code usage/perplexity
- reconstruction or latent R2
- next-code top1 vs copy baseline
- code transition entropy
- code stability under small visual noise

### 9.2 Action Relevance

- code + proprio action prediction > proprio-only
- image perturbation changes code and action
- state-code ablation degrades action prediction
- action-conditioned dynamics > proprio-only dynamics

### 9.3 Robot Outcome

- closed-loop success rate
- recovery from visual perturbation
- task-stage correctness
- inference cost
- failure modes: no movement, no braking, wrong object, wrong placement

## 10. 推荐推进顺序

### Phase 0: 文档和口径统一

- 明确 CodeWAM-compatible 和 CodeWAM-native 两条线
- 更新旧文档中 "FastWAM patch" 口径
- 把 Package Scan v6 定位为本机 native 小实验数据

### Phase 1: Native probe

- 训练/冻结 tokenizer
- 跑 E1-E5 小实验
- 建立 action relevance 诊断
- 输出 code visualization 和 nearest-neighbor examples

### Phase 2: Native policy prototype

- 小型 belief + policy
- action-conditioned code dynamics
- stage/value head
- 在 Package Scan v6 做 offline 验证

### Phase 3: Compatible cluster training

- 生成 ActionDiT backbone
- 用同数据/同评测对比 FastWAM 和 CodeWAM-compatible
- 做 state-token/dynamics/frozen-tokenizer ablation

### Phase 4: Planning and retrieval

- 加 code-space retrieval
- 加 short-horizon candidate action scoring
- 验证是否提升闭环稳定性

## 11. 当前结论

CodeWAM 的机会不在于 "给 FastWAM 多加一个码本",而在于:

```text
离散行为状态空间
+ 动作条件状态转移
+ 行为相关 policy
+ code-space diagnostics/planning
```

FastWAM 是重要的预训练资源和公平对照框架,但不应成为 CodeWAM-native 的结构边界。
下一阶段应以 `package_scan_v6` 为本机快速实验场,先证明 code 是 action-relevant state,
再决定如何把 native 结构放大到集群训练。
