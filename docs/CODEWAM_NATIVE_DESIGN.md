# CodeWAM Native Design

本文件是 CodeWAM 的技术规划主文档。它的目标不是描述当前代码已经实现了什么,
而是帮助我们决定 CodeWAM 应该长成什么样。

当前最重要的判断是: CodeWAM 不应被限定为 "FastWAM + codebook"。FastWAM 是
预训练资源、工程参考和公平对照,但 CodeWAM 的核心问题应该更独立:

```text
机器人控制中,能否用离散、可预测、行为相关的 state code
替代连续视觉 latent,成为 policy、dynamics、diagnostics 和 planning 的主状态?
```

如果这个判断成立,CodeWAM 的价值不是多一个 regularizer,而是建立一个可以被动作模型
真正使用的离散行为状态空间。

## 1. 研究依据

这版规划吸收了 `EmbodiedAI-Research` 中几条相关路线。它们共同指向一个结论:
控制不需要漂亮的未来视频,而需要动作可用的世界表征。

### 1.1 Fast-WAM

核心启发:

```text
training-time world modeling > test-time future imagination
```

Fast-WAM 表明,WAM 的主要收益可能来自训练期视频建模对表征的塑形,而不是测试期真的
生成未来帧。测试时跳过未来视频分支,直接用单次 video DiT 表征预测动作,可以显著降低
延迟。

对 CodeWAM 的含义:

- 不必追求 pixel-space rollout。
- 未来建模可以保留为训练信号。
- CodeWAM 可以把未来建模搬到更便宜的离散 code 空间。
- 必须做 `with/without future-code prediction` 消融,对应 Fast-WAM 的 video co-training 消融。

### 1.2 DiT4DiT

核心启发:

```text
intermediate video-denoising features > reconstructed future frames
```

DiT4DiT 证明,视频扩散模型中间层/中间 denoising step 的 hidden features 比最终生成的未来帧
更适合作为 action condition。

对 CodeWAM 的含义:

- Code 不应只是 VAE latent 的普通压缩,而应对齐 video dynamics hidden state。
- 可以把 DiT hidden features 离散化,而不只量化 VAE latent。
- 需要做 `source ablation`: VAE latent code vs DiT hidden feature code vs mixed code。
- 需要做 `layer/depth ablation`: 不同层/不同 code level 对动作是否有用。

### 1.3 VLANeXt

核心启发:

```text
先把 VLA 基础件搭稳,再谈 WAM 创新
```

VLANeXt 的价值是 recipe checklist: temporal history、action chunking、flow matching、
multi-view/proprio 处理、VLM-policy bridge、便宜辅助目标。

对 CodeWAM 的含义:

- Native 方案不能只讨论 codebook,也要保证 policy 基础结构合理。
- History/belief 应是一级模块,不是附属技巧。
- World-model objective 要比较成本: future image 很贵,code/time-series 目标更适合。
- 多视角和 proprio 的融合方式本身要做消融。

### 1.4 RoboDojo

核心启发:

```text
LIBERO/RoboTwin 高分不足以证明 WAM 可用
```

RoboDojo 把能力拆成 Generalization、Memory、Precision、Long-Horizon、Open 等维度。
Fast-WAM 在 RoboDojo 上分数较低,说明常规 benchmark 高分不能代表可靠操作。

对 CodeWAM 的含义:

- CodeWAM 的 claim 不能只停在 loss 和 simulation average score。
- 要建立 Memory/Open/Precision/Generalization 风格的内部诊断。
- `visual sensitivity`、`proprio shortcut`、`state memory`、`open instruction grounding`
  必须成为评估项。

### 1.5 EgoSteer

核心启发:

```text
full stack matters: data, embodiment, model, masking, HITL
```

EgoSteer 的重要点不是单个模型,而是人类视频预训练、真实机器人数据、proprio history masking、
camera dropout、DAgger/HITL 和部署栈组成闭环。

对 CodeWAM 的含义:

- 为了抗 proprio shortcut,可以显式加入 proprio masking、camera dropout、code dropout。
- Package Scan v6 只是本机起点,后续要考虑数据闭环和失败样本回流。
- Code 要服务机器人实际执行,而不是只服务离线指标。

### 1.6 CHORD

核心启发:

```text
视觉未来不是完整世界状态,接触力学也是状态
```

CHORD 强调 contact wrench: 很多灵巧操作失败不是因为看不见,而是因为没有建模接触力和物体受力效果。

对 CodeWAM 的含义:

- CodeWAM 的 state code 不能只等同视觉 appearance code。
- 对 package scan 这类任务,至少要考虑 grasp/contact/stability 的 latent proxy。
- 未来可以扩展出 `contact-aware code` 或 `contact/event head`。

### 1.7 DexVerse 和 SimFoundry

核心启发:

```text
能力 taxonomy 和场景扩展比单一 demo 更重要
```

DexVerse 给任务能力分类,SimFoundry 给 real-to-sim / digital cousin 评测扩展思路。

对 CodeWAM 的含义:

- 后续评估要按任务能力分类,不要只说 "成功率"。
- 如果有真实 demo 场景,可以考虑构造仿真近邻场景测试泛化。

## 2. 当前核心假设

CodeWAM 的核心假设应该收束成四句话:

```text
H1: 机器人操作中存在比连续视觉 latent 更紧凑的离散行为状态。
H2: 这个状态可以通过未来预测、动作预测和扰动诊断被训练成 action-relevant。
H3: 一旦 state code 足够稳定,policy/dynamics/planning 都可以围绕它组织。
H4: state vocabulary 必须先离线收敛并冻结/版本化,downstream policy/dynamics 不应追逐漂移的码本。
```

这和当前 compatible 版不同。当前代码是:

```text
video latent + proprio + state code token -> ActionDiT
state code -> future code auxiliary loss
```

Native 目标应该是:

```text
observation -> state code
state code + proprio + task + history -> action
state code + action -> next state code
future state code -> value / stage / success / contact
```

重点变化:

- Code 从辅助 token 变成主状态。
- Dynamics 从辅助 regularizer 变成世界模型核心。
- Evaluation 从离线 loss 扩展到 action relevance 和 robot outcome。

## 3. Code 的优势和失败条件

### 3.1 优势

**抗 shortcut**

离散 code 是瓶颈,可以迫使视觉信息以更紧凑的方式进入 policy。它必须证明:

```text
code + proprio -> action  >  proprio-only -> action
```

**低成本 dynamics**

与 pixel rollout 相比,code-space transition 更便宜:

```text
c_t, a_t -> c_{t+1}
c_t, a_{t:t+k} -> c_{t+k}
```

**可诊断**

可以统计 usage、perplexity、transition、nearest neighbor、stage clustering、扰动响应。

**可规划**

离散状态支持 retrieval、beam search、candidate action scoring。

**可复用**

如果 code 表示的是行为状态,新任务可以复用 tokenizer/dynamics,只换 policy/value/goal head。

### 3.2 失败条件

如果出现下面情况,CodeWAM-native 的假设就不成立或需要换路线:

- `code + proprio` 不强于 `proprio-only`。
- 图像扰动改变了画面,但 code/action 不变。
- code usage 看起来健康,但 nearest neighbor 只按背景/光照聚类。
- next-code top1 高,但只是复制当前 code 或利用 proprio shortcut。
- policy 加 code 后离线 loss 下降,闭环仍然不动/不刹车/不抓取。
- code bottleneck 丢失关键几何细节,导致动作抖动或接触失败。

这意味着评估不能只看 codebook 本身,必须看 action-level 和 outcome-level。

### 3.3 码本生命周期原则

RQ-VAE/RVQ 的价值不是让 policy 训练时继续流式改聚类中心,而是先把观察空间翻译成
一套稳定的机器人状态语言。这个语言一旦进入 policy、dynamics、retrieval 或 planning,
就必须像 tokenizer vocabulary 一样被版本化和冻结。

允许发生更新的阶段:

```text
offline tokenizer pretraining:
    encoder / quantizer / EMA centers 可以正常训练和收敛
```

不应该发生更新的阶段:

```text
downstream policy/dynamics/value/retrieval training:
    frozen encoder + frozen codebook centers + fixed exported codes
```

原因:

- 如果 codebook center 边训边漂移,同一个 index 的语义会变,policy 学到的状态语言不稳定。
- Dynamics 的监督目标会非平稳,`c_t + a_t -> c_{t+1}` 不再是固定分类问题。
- Retrieval/planning 的历史索引会失效,nearest neighbor 和 rollout 结果不可比较。
- Diagnostics 会变得混乱,perplexity、usage、transition entropy 无法跨 epoch 对齐。
- 小数据本地 demo 容易被 batch bias 带偏,码本会追逐近期样本而不是形成全局状态词表。

更合理的做法是:

```text
codebook_v1:
    train offline -> freeze -> export codes -> train downstream

new data arrives:
    train codebook_v2 offline -> re-tokenize -> compare/migrate
```

每个 codebook 版本至少应记录:

```text
feature source: Wan-VAE latent / video-DiT hidden / mixed
feature config: camera, frame window, resolution, layer, stride
quantizer config: RQ levels, codebook size, commitment weight, EMA setting
dataset manifest: trajectory list, split, hash/version
frozen artifacts: encoder checkpoint, quantizer centers, normalization stats
exports: per-trajectory code arrays, timestamps, mask/valid flags
stats: usage, perplexity, dead-code ratio, nearest-neighbor samples
```

所以第一版 CodeWAM-native 应把 RQ-VAE 当成离线 state tokenizer,而不是 policy 训练环路中的
可变聚类模块。

### 3.4 当前阶段门: 先证明码本

当前工作顺序应调整为:

```text
先评估 codebook 是否是有用的状态度量,
再设计 code embedding / register token 如何进入 policy。
```

Package Scan v6 只作为本机链路检查。真正有借鉴意义的第一阶段评估应放在公开数据集上:

```text
LIBERO / robomimic:
    小而干净,适合验证 usage、phase、action relevance。

CALVIN / BridgeData V2:
    更长时序和真实视频,适合验证 temporal code 是否稳定且有阶段意义。

DROID / Open X-Embodiment subset:
    多场景压力测试,不应作为第一步全量入口。
```

第一版 codebook evaluation 已按统一 latent cache 组织:

```text
public robot dataset
-> Wan-VAE latent cache [N,C,T,H,W]
-> temporal interleaved descriptor
-> KMeans / RQ candidates
-> metrics and frozen artifacts
```

当前推荐 descriptor 不再额外训练重 State Encoder,而是直接度量 Wan latent:

```text
desc_s(t) = concat(pool(f_t), pool(f_{t+s}), pool(f_{t+s}) - pool(f_t))
s in {2, 3, 5}
```

这样每个当前状态都有三套互质时间间隔码本:

```text
c_t = {RQ_2(desc_2(t)), RQ_3(desc_3(t)), RQ_5(desc_5(t))}
```

每套码本暂定 3 层 RQ。`K=256/512/1024` 应由 usage、perplexity、dead code、重构误差、
temporal transition、action relevance 和 retrieval sanity 共同决定,而不是提前拍定。

## 4. 设计选择总览

下面是供我们取舍的几条路线。它们不是互斥的,但阶段上应有先后。

```text
Option A: Compatible CodeWAM
Option B: Hidden-Feature CodeWAM
Option C: Native Code-State Policy
Option D: Object/Contact-Centric CodeWAM
Option E: Retrieval/Planning CodeWAM
```

### 4.1 Option A: Compatible CodeWAM

结构:

```text
Wan-VAE latent -> RQ code -> context token
FastWAM ActionDiT/MoT -> action
RQ dynamics -> auxiliary future-code loss
```

优点:

- 最容易复用当前代码。
- 和 FastWAM 做公平对照最直接。
- 适合集群大模型实验。

风险:

- code 可能继续是可忽略 token。
- policy 仍可能依赖 proprio。
- 很难证明 code 是主状态。

适用阶段:

- 作为 baseline 和 large-scale compatible 对照保留。

推荐程度:

```text
必须保留,但不应作为最终形态。
```

### 4.2 Option B: Hidden-Feature CodeWAM

结构:

```text
video DiT intermediate hidden feature -> tokenizer -> state code
state code + proprio + task -> action
state code + action -> next state code
```

来源:

- DiT4DiT 的 intermediate denoising feature 思路。

优点:

- code 源头更接近 action-relevant video dynamics。
- 比直接量化 VAE latent 更可能捕捉时空变化。
- 可以和 DiT4DiT 做清晰理论对照: continuous hidden feature vs discrete code。

风险:

- 需要跑 video DiT 抽特征,成本高于 VAE latent。
- 不同层/denoise step 的选择需要实验。
- 依赖特定 video backbone,迁移性要验证。

适用阶段:

- 在 Package Scan v6 上离线抽特征做 tokenizer/source ablation。

推荐程度:

```text
高。它可能是 CodeWAM 从 "VAE 压缩器" 走向 "动作动态状态码" 的关键。
```

### 4.3 Option C: Native Code-State Policy

结构:

```text
image/video -> tokenizer -> code
code history + proprio history + task -> belief
belief -> action chunk
belief + action -> future code
future code -> stage/value/success
```

优点:

- 最符合 CodeWAM-native 的定义。
- code 是 policy 主输入,不再只是附加 token。
- 可以低成本快速迭代,不必一开始接 5B 大模型。

风险:

- 小模型结果不一定能直接放大到 FastWAM 级别。
- 如果 tokenizer 不够好,policy 上限会受限。
- 需要自己搭 action head、belief、diagnostics。

适用阶段:

- Package Scan v6 本机/小集群实验。

推荐程度:

```text
最高。它是验证 CodeWAM 核心假设的主线。
```

### 4.4 Option D: Object/Contact-Centric CodeWAM

结构:

```text
image/video -> object/region/contact tokens -> hierarchical code
object/contact code + proprio -> action
object/contact code + action -> next relation/contact state
```

来源:

- CHORD 的 contact wrench 思路。
- DexVerse 的 contact-rich/task taxonomy。
- Package Scan v6 中 package、barcode、gripper、pallet/box 是关键对象。

优点:

- 更贴近真实任务关键变量。
- 可以减少背景/光照对 code 的污染。
- 对条码朝向、抓取状态、放置状态更自然。

风险:

- 需要 object/region prior,工程复杂。
- 没有标注时容易引入噪声。
- contact proxy 难以从纯视觉稳定估计。

适用阶段:

- 在基础 native policy 证明 code 有用之后,作为增强路线。

推荐程度:

```text
中高。不要第一步就做复杂检测,但应保留为第二阶段方向。
```

### 4.5 Option E: Retrieval/Planning CodeWAM

结构:

```text
current code -> retrieve similar trajectories
candidate action -> code dynamics rollout
future code -> value/stage/success score
select action or bias policy
```

优点:

- 最能体现离散 code 的独特价值。
- 适合小数据真机任务,历史相似片段很有用。
- 可解释性强。

风险:

- 依赖可靠 code dynamics/value。
- 闭环部署复杂。
- 若 code 粗糙,检索会误导 policy。

适用阶段:

- tokenizer、policy、dynamics 都初步可靠之后。

推荐程度:

```text
作为后续亮点,不作为第一阶段核心。
```

## 5. 模块级选择

### 5.1 Tokenizer 选择

| 方案 | 优势 | 风险 | 适用 |
|---|---|---|---|
| RQ/RVQ over Wan-VAE latent | 当前已有,简单直接,可解释 | 可能只是视觉压缩,不够 action-relevant | 第一基线 |
| RQ/RVQ over video DiT hidden | 更接近 dynamics/action feature | 抽特征成本高,需选层 | 强推荐 ablation |
| FSQ/LFQ | 可能更稳,死码风险低 | 需要重写 tokenizer/dynamics 适配 | RQ 不稳时 |
| Hierarchical code | coarse/fine 兼顾阶段和几何 | 训练复杂 | 第二阶段 |
| Object/region code | 更贴近任务变量 | 需要 region prior | Package Scan 增强 |
| Contact/event code | 抓取/接触更明确 | 纯视觉估计难 | contact-rich 任务 |

建议第一轮不要只做一个 tokenizer,至少比较:

```text
T1: Wan-VAE latent + RQ
T2: video DiT hidden + RQ
T3: Wan-VAE latent + FSQ/LFQ
```

但无论选哪种 tokenizer,第一阶段都应遵循同一个生命周期:

```text
L0: collect latent/hidden features
L1: offline train RQ/FSQ/LFQ tokenizer
L2: freeze encoder and quantizer centers
L3: export fixed codes for all trajectories
L4: train policy/dynamics/value/retrieval on frozen codes
L5: optional offline codebook_v2 retraining and re-tokenization
```

其中 `L4` 不更新码本。后续如果确实想让 encoder 适配 policy,也应先做
`frozen centers + encoder consistency/distillation` 的受控实验,而不是直接让聚类中心流式更新。

### 5.2 Belief 选择

| 方案 | 优势 | 风险 |
|---|---|---|
| Single-frame code | 简单,便于诊断 | 时间歧义严重 |
| Code history Transformer | 适合 chunk policy,表达强 | 数据少时可能过拟合 |
| GRU/SSM belief | 小数据稳,低延迟 | 表达力弱于 Transformer |
| Object memory | 对阶段/目标保持好 | 需要 object code 稳定 |

建议:

```text
先做 single-frame vs short-history 对照。
如果 history 明显提升,belief 就是 native 主干必需模块。
```

### 5.3 Dynamics 选择

| 方案 | 形式 | 用途 |
|---|---|---|
| Unconditioned dynamics | c_t -> c_{t+1} | 检查视觉状态可预测性 |
| Action-conditioned dynamics | c_t, a_t -> c_{t+1} | 真正 world-action model |
| Multi-step dynamics | c_t, a_{t:t+k} -> c_{t+k} | planning/retrieval |
| Contrastive dynamics | positive future vs negatives | 降低离散 CE 噪声 |
| Stochastic dynamics | p(c_{t+1} | c_t, a_t) | 多未来分支 |

建议:

```text
第一轮必须从 action-conditioned dynamics 开始。
只做 c_t -> c_{t+1} 不足以支撑 WAM claim。
```

### 5.4 Policy 选择

| 方案 | 说明 | 判断 |
|---|---|---|
| code as extra token | 当前 compatible 路线 | baseline |
| code bottleneck policy | policy 主要看 code/proprio/task | native 主线 |
| code + continuous residual | 防止几何信息损失 | 推荐保留 |
| MoE by code/stage | 不同状态激活不同 action expert | 后续探索 |
| retrieval-biased policy | 相似历史动作作为 prior | 小数据增强 |

建议:

```text
native policy 不要完全禁止 continuous residual。
更合理的是 code 为主、低维 residual 为辅,避免瓶颈过强。
```

### 5.5 Anti-shortcut 训练技巧

从 EgoSteer 和当前问题出发,应系统加入:

- proprio history masking
- camera dropout
- code dropout
- code-only / proprio-only auxiliary heads
- image perturbation consistency
- action sensitivity regularization
- balanced decision-frame sampling

关键不是让模型看不到 proprio,而是避免它只用 proprio。

## 6. 推荐主路线

我的建议是采用 "双线并行,Native 优先验证"。

```text
Line A: Compatible
    保留当前 FastWAM-compatible 结构,用于大模型对照。

Line B: Native
    在 Package Scan v6 上快速验证 code 是否能成为 action-relevant state。
```

其中 Line B 应该优先。

### 6.1 第一阶段: Package Scan v6 Native Probe

目标:

```text
证明 code 不是视觉压缩玩具,而是 action-relevant state。
```

实验:

```text
F0: offline train tokenizer and freeze codebook_v1
F1: export fixed trajectory codes
E1: proprio-only -> action
E2: code-only -> action
E3: code + proprio -> action
E4: code + proprio + history -> action
E5: code + action -> next frozen code
E6: future frozen code -> stage/success/contact proxy
```

必要扰动:

```text
P1: mask image / blur image / camera dropout
P2: mask proprio history
P3: shuffle code across batch
P4: replace current code with nearest-neighbor code
P5: action-conditioned future-code rollout
```

通过条件:

```text
E3 > E1
E4 >= E3 and more stable
E5 > copy-current-code and proprio-only dynamics
image/code perturbation changes action sensibly
nearest neighbors cluster by task state, not background
downstream training never updates codebook centers
```

### 6.2 第二阶段: Hidden Feature Code

目标:

```text
验证 DiT4DiT 式 intermediate dynamics features 是否比 VAE latent 更适合作为 code source。
```

实验:

```text
S1: Wan-VAE latent code
S2: video DiT middle-layer hidden code
S3: video DiT late-layer hidden code
S4: mixed VAE + DiT hidden code
```

判断:

```text
action relevance
future-code predictability
visual perturbation sensitivity
runtime cost
```

如果 S2 明显更强,CodeWAM 的定义应从 "VAE latent codebook" 升级为:

```text
actionable video-dynamics state tokenizer
```

### 6.3 第三阶段: Compatible Cluster Training

目标:

```text
在同数据/同评测下,比较 FastWAM continuous latent 和 CodeWAM discrete state code。
```

对照:

```text
C1: FastWAM baseline
C2: Code token only, no dynamics
C3: Dynamics only, no code token
C4: Code token + dynamics
C5: frozen tokenizer vs trainable tokenizer
C6: VAE-code vs hidden-feature-code
```

必须保留 action-level 诊断,否则 compatible 结果难解释。

### 6.4 第四阶段: Object/Contact 和 Planning

目标:

```text
让 code 从 "视觉状态" 走向 "任务/接触/对象关系状态"。
```

方向:

- package/gripper/barcode/pallet object code
- grasp/contact/stability event head
- code nearest-neighbor retrieval
- candidate action code rollout
- future-code value/stage scoring

## 7. 决策菜单

下面是需要你最终拍板的关键选择。

### 7.1 Code 源头

**选择 A: Wan-VAE latent**

- 最稳、最快、当前代码延续性最好。
- 风险是 action relevance 可能不够。

**选择 B: video DiT hidden feature**

- 更贴近 DiT4DiT 的 actionable dynamics。
- 成本更高,但理论更强。

**选择 C: VAE + hidden feature mixed**

- 兼顾几何和动态。
- 结构更复杂。

我的倾向:

```text
先 A+B 对照,不要直接上 C。
```

### 7.2 Code 粒度

**选择 A: 全局 pooled code**

- 简单。
- 容易丢几何。

**选择 B: patch/grid code**

- 保留空间关系。
- token 更多,dynamics 更难。

**选择 C: object/region code**

- 最贴任务。
- 需要 region prior。

我的倾向:

```text
Package Scan v6 先用 grid code,后续再转 object/region code。
```

### 7.3 Policy 是否必须经过 code bottleneck

**选择 A: soft bottleneck**

```text
policy = code + proprio + low-dim residual
```

优点是安全,不易丢信息。

**选择 B: hard bottleneck**

```text
policy = code + proprio only
```

优点是检验最干净,风险是性能受限。

我的倾向:

```text
实验上两者都做;工程默认 soft bottleneck。
```

### 7.4 是否早期引入 planning

**选择 A: 先不 planning**

- 专注证明 code action relevance。

**选择 B: 早期加 retrieval**

- 小数据可能很有帮助。

**选择 C: 早期加 rollout planning**

- 最有野心,但依赖 dynamics 可靠。

我的倾向:

```text
先做 retrieval 诊断,不急着闭环 rollout planning。
```

### 7.5 是否继续重投入 FastWAM-compatible

**选择 A: Compatible 为主**

- 快速接大模型和论文 baseline。
- 风险是继续被框住。

**选择 B: Native 为主,Compatible 保底**

- 最符合 CodeWAM 新结构探索。
- 风险是短期工程成果不如直接训大模型明显。

我的倾向:

```text
Native 为主,Compatible 保底。
```

### 7.6 码本是否允许 downstream 更新

**选择 A: fully frozen tokenizer/codebook**

```text
offline train -> freeze encoder + centers -> export fixed codes -> train downstream
```

优点是语义稳定、诊断清楚、dynamics 目标固定。缺点是 tokenizer 不会自动适配 policy。

**选择 B: frozen centers + encoder-only fine-tune**

```text
centers 固定,encoder 在 consistency/distillation 约束下微调
```

优点是允许轻微 domain adaptation。风险是仍可能引入 code assignment drift,需要严格监控。

**选择 C: downstream streaming EMA / online cluster update**

```text
policy/dynamics 训练时继续更新 center
```

优点是看似自适应,但会破坏 state vocabulary 稳定性。只适合作为研究消融,不适合作为默认方案。

我的倾向:

```text
第一版采用 A。等 frozen 方案跑通后,再用 B 做受控增强;C 不作为主路线。
```

## 8. 评估体系

CodeWAM 的评估必须三层并行。

### 8.1 Representation

- codebook version id and dataset manifest
- code usage/perplexity
- dead code ratio
- assignment stability under frozen encoder/centers
- reconstruction / latent R2
- next-code top1 vs copy baseline
- transition entropy
- nearest-neighbor visualization
- code stability under mild image noise

### 8.2 Action Relevance

- code + proprio > proprio-only
- code-only 是否有非平凡预测力
- state-code ablation 是否伤害 action
- image/code perturbation 是否改变 action
- action-conditioned dynamics > proprio-only dynamics
- decision-frame action sensitivity 是否提升

### 8.3 Robot/Task Outcome

- closed-loop success
- no movement / no braking / wrong object / wrong placement
- Generalization-random
- Memory/state retention
- Precision/contact stability
- Open instruction grounding
- inference latency

RoboDojo 的提醒是: 最终不能只看平均成功率,要看能力维度。

## 9. 推荐执行顺序

### Phase 0: 文档和实验定义

- 把 CodeWAM-compatible / native / hidden-feature-code 三条线说清楚。
- 定义 Package Scan v6 的 native probe 指标。
- 明确哪些结果会否定当前假设。

### Phase 1: Native Probe

- 实现 tokenizer source ablation: VAE latent vs video hidden feature。
- 实现 offline codebook_v1 训练、冻结和 trajectory code export。
- 实现 proprio-only/code-only/code+proprio/code+history action heads。
- 实现 action-conditioned code dynamics。
- 实现 perturbation diagnostics。
- 明确禁止 downstream 训练更新 codebook centers。

### Phase 2: Native Prototype

- 小型 belief state。
- soft bottleneck policy。
- future-code stage/value/contact proxy。
- nearest-neighbor retrieval diagnostics。

### Phase 3: Compatible 对照

- 保留 FastWAM-compatible 大模型路线。
- 做 state token/dynamics/source ablation,以及 frozen tokenizer vs encoder-only adaptation 消融。
- 输出和 native probe 对齐的 action relevance 报告。

### Phase 4: 扩展能力

- object/region code。
- contact/event code。
- code-space retrieval。
- candidate action scoring。
- RoboDojo/DexVerse 风格能力评估。

## 10. 当前推荐方案

如果现在必须选一个最合理的默认方案,我建议:

```text
CodeWAM-native v0:

1. observation -> Wan-VAE latent and optional video-DiT hidden feature
2. offline train/version grid RQ codebook, then freeze encoder + centers and export fixed codes
3. code history + proprio history + task -> small belief transformer
4. belief -> action chunk
5. belief + action -> next frozen code
6. future code -> stage/contact/success proxy
7. diagnostics: proprio-only, code-only, perturbation, nearest-neighbor, copy baseline
```

同时保留:

```text
CodeWAM-compatible:
    FastWAM + state token + future-code loss
```

两条线的关系:

```text
Native 线回答: code 是否真的能成为 action-relevant state?
Compatible 线回答: 这个 state code 放进 FastWAM 大模型后是否能带来实际收益?
```

这样我们不会被 FastWAM 框住,也不会失去和 FastWAM 公平对照的工程基础。

## 11. 需要讨论的最终取舍

下面这些问题建议我们下一轮逐个定:

1. 第一轮 tokenizer 是否只做 Wan-VAE latent,还是同时抽 video DiT hidden feature?
2. 第一轮 code 粒度用 pooled code、grid code,还是直接尝试 object/region code?
3. Native policy 默认使用 hard bottleneck 还是 soft bottleneck?
4. 是否把 proprio masking/camera dropout/code dropout 作为第一轮必做?
5. Package Scan v6 上是否先只做 offline action prediction,还是直接做 small closed-loop replay/rollout?
6. Compatible 大模型训练是并行推进,还是等 native probe 给出明确正结果后再推进?
7. 码本生命周期是否采用 fully frozen v1,还是允许 frozen-centers encoder-only fine-tune?
8. 我们的论文/项目 claim 更偏:
   - cheap discrete WAM,
   - anti-proprio-shortcut state learning,
   - actionable video-dynamics tokenizer,
   - or code-space planning?

我的当前偏好是:

```text
主 claim: actionable discrete state for robot world-action modeling.
第一阶段重点: offline frozen RQ state vocabulary + anti-proprio-shortcut + action-conditioned code dynamics.
第二阶段亮点: hidden-feature code + retrieval/planning.
```

这条线既吸收 Fast-WAM/DiT4DiT 的 WAM 思路,又保留 CodeWAM 自己的结构野心。
