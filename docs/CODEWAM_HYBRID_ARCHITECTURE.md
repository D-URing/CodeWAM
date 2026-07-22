# CodeWAM Hybrid Architecture

> 本文保留设计动机和完整备选细节。当前 v1 的结构接口、实验门和通过标准以
> `docs/CODEWAM_V1_PLAN.md` 为准。

本文件记录当前推荐的 CodeWAM 新结构。核心判断是:

```text
CodeWAM 不应是 code-only policy。
CodeWAM 应是 continuous latent + discrete visual RQ code 的 hybrid world-action model。
```

也就是说,离散码本不替代连续视觉信息。它负责状态结构、动态模式和可检索坐标;
连续 latent 负责精确几何、姿态、接触细节和动作微调。

## 1. 直觉

机器人操作需要两类信息:

```text
连续信息:
    物体到底偏了几毫米,手爪角度差多少,接触边缘在哪里。

离散信息:
    当前处于什么操作状态,正在接近/接触/推动/插入/完成中的哪一类动态模式。
```

只用连续 latent,模型容易把所有东西混在一个高维向量里,可诊断性差,也容易走 proprio
shortcut。只用 code,模型会学到模式,但丢掉精确调节能力。

所以 CodeWAM 的基本形式是:

```text
Action = f(
    continuous visual latent,
    multi-prime RQ visual state codes,
    action/proprio history,
    language/task goal
)
```

## 2. 模块

### 2.1 Wan-VAE latent

Wan-VAE 是视觉压缩器。输入视频或多帧图像,输出时空 latent:

```text
video frames -> Wan-VAE encoder -> z_t
```

`z_t` 仍然保留位置、尺度、形状、遮挡、纹理和接触线索。CodeWAM 不能把它压成一个
全局均值后就丢掉空间结构。第一版应至少比较:

```text
pooled state descriptor
spatial grid descriptor
short temporal context descriptor
```

### 2.2 Multi-prime visual RQ codebook

RQ 是多层查字典:

```text
z -> code level 1 + code level 2 + code level 3
```

当前推荐用三套互质时间间隔码本:

```text
C2: 2-step visual state code
C3: 3-step visual state code
C5: 5-step visual state code
```

它们不是 delta-only code。它们仍然给当前视觉状态定位,只是定位时参考不同时间尺度:

```text
state_code_s(t) = RQ_s(visual_context centered at t, stride=s)
s in {2, 3, 5}
```

直觉:

```text
C2: 局部接触、短期移动、细微位移
C3: 连续运动趋势、局部阶段
C5: 更长状态迁移、任务阶段变化
```

每个时刻的状态不是一个 code,而是一组混合坐标:

```text
State(t) = {
    z_t,        # continuous precision
    c2_t,       # short-scale discrete state
    c3_t,       # mid-scale discrete state
    c5_t        # longer-scale discrete state
}
```

### 2.3 Fusion module

Fusion module 的职责不是简单拼接,而是分配信息角色:

```text
latent z_t:
    提供精确几何和动作微调信息。

RQ codes:
    提供状态模式、阶段、可检索坐标和动态先验。

proprio/action history:
    提供机器人自身连续状态和控制惯性。

language/task:
    提供目标方向和约束。
```

推荐形式:

```text
continuous latent tokens
    + code embeddings / code registers
    + proprio tokens
    + task tokens
    -> belief/action transformer
```

### 2.4 Code transition model

CodeWAM 的 world model 不应只预测像素或 latent,而应预测未来状态坐标:

```text
c_t, z_t, a_t -> c_{t+1}
c_t, z_t, a_{t:t+k} -> c_{t+k}
```

这让动作变成:

```text
当前状态坐标应该如何迁移?
为了实现这个迁移,连续动作应该是多少?
```

### 2.5 Action realizer

最终动作头仍然必须读取连续 latent:

```text
current z_t
+ current code coordinates
+ predicted/desired code transition
+ proprio/action history
+ task
-> action chunk
```

这一步负责精确调节。code 只提供模式和方向,不承担毫米级控制。

## 3. 来自 Genie 和 MaskViT 的借鉴

### 3.1 Genie

Genie 的可借鉴点是模块拆分:

```text
video tokenizer -> latent action model -> action-conditioned dynamics
```

对 CodeWAM 的含义:

- state tokenizer 和 action interface 要分开,不要一开始把视觉码本做成动作码本。
- latent action 可以作为后续扩展,但第一主线应是 visual state RQ。
- dynamics 要被设计成可控的状态迁移模型,而不是只生成好看的未来。
- Genie 发现 raw-pixel LAM 比 tokenized image LAM 更利于 controllability,提醒我们:
  码本不能丢掉关键运动细节,continuous latent 必须保留。

### 3.2 MaskViT

MaskViT 的可借鉴点是 masked token prediction:

```text
known context tokens + masked future tokens -> predict future tokens
```

对 CodeWAM 的含义:

- 可以把 future-code prediction 设计成 masked prediction,而不是只做下一步分类。
- variable mask ratio 可以让模型习惯不同程度的未来缺失。
- iterative refinement 可以作为未来 code rollout 的高效推理方式。
- MaskViT 的 per-frame tokenizer 出现 flicker,提醒我们:
  CodeWAM 的 visual RQ 必须重视 temporal consistency。

### 3.3 FastWAM

FastWAM 的可借鉴点是:

```text
training-time world modeling can help action prediction;
inference-time explicit future rollout is not always required。
```

对 CodeWAM 的含义:

- 未来预测可以作为训练信号,推理时 action head 可以直接输出动作。
- 但 CodeWAM 不应被限制为 FastWAM + token。FastWAM 只是 compatible baseline 和对照。

## 4. Mask 设计

要让每类信息正确发挥作用,只靠把所有 token 拼在一起是不够的。需要 mask、dropout、
loss 和诊断共同约束。

### 4.1 信息可见性 mask

原则:

```text
Action head 在训练时不能看到推理时看不到的信息。
```

尤其要隔离 ground-truth future:

```text
允许:
    action tokens attend to current/past visual latent, current/past codes, task, proprio history

禁止:
    action tokens attend to ground-truth future visual latent/codes/actions
```

训练时 video/future branch 可以使用 future target 做 denoising 或 masked prediction,但这些
future target 不能通过 attention 泄漏给 action branch。

### 4.2 MaskViT-style future code mask

用于训练 code dynamics:

```text
known:
    current/past codes, current latent, action chunk/task

masked:
    part or all future codes

target:
    predict masked future codes
```

mask ratio 不固定,而是在一个范围内采样:

```text
mask_ratio ~ Uniform(0.5, 1.0)
```

这样模型既学一步迁移,也学在未来信息很少时补全状态轨迹。

### 4.3 Modality dropout

为了避免某一路信息独大:

```text
latent dropout:
    部分训练步弱化连续视觉,检查 code 是否有用。

code dropout:
    随机丢掉 C2/C3/C5 或 RQ 某一层,避免模型只记单一码本。

camera dropout:
    多视角中随机丢一路,避免过拟合某个视角。

proprio dropout / history masking:
    降低 proprio shortcut,逼迫模型使用视觉。
```

这些 dropout 不是为了让模型永远缺信息,而是训练它在信息不完整时仍能合理分配权重。

### 4.4 Code family mask

三套互质码本需要单独控制:

```text
drop C2 only:
    检查短期局部动态是否可替代。

drop C3 only:
    检查中期趋势是否关键。

drop C5 only:
    检查阶段信息是否关键。

drop all codes:
    回到 continuous latent baseline。
```

如果丢掉某一套码本几乎不影响结果,说明它和其他码本冗余,需要重新训练或换时间间隔。

### 4.5 Code-level mask

RQ 的三层残差也应可 mask:

```text
level 1 only:
    粗状态是否足够?

level 1 + 2:
    加细节是否提升?

level 1 + 2 + 3:
    完整 RQ 是否带来动作精度?
```

这能回答每层 RQ 是否真的有用,而不是只增加参数。

### 4.6 Counterfactual mask / shuffle

训练和评估时加入反事实扰动:

```text
shuffle codes across batch:
    如果 action 几乎不变,说明模型没用 code。

replace code with nearest neighbor:
    action 应小幅变化或保持同阶段一致。

replace code with far neighbor:
    action 应明显改变或置信下降。

mask image but keep code:
    检查 code 是否承载状态模式。

keep image but mask code:
    检查 continuous latent 是否能补足精细动作。
```

这些不是主训练目标,但应进入固定诊断。

## 5. 训练目标

推荐第一版使用多目标,但明确主次:

```text
L_action:
    action chunk prediction / denoising,主目标。

L_code_recon:
    RQ 对 visual latent/context 的重构质量,离线 tokenizer 阶段使用。

L_future_code:
    action-conditioned masked future-code prediction。

L_temporal_consistency:
    相邻状态 code 不能无意义乱跳,但也不能全部复制当前 code。

L_anti_shortcut:
    proprio-only / code-only / latent-only heads 的对照和惩罚项。
```

第一版下游训练不更新 codebook centers:

```text
offline train codebook -> freeze -> export fixed codes -> downstream train
```

## 6. 最小可行版本

第一版 CodeWAM-hybrid v0:

```text
1. Export Wan-VAE latent z for trajectories.
2. Train frozen C2/C3/C5 visual RQ codebooks offline.
3. Export fixed codes for every trajectory timestamp.
4. Train action model with:
       z_t + c2_t/c3_t/c5_t + proprio history + task
5. Add masked future-code prediction as auxiliary world objective.
6. Run ablations:
       latent-only
       code-only
       latent+code
       latent+code+proprio
       code family dropout
       RQ level dropout
       future-code objective on/off
```

通过标准:

```text
latent + code > latent-only
latent + code + proprio > proprio-only
future-code objective improves action or diagnostics
code perturbation changes action sensibly
nearest-neighbor code groups by task state, not background
```

## 7. 当前定义

CodeWAM 当前定义为:

```text
A hybrid world-action model that uses frozen multi-scale visual RQ codes
as a discrete state coordinate system, while preserving continuous visual
latents for precise robot control.
```

中文表述:

```text
CodeWAM 是一个连续视觉 latent 与离散视觉 RQ 状态坐标共同驱动的世界动作模型。
码本提供状态结构和动态模式,latent 保留精确几何和操作细节。
```
