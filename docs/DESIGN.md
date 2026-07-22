# World Fast Codebook — FastWAM 的 RQ 离散状态码本扩展

> **Legacy prototype:** 本文只记录早期 `StateEncoder + online EMA RQ + single token` 兼容实现。
> 该路径现已默认关闭,不代表 CodeWAM v1。三套独立 frozen RQ、九个只读测量 token、连续状态
> 路径和 mode-specific mask 的唯一规范见 `docs/CODEWAM_V1_PLAN.md`。

> 本文件记录在 FastWAM 之上引入的 **World Fast Codebook**(RQ 离散状态码本)扩展:动机、
> 设计、代码落点、判定实验结果、如何启用。核心代码全部为**可选**(默认关闭,与原 FastWAM 行为一致)。

## 1. 动机

在真机(Piper 单臂 7D)小数据微调时,FastWAM 的头号病是 **proprio 捷径**:动作专家直接从本体
感受(当前关节角)外推动作,几乎忽略图像与语言 → 闭环不动/不刹车。根因是动作 loss 被
"proprio 可推的脚本化够向"主导,视觉决策帧梯度弱。

借鉴 ElimWM 的**可迁移内核 = RQ 离散潜码本**(不是不可达剪枝),把 FastWAM 结合处传递的
"连续、庞大、易被绕过的视觉状态"换成一个**离散、紧凑、被迫承载任务信息的"状态词表"**。

类比 LLM:RQ 码本 = 学出来的"状态 tokenizer";一个共享码本同时充当三角色:
- **A 条件**:当前观测 → 状态 token → 动作专家 cross-attend(理解通路)。
- **B 离散世界模型**:从当前状态码预测未来帧的状态码(交叉熵),取代/补充像素扩散想象(廉价世界模型)。
- **C 小数据微调**:大数据预训码本,真机冻结码本只调动作头(强正则、抗过拟合)。

三者互锁:B 的"预测未来码"损失强迫码编码可控/因果动力学 → 修 A 的"瓶颈≠强制使用";
A 给码任务接地;C 解小数据覆盖不足。

## 2. 判定实验(接入前的必要条件门)

在动大结构前,先在真机 `package_scan_v6` 的 Wan-VAE latent 上验证三条 linchpin 假设
(脚本:`sample_dump/state_codebook_probe.py`):

- **P1 可离散性**:池化 latent 能否被 RQ 干净量化。
- **P2 动力学可预测**:从当前状态码预测下一 latent 帧的码,top1 必须打过"复制当前码"基线。
- **P4 因果信息**:视觉状态码是否带 proprio 之外的信息(破捷径的必要条件)。

**结果(N=800, K=64, pool=2, 冻结码同预算公平比,held-out 窗口):**

| 指标 | 结果 |
|---|---|
| P1 每级码字利用率 | 0.94 / 0.88 / 0.89(困惑度 ~50/38/40) |
| P1 量化相对误差 / 4×4 全空间重构 R² | 0.069 / 0.679 |
| P2 下一码 top1:视觉 vs 复制基线 | 0.550 vs 0.119(**+0.431**) |
| P4 下一码 top1:视觉 vs proprio-only | 0.550 vs 0.174(**视觉 +0.376**) |

结论:真机视觉状态在 Wan-VAE latent 上**可离散化、可预测、且带 proprio 之外的因果信息**,门通过 → 接入。

**关键教训(务必保留)**:全局平均池化(pool=1)+ 通道均值重构会导致 (a) 码本坍塌(利用率→0.05)
(b) proprio 反而比视觉码更能预测下一码(假性捷径)。必须 **pool≥2 保空间细节 + 重构全空间 latent
+ 冻结码本后同预算公平比 视觉/proprio 头**。ElimWM 的 σ/VAE 不确定性机制在其项目里基本失败,不予移植。

## 3. 代码落点

新增/改动(均可选,`state_codebook.enabled=false` 时与原 FastWAM 完全一致):

- `src/fastwam/models/wan22/state_codebook.py`(**新增**):
  - `StateEncoder`(可配 `pool` 保留空间网格)、`ResidualQuantizer`(EMA 码本 + commitment + **死码重置**,直击坍塌)、
    `DynamicsHead`(预测未来状态码)、`StateCodebook`(封装)。
- `src/fastwam/models/wan22/fastwam.py`:
  - `__init__` / `from_wan22_pretrained` 加 `state_codebook` 参数;码本保持 **fp32**(EMA 精度),`to()` 里 `.float()` 防 bf16 转换。
  - **A**:`_encode_state_and_append_context`(在 `build_inputs` 里、proprio 之后)——首帧 latent→RQ 状态码→拼进 context。
  - **B**:`training_loss` 末尾加 `λ_vq·vq + λ_dyn·CE(未来帧码)`;`loss_dict` 新增 `loss_vq/loss_dyn/sc_dyn_top1/sc_usage`。
  - `save/load_checkpoint` 存取 `state_codebook`。
- `src/fastwam/runtime.py`:`create_fastwam` 透传 `state_codebook`。
- `configs/model/fastwam.yaml`:新增 `state_codebook` 块(默认 `enabled: false`)。
- `configs/task/real_robot_joint_2cam224_v6_codebook.yaml`(**新增**):继承 `v6_clean` 并启用码本。

验证:CPU 单测(ST 梯度 / EMA / 死码重置)、全模型 1-step smoke(`loss_dyn/loss_vq` 生效、梯度到 encoder+dynamics、码本保持 fp32)、hydra 配置合成 均通过。

## 4. 如何启用

```bash
bash scripts/train_zero1.sh 8 \
  task=real_robot_joint_2cam224_v6_codebook \
  model.model_id=Wan2.2-TI2V-5B model.tokenizer_model_id=Wan2.2-TI2V-5B \
  model.redirect_common_files=false model.load_text_encoder=false \
  model.skip_dit_load_from_pretrain=false model.action_dit_pretrained_path=null \
  model.mot_checkpoint_mixed_attn=false mixed_precision=bf16 wandb.enabled=false
```

码本超参在 `configs/model/fastwam.yaml` 的 `state_codebook` 块:`dim/n_levels/codebook_size/pool/dynamics_future_k/loss_lambda_dyn/loss_lambda_vq`。

## 5. 下一步真正判据

窗口内下一帧可预测(P2)只是**必要条件**。真正证明"破动作级 proprio 捷径",要训一段后用
`sample_dump/` 的三通道动作敏感性诊断,看**图像通道 `max|Δ|` 是否从 ~2.45° 抬升**——这才是最终证据。
