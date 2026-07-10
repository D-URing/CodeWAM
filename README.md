# CodeWAM — Codebook World-Action Model

CodeWAM 是一个独立的世界动作模型(world-action model)方法。它与 FastWAM **复用同一套基座专家**
——Wan 视频专家 + ActionDiT 动作专家 + MoT 混合注意力 + Wan-VAE + flow-matching——但采用**不同的
组合形式**:引入一个 **RQ 离散状态码本**作为表示层核心。

CodeWAM 不是对 FastWAM 的补丁或子类;它是一个平行方法,最终与 FastWAM **做对照**。

## 核心思想

把"当前观测"压成一个**离散、紧凑、可复用的状态码**(学出来的"状态词表"),让它同时承担:

- **A 状态条件**:当前帧 latent → RQ 状态码 → 作为额外 token 供动作专家 cross-attend。
- **B 廉价世界模型**:从当前状态码预测**未来帧的状态码**(离散动力学,交叉熵)+ vq 码本损失。
  这是对 FastWAM 路线的直接对照——FastWAM 在 test-time 逐步**想象未来像素**(贵),
  CodeWAM 只在**离散码空间**里预测未来状态(便宜)。

动机:真机小数据微调下,连续视觉状态容易被"proprioception 捷径"绕过(动作≈当前本体感受、
忽略视觉)。离散码本 + "被迫预测未来"的监督,逼模型用视觉、抗过拟合。

## 判定实验(接入前的必要条件门,已通过)

在真机数据的 Wan-VAE latent 上先验证三条 linchpin 假设(`codewam/probe.py`):

| 指标 | 结果(N=800, K=64, pool=2, 公平协议) |
|---|---|
| P1 可离散性:每级码字利用率 / 困惑度 | 0.94 / 0.88 / 0.89,困惑度 ~50/38/40 |
| P1 量化相对误差 / 4×4 全空间重构 R² | 0.069 / 0.679 |
| P2 下一码 top1:视觉 vs 复制基线 | **0.550 vs 0.119** |
| P4 下一码 top1:视觉 vs proprio-only | **0.550 vs 0.174** |

→ 真机视觉状态**可离散化、可预测、且带 proprio 之外的因果信息**。
（关键教训:必须 pool≥2 保空间细节 + 全空间重构 + 冻结码本后同预算公平比,否则会坍塌/假阴性。）

## 结构

```
codewam/
├── codebook.py   # StateEncoder(可配 pool) + ResidualQuantizer(EMA + 死码重置) + DynamicsHead + StateCodebook
├── model.py      # class CodeWAM: 组装基座专家 + 码本; build_inputs(A) / training_loss(B) / infer
├── runtime.py    # create_codewam 工厂(hydra _target_)
└── probe.py      # P1-P4 判定探针(在冻结 Wan-VAE latent 上)
configs/          # model / task 配置(与 FastWAM 共享数据管线以便对照)
docs/DESIGN.md    # 设计、判定实验、工程要点(反坍塌等)
```

## 依赖与安装

CodeWAM 复用 Wan 基座专家的构件,由 FastWAM 参考实现提供:

```bash
bash scripts/bootstrap_fastwam.sh   # sparse checkout FastWAM 到 external/FastWAM 并 editable install
pip install -e .                    # codewam; bootstrap 脚本默认也会执行
```

二者共享数据与评测,便于 **CodeWAM vs FastWAM** 的 apples-to-apples 对照。

推荐的完整准备顺序:

```bash
# 1) 拉取固定版本 FastWAM 依赖
bash scripts/bootstrap_fastwam.sh

# 2) 下载 Wan/FastWAM 模型文件到 checkpoints/
bash scripts/download_models.sh

# 3) 从 Wan DiT 预生成 ActionDiT backbone
bash scripts/prepare_action_dit.sh

# 4) 检查环境与模型文件
python3 scripts/check_environment.py --mode local

# 5) 训练示例
bash scripts/train_zero1.sh 8 task=libero_codewam_2cam224
```

外部依赖和模型固定在 [`upstreams.yaml`](./upstreams.yaml)。更多说明见
[`docs/BOOTSTRAP.md`](./docs/BOOTSTRAP.md) 和 [`docs/TRAINING.md`](./docs/TRAINING.md)。

## 本机 Package Scan v6 Demo

`package_scan_v6/` 是当前本机真机小 demo 数据目录,不入库。链路检查:

```bash
python scripts/demo_package_scan_v6.py
```

脚本会读取 LeRobot v3 parquet 元数据、解码 top/wrist 两路 AV1 视频、构造 CodeWAM 风格窗口,并在
`runs/package_scan_v6_demo/` 下保存预览条。

## 状态

- 已通过:码本模块 CPU 单测、P1/P2/P4 判定实验(真机 latent)、模型构建 + 1-step 前反向 smoke。
- 下一步(真正判据):训一段后用三通道动作敏感性诊断看**图像通道 max|Δ| 是否抬升**——
  这才是"破动作级 proprio 捷径"的证据;并与 FastWAM 在同数据/同评测下对照成功率与推理代价。
- 工程整理:已补齐外部上游 sparse checkout、模型下载、ActionDiT 预处理、Hydra 训练配置和
  CodeWAM 训练入口;外部代码/模型默认不入库。

详见 [`docs/DESIGN.md`](./docs/DESIGN.md)。
