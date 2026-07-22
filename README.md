# CodeWAM — Codebook World-Action Model

CodeWAM 是一个独立的世界动作模型(world-action model)方法。它复用 Wan-VAE、Video DiT、
ActionDiT 和 flow-matching 等基座能力,但不把 FastWAM 的对称 MoT 拓扑当作结构边界。核心是用
冻结的离散视觉状态码本,为连续 Wan latent 建立可度量、可预测、可干预的机器人状态坐标。

CodeWAM 不是对 FastWAM 的补丁或子类;它是一个平行方法,最终与 FastWAM **做对照**。

## 核心思想

CodeWAM v1 已锁定为连续与离散双路径:

```text
unquantized Wan latent -> continuous state H -> 精确几何与动作微调
three frozen RQ-3     -> 9 code tokens     -> 多时间尺度状态坐标
H + code + L/P        -> belief B          -> continuous action policy
shared state + action -> world expert      -> training-only future-code objective
```

三套码本基于因果窗口 `[t-2s,t-s,t]`,其中 `s in {2,3,5}`。它们彼此独立,三级 residual
centers 不共享,九个 code measurement tokens 不求和、不流式更新。Policy、Forward Dynamics
和 Video Prior 使用显式不同的 mask program,动作分支永远看不到未来 target。

实现仍按实验门推进,先建立可信的离线 codebook evaluation:

```text
public robot dataset
-> Wan-VAE latent cache
-> causal temporal descriptors at stride 2/3/5
-> three independent 3-level RQ codebooks
-> usage / reconstruction / temporal / action relevance metrics
```

完整结构、已知/未知信息边界和实现顺序见
[`docs/CODEWAM_V1_PLAN.md`](./docs/CODEWAM_V1_PLAN.md)。

## 结构

```
codewam/
├── codebook.py    # legacy online EMA 原型,默认关闭
├── codebook_eval/ # 早期离线 evaluator,待迁移到 canonical causal contract
├── model.py       # 当前 FastWAM-compatible 模型原型,不是最终 v1 topology
├── runtime.py     # create_codewam 工厂(hydra _target_)
└── probe.py       # 早期兼容探针
configs/          # model / task 配置(与 FastWAM 共享数据管线以便对照)
docs/CODEWAM_V1_PLAN.md # canonical architecture + mask program
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
CodeWAM 从 compatible 接入走向 native 架构的规划见
[`docs/CODEWAM_NATIVE_DESIGN.md`](./docs/CODEWAM_NATIVE_DESIGN.md)。当前推荐的 hybrid 架构见
[`docs/CODEWAM_HYBRID_ARCHITECTURE.md`](./docs/CODEWAM_HYBRID_ARCHITECTURE.md)。当前唯一的 v1
结构与实验决策规范见 [`docs/CODEWAM_V1_PLAN.md`](./docs/CODEWAM_V1_PLAN.md)。

## 本机 Package Scan v6 Demo

`package_scan_v6/` 是当前本机真机小 demo 数据目录,不入库。链路检查:

```bash
python scripts/demo_package_scan_v6.py
```

脚本会读取 LeRobot v3 parquet 元数据、解码 top/wrist 两路 AV1 视频、构造 CodeWAM 风格窗口,并在
`runs/package_scan_v6_demo/` 下保存预览条。

## 离线码本评估

本机先验证训练器:

```bash
python scripts/codebook_eval.py synthetic-smoke
```

从公开数据集的 Wan latent shards 一键训练候选码本:

```bash
bash scripts/train_codebooks.sh configs/codebook_eval/public_latent_codebooks.yaml
```

详见 [`docs/CODEBOOK_EVAL.md`](./docs/CODEBOOK_EVAL.md)。

## 状态

- 已准备:外部上游 sparse checkout、模型下载脚本、ActionDiT 预处理、Hydra 训练配置、
  本机 Package Scan v6 demo reader。
- 已锁定:三套独立 causal RQ、九个只读 code measurements、连续状态路径、belief 聚合器、
  mode-specific Policy/FD/Prior masks 和可选 MemoryPort。
- 当前边界:`codewam/codebook.py` 的在线 EMA 单 token 原型已默认关闭;不能作为 v1 实验结果。
- 下一步:迁移 evaluator 到 episode split、train-only normalization、held-out probe、retrieval
  montage 和几何扰动测试,先完成 Gate 0/1/2,再实现模型接口与 mask 单测。

项目决策以 [`docs/CODEWAM_V1_PLAN.md`](./docs/CODEWAM_V1_PLAN.md) 为准;早期兼容原型说明见
[`docs/DESIGN.md`](./docs/DESIGN.md)。
