# CodeWAM — Codebook World-Action Model

CodeWAM 是一个独立的世界动作模型(world-action model)方法。它与 FastWAM **复用同一套基座专家**
——Wan 视频专家 + ActionDiT 动作专家 + MoT 混合注意力 + Wan-VAE + flow-matching——但采用**不同的
组合形式**:围绕离散状态码本构建可度量、可预测、可复用的机器人状态空间。

CodeWAM 不是对 FastWAM 的补丁或子类;它是一个平行方法,最终与 FastWAM **做对照**。

## 核心思想

当前阶段的主线不是先把 codebook 接进 policy,而是先回答:

```text
Wan-VAE / DiT latent 能否被离散化成有行为意义的 state code?
```

为此,CodeWAM 先建立离线 codebook evaluation:

```text
public robot dataset
-> Wan-VAE latent cache
-> temporal interleaved descriptors, e.g. stride 2/3/5
-> KMeans / 3-level RQ codebooks
-> usage / reconstruction / temporal / action relevance metrics
```

如果离线指标证明 codebook 确实形成了有用的状态度量,下一阶段再讨论 code embedding /
register token 应该如何注入 ActionDiT 或 MoT 中间层。

## 结构

```
codewam/
├── codebook.py   # StateEncoder(可配 pool) + ResidualQuantizer(EMA + 死码重置) + DynamicsHead + StateCodebook
├── codebook_eval/ # 离线 latent descriptor、KMeans/RQ 训练与指标
├── model.py      # class CodeWAM: 组装基座专家 + 码本; build_inputs(A) / training_loss(B) / infer
├── runtime.py    # create_codewam 工厂(hydra _target_)
└── probe.py      # 早期兼容探针,后续会被 codebook_eval 路线替代
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
CodeWAM 从 compatible 接入走向 native 架构的规划见
[`docs/CODEWAM_NATIVE_DESIGN.md`](./docs/CODEWAM_NATIVE_DESIGN.md)。

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
- 当前推进:公开数据集 latent cache -> 多互质时间间隔 descriptor -> KMeans/RQ 码本候选 ->
  离线指标对比。
- 下一步:补充公开数据集 adapter/latent export,并加入 action relevance 与 retrieval sanity check。
- 工程整理:已补齐外部上游 sparse checkout、模型下载、ActionDiT 预处理、Hydra 训练配置和
  CodeWAM 训练入口;外部代码/模型默认不入库。

详见 [`docs/DESIGN.md`](./docs/DESIGN.md)。
