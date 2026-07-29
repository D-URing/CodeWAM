# Configs

当前目录保存 FastWAM-compatible legacy 配置和 codebook evaluation 配置:

```text
configs/
├── train.yaml
├── data/
│   ├── libero_2cam.yaml
│   ├── robotwin.yaml
│   └── package_scan_v6.yaml
├── model/
│   └── codewam.yaml
└── task/
    ├── libero_codewam_2cam224.yaml
    ├── robotwin_codewam_3cam384.yaml
    └── package_scan_v6_demo.yaml
```

`configs/model/codewam.yaml` 当前保留早期兼容原型的参数,但默认关闭:

```yaml
_target_: codewam.runtime.create_codewam
state_codebook:
  enabled: false  # legacy online EMA prototype
  dim: 128
  n_levels: 3
  codebook_size: 64
  pool: 2
  dynamics_future_k: 1
  loss_lambda_dyn: 1.0
  loss_lambda_vq: 1.0
```

canonical v1 不在 policy 训练中拟合该模块。正式结构使用三套离线训练、冻结且彼此独立的
`Q2/Q3/Q5` RQ artifacts;五模块接口和可见性 contract 见 `docs/ARCHITECTURE.md`,数据与训练
contract 见 `docs/CODEBOOK.md`。独立 `codewam/models/` 与对应 config 尚未实现,所以下列
命令只验证 FastWAM-compatible legacy/F0 链路。

训练示例:

```bash
bash scripts/train_zero1.sh 8 task=libero_codewam_2cam224
bash scripts/train_zero1.sh 8 task=robotwin_codewam_3cam384
```

数据路径沿用 FastWAM 约定,默认在 `data/` 下;模型路径默认在 `checkpoints/` 下。
`package_scan_v6.yaml` 是本机 Package Scan v6 小 demo 数据入口,默认读取仓库根目录下被 git 忽略的
`package_scan_v6/`。
