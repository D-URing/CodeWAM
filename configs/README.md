# Configs

当前目录同时保存 structured CodeWAM v2、retained v1 baseline、FastWAM-compatible legacy 和
codebook evaluation 配置:

```text
configs/
├── train.yaml
├── data/
│   ├── libero_2cam.yaml
│   ├── robotwin.yaml
│   └── package_scan_v6.yaml
├── model/
│   ├── codewam_v1.yaml
│   ├── codewam_v2.yaml
│   └── codewam.yaml
├── gate2/
│   └── droid_joint_v1.yaml
├── policy/
│   ├── droid_c012_v1.yaml
│   └── droid_c012_v2.yaml
└── task/
    ├── libero_codewam_2cam224.yaml
    ├── robotwin_codewam_3cam384.yaml
    └── package_scan_v6_demo.yaml
```

`configs/model/codewam_v1.yaml` 与 `codewam_v2.yaml` 的 `_target_` 都是 `CodeWAMConfig`;
具体由 `build_codewam_v1` 或 `build_codewam_v2` 构造,不能只看 dataclass 猜测架构。
真实 trainer 还必须显式加载 chart-local frozen artifacts,并按数据 adapter 核对
`proprio_dim/action_dim/language_dim`。它不是 legacy Hydra 训练入口。

`configs/model/codewam.yaml` 保留早期兼容原型的参数,但默认关闭:

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

canonical v2 不在 policy 训练中拟合该模块。正式结构使用三套离线训练、冻结且彼此独立的
`Q2/Q3/Q5` RQ artifacts;五模块接口和可见性 contract 见 `docs/ARCHITECTURE.md`,数据与训练
contract 见 `docs/CODEBOOK.md`。

`configs/gate2/droid_joint_v1.yaml` 是真实 joint cache 上的 dynamics gate,只训练 future-code
CE,固定 PERSIST/NOACT/TRUE/SHUFFLE 和动作干预口径。它不是 policy trainer;artifact/cache
路径必须由 `scripts/run_gate2.py` 显式提供。

下列训练命令仍只运行 FastWAM-compatible legacy/F0 链路:

```bash
bash scripts/train_zero1.sh 8 task=libero_codewam_2cam224
bash scripts/train_zero1.sh 8 task=robotwin_codewam_3cam384
```

数据路径沿用 FastWAM 约定,默认在 `data/` 下;模型路径默认在 `checkpoints/` 下。
`package_scan_v6.yaml` 是本机 Package Scan v6 小 demo 数据入口,默认读取仓库根目录下被 git 忽略的
`package_scan_v6/`。
