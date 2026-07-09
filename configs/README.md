# Configs

CodeWAM 使用与 FastWAM 相同的 Hydra 组织方式:

```text
configs/
├── train.yaml
├── data/
│   ├── libero_2cam.yaml
│   └── robotwin.yaml
├── model/
│   └── codewam.yaml
└── task/
    ├── libero_codewam_2cam224.yaml
    └── robotwin_codewam_3cam384.yaml
```

`configs/model/codewam.yaml` 使用:

```yaml
_target_: codewam.runtime.create_codewam
state_codebook:
  enabled: true
  dim: 128
  n_levels: 3
  codebook_size: 64
  pool: 2
  dynamics_future_k: 1
  loss_lambda_dyn: 1.0
  loss_lambda_vq: 1.0
```

训练示例:

```bash
bash scripts/train_zero1.sh 8 task=libero_codewam_2cam224
bash scripts/train_zero1.sh 8 task=robotwin_codewam_3cam384
```

数据路径沿用 FastWAM 约定,默认在 `data/` 下;模型路径默认在 `checkpoints/` 下。
