# Configs

CodeWAM 与 FastWAM 共享数据与评测管线以便公平对照。训练配置沿用 FastWAM 的 hydra 结构:
`model` 用 `_target_: codewam.runtime.create_codewam` + `state_codebook` 块;`data`/`task`
复用 FastWAM 的真机数据配置(如 `real_robot_joint_2cam_v6`)。

启用码本训练的最小 `state_codebook` 块示例:

```yaml
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

(具体 model/task yaml 待与 FastWAM 对照实验一起定稿。)
