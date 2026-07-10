# Project Layout

```text
CodeWAM/
├── codewam/                  # CodeWAM package
│   ├── codebook.py            # StateEncoder, RQ, DynamicsHead, StateCodebook
│   ├── model.py               # CodeWAM assembly, training, inference, checkpoints
│   ├── probe.py               # P1/P2/P4 latent probe
│   └── runtime.py             # Hydra factory
├── configs/                   # CodeWAM Hydra configs
├── scripts/                   # Bootstrap, download, train launchers
├── requirements/               # Local-dev and cluster CUDA dependency sets
├── docs/                      # Design and setup docs
├── external/FastWAM/          # Local sparse checkout, ignored by git
├── checkpoints/               # Model files, ignored by git
├── data/                      # Datasets, ignored by git
└── runs/                      # Training outputs, ignored by git
```

## Boundary With FastWAM

CodeWAM owns:

- RQ state codebook
- CodeWAM-specific context-token injection
- future-code dynamics loss
- CodeWAM Hydra model/task configs

FastWAM remains the provider for:

- Wan video expert
- ActionDiT
- MoT
- Wan-VAE
- flow-matching schedulers
- dataset processors
- training runtime

The chosen FastWAM subtree and model repositories are pinned in
`upstreams.yaml`.
