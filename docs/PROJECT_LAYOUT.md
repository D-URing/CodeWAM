# Project Layout

```text
CodeWAM/
├── codewam/                   # CodeWAM package
│   ├── codebook.py             # StateEncoder, RQ, DynamicsHead, StateCodebook
│   ├── model.py                # CodeWAM assembly, training, inference, checkpoints
│   ├── probe.py                # P1/P2/P4 latent probe
│   └── runtime.py              # Hydra factory
├── configs/                    # Hydra configs for model/data/task/train
├── scripts/                    # Bootstrap, download, train, environment checks
│   ├── accelerate_configs/      # accelerate launch configs
│   └── ds_configs/              # DeepSpeed configs
├── requirements/                # Local-dev and cluster CUDA dependency sets
├── docs/                       # Design, setup, training, and layout docs
├── external/                   # External source checkouts; contents ignored by git
├── checkpoints/                # Model files; ignored by git
├── data/                       # Datasets; ignored by git
├── runs/                       # Training outputs; ignored by git
└── upstreams.yaml              # Pinned upstream repositories and commits
```

## Local-Only Directories

These directories may exist during development but are intentionally not tracked:

```text
.venv/                         # Local Python 3.10 development environment
external/FastWAM/               # Sparse checkout of pinned FastWAM upstream
checkpoints/                    # Wan/FastWAM/ActionDiT model files
data/ or datasets/              # Local or cluster-mounted datasets
runs/, outputs/, logs/, wandb/   # Training and evaluation artifacts
.hf/                            # Legacy project-local Hugging Face cache
```

Generated Python artifacts such as `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`,
and `*.egg-info/` should be treated as disposable local state.

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
