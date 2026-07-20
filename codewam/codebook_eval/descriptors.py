from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DescriptorBatch:
    name: str
    family: str
    stride: int
    vectors: torch.Tensor
    sample_index: torch.Tensor
    time_index: torch.Tensor


def _pool_latents(latents: torch.Tensor, pool: int) -> torch.Tensor:
    if latents.ndim != 5:
        raise ValueError(f"`latents` must be [N,C,T,H,W], got {tuple(latents.shape)}")
    if pool <= 0:
        raise ValueError(f"`pool` must be positive, got {pool}")

    n, c, t, h, w = latents.shape
    x = latents.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w).float()
    if pool == 1:
        x = x.mean(dim=(2, 3), keepdim=True)
    else:
        x = F.adaptive_avg_pool2d(x, (pool, pool))
    return x.flatten(1).reshape(n, t, -1)


def build_temporal_descriptors(
    latents: torch.Tensor,
    strides: Iterable[int],
    pool: int = 2,
    family: str = "transition",
    include_current: bool = True,
    include_future: bool = True,
    include_delta: bool = True,
    normalize: bool = True,
) -> list[DescriptorBatch]:
    """Build multi-stride descriptors from Wan-VAE latent windows.

    Each descriptor is aligned to a current latent time index `t` and a future
    time index `t + stride`. The default vector is `[f_t, f_{t+s}, f_{t+s}-f_t]`.
    """

    pooled = _pool_latents(latents, pool=pool)
    n, t, _ = pooled.shape
    batches: list[DescriptorBatch] = []

    for stride in [int(s) for s in strides]:
        if stride <= 0:
            raise ValueError(f"Stride must be positive, got {stride}")
        if t <= stride:
            continue

        current = pooled[:, :-stride]
        future = pooled[:, stride:]
        parts = []
        if include_current:
            parts.append(current)
        if include_future:
            parts.append(future)
        if include_delta:
            parts.append(future - current)
        if not parts:
            raise ValueError("At least one descriptor component must be enabled.")

        vectors = torch.cat(parts, dim=-1).reshape(n * (t - stride), -1)
        if normalize:
            mean = vectors.mean(dim=0, keepdim=True)
            std = vectors.std(dim=0, keepdim=True).clamp_min(1e-6)
            vectors = (vectors - mean) / std

        sample_index = torch.arange(n).repeat_interleave(t - stride)
        time_index = torch.arange(t - stride).repeat(n)
        batches.append(
            DescriptorBatch(
                name=f"latent_stride{stride}_pool{pool}",
                family=family,
                stride=stride,
                vectors=vectors.contiguous(),
                sample_index=sample_index,
                time_index=time_index,
            )
        )

    if not batches:
        raise ValueError(
            f"No descriptors were built. Check latent temporal length T={t} and strides={list(strides)}."
        )
    return batches
