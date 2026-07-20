from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class KMeansResult:
    centers: torch.Tensor
    codes: torch.Tensor
    distances: torch.Tensor
    inertia: float


@dataclass
class RQResult:
    centers: list[torch.Tensor]
    codes: torch.Tensor
    quantized: torch.Tensor
    residual_norms: list[float]


def squared_l2(x: torch.Tensor, centers: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
    outs = []
    centers_t = centers.t()
    center_norm = centers.square().sum(dim=1).unsqueeze(0)
    for start in range(0, x.shape[0], chunk_size):
        xb = x[start : start + chunk_size]
        dist = xb.square().sum(dim=1, keepdim=True) - 2.0 * xb @ centers_t + center_norm
        outs.append(dist)
    return torch.cat(outs, dim=0)


def assign_codes(x: torch.Tensor, centers: torch.Tensor, chunk_size: int = 8192) -> tuple[torch.Tensor, torch.Tensor]:
    codes_out = []
    distances_out = []
    centers_t = centers.t()
    center_norm = centers.square().sum(dim=1).unsqueeze(0)
    for start in range(0, x.shape[0], chunk_size):
        xb = x[start : start + chunk_size]
        dist = xb.square().sum(dim=1, keepdim=True) - 2.0 * xb @ centers_t + center_norm
        distances, codes = dist.min(dim=1)
        codes_out.append(codes)
        distances_out.append(distances)
    return torch.cat(codes_out, dim=0), torch.cat(distances_out, dim=0)


def _init_centers(x: torch.Tensor, k: int, generator: torch.Generator) -> torch.Tensor:
    if x.shape[0] < k:
        raise ValueError(f"Need at least K samples for KMeans, got N={x.shape[0]}, K={k}")
    perm = torch.randperm(x.shape[0], generator=generator, device=x.device)[:k]
    return x[perm].clone()


def kmeans(
    x: torch.Tensor,
    k: int,
    iters: int = 50,
    seed: int = 0,
    chunk_size: int = 8192,
    tol: float = 1e-5,
) -> KMeansResult:
    """Small dependency-free Lloyd KMeans for offline codebook probes."""

    if x.ndim != 2:
        raise ValueError(f"`x` must be [N,D], got {tuple(x.shape)}")
    if k <= 1:
        raise ValueError(f"`k` must be > 1, got {k}")

    x = x.float().contiguous()
    generator = torch.Generator(device=x.device).manual_seed(int(seed))
    centers = _init_centers(x, k=k, generator=generator)
    prev_inertia: float | None = None

    for _ in range(int(iters)):
        codes, distances = assign_codes(x, centers, chunk_size=chunk_size)
        counts = torch.bincount(codes, minlength=k).to(dtype=x.dtype).clamp_min(1.0)
        new_centers = torch.zeros_like(centers)
        new_centers.index_add_(0, codes, x)
        new_centers = new_centers / counts.unsqueeze(1)

        empty = torch.bincount(codes, minlength=k) == 0
        if empty.any():
            refill = torch.randperm(x.shape[0], generator=generator, device=x.device)[: int(empty.sum())]
            new_centers[empty] = x[refill]

        centers = new_centers
        inertia = float(distances.mean().item())
        if prev_inertia is not None:
            improvement = abs(prev_inertia - inertia) / max(abs(prev_inertia), 1e-12)
            if improvement < tol:
                break
        prev_inertia = inertia

    codes, distances = assign_codes(x, centers, chunk_size=chunk_size)
    return KMeansResult(
        centers=centers.detach(),
        codes=codes.detach(),
        distances=distances.detach(),
        inertia=float(distances.mean().item()),
    )


def train_residual_quantizer(
    x: torch.Tensor,
    k: int,
    levels: int = 3,
    iters: int = 50,
    seed: int = 0,
    chunk_size: int = 8192,
) -> RQResult:
    if levels <= 0:
        raise ValueError(f"`levels` must be positive, got {levels}")

    residual = x.float().contiguous()
    quantized = torch.zeros_like(residual)
    all_centers: list[torch.Tensor] = []
    all_codes: list[torch.Tensor] = []
    residual_norms = [float(residual.square().mean().item())]

    for level in range(levels):
        km = kmeans(
            residual,
            k=k,
            iters=iters,
            seed=int(seed) + level * 9973,
            chunk_size=chunk_size,
        )
        centers = km.centers.to(device=residual.device, dtype=residual.dtype)
        codes = km.codes.to(device=residual.device)
        q = centers[codes]
        quantized = quantized + q
        residual = residual - q
        all_centers.append(km.centers.detach())
        all_codes.append(km.codes.detach())
        residual_norms.append(float(residual.square().mean().item()))

    return RQResult(
        centers=all_centers,
        codes=torch.stack(all_codes, dim=1),
        quantized=quantized.detach(),
        residual_norms=residual_norms,
    )


def pack_codebook(result: KMeansResult | RQResult, meta: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"meta": dict(meta)}
    if isinstance(result, KMeansResult):
        payload.update(
            {
                "type": "kmeans",
                "centers": result.centers.detach().cpu(),
                "codes": result.codes.detach().cpu(),
            }
        )
    else:
        payload.update(
            {
                "type": "rq",
                "centers": [center.detach().cpu() for center in result.centers],
                "codes": result.codes.detach().cpu(),
            }
        )
    return payload
