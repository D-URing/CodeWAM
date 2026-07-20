from __future__ import annotations

import math
from typing import Any

import torch

from .clustering import RQResult, assign_codes
from .descriptors import DescriptorBatch


def usage_metrics(codes: torch.Tensor, k: int) -> dict[str, float]:
    codes = codes.reshape(-1).cpu()
    counts = torch.bincount(codes, minlength=k).float()
    probs = counts / counts.sum().clamp_min(1.0)
    nonzero = probs[probs > 0]
    entropy = float(-(nonzero * nonzero.log()).sum().item())
    perplexity = float(math.exp(entropy))
    return {
        "used": int((counts > 0).sum().item()),
        "usage": float((counts > 0).float().mean().item()),
        "dead": int((counts == 0).sum().item()),
        "dead_frac": float((counts == 0).float().mean().item()),
        "perplexity": perplexity,
        "perplexity_frac": perplexity / float(k),
        "max_count_frac": float(counts.max().item() / counts.sum().clamp_min(1.0).item()),
    }


def reconstruction_metrics(x: torch.Tensor, quantized: torch.Tensor) -> dict[str, float]:
    x = x.float().cpu()
    quantized = quantized.float().cpu()
    err = (x - quantized).square().mean()
    baseline = x.square().mean().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(x, quantized, dim=1).mean()
    return {
        "mse": float(err.item()),
        "relative_mse": float((err / baseline).item()),
        "r2_like": float((1.0 - err / baseline).item()),
        "mean_cosine": float(cosine.item()),
    }


def temporal_metrics(batch: DescriptorBatch, codes: torch.Tensor) -> dict[str, float]:
    """Measure local stability without assuming episode-level global ids."""

    if codes.ndim == 2:
        level_codes = codes[:, 0]
    else:
        level_codes = codes
    sample = batch.sample_index.cpu()
    time = batch.time_index.cpu()
    order = torch.argsort(sample * (int(time.max().item()) + 1) + time)
    s = sample[order]
    t = time[order]
    c = level_codes.cpu()[order]
    same_sample_next = (s[1:] == s[:-1]) & (t[1:] == t[:-1] + 1)
    if not same_sample_next.any():
        return {"same_next_frac": float("nan"), "change_next_frac": float("nan")}
    same_code = c[1:][same_sample_next] == c[:-1][same_sample_next]
    same_frac = float(same_code.float().mean().item())
    return {"same_next_frac": same_frac, "change_next_frac": 1.0 - same_frac}


def action_relevance_metrics(
    batch: DescriptorBatch,
    codes: torch.Tensor,
    actions: torch.Tensor | None,
    latent_t: int,
    k: int,
) -> dict[str, float]:
    """Estimate how much action variation is separated by the first-level code.

    This is an in-sample diagnostic, not a policy score. It asks whether windows
    sharing the same visual code also share a similar action segment.
    """

    if actions is None:
        return {}
    if actions.ndim != 3:
        raise ValueError(f"`actions` must be [N,T,D], got {tuple(actions.shape)}")
    if latent_t <= 1:
        return {}

    labels = codes[:, 0] if codes.ndim == 2 else codes
    labels = labels.cpu().long()
    actions = actions.float().cpu()
    action_t = actions.shape[1]
    targets = []
    for sample_idx, time_idx in zip(batch.sample_index.tolist(), batch.time_index.tolist()):
        start = int(round(float(time_idx) / float(latent_t - 1) * action_t))
        end = int(round(float(time_idx + batch.stride) / float(latent_t - 1) * action_t))
        start = max(0, min(start, action_t - 1))
        end = max(start + 1, min(end, action_t))
        targets.append(actions[int(sample_idx), start:end].mean(dim=0))
    y = torch.stack(targets, dim=0)

    total_var = y.var(dim=0, unbiased=False).mean().clamp_min(1e-12)
    centroids = torch.zeros(k, y.shape[1])
    counts = torch.bincount(labels, minlength=k).float().clamp_min(1.0)
    centroids.index_add_(0, labels, y)
    centroids = centroids / counts.unsqueeze(1)
    pred = centroids[labels]
    mse = (y - pred).square().mean()
    return {
        "action_code_r2_in_sample": float((1.0 - mse / total_var).item()),
        "action_code_mse_in_sample": float(mse.item()),
    }


def rq_metrics(batch: DescriptorBatch, result: RQResult, k: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "descriptor": batch.name,
        "stride": batch.stride,
        "n_vectors": int(batch.vectors.shape[0]),
        "dim": int(batch.vectors.shape[1]),
        "levels": int(result.codes.shape[1]),
        "k": int(k),
        "residual_norms": result.residual_norms,
    }
    out.update(reconstruction_metrics(batch.vectors, result.quantized))
    out.update({f"temporal_{key}": value for key, value in temporal_metrics(batch, result.codes).items()})
    for level in range(result.codes.shape[1]):
        prefix = f"level{level + 1}"
        out.update({f"{prefix}_{key}": value for key, value in usage_metrics(result.codes[:, level], k).items()})
    return out


def kmeans_metrics(batch: DescriptorBatch, centers: torch.Tensor, codes: torch.Tensor, k: int) -> dict[str, Any]:
    centers = centers.float()
    codes_device, _ = assign_codes(batch.vectors.float(), centers.to(batch.vectors.device))
    quantized = centers[codes_device.cpu()]
    out: dict[str, Any] = {
        "descriptor": batch.name,
        "stride": batch.stride,
        "n_vectors": int(batch.vectors.shape[0]),
        "dim": int(batch.vectors.shape[1]),
        "levels": 1,
        "k": int(k),
    }
    out.update(reconstruction_metrics(batch.vectors, quantized))
    out.update({f"temporal_{key}": value for key, value in temporal_metrics(batch, codes).items()})
    out.update({f"level1_{key}": value for key, value in usage_metrics(codes, k).items()})
    return out
