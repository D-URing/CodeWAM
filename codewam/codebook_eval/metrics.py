from __future__ import annotations

import math
from typing import Any

import torch

from .clustering import RQResult
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


def empirical_label_metrics(labels: torch.Tensor, prefix: str) -> dict[str, float]:
    labels = labels.reshape(-1).cpu().long()
    if labels.numel() == 0:
        return {
            f"{prefix}_unique": 0,
            f"{prefix}_unique_frac": float("nan"),
            f"{prefix}_perplexity": float("nan"),
            f"{prefix}_perplexity_frac": float("nan"),
            f"{prefix}_max_count_frac": float("nan"),
            f"{prefix}_entropy": float("nan"),
        }
    unique, counts = torch.unique(labels, return_counts=True)
    probs = counts.float() / counts.sum().float().clamp_min(1.0)
    entropy = float(-(probs * probs.log()).sum().item())
    perplexity = float(math.exp(entropy))
    return {
        f"{prefix}_unique": int(unique.numel()),
        f"{prefix}_unique_frac": float(unique.numel() / labels.numel()),
        f"{prefix}_perplexity": perplexity,
        f"{prefix}_perplexity_frac": float(perplexity / labels.numel()),
        f"{prefix}_max_count_frac": float(counts.max().item() / counts.sum().item()),
        f"{prefix}_entropy": entropy,
    }


def joint_codes(codes: torch.Tensor, k: int) -> torch.Tensor:
    codes = codes.cpu().long()
    if codes.ndim == 1:
        return codes
    bases = (int(k) ** torch.arange(codes.shape[1], dtype=torch.long)).view(1, -1)
    return (codes * bases).sum(dim=1)


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

    level_codes = codes[:, 0] if codes.ndim == 2 else codes
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
    out = {"same_next_frac": same_frac, "change_next_frac": 1.0 - same_frac}

    cur = c[:-1][same_sample_next]
    nxt = c[1:][same_sample_next]
    transition = torch.stack([cur, nxt], dim=1)
    _, transition_counts = torch.unique(transition, dim=0, return_counts=True)
    probs = transition_counts.float() / transition_counts.sum().float().clamp_min(1.0)
    entropy = float(-(probs * probs.log()).sum().item())
    out.update(
        {
            "transition_unique": int(transition_counts.numel()),
            "transition_entropy": entropy,
            "transition_perplexity": float(math.exp(entropy)),
        }
    )
    return out


def joint_temporal_metrics(batch: DescriptorBatch, codes: torch.Tensor, k: int) -> dict[str, float]:
    labels = joint_codes(codes, k=k)
    return {f"joint_{key}": value for key, value in temporal_metrics(batch, labels).items()}


def action_targets(batch: DescriptorBatch, actions: torch.Tensor | None, latent_t: int) -> torch.Tensor | None:
    if actions is None:
        return None
    if actions.ndim != 3:
        raise ValueError(f"`actions` must be [N,T,D], got {tuple(actions.shape)}")
    if latent_t <= 1:
        return None

    actions = actions.float().cpu()
    action_t = actions.shape[1]
    targets = []
    for sample_idx, time_idx in zip(batch.sample_index.tolist(), batch.time_index.tolist()):
        start = int(round(float(time_idx) / float(latent_t - 1) * action_t))
        end = int(round(float(time_idx + batch.stride) / float(latent_t - 1) * action_t))
        start = max(0, min(start, action_t - 1))
        end = max(start + 1, min(end, action_t))
        targets.append(actions[int(sample_idx), start:end].mean(dim=0))
    return torch.stack(targets, dim=0)


def grouped_target_r2(labels: torch.Tensor, targets: torch.Tensor, prefix: str) -> dict[str, float]:
    labels = labels.cpu().long().reshape(-1)
    y = targets.float().cpu()
    total_var = y.var(dim=0, unbiased=False).mean().clamp_min(1e-12)

    unique, inverse = torch.unique(labels, return_inverse=True)
    centroids = torch.zeros(unique.numel(), y.shape[1])
    counts = torch.bincount(inverse, minlength=unique.numel()).float().clamp_min(1.0)
    centroids.index_add_(0, inverse, y)
    centroids = centroids / counts.unsqueeze(1)
    pred = centroids[inverse]
    mse = (y - pred).square().mean()
    return {
        f"{prefix}_r2_in_sample": float((1.0 - mse / total_var).item()),
        f"{prefix}_mse_in_sample": float(mse.item()),
        f"{prefix}_groups": int(unique.numel()),
    }


def action_relevance_metrics(
    batch: DescriptorBatch,
    codes: torch.Tensor,
    actions: torch.Tensor | None,
    latent_t: int,
    k: int,
) -> dict[str, float]:
    """Estimate how much action variation is separated by the code.

    This is an in-sample diagnostic, not a policy score. It asks whether windows
    sharing the same visual code also share a similar action segment.
    """

    targets = action_targets(batch, actions=actions, latent_t=latent_t)
    if targets is None:
        return {}
    out: dict[str, float] = {}
    if codes.ndim == 2:
        for level in range(codes.shape[1]):
            out.update(grouped_target_r2(codes[:, level], targets, prefix=f"action_level{level + 1}"))
        out.update(grouped_target_r2(joint_codes(codes, k=k), targets, prefix="action_joint"))
        # Backward-compatible summary columns.
        out["action_code_r2_in_sample"] = out["action_joint_r2_in_sample"]
        out["action_code_mse_in_sample"] = out["action_joint_mse_in_sample"]
    else:
        out.update(grouped_target_r2(codes, targets, prefix="action_code"))
    return out


def rq_metrics(batch: DescriptorBatch, result: RQResult, k: int) -> dict[str, Any]:
    residual_norms = [float(v) for v in result.residual_norms]
    out: dict[str, Any] = {
        "descriptor": batch.name,
        "stride": batch.stride,
        "n_vectors": int(batch.vectors.shape[0]),
        "dim": int(batch.vectors.shape[1]),
        "levels": int(result.codes.shape[1]),
        "k": int(k),
        "residual_norms": residual_norms,
    }
    out.update(reconstruction_metrics(batch.vectors, result.quantized))
    out.update({f"temporal_{key}": value for key, value in temporal_metrics(batch, result.codes).items()})
    out.update(joint_temporal_metrics(batch, result.codes, k=k))
    out.update(empirical_label_metrics(joint_codes(result.codes, k=k), prefix="joint_code"))
    if residual_norms:
        for level in range(1, len(residual_norms)):
            prev = max(residual_norms[level - 1], 1e-12)
            out[f"residual_reduction_l{level}"] = 1.0 - residual_norms[level] / prev
        out["residual_total_reduction"] = 1.0 - residual_norms[-1] / max(residual_norms[0], 1e-12)
    for level in range(result.codes.shape[1]):
        prefix = f"level{level + 1}"
        out.update({f"{prefix}_{key}": value for key, value in usage_metrics(result.codes[:, level], k).items()})
    return out


def mutual_information_metrics(labels_a: torch.Tensor, labels_b: torch.Tensor, prefix: str) -> dict[str, float]:
    a = labels_a.cpu().long().reshape(-1)
    b = labels_b.cpu().long().reshape(-1)
    if a.numel() != b.numel():
        raise ValueError(f"Label lengths must match, got {a.numel()} and {b.numel()}")
    if a.numel() == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mi": float("nan"),
            f"{prefix}_nmi_min": float("nan"),
            f"{prefix}_nmi_sqrt": float("nan"),
            f"{prefix}_unique_a": 0,
            f"{prefix}_unique_b": 0,
        }

    unique_a, inv_a = torch.unique(a, return_inverse=True)
    unique_b, inv_b = torch.unique(b, return_inverse=True)
    joint = inv_a * unique_b.numel() + inv_b
    _, counts_ab = torch.unique(joint, return_counts=True)
    counts_a = torch.bincount(inv_a, minlength=unique_a.numel()).float()
    counts_b = torch.bincount(inv_b, minlength=unique_b.numel()).float()
    n = float(a.numel())
    pa = counts_a / n
    pb = counts_b / n
    pab = counts_ab.float() / n
    ha = float(-(pa[pa > 0] * pa[pa > 0].log()).sum().item())
    hb = float(-(pb[pb > 0] * pb[pb > 0].log()).sum().item())
    hab = float(-(pab[pab > 0] * pab[pab > 0].log()).sum().item())
    mi = ha + hb - hab
    return {
        f"{prefix}_n": int(a.numel()),
        f"{prefix}_mi": float(mi),
        f"{prefix}_nmi_min": float(mi / max(min(ha, hb), 1e-12)),
        f"{prefix}_nmi_sqrt": float(mi / max(math.sqrt(max(ha, 0.0) * max(hb, 0.0)), 1e-12)),
        f"{prefix}_unique_a": int(unique_a.numel()),
        f"{prefix}_unique_b": int(unique_b.numel()),
    }


def kmeans_metrics(batch: DescriptorBatch, centers: torch.Tensor, codes: torch.Tensor, k: int) -> dict[str, Any]:
    centers = centers.float()
    quantized = centers[codes.to(device=centers.device)].cpu()
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
