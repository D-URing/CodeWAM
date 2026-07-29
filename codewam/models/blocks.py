from __future__ import annotations

import math

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim: int, multiplier: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = max(dim, int(round(dim * multiplier)))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class QueryCrossAttentionBlock(nn.Module):
    """Pre-norm query self-attention followed by read-only cross-attention."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.0,
        ffn_multiplier: float = 4.0,
    ):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, multiplier=ffn_multiplier, dropout=dropout)

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        *,
        query_valid: torch.Tensor | None = None,
        context_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_padding = self._safe_padding_mask(query_valid)
        context_padding = self._safe_padding_mask(context_valid)
        normalized = self.self_norm(queries)
        update, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=query_padding,
            need_weights=False,
        )
        queries = queries + update
        update, _ = self.cross_attention(
            self.cross_norm(queries),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=context_padding,
            need_weights=False,
        )
        queries = queries + update
        return queries + self.ffn(self.ffn_norm(queries))

    @staticmethod
    def _safe_padding_mask(valid: torch.Tensor | None) -> torch.Tensor | None:
        if valid is None:
            return None
        if valid.dtype != torch.bool or valid.ndim != 2:
            raise ValueError("Attention validity must be bool [B,N].")
        if valid.shape[1] == 0:
            raise ValueError("Attention sequences must contain at least one token.")
        padding = ~valid
        all_missing = padding.all(dim=1)
        padding = padding.clone()
        padding[all_missing, 0] = False
        return padding


def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError(f"Sinusoidal input must be [B], got {tuple(values.shape)}.")
    half = dim // 2
    if half == 0:
        return values[:, None]
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=values.device, dtype=values.dtype)
        / max(half - 1, 1)
    )
    angles = values[:, None] * frequencies[None]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if embedding.shape[1] < dim:
        embedding = torch.nn.functional.pad(embedding, (0, dim - embedding.shape[1]))
    return embedding


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape[: mask.ndim] != mask.shape:
        raise ValueError(
            f"Mask {tuple(mask.shape)} is not a prefix of {tuple(values.shape)}."
        )
    weights = mask.to(dtype=values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    numerator = (values * weights).sum()
    denominator = weights.expand_as(values).sum().clamp_min(1.0)
    return numerator / denominator
