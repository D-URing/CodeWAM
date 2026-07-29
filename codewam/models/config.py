from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from .code_dynamics import DynamicsMode


Variant = Literal["C0", "C1", "C2"]


@dataclass(frozen=True)
class CodeWAMConfig:
    variant: Variant = "C2"
    latent_channels: int = 48
    proprio_dim: int = 14
    action_dim: int = 7
    language_dim: int = 4096
    dim: int = 512
    heads: int = 8
    patch_size: int = 2
    max_time: int = 32
    max_cameras: int = 8
    max_spatial_tokens: int = 1024
    max_action_horizon: int = 32
    state_spatial_layers: int = 2
    state_temporal_layers: int = 2
    belief_queries: int = 8
    belief_layers: int = 3
    action_layers: int = 4
    dynamics_layers: int = 3
    dynamics_action_layers: int = 1
    dynamics_mode: DynamicsMode = "independent"
    dropout: float = 0.0
    lambda_code: float = 1.0

    def __post_init__(self) -> None:
        if self.variant not in {"C0", "C1", "C2"}:
            raise ValueError(f"Unsupported CodeWAM variant `{self.variant}`.")
        if self.dynamics_mode not in {"independent", "prefix"}:
            raise ValueError(
                f"Unsupported dynamics mode `{self.dynamics_mode}`."
            )
        dimensions = (
            self.latent_channels,
            self.proprio_dim,
            self.action_dim,
            self.language_dim,
            self.dim,
            self.heads,
            self.patch_size,
            self.max_time,
            self.max_cameras,
            self.max_spatial_tokens,
            self.max_action_horizon,
            self.state_spatial_layers,
            self.state_temporal_layers,
            self.belief_queries,
            self.belief_layers,
            self.action_layers,
            self.dynamics_layers,
            self.dynamics_action_layers,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("All CodeWAM dimensions and layer counts must be positive.")
        if self.dim % self.heads:
            raise ValueError("CodeWAM dim must be divisible by heads.")
        if not isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("`dropout` must be finite and in [0,1).")
        if not isfinite(self.lambda_code) or self.lambda_code < 0:
            raise ValueError("`lambda_code` must be finite and non-negative.")
