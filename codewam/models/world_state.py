from __future__ import annotations

import torch
from torch import nn

from .blocks import QueryCrossAttentionBlock, sinusoidal_embedding
from .contracts import (
    ContinuousState,
    MultiClockCodeState,
    StateInputs,
    StructuredWorldState,
    WorldBelief,
)


class StructuredWorldBuilder(nn.Module):
    """Builds task-free global state, then applies a gated code correction."""

    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        proprio_dim: int,
        action_dim: int,
        families: tuple[str, ...],
        queries: int = 8,
        layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if min(dim, heads, proprio_dim, action_dim, queries, layers) <= 0:
            raise ValueError("World-builder dimensions and layers must be positive.")
        if dim % heads:
            raise ValueError("World-builder dim must be divisible by heads.")
        if not families or len(set(families)) != len(families):
            raise ValueError(
                "World-builder clock families must be nonempty and unique."
            )
        self.dim = int(dim)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.families = tuple(families)
        self.queries = nn.Parameter(torch.randn(queries, dim) * (dim**-0.5))
        self.proprio_projection = nn.Linear(proprio_dim, dim)
        self.action_projection = nn.Linear(action_dim, dim)
        self.time_projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.modality_embedding = nn.Parameter(torch.randn(3, dim) * (dim**-0.5))
        self.blocks = nn.ModuleList(
            [
                QueryCrossAttentionBlock(
                    dim=dim,
                    heads=heads,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )
        self.code_fusion = nn.Sequential(
            nn.LayerNorm(len(families) * dim),
            nn.Linear(len(families) * dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.code_update = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        # C1/C2 begin from the same global state as C0; training must earn the update.
        self.code_gate = nn.Parameter(torch.zeros(queries, 1))
        self.output_norm = nn.LayerNorm(dim)

    @staticmethod
    def _valid_or_ones(
        valid: torch.Tensor | None,
        values: torch.Tensor,
    ) -> torch.Tensor:
        if valid is not None:
            return valid
        return torch.ones(
            values.shape[:2],
            dtype=torch.bool,
            device=values.device,
        )

    def _time_tokens(
        self,
        offsets: torch.Tensor | None,
        *,
        batch: int,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
        include_current: bool,
    ) -> torch.Tensor:
        if offsets is None:
            stop = 1 if include_current else 0
            values = torch.arange(
                stop - length,
                stop,
                device=device,
                dtype=dtype,
            )[None].expand(batch, -1)
        else:
            values = offsets.to(device=device, dtype=dtype)
        embedded = sinusoidal_embedding(values.reshape(-1), self.dim)
        return self.time_projection(embedded).reshape(batch, length, self.dim)

    def forward(
        self,
        continuous: ContinuousState,
        state: StateInputs,
        codes: MultiClockCodeState | None = None,
    ) -> StructuredWorldState:
        batch = state.batch_size
        if (
            continuous.tokens.shape[0] != batch
            or continuous.tokens.shape[2] != self.dim
        ):
            raise ValueError("Continuous detail does not match world-builder shape.")
        if state.proprio_history.shape[2] != self.proprio_dim:
            raise ValueError(
                f"Expected proprio dim {self.proprio_dim}, "
                f"got {state.proprio_history.shape[2]}."
            )
        if state.past_actions.shape[2] != self.action_dim:
            raise ValueError(
                f"Expected past-action dim {self.action_dim}, "
                f"got {state.past_actions.shape[2]}."
            )

        parts = [continuous.tokens + self.modality_embedding[0]]
        masks = [continuous.valid]
        proprio = self.proprio_projection(state.proprio_history)
        proprio = proprio + self._time_tokens(
            state.proprio_time_offsets,
            batch=batch,
            length=proprio.shape[1],
            device=proprio.device,
            dtype=proprio.dtype,
            include_current=True,
        )
        parts.append(proprio + self.modality_embedding[1])
        masks.append(self._valid_or_ones(state.proprio_valid, proprio))

        if state.past_actions.shape[1]:
            actions = self.action_projection(state.past_actions)
            actions = actions + self._time_tokens(
                state.past_action_time_offsets,
                batch=batch,
                length=actions.shape[1],
                device=actions.device,
                dtype=actions.dtype,
                include_current=False,
            )
            parts.append(actions + self.modality_embedding[2])
            masks.append(self._valid_or_ones(state.past_action_valid, actions))

        context = torch.cat(parts, dim=1)
        context_valid = torch.cat(masks, dim=1)
        if context_valid.device.type == "cpu" and (~context_valid).all(dim=1).any():
            raise ValueError("Every sample needs at least one valid world measurement.")
        belief = self.queries[None].expand(batch, -1, -1)
        for block in self.blocks:
            belief = block(belief, context, context_valid=context_valid)

        if codes is not None:
            if codes.families != self.families:
                raise ValueError("Code families do not match the world builder.")
            if codes.tokens.shape != (batch, len(self.families), self.dim):
                raise ValueError("Code state does not match world-builder shape.")
            masked_codes = codes.tokens * codes.valid[:, :, None].to(codes.tokens.dtype)
            summary = self.code_fusion(masked_codes.flatten(1, 2))
            update = self.code_update(belief + summary[:, None])
            has_code = codes.valid.any(dim=1)[:, None, None].to(update.dtype)
            belief = belief + has_code * torch.tanh(self.code_gate)[None] * update

        return StructuredWorldState(
            belief=WorldBelief(tokens=self.output_norm(belief)),
            continuous=continuous,
            codes=codes,
        )
