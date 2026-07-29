from __future__ import annotations

import torch
from torch import nn

from .blocks import QueryCrossAttentionBlock
from .contracts import CodeTokens, ContinuousState, StateInputs, WorldBelief


class WorldBeliefCore(nn.Module):
    """Task-free learned queries over continuous and discrete world evidence."""

    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        proprio_dim: int,
        action_dim: int,
        queries: int = 8,
        layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        if min(dim, heads, proprio_dim, action_dim, queries, layers) <= 0:
            raise ValueError("World-belief dimensions and layer counts must be positive.")
        if dim % heads:
            raise ValueError("World-belief dim must be divisible by heads.")
        self.dim = int(dim)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.queries = nn.Parameter(torch.randn(queries, dim) * (dim**-0.5))
        self.proprio_projection = nn.Linear(proprio_dim, dim)
        self.action_projection = nn.Linear(action_dim, dim)
        self.modality_embedding = nn.Parameter(torch.randn(4, dim) * (dim**-0.5))
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
        self.output_norm = nn.LayerNorm(dim)

    def forward(
        self,
        continuous: ContinuousState,
        state: StateInputs,
        codes: CodeTokens | None = None,
    ) -> WorldBelief:
        batch = state.batch_size
        if continuous.tokens.shape[0] != batch:
            raise ValueError("Continuous state and raw state must be batch aligned.")
        if continuous.tokens.shape[2] != self.dim:
            raise ValueError(
                f"Continuous-state width must be {self.dim}, "
                f"got {continuous.tokens.shape[2]}."
            )
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
        if codes is not None:
            if codes.tokens.shape[0] != batch or codes.tokens.shape[2] != self.dim:
                raise ValueError("Code tokens do not match belief batch/width.")
            parts.append(codes.tokens + self.modality_embedding[1])
            masks.append(codes.valid)

        proprio = self.proprio_projection(state.proprio_history)
        parts.append(proprio + self.modality_embedding[2])
        masks.append(
            state.proprio_valid
            if state.proprio_valid is not None
            else torch.ones(
                proprio.shape[:2],
                dtype=torch.bool,
                device=proprio.device,
            )
        )
        if state.past_actions.shape[1]:
            actions = self.action_projection(state.past_actions)
            parts.append(actions + self.modality_embedding[3])
            masks.append(
                state.past_action_valid
                if state.past_action_valid is not None
                else torch.ones(
                    actions.shape[:2],
                    dtype=torch.bool,
                    device=actions.device,
                )
            )

        context = torch.cat(parts, dim=1)
        context_valid = torch.cat(masks, dim=1)
        if (
            context_valid.device.type == "cpu"
            and (~context_valid).all(dim=1).any()
        ):
            raise ValueError("Every sample needs at least one valid world measurement.")
        belief = self.queries[None].expand(batch, -1, -1)
        for block in self.blocks:
            belief = block(
                belief,
                context,
                context_valid=context_valid,
            )
        return WorldBelief(tokens=self.output_norm(belief))
