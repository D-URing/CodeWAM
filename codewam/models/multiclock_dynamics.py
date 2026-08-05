from __future__ import annotations

from math import prod

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import sinusoidal_embedding
from .code_dynamics import (
    DynamicsMode,
    FutureCodePrediction,
    encode_prefix_ids,
)
from .contracts import (
    ActionBatch,
    FutureCodeTargets,
    MultiClockCodeState,
    TransitionSchedule,
    WorldBelief,
)


class _ResidualMLP(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.net(values)


class MultiClockTransitionModel(nn.Module):
    """Order-aware action-conditioned transition without future-token attention."""

    def __init__(
        self,
        *,
        dim: int,
        action_dim: int,
        max_horizon: int,
        families: tuple[str, ...],
        codebook_sizes: tuple[tuple[int, ...], ...],
        mode: DynamicsMode = "prefix",
        layers: int = 2,
        action_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if min(dim, action_dim, max_horizon, layers, action_layers) <= 0:
            raise ValueError("Transition dimensions and layers must be positive.")
        if mode not in {"independent", "prefix"}:
            raise ValueError(f"Unsupported transition mode `{mode}`.")
        if len(families) != len(codebook_sizes) or not families:
            raise ValueError("Transition families and vocabularies must align.")
        depths = {len(sizes) for sizes in codebook_sizes}
        if len(depths) != 1 or next(iter(depths)) <= 0:
            raise ValueError("Transition families need one common RQ depth.")
        if any(size <= 1 for sizes in codebook_sizes for size in sizes):
            raise ValueError("Every transition vocabulary needs at least two IDs.")
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.max_horizon = int(max_horizon)
        self.families = tuple(families)
        self.codebook_sizes = tuple(
            tuple(int(size) for size in sizes) for sizes in codebook_sizes
        )
        self.levels = next(iter(depths))
        self.mode: DynamicsMode = mode
        self.action_projection = nn.Linear(action_dim, dim)
        self.action_encoder = nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=action_layers,
            batch_first=True,
            dropout=dropout if action_layers > 1 else 0.0,
        )
        self.no_action = nn.Parameter(torch.randn(dim) * (dim**-0.5))
        self.delta_time_projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.family_embedding = nn.Parameter(
            torch.randn(len(families), dim) * (dim**-0.5)
        )
        self.input_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(
            [_ResidualMLP(dim, dropout) for _ in range(layers)]
        )
        self.output_norm = nn.LayerNorm(dim)
        if mode == "independent":
            output_sizes = [
                size for family_sizes in self.codebook_sizes for size in family_sizes
            ]
        else:
            output_sizes = [prod(sizes) for sizes in self.codebook_sizes]
        self.heads = nn.ModuleList([nn.Linear(dim, size) for size in output_sizes])

    def _action_prefix_states(
        self,
        actions: ActionBatch,
        schedule: TransitionSchedule,
    ) -> torch.Tensor:
        batch, horizon, action_dim = actions.values.shape
        if action_dim != self.action_dim:
            raise ValueError(
                f"Expected transition action dim {self.action_dim}, got {action_dim}."
            )
        if horizon > self.max_horizon:
            raise ValueError(
                f"Transition action horizon {horizon} exceeds {self.max_horizon}."
            )
        if schedule.batch_size != batch or schedule.families != len(self.families):
            raise ValueError("Transition schedule does not match actions/families.")
        lengths = schedule.action_prefix_lengths.to(device=actions.values.device)
        if lengths.device.type == "cpu" and (lengths > horizon).any():
            raise ValueError("Transition action prefix exceeds the action chunk.")
        lengths = lengths.clamp(min=0, max=horizon)
        action_valid = (
            actions.valid
            if actions.valid is not None
            else torch.ones(
                (batch, horizon),
                dtype=torch.bool,
                device=actions.values.device,
            )
        )
        contiguous = action_valid.long().cumprod(dim=1).sum(dim=1)
        if lengths.device.type == "cpu" and (lengths > contiguous[:, None]).any():
            raise ValueError("Transition action prefix contains invalid action steps.")

        projected = self.action_projection(actions.values)
        projected = projected * action_valid[:, :, None].to(projected.dtype)
        encoded, _ = self.action_encoder(projected)
        prefix_states = torch.cat(
            (
                self.no_action[None, None].expand(batch, 1, -1),
                encoded,
            ),
            dim=1,
        )
        gather = lengths[:, :, None].expand(-1, -1, self.dim)
        return prefix_states.gather(1, gather)

    def forward(
        self,
        belief: WorldBelief,
        codes: MultiClockCodeState,
        actions: ActionBatch,
        schedule: TransitionSchedule,
    ) -> FutureCodePrediction:
        batch = belief.tokens.shape[0]
        expected_codes = (batch, len(self.families), self.dim)
        if belief.tokens.shape[2] != self.dim:
            raise ValueError("World belief does not match transition width.")
        if codes.tokens.shape != expected_codes or codes.families != self.families:
            raise ValueError("Current multi-clock state does not match transition.")
        action_state = self._action_prefix_states(actions, schedule)
        delta_times = schedule.delta_times.to(
            device=belief.tokens.device,
            dtype=belief.tokens.dtype,
        )
        delta_state = self.delta_time_projection(
            sinusoidal_embedding(delta_times.reshape(-1), self.dim)
        ).reshape(batch, len(self.families), self.dim)
        world = belief.tokens.mean(dim=1)[:, None]
        current_codes = codes.tokens * codes.valid[:, :, None].to(codes.tokens.dtype)
        values = self.input_norm(
            world
            + current_codes
            + action_state
            + delta_state
            + self.family_embedding[None]
        )
        for block in self.blocks:
            values = block(values)
        values = self.output_norm(values)

        if self.mode == "independent":
            logits = []
            head = 0
            for family in range(len(self.families)):
                for _ in range(self.levels):
                    logits.append(self.heads[head](values[:, family]))
                    head += 1
        else:
            logits = [
                head(values[:, family])
                for family, head in enumerate(self.heads)
            ]
        return FutureCodePrediction(
            mode=self.mode,
            logits=tuple(logits),
            families=self.families,
            codebook_sizes=self.codebook_sizes,
        )

    def loss(
        self,
        prediction: FutureCodePrediction,
        targets: FutureCodeTargets,
        *,
        sample_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, families, levels = targets.code_ids.shape
        if (
            prediction.mode != self.mode
            or prediction.families != self.families
            or prediction.codebook_sizes != self.codebook_sizes
        ):
            raise ValueError("Future-code prediction does not belong to this model.")
        if (families, levels) != (len(self.families), self.levels):
            raise ValueError("Future-code targets do not match transition layout.")
        if sample_valid.dtype != torch.bool or sample_valid.shape != (batch,):
            raise ValueError("Transition supervision must be bool [B].")

        losses: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        if self.mode == "independent":
            head = 0
            for family in range(families):
                valid = targets.available[:, family] & sample_valid
                for level in range(levels):
                    target = torch.where(
                        valid,
                        targets.code_ids[:, family, level],
                        torch.zeros_like(targets.code_ids[:, family, level]),
                    )
                    losses.append(
                        F.cross_entropy(
                            prediction.logits[head],
                            target,
                            reduction="none",
                        )
                    )
                    masks.append(valid)
                    head += 1
        else:
            for family, sizes in enumerate(self.codebook_sizes):
                valid = targets.available[:, family] & sample_valid
                target = torch.where(
                    valid[:, None],
                    targets.code_ids[:, family],
                    torch.zeros_like(targets.code_ids[:, family]),
                )
                losses.append(
                    F.cross_entropy(
                        prediction.logits[family],
                        encode_prefix_ids(target, sizes),
                        reduction="none",
                    )
                    / float(self.levels)
                )
                masks.append(valid)
        stacked_loss = torch.stack(losses, dim=1)
        weights = torch.stack(masks, dim=1).to(stacked_loss.dtype)
        return (stacked_loss * weights).sum() / weights.sum().clamp_min(1.0)
