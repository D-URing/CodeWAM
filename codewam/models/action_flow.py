from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .blocks import QueryCrossAttentionBlock, masked_mean, sinusoidal_embedding
from .contracts import ActionBatch, PolicyCondition, StateInputs, WorldBelief


@dataclass(frozen=True)
class FlowMatchingOutput:
    loss: torch.Tensor
    velocity: torch.Tensor
    noised_actions: torch.Tensor
    target_velocity: torch.Tensor
    flow_time: torch.Tensor
    noise: torch.Tensor


class ActionFlowDecoder(nn.Module):
    """Continuous action-chunk flow model reading completed world belief."""

    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        action_dim: int,
        proprio_dim: int,
        language_dim: int,
        max_horizon: int,
        layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if min(
            dim,
            heads,
            action_dim,
            proprio_dim,
            language_dim,
            max_horizon,
            layers,
        ) <= 0:
            raise ValueError("Action-flow dimensions and layer counts must be positive.")
        if dim % heads:
            raise ValueError("Action-flow dim must be divisible by heads.")
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.language_dim = int(language_dim)
        self.max_horizon = int(max_horizon)
        self.action_projection = nn.Linear(action_dim, dim)
        self.language_projection = nn.Linear(language_dim, dim)
        self.proprio_projection = nn.Linear(proprio_dim, dim)
        self.flow_time_projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.position_embedding = nn.Parameter(
            torch.randn(max_horizon, dim) * (dim**-0.5)
        )
        self.context_identity = nn.Parameter(torch.randn(3, dim) * (dim**-0.5))
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
        self.output = nn.Linear(dim, action_dim)

    def _context(
        self,
        belief: WorldBelief,
        policy: PolicyCondition,
        current_proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = belief.tokens.shape[0]
        if policy.batch_size != batch or current_proprio.shape[0] != batch:
            raise ValueError("Action-flow contexts must be batch aligned.")
        if policy.language.shape[2] != self.language_dim:
            raise ValueError(
                f"Expected language dim {self.language_dim}, "
                f"got {policy.language.shape[2]}."
            )
        if current_proprio.shape != (batch, self.proprio_dim):
            raise ValueError(
                f"Current proprio must be {(batch, self.proprio_dim)}, "
                f"got {tuple(current_proprio.shape)}."
            )
        language = self.language_projection(policy.language)
        proprio = self.proprio_projection(current_proprio)[:, None]
        context = torch.cat(
            (
                belief.tokens + self.context_identity[0],
                language + self.context_identity[1],
                proprio + self.context_identity[2],
            ),
            dim=1,
        )
        language_valid = (
            policy.language_valid
            if policy.language_valid is not None
            else torch.ones(
                policy.language.shape[:2],
                dtype=torch.bool,
                device=policy.language.device,
            )
        )
        valid = torch.cat(
            (
                torch.ones(
                    belief.tokens.shape[:2],
                    dtype=torch.bool,
                    device=belief.tokens.device,
                ),
                language_valid,
                torch.ones((batch, 1), dtype=torch.bool, device=belief.tokens.device),
            ),
            dim=1,
        )
        return context, valid

    def velocity(
        self,
        noised_actions: torch.Tensor,
        flow_time: torch.Tensor,
        *,
        belief: WorldBelief,
        policy: PolicyCondition,
        current_proprio: torch.Tensor,
        action_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noised_actions.ndim != 3 or noised_actions.shape[2] != self.action_dim:
            raise ValueError(
                f"Noised actions must be [B,H,{self.action_dim}], "
                f"got {tuple(noised_actions.shape)}."
            )
        batch, horizon, _ = noised_actions.shape
        if horizon <= 0 or horizon > self.max_horizon:
            raise ValueError(
                f"Action horizon {horizon} is outside [1,{self.max_horizon}]."
            )
        if flow_time.shape != (batch,):
            raise ValueError(f"Flow time must be [B], got {tuple(flow_time.shape)}.")
        if action_valid is not None and (
            action_valid.dtype != torch.bool
            or action_valid.shape != (batch, horizon)
        ):
            raise ValueError("Action validity must be bool [B,H].")

        context, context_valid = self._context(
            belief,
            policy,
            current_proprio,
        )
        tokens = (
            self.action_projection(noised_actions)
            + self.position_embedding[:horizon][None]
            + self.flow_time_projection(
                sinusoidal_embedding(
                    flow_time.to(dtype=noised_actions.dtype),
                    self.dim,
                )
            )[:, None]
        )
        for block in self.blocks:
            tokens = block(
                tokens,
                context,
                query_valid=action_valid,
                context_valid=context_valid,
            )
        return self.output(self.output_norm(tokens))

    def flow_matching_loss(
        self,
        actions: ActionBatch,
        *,
        belief: WorldBelief,
        policy: PolicyCondition,
        state: StateInputs,
        sample_valid: torch.Tensor,
        noise: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> FlowMatchingOutput:
        target = actions.values
        batch, horizon, _ = target.shape
        if target.shape[2] != self.action_dim:
            raise ValueError(
                f"Expected action dim {self.action_dim}, got {target.shape[2]}."
            )
        if sample_valid.dtype != torch.bool or sample_valid.shape != (batch,):
            raise ValueError("Action supervision must be bool [B].")
        if noise is None:
            noise = torch.randn(
                target.shape,
                device=target.device,
                dtype=target.dtype,
                generator=generator,
            )
        if noise.shape != target.shape:
            raise ValueError("Action noise and target must have identical shapes.")
        if noise.device != target.device or noise.dtype != target.dtype:
            raise ValueError("Action noise must match target device and dtype.")
        if flow_time is None:
            flow_time = torch.rand(
                (batch,),
                device=target.device,
                dtype=target.dtype,
                generator=generator,
            )
        if flow_time.shape != (batch,):
            raise ValueError("Flow time must be [B].")
        if flow_time.device != target.device or flow_time.dtype != target.dtype:
            raise ValueError("Flow time must match target device and dtype.")
        interpolation = flow_time[:, None, None]
        noised = (1.0 - interpolation) * noise + interpolation * target
        target_velocity = target - noise
        velocity = self.velocity(
            noised,
            flow_time,
            belief=belief,
            policy=policy,
            current_proprio=state.current_proprio,
            action_valid=actions.valid,
        )
        valid = (
            actions.valid
            if actions.valid is not None
            else torch.ones(
                (batch, horizon),
                dtype=torch.bool,
                device=target.device,
            )
        )
        valid = valid & sample_valid[:, None]
        loss = masked_mean((velocity - target_velocity).square(), valid)
        return FlowMatchingOutput(
            loss=loss,
            velocity=velocity,
            noised_actions=noised,
            target_velocity=target_velocity,
            flow_time=flow_time,
            noise=noise,
        )

    @torch.no_grad()
    def sample(
        self,
        *,
        belief: WorldBelief,
        policy: PolicyCondition,
        current_proprio: torch.Tensor,
        horizon: int,
        steps: int = 10,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if horizon <= 0 or horizon > self.max_horizon or steps <= 0:
            raise ValueError("Action horizon/steps are outside configured limits.")
        batch = belief.tokens.shape[0]
        if initial_noise is None:
            values = torch.randn(
                (batch, horizon, self.action_dim),
                device=belief.tokens.device,
                dtype=belief.tokens.dtype,
                generator=generator,
            )
        else:
            expected = (batch, horizon, self.action_dim)
            if tuple(initial_noise.shape) != expected:
                raise ValueError(
                    f"Initial action noise must be {expected}, "
                    f"got {tuple(initial_noise.shape)}."
                )
            values = initial_noise.clone()
        step_size = 1.0 / float(steps)
        for step in range(steps):
            flow_time = torch.full(
                (batch,),
                step * step_size,
                device=values.device,
                dtype=values.dtype,
            )
            values = values + step_size * self.velocity(
                values,
                flow_time,
                belief=belief,
                policy=policy,
                current_proprio=current_proprio,
            )
        return values
