from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from codewam.codebook_eval.streaming import FrozenRQArtifact

from .action_flow import ActionFlowDecoder, FlowMatchingOutput
from .belief_core import WorldBeliefCore
from .code_dynamics import (
    CodeDynamicsDecoder,
    FutureCodePrediction,
)
from .config import CodeWAMConfig
from .contracts import (
    ActionBatch,
    CodeMeasurements,
    CodeWAMBatch,
    PolicyCondition,
    StateInputs,
    WorldBelief,
)
from .continuous_state import ContinuousStateEncoder
from .frozen_codebook import FrozenCodebookAdapter


@dataclass(frozen=True)
class CodeWAMLossOutput:
    total: torch.Tensor
    action: torch.Tensor
    code: torch.Tensor
    belief: WorldBelief
    flow: FlowMatchingOutput
    future: FutureCodePrediction | None


class CodeWAMV1(nn.Module):
    """Independent five-module CodeWAM with explicit C0/C1/C2 ablations."""

    def __init__(
        self,
        *,
        config: CodeWAMConfig,
        continuous_state: ContinuousStateEncoder,
        frozen_codebook: FrozenCodebookAdapter,
        belief_core: WorldBeliefCore,
        action_flow: ActionFlowDecoder,
        code_dynamics: CodeDynamicsDecoder,
    ):
        super().__init__()
        self.config = config
        self.continuous_state = continuous_state
        self.frozen_codebook = frozen_codebook
        self.belief_core = belief_core
        self.action_flow = action_flow
        self.code_dynamics = code_dynamics

    def build_belief(
        self,
        state: StateInputs,
        codes: CodeMeasurements | None,
    ) -> WorldBelief:
        continuous = self.continuous_state(state)
        code_tokens = None
        if self.config.variant in {"C1", "C2"}:
            if codes is None:
                raise ValueError(f"{self.config.variant} requires frozen code measurements.")
            code_tokens = self.frozen_codebook(codes)
        return self.belief_core(continuous, state, code_tokens)

    def policy_velocity(
        self,
        *,
        state: StateInputs,
        codes: CodeMeasurements | None,
        policy: PolicyCondition,
        noised_actions: torch.Tensor,
        flow_time: torch.Tensor,
        action_valid: torch.Tensor | None = None,
    ) -> tuple[WorldBelief, torch.Tensor]:
        belief = self.build_belief(state, codes)
        velocity = self.action_flow.velocity(
            noised_actions,
            flow_time,
            belief=belief,
            policy=policy,
            current_proprio=state.current_proprio,
            action_valid=action_valid,
        )
        return belief, velocity

    def compute_losses(
        self,
        batch: CodeWAMBatch,
        *,
        noise: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> CodeWAMLossOutput:
        belief = self.build_belief(batch.state, batch.codes)
        flow = self.action_flow.flow_matching_loss(
            batch.actions,
            belief=belief,
            policy=batch.policy,
            state=batch.state,
            sample_valid=batch.supervision.action,
            noise=noise,
            flow_time=flow_time,
            generator=generator,
        )
        future = None
        code_loss = flow.loss.new_zeros(())
        if self.config.variant == "C2":
            if batch.future_codes is None:
                raise ValueError("C2 requires future-code labels.")
            future = self.code_dynamics(belief, batch.actions)
            code_loss = self.code_dynamics.loss(
                future,
                batch.future_codes,
                sample_valid=batch.supervision.dynamics,
            )
        total = flow.loss + self.config.lambda_code * code_loss
        return CodeWAMLossOutput(
            total=total,
            action=flow.loss,
            code=code_loss,
            belief=belief,
            flow=flow,
            future=future,
        )

    def forward(
        self,
        batch: CodeWAMBatch,
        *,
        noise: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
    ) -> CodeWAMLossOutput:
        return self.compute_losses(batch, noise=noise, flow_time=flow_time)

    @torch.no_grad()
    def infer_actions(
        self,
        *,
        state: StateInputs,
        policy: PolicyCondition,
        codes: CodeMeasurements | None,
        horizon: int,
        steps: int = 10,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        belief = self.build_belief(state, codes)
        return self.action_flow.sample(
            belief=belief,
            policy=policy,
            current_proprio=state.current_proprio,
            horizon=horizon,
            steps=steps,
            initial_noise=initial_noise,
            generator=generator,
        )

    def predict_future_codes(
        self,
        *,
        state: StateInputs,
        codes: CodeMeasurements | None,
        candidate_actions: ActionBatch,
    ) -> FutureCodePrediction:
        belief = self.build_belief(state, codes)
        return self.code_dynamics(belief, candidate_actions)


def build_codewam_v1(
    config: CodeWAMConfig,
    artifact_sets: Mapping[str, Sequence[FrozenRQArtifact]],
) -> CodeWAMV1:
    adapter = FrozenCodebookAdapter(artifact_sets, dim=config.dim)
    continuous = ContinuousStateEncoder(
        latent_channels=config.latent_channels,
        dim=config.dim,
        heads=config.heads,
        patch_size=config.patch_size,
        spatial_layers=config.state_spatial_layers,
        temporal_layers=config.state_temporal_layers,
        max_time=config.max_time,
        max_cameras=config.max_cameras,
        max_spatial_tokens=config.max_spatial_tokens,
        dropout=config.dropout,
    )
    belief = WorldBeliefCore(
        dim=config.dim,
        heads=config.heads,
        proprio_dim=config.proprio_dim,
        action_dim=config.action_dim,
        queries=config.belief_queries,
        layers=config.belief_layers,
        dropout=config.dropout,
    )
    action = ActionFlowDecoder(
        dim=config.dim,
        heads=config.heads,
        action_dim=config.action_dim,
        proprio_dim=config.proprio_dim,
        language_dim=config.language_dim,
        max_horizon=config.max_action_horizon,
        layers=config.action_layers,
        dropout=config.dropout,
    )
    dynamics = CodeDynamicsDecoder(
        dim=config.dim,
        heads=config.heads,
        action_dim=config.action_dim,
        max_horizon=config.max_action_horizon,
        families=adapter.families,
        codebook_sizes=adapter.codebook_sizes,
        mode=config.dynamics_mode,
        layers=config.dynamics_layers,
        action_layers=config.dynamics_action_layers,
        dropout=config.dropout,
    )
    return CodeWAMV1(
        config=config,
        continuous_state=continuous,
        frozen_codebook=adapter,
        belief_core=belief,
        action_flow=action,
        code_dynamics=dynamics,
    )
