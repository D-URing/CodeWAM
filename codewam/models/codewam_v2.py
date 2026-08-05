from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from codewam.codebook_eval.streaming import FrozenRQArtifact

from .action_flow import ActionFlowDecoder
from .code_dynamics import FutureCodePrediction
from .codewam_v1 import CodeWAMLossOutput
from .config import CodeWAMConfig
from .contracts import (
    ActionBatch,
    CodeMeasurements,
    CodeWAMBatch,
    MultiClockCodeState,
    PolicyCondition,
    StateInputs,
    StructuredWorldState,
    TransitionSchedule,
)
from .continuous_state import ContinuousStateEncoder
from .frozen_codebook import FrozenCodebookAdapter
from .multiclock_dynamics import MultiClockTransitionModel
from .world_state import StructuredWorldBuilder


class CodeWAMV2(nn.Module):
    """Structured continuous/discrete world state with multi-clock dynamics."""

    def __init__(
        self,
        *,
        config: CodeWAMConfig,
        continuous_state: ContinuousStateEncoder,
        frozen_codebook: FrozenCodebookAdapter,
        world_builder: StructuredWorldBuilder,
        action_flow: ActionFlowDecoder,
        transition: MultiClockTransitionModel,
    ):
        super().__init__()
        if frozen_codebook.layout != "hierarchical":
            raise ValueError("CodeWAMV2 requires hierarchical RQ code state.")
        self.config = config
        self.continuous_state = continuous_state
        self.frozen_codebook = frozen_codebook
        self.world_builder = world_builder
        self.action_flow = action_flow
        self.transition = transition

    def build_world_state(
        self,
        state: StateInputs,
        codes: CodeMeasurements | None,
    ) -> StructuredWorldState:
        continuous = self.continuous_state(state)
        code_state = None
        if self.config.variant in {"C1", "C2"}:
            if codes is None:
                raise ValueError(
                    f"{self.config.variant} requires frozen code measurements."
                )
            projected = self.frozen_codebook(codes)
            if not isinstance(projected, MultiClockCodeState):
                raise TypeError("Hierarchical adapter returned an invalid code state.")
            code_state = projected
        return self.world_builder(continuous, state, code_state)

    def policy_velocity(
        self,
        *,
        state: StateInputs,
        codes: CodeMeasurements | None,
        policy: PolicyCondition,
        noised_actions: torch.Tensor,
        flow_time: torch.Tensor,
        action_valid: torch.Tensor | None = None,
    ) -> tuple[StructuredWorldState, torch.Tensor]:
        world = self.build_world_state(state, codes)
        velocity = self.action_flow.velocity(
            noised_actions,
            flow_time,
            belief=world.belief,
            policy=policy,
            current_proprio=state.current_proprio,
            action_valid=action_valid,
            continuous=world.continuous,
            codes=world.codes,
        )
        return world, velocity

    def compute_losses(
        self,
        batch: CodeWAMBatch,
        *,
        noise: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> CodeWAMLossOutput:
        world = self.build_world_state(batch.state, batch.codes)
        flow = self.action_flow.flow_matching_loss(
            batch.actions,
            belief=world.belief,
            policy=batch.policy,
            state=batch.state,
            sample_valid=batch.supervision.action,
            noise=noise,
            flow_time=flow_time,
            generator=generator,
            continuous=world.continuous,
            codes=world.codes,
        )
        future = None
        code_loss = flow.loss.new_zeros(())
        if self.config.variant == "C2":
            if batch.future_codes is None or batch.future_codes.schedule is None:
                raise ValueError(
                    "C2 requires future-code labels and transition schedule."
                )
            if world.codes is None:
                raise RuntimeError("C2 world state is missing current clock codes.")
            future = self.transition(
                world.belief,
                world.codes,
                batch.actions,
                batch.future_codes.schedule,
            )
            code_loss = self.transition.loss(
                future,
                batch.future_codes,
                sample_valid=batch.supervision.dynamics,
            )
        total = flow.loss + self.config.lambda_code * code_loss
        return CodeWAMLossOutput(
            total=total,
            action=flow.loss,
            code=code_loss,
            belief=world.belief,
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
        world = self.build_world_state(state, codes)
        return self.action_flow.sample(
            belief=world.belief,
            policy=policy,
            current_proprio=state.current_proprio,
            horizon=horizon,
            steps=steps,
            initial_noise=initial_noise,
            generator=generator,
            continuous=world.continuous,
            codes=world.codes,
        )

    def predict_future_codes(
        self,
        *,
        state: StateInputs,
        codes: CodeMeasurements,
        candidate_actions: ActionBatch,
        schedule: TransitionSchedule,
    ) -> FutureCodePrediction:
        world = self.build_world_state(state, codes)
        if world.codes is None:
            raise ValueError("Future-code prediction requires current clock codes.")
        return self.transition(
            world.belief,
            world.codes,
            candidate_actions,
            schedule,
        )


def build_codewam_v2(
    config: CodeWAMConfig,
    artifact_sets: Mapping[str, Sequence[FrozenRQArtifact]],
) -> CodeWAMV2:
    adapter = FrozenCodebookAdapter(
        artifact_sets,
        dim=config.dim,
        layout="hierarchical",
    )
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
        use_relative_time=True,
        dropout=config.dropout,
    )
    world_builder = StructuredWorldBuilder(
        dim=config.dim,
        heads=config.heads,
        proprio_dim=config.proprio_dim,
        action_dim=config.action_dim,
        families=adapter.families,
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
        include_continuous=True,
        include_codes=True,
        dropout=config.dropout,
    )
    transition = MultiClockTransitionModel(
        dim=config.dim,
        action_dim=config.action_dim,
        max_horizon=config.max_action_horizon,
        families=adapter.families,
        codebook_sizes=adapter.codebook_sizes,
        mode=config.dynamics_mode,
        layers=config.dynamics_layers,
        action_layers=config.dynamics_action_layers,
        dropout=config.dropout,
    )
    return CodeWAMV2(
        config=config,
        continuous_state=continuous,
        frozen_codebook=adapter,
        world_builder=world_builder,
        action_flow=action,
        transition=transition,
    )
