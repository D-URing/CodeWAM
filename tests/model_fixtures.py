from __future__ import annotations

import torch

from codewam.codebook_eval.streaming import (
    CausalDescriptorSpec,
    FrozenRQArtifact,
    NormalizationStats,
)
from codewam.models import (
    ActionBatch,
    CodeMeasurements,
    CodeWAMBatch,
    CodeWAMConfig,
    FutureCodeTargets,
    PolicyCondition,
    StateInputs,
    SupervisionMasks,
    TransitionSchedule,
)


def synthetic_artifacts(
    chart: str,
    *,
    offset: float = 0.0,
    k: int = 4,
    levels: int = 3,
    descriptor_dim: int = 12,
) -> tuple[FrozenRQArtifact, ...]:
    generator = torch.Generator().manual_seed(
        sum(ord(value) for value in chart) + 17
    )
    artifacts = []
    for stride in (2, 3, 5):
        artifacts.append(
            FrozenRQArtifact(
                family=f"Q{stride}",
                descriptor=CausalDescriptorSpec(
                    stride=stride,
                    pool=1,
                    camera_ids=("wrist",),
                ),
                normalization=NormalizationStats(
                    count=32,
                    mean=torch.zeros(descriptor_dim),
                    std=torch.ones(descriptor_dim),
                ),
                centers=tuple(
                    torch.randn(
                        (k, descriptor_dim),
                        generator=generator,
                    )
                    + float(offset)
                    for _ in range(levels)
                ),
                metadata={
                    "manifest_fingerprint": f"synthetic-{chart}",
                    "dataset_revision": f"{chart}-v1",
                    "wan_model_id": "synthetic-wan",
                    "wan_revision": "synthetic-wan-revision",
                    "preprocess_revision": "synthetic-preprocess",
                    "config_hash": "synthetic-config",
                    "source_checksums": [f"synthetic-{chart}-source"],
                },
            )
        )
    return tuple(artifacts)


def small_config(
    *,
    variant: str = "C2",
    dynamics_mode: str = "independent",
) -> CodeWAMConfig:
    return CodeWAMConfig(
        variant=variant,
        latent_channels=4,
        proprio_dim=6,
        action_dim=7,
        language_dim=8,
        dim=16,
        heads=4,
        patch_size=2,
        max_time=8,
        max_cameras=4,
        max_spatial_tokens=64,
        max_action_horizon=4,
        state_spatial_layers=1,
        state_temporal_layers=1,
        belief_queries=3,
        belief_layers=1,
        action_layers=1,
        dynamics_layers=1,
        dynamics_action_layers=1,
        dynamics_mode=dynamics_mode,
        dropout=0.0,
    )


def small_batch(
    *,
    chart_names: tuple[str, ...] = ("droid", "droid"),
    action_supervision: torch.Tensor | None = None,
    dynamics_supervision: torch.Tensor | None = None,
) -> CodeWAMBatch:
    batch = len(chart_names)
    generator = torch.Generator().manual_seed(101)
    state = StateInputs(
        latents=torch.randn((batch, 5, 2, 4, 8, 8), generator=generator),
        proprio_history=torch.randn((batch, 3, 6), generator=generator),
        past_actions=torch.randn((batch, 2, 7), generator=generator),
        latent_time_offsets=torch.tensor(
            [[-0.4, -0.3, -0.2, -0.1, 0.0]] * batch
        ),
        proprio_time_offsets=torch.tensor([[-0.2, -0.1, 0.0]] * batch),
        past_action_time_offsets=torch.tensor([[-0.2, -0.1]] * batch),
    )
    policy = PolicyCondition(
        language=torch.randn((batch, 4, 8), generator=generator),
    )
    actions = ActionBatch(
        values=torch.randn((batch, 3, 7), generator=generator),
    )
    codes = CodeMeasurements(
        code_ids=torch.randint(0, 4, (batch, 3, 3), generator=generator),
        available=torch.ones((batch, 3), dtype=torch.bool),
        chart_names=chart_names,
    )
    future = FutureCodeTargets(
        code_ids=torch.randint(0, 4, (batch, 3, 3), generator=generator),
        available=torch.ones((batch, 3), dtype=torch.bool),
        schedule=TransitionSchedule(
            action_prefix_lengths=torch.tensor([[1, 2, 3]] * batch),
            delta_times=torch.tensor([[0.1, 0.2, 0.3]] * batch),
        ),
    )
    return CodeWAMBatch(
        state=state,
        policy=policy,
        actions=actions,
        codes=codes,
        future_codes=future,
        supervision=SupervisionMasks(
            temporal=torch.ones((batch,), dtype=torch.bool),
            action=(
                torch.ones((batch,), dtype=torch.bool)
                if action_supervision is None
                else action_supervision
            ),
            dynamics=(
                torch.ones((batch,), dtype=torch.bool)
                if dynamics_supervision is None
                else dynamics_supervision
            ),
        ),
    )
