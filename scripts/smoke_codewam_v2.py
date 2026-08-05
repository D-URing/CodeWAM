#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

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
    build_codewam_v2,
)


SMOKE_SCHEMA = "codewam.model-smoke.v2"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a synthetic engineering smoke for structured CodeWAM v2."
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    return device


def _artifacts(seed: int) -> tuple[FrozenRQArtifact, ...]:
    generator = torch.Generator().manual_seed(seed)
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
                    count=64,
                    mean=torch.zeros(12),
                    std=torch.ones(12),
                ),
                centers=tuple(
                    torch.randn((4, 12), generator=generator) for _ in range(3)
                ),
                metadata={
                    "manifest_fingerprint": "synthetic-v2",
                    "dataset_revision": "synthetic-v2",
                    "wan_model_id": "synthetic-wan",
                    "wan_revision": "synthetic-wan-v1",
                    "preprocess_revision": "synthetic-preprocess-v1",
                    "config_hash": "synthetic-v2-config",
                    "source_checksums": ["synthetic-v2-source"],
                },
            )
        )
    return tuple(artifacts)


def _config() -> CodeWAMConfig:
    return CodeWAMConfig(
        variant="C2",
        dynamics_mode="prefix",
        latent_channels=4,
        proprio_dim=6,
        action_dim=7,
        language_dim=8,
        dim=16,
        heads=4,
        patch_size=2,
        max_time=8,
        max_cameras=2,
        max_spatial_tokens=64,
        max_action_horizon=4,
        state_spatial_layers=1,
        state_temporal_layers=1,
        belief_queries=3,
        belief_layers=1,
        action_layers=1,
        dynamics_layers=1,
        dynamics_action_layers=1,
        dropout=0.0,
        lambda_code=0.1,
    )


def _batch(device: torch.device, seed: int) -> CodeWAMBatch:
    generator = torch.Generator().manual_seed(seed)

    def randn(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=generator).to(device)

    available = torch.ones((2, 3), dtype=torch.bool, device=device)
    schedule = TransitionSchedule(
        action_prefix_lengths=torch.tensor(
            [[1, 2, 3], [1, 2, 3]],
            dtype=torch.long,
            device=device,
        ),
        delta_times=torch.tensor(
            [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]],
            device=device,
        ),
    )
    return CodeWAMBatch(
        state=StateInputs(
            latents=randn((2, 5, 2, 4, 8, 8)),
            proprio_history=randn((2, 3, 6)),
            past_actions=randn((2, 2, 7)),
            latent_time_offsets=torch.tensor(
                [[-0.4, -0.3, -0.2, -0.1, 0.0]] * 2,
                device=device,
            ),
            proprio_time_offsets=torch.tensor(
                [[-0.2, -0.1, 0.0]] * 2,
                device=device,
            ),
            past_action_time_offsets=torch.tensor(
                [[-0.2, -0.1]] * 2,
                device=device,
            ),
        ),
        codes=CodeMeasurements(
            code_ids=torch.randint(
                0,
                4,
                (2, 3, 3),
                generator=generator,
                device=device,
            ),
            available=available,
            chart_names=("synthetic", "synthetic"),
        ),
        policy=PolicyCondition(language=randn((2, 4, 8))),
        actions=ActionBatch(values=randn((2, 3, 7))),
        future_codes=FutureCodeTargets(
            code_ids=torch.randint(
                0,
                4,
                (2, 3, 3),
                generator=generator,
                device=device,
            ),
            available=available,
            schedule=schedule,
        ),
        supervision=SupervisionMasks(
            temporal=torch.ones(2, dtype=torch.bool, device=device),
            action=torch.ones(2, dtype=torch.bool, device=device),
            dynamics=torch.ones(2, dtype=torch.bool, device=device),
        ),
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def run_smoke(device: torch.device, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = build_codewam_v2(
        _config(),
        {"synthetic": _artifacts(seed + 1)},
    ).to(device)
    batch = _batch(device, seed + 2)
    centers = {
        name: getattr(model.frozen_codebook, name).detach().clone()
        for name in model.frozen_codebook._buffer_names.values()
    }
    noise = torch.randn_like(batch.actions.values)
    flow_time = torch.tensor([0.25, 0.75], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    output = model(batch, noise=noise, flow_time=flow_time)
    optimizer.zero_grad(set_to_none=True)
    output.total.backward()
    optimizer.step()

    centers_frozen = all(
        torch.equal(getattr(model.frozen_codebook, name), value)
        for name, value in centers.items()
    )
    parameters_finite = all(
        bool(torch.isfinite(parameter).all().item()) for parameter in model.parameters()
    )

    model.eval()
    changed_targets = FutureCodeTargets(
        code_ids=(batch.future_codes.code_ids + 1).remainder(4),
        available=batch.future_codes.available,
        schedule=batch.future_codes.schedule,
    )
    changed_batch = CodeWAMBatch(
        state=batch.state,
        policy=batch.policy,
        actions=batch.actions,
        supervision=batch.supervision,
        codes=batch.codes,
        future_codes=changed_targets,
    )
    first = model(batch, noise=noise, flow_time=flow_time)
    second = model(changed_batch, noise=noise, flow_time=flow_time)
    future_label_isolated = torch.equal(first.flow.velocity, second.flow.velocity)

    world = model.build_world_state(batch.state, batch.codes)
    changed_actions = batch.actions.values.clone()
    changed_actions[:, 2] += 50.0
    changed_action_batch = ActionBatch(values=changed_actions)
    first_transition = model.transition(
        world.belief,
        world.codes,
        batch.actions,
        batch.future_codes.schedule,
    )
    second_transition = model.transition(
        world.belief,
        world.codes,
        changed_action_batch,
        batch.future_codes.schedule,
    )
    prefix_isolated = torch.equal(
        first_transition.logits[0],
        second_transition.logits[0],
    ) and torch.equal(first_transition.logits[1], second_transition.logits[1])

    _sync(device)
    started = time.perf_counter()
    sampled = model.infer_actions(
        state=batch.state,
        policy=batch.policy,
        codes=batch.codes,
        horizon=3,
        steps=2,
        initial_noise=torch.zeros_like(batch.actions.values),
    )
    _sync(device)
    latency = time.perf_counter() - started
    report = {
        "schema": SMOKE_SCHEMA,
        "device": str(device),
        "seed": seed,
        "loss": {
            "total": float(output.total.detach().cpu()),
            "action": float(output.action.detach().cpu()),
            "code": float(output.code.detach().cpu()),
        },
        "optimizer_step": True,
        "parameters_finite": parameters_finite,
        "frozen_centers_unchanged": centers_frozen,
        "future_label_isolated_from_policy": future_label_isolated,
        "action_prefix_isolated_by_clock": prefix_isolated,
        "sample": {
            "shape": list(sampled.shape),
            "finite": bool(torch.isfinite(sampled).all().item()),
            "two_step_seconds": latency,
        },
        "parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
        },
    }
    required = (
        report["optimizer_step"],
        report["parameters_finite"],
        report["frozen_centers_unchanged"],
        report["future_label_isolated_from_policy"],
        report["action_prefix_isolated_by_clock"],
        report["sample"]["finite"],
    )
    if not all(required):
        raise RuntimeError(f"CodeWAM v2 smoke failed: {report}")
    return report


def main() -> None:
    args = _parse_args()
    report = run_smoke(_device(args.device), args.seed)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
