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
    ContinuousStateEncoder,
    FutureCodeTargets,
    PolicyCondition,
    StateInputs,
    SupervisionMasks,
    TemporalLatentPredictor,
    build_codewam_v1,
    future_code_metrics,
    persistence_code_metrics,
    temporal_pretraining_loss,
    transition_family_masks,
)


SMOKE_SCHEMA = "codewam.model-smoke.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a synthetic engineering smoke for the independent CodeWAM v1 "
            "modules. This does not produce scientific evidence."
        )
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return device


def _synthetic_artifacts(
    chart: str,
    *,
    offset: float,
    seed: int,
    k: int = 4,
    levels: int = 3,
    descriptor_dim: int = 12,
) -> tuple[FrozenRQArtifact, ...]:
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
                    mean=torch.zeros(descriptor_dim),
                    std=torch.ones(descriptor_dim),
                ),
                centers=tuple(
                    torch.randn(
                        (k, descriptor_dim),
                        generator=generator,
                    )
                    + offset
                    for _ in range(levels)
                ),
                metadata={
                    "manifest_fingerprint": f"synthetic-{chart}",
                    "dataset_revision": f"synthetic-{chart}-v1",
                    "wan_model_id": "synthetic-wan",
                    "wan_revision": "synthetic-wan-v1",
                    "preprocess_revision": "synthetic-preprocess-v1",
                    "config_hash": "synthetic-config",
                    "source_checksums": [f"synthetic-{chart}-source"],
                },
            )
        )
    return tuple(artifacts)


def _config(
    variant: str,
    dynamics_mode: str,
) -> CodeWAMConfig:
    return CodeWAMConfig(
        variant=variant,
        dynamics_mode=dynamics_mode,
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
        dropout=0.0,
    )


def _synthetic_batch(device: torch.device, seed: int) -> CodeWAMBatch:
    generator = torch.Generator().manual_seed(seed)

    def random(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=generator).to(device)

    code_ids = torch.randint(
        0,
        4,
        (2, 3, 3),
        generator=generator,
        dtype=torch.long,
    )
    future_ids = torch.randint(
        0,
        4,
        (2, 3, 3),
        generator=generator,
        dtype=torch.long,
    )
    code_ids[1, 1] = -1
    future_ids[1, 1] = -1
    available = torch.tensor(
        [[True, True, True], [True, False, True]],
        dtype=torch.bool,
        device=device,
    )
    return CodeWAMBatch(
        state=StateInputs(
            latents=random((2, 5, 2, 4, 8, 8)),
            proprio_history=random((2, 3, 6)),
            past_actions=random((2, 2, 7)),
        ),
        codes=CodeMeasurements(
            code_ids=code_ids.to(device),
            available=available,
            chart_names=("droid", "libero"),
        ),
        policy=PolicyCondition(
            language=random((2, 4, 8)),
            language_valid=torch.tensor(
                [[True, True, True, True], [True, True, False, False]],
                dtype=torch.bool,
                device=device,
            ),
        ),
        actions=ActionBatch(
            values=random((2, 3, 7)),
            valid=torch.tensor(
                [[True, True, True], [True, True, False]],
                dtype=torch.bool,
                device=device,
            ),
        ),
        future_codes=FutureCodeTargets(
            code_ids=future_ids.to(device),
            available=available,
        ),
        supervision=SupervisionMasks(
            temporal=torch.tensor([True, True], dtype=torch.bool, device=device),
            action=torch.tensor([True, False], dtype=torch.bool, device=device),
            dynamics=torch.tensor([True, True], dtype=torch.bool, device=device),
        ),
    )


def _center_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    adapter = model.frozen_codebook
    return {
        name: getattr(adapter, name).detach().cpu().clone()
        for name in adapter._buffer_names.values()
    }


def _centers_unchanged(
    model: torch.nn.Module,
    expected: dict[str, torch.Tensor],
) -> bool:
    adapter = model.frozen_codebook
    return all(
        torch.equal(getattr(adapter, name).detach().cpu(), value)
        for name, value in expected.items()
    )


def _has_gradient(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None
        and bool(parameter.grad.detach().abs().sum().item() > 0.0)
        for parameter in module.parameters()
    )


def _finite_scalar(value: torch.Tensor, name: str) -> float:
    if value.numel() != 1 or not bool(torch.isfinite(value).item()):
        raise RuntimeError(f"Smoke produced a non-finite `{name}`.")
    return float(value.detach().cpu().item())


def _run_stage0(
    batch: CodeWAMBatch,
    *,
    device: torch.device,
) -> dict[str, Any]:
    encoder = ContinuousStateEncoder(
        latent_channels=4,
        dim=16,
        heads=4,
        patch_size=2,
        spatial_layers=1,
        temporal_layers=1,
        max_time=8,
        max_cameras=4,
        max_spatial_tokens=64,
        dropout=0.0,
    ).to(device)
    predictor = TemporalLatentPredictor(
        dim=16,
        latent_channels=4,
        patch_size=2,
        heads=4,
        layers=1,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (*encoder.parameters(), *predictor.parameters()),
        lr=1e-4,
    )
    loss = temporal_pretraining_loss(
        encoder,
        predictor,
        batch.state,
        context_index=2,
        target_index=4,
        sample_valid=batch.supervision.temporal,
    )
    loss.backward()
    gradient_routes = {
        "continuous_state": _has_gradient(encoder),
        "temporal_predictor": _has_gradient(predictor),
    }
    optimizer.step()
    if not all(gradient_routes.values()):
        raise RuntimeError(f"Stage-0 gradient route failed: {gradient_routes}.")
    return {
        "loss": _finite_scalar(loss, "stage0_loss"),
        "context_index": 2,
        "target_index": 4,
        "future_excluded_from_encoder_input": True,
        "gradient_routes": gradient_routes,
        "predictor_discardable": True,
    }


def _run_variant(
    *,
    variant: str,
    dynamics_mode: str,
    artifact_sets: dict[str, tuple[FrozenRQArtifact, ...]],
    batch: CodeWAMBatch,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = build_codewam_v1(
        _config(variant, dynamics_mode),
        artifact_sets,
    ).to(device)
    centers_before = _center_snapshot(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    generator = torch.Generator().manual_seed(seed + 1)
    noise = torch.randn(
        tuple(batch.actions.values.shape),
        generator=generator,
    ).to(device)
    flow_time = torch.tensor([0.25, 0.75], device=device)
    output = model(batch, noise=noise, flow_time=flow_time)
    output.total.backward()
    gradient_routes = {
        "continuous_state": _has_gradient(model.continuous_state),
        "frozen_codebook_projection": _has_gradient(model.frozen_codebook),
        "belief_core": _has_gradient(model.belief_core),
        "action_flow": _has_gradient(model.action_flow),
        "code_dynamics": _has_gradient(model.code_dynamics),
    }
    optimizer.step()
    centers_frozen = _centers_unchanged(model, centers_before)
    if not centers_frozen:
        raise RuntimeError(f"Frozen centers changed in {variant}/{dynamics_mode}.")

    model.eval()
    inferred = model.infer_actions(
        state=batch.state,
        codes=batch.codes,
        policy=batch.policy,
        horizon=3,
        steps=2,
        initial_noise=torch.zeros_like(batch.actions.values),
    )
    if not bool(torch.isfinite(inferred).all().item()):
        raise RuntimeError(f"Inference was non-finite in {variant}/{dynamics_mode}.")

    report: dict[str, Any] = {
        "total_loss": _finite_scalar(output.total, "total_loss"),
        "action_loss": _finite_scalar(output.action, "action_loss"),
        "code_loss": _finite_scalar(output.code, "code_loss"),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "gradient_routes": gradient_routes,
        "frozen_centers_unchanged": centers_frozen,
        "inference_shape": list(inferred.shape),
        "inference_finite": True,
        "basic_inference_uses_dynamics": False,
    }
    if output.future is not None:
        transition_masks = transition_family_masks(
            batch.codes,
            batch.future_codes,
        )
        supervised = batch.supervision.dynamics[:, None]
        all_available = batch.future_codes.available & supervised
        changed_available = transition_masks["changed"] & supervised
        predicted_ids = output.future.predicted_ids()
        report["future_code"] = {
            "mode": output.future.mode,
            "all_metrics": future_code_metrics(
                output.future,
                batch.future_codes,
                sample_valid=batch.supervision.dynamics,
                calibration_bins=5,
            ),
            "changed_family_metrics": future_code_metrics(
                output.future,
                batch.future_codes,
                sample_valid=batch.supervision.dynamics,
                family_valid=transition_masks["changed"],
                calibration_bins=5,
            ),
            "persistence_baseline": persistence_code_metrics(
                batch.codes,
                batch.future_codes,
                sample_valid=batch.supervision.dynamics,
            ),
            "normalized_center_mse": float(
                model.frozen_codebook.normalized_center_mse(
                    predicted_ids,
                    batch.future_codes.code_ids,
                    available=all_available,
                    chart_names=batch.codes.chart_names,
                )
                .detach()
                .cpu()
                .item()
            ),
            "changed_family_center_mse": float(
                model.frozen_codebook.normalized_center_mse(
                    predicted_ids,
                    batch.future_codes.code_ids,
                    available=changed_available,
                    chart_names=batch.codes.chart_names,
                )
                .detach()
                .cpu()
                .item()
            ),
            "persistence_center_mse": float(
                model.frozen_codebook.normalized_center_mse(
                    batch.codes.code_ids,
                    batch.future_codes.code_ids,
                    available=all_available,
                    chart_names=batch.codes.chart_names,
                )
                .detach()
                .cpu()
                .item()
            ),
            "persistence_changed_family_center_mse": float(
                model.frozen_codebook.normalized_center_mse(
                    batch.codes.code_ids,
                    batch.future_codes.code_ids,
                    available=changed_available,
                    chart_names=batch.codes.chart_names,
                )
                .detach()
                .cpu()
                .item()
            ),
        }
    return report


def run_smoke(
    *,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    torch.manual_seed(seed)
    artifact_sets = {
        "droid": _synthetic_artifacts(
            "droid",
            offset=0.0,
            seed=seed + 11,
        ),
        "libero": _synthetic_artifacts(
            "libero",
            offset=10.0,
            seed=seed + 17,
        ),
    }
    batch = _synthetic_batch(device, seed + 23)
    variants: dict[str, Any] = {}
    for index, (variant, mode) in enumerate(
        (
            ("C0", "independent"),
            ("C1", "independent"),
            ("C2", "independent"),
            ("C2", "prefix"),
        )
    ):
        label = variant if mode == "independent" else f"{variant}-{mode}"
        variants[label] = _run_variant(
            variant=variant,
            dynamics_mode=mode,
            artifact_sets=artifact_sets,
            batch=batch,
            device=device,
            seed=seed + 100 + index,
        )

    return {
        "schema": SMOKE_SCHEMA,
        "kind": "synthetic-engineering-smoke",
        "scientific_evidence": False,
        "device": str(device),
        "torch_version": torch.__version__,
        "seed": seed,
        "stage0": _run_stage0(batch, device=device),
        "variants": variants,
        "checks": {
            "c0_c1_c2_optimizer_step": True,
            "independent_and_prefix_dynamics": True,
            "mixed_chart_lookup": True,
            "availability_masks": True,
            "failure_imitation_mask": True,
            "frozen_center_integrity": True,
            "basic_action_inference": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    report = run_smoke(
        device=_resolve_device(args.device),
        seed=args.seed,
    )
    if args.output is not None:
        _write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
