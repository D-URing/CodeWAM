from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from codewam.data.droid_manifest import write_json_report
from codewam.data.droid_rlds import read_manifest_droid_rlds_frames

from .manifest import EpisodeManifest, EpisodeRecord
from .shards import (
    PooledFeatureEpisode,
    file_sha256,
    load_torch_payload,
)
from .streaming import FrozenRQArtifact, encode_residual_quantizer
from .wan_probe_export import (
    WanProbeExportConfig,
    _load_wan_vae,
    _torch_dtype,
)


VISUAL_PERTURBATION_CONTRACT_SCHEMA = (
    "codewam.visual-perturbation-contract.v3"
)
VISUAL_PERTURBATION_REPORT_SCHEMA = (
    "codewam.visual-perturbation-report.v3"
)
TARGET_LATENT_TICK = 10
CLIP_FRAMES = 45
EXPECTED_LATENT_TICKS = 12
ENDPOINT_FRAME_START = 4 * TARGET_LATENT_TICK - 3
ENDPOINT_FRAME_STOP = 4 * TARGET_LATENT_TICK + 1


@dataclass(frozen=True)
class RGBCondition:
    name: str
    category: str
    scope: str
    operation: str
    value: float
    axis: str | None = None


RGB_CONDITIONS = (
    RGBCondition("identity", "baseline", "all", "identity", 0.0),
    RGBCondition(
        "uniform_brightness_085",
        "photometric_nuisance",
        "all",
        "brightness",
        0.85,
    ),
    RGBCondition(
        "uniform_brightness_115",
        "photometric_nuisance",
        "all",
        "brightness",
        1.15,
    ),
    RGBCondition(
        "uniform_contrast_085",
        "photometric_nuisance",
        "all",
        "contrast",
        0.85,
    ),
    RGBCondition(
        "uniform_contrast_115",
        "photometric_nuisance",
        "all",
        "contrast",
        1.15,
    ),
    RGBCondition(
        "uniform_translate_x_negative_8",
        "global_geometry",
        "all",
        "translate",
        -8.0,
        "x",
    ),
    RGBCondition(
        "uniform_translate_x_positive_8",
        "global_geometry",
        "all",
        "translate",
        8.0,
        "x",
    ),
    RGBCondition(
        "uniform_scale_090",
        "global_geometry",
        "all",
        "scale",
        0.90,
    ),
    RGBCondition(
        "uniform_scale_110",
        "global_geometry",
        "all",
        "scale",
        1.10,
    ),
    RGBCondition(
        "endpoint_translate_x_negative_4",
        "endpoint_geometry",
        "endpoint",
        "translate",
        -4.0,
        "x",
    ),
    RGBCondition(
        "endpoint_translate_x_positive_4",
        "endpoint_geometry",
        "endpoint",
        "translate",
        4.0,
        "x",
    ),
    RGBCondition(
        "endpoint_translate_x_negative_8",
        "endpoint_geometry",
        "endpoint",
        "translate",
        -8.0,
        "x",
    ),
    RGBCondition(
        "endpoint_translate_x_positive_8",
        "endpoint_geometry",
        "endpoint",
        "translate",
        8.0,
        "x",
    ),
    RGBCondition(
        "endpoint_translate_y_negative_8",
        "endpoint_geometry",
        "endpoint",
        "translate",
        -8.0,
        "y",
    ),
    RGBCondition(
        "endpoint_translate_y_positive_8",
        "endpoint_geometry",
        "endpoint",
        "translate",
        8.0,
        "y",
    ),
    RGBCondition(
        "endpoint_scale_090",
        "endpoint_geometry",
        "endpoint",
        "scale",
        0.90,
    ),
    RGBCondition(
        "endpoint_scale_110",
        "endpoint_geometry",
        "endpoint",
        "scale",
        1.10,
    ),
)
OPPOSITE_CONDITION_PAIRS = (
    (
        "endpoint_translate_x_negative_4",
        "endpoint_translate_x_positive_4",
    ),
    (
        "endpoint_translate_x_negative_8",
        "endpoint_translate_x_positive_8",
    ),
    (
        "endpoint_translate_y_negative_8",
        "endpoint_translate_y_positive_8",
    ),
    ("endpoint_scale_090", "endpoint_scale_110"),
)


@dataclass(frozen=True)
class VisualClip:
    clip_id: str
    source: str
    split: str
    scene_id: str | None
    task_id: str | None
    frames: torch.Tensor
    canonical_pooled: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.clip_id or not self.source or not self.split:
            raise ValueError("Visual clip identity fields must be nonempty.")
        if (
            self.frames.ndim != 4
            or self.frames.shape[-1] != 3
            or self.frames.dtype != torch.uint8
        ):
            raise ValueError("Visual clip frames must be uint8 [T,H,W,3].")
        if int(self.frames.shape[0]) != CLIP_FRAMES:
            raise ValueError(
                f"Visual clip must contain {CLIP_FRAMES} frames."
            )
        if self.canonical_pooled is not None and (
            self.canonical_pooled.ndim != 4
            or self.canonical_pooled.shape[0] < EXPECTED_LATENT_TICKS
            or tuple(self.canonical_pooled.shape[1:]) != (48, 4, 4)
        ):
            raise ValueError(
                "Canonical pooled reference must be [T>=12,48,4,4]."
            )
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_score(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def _resize_float(
    frames: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    values = frames.permute(0, 3, 1, 2).float()
    resized = F.interpolate(
        values,
        size=(int(height), int(width)),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized.clamp(0, 255).permute(0, 2, 3, 1).contiguous()


def _resize_uint8(
    frames: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    return (
        _resize_float(frames, height, width)
        .round()
        .to(torch.uint8)
        .contiguous()
    )


def _restore_frame_dtype(
    values: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    values = values.clamp(0, 255)
    if reference.is_floating_point():
        return values.to(dtype=reference.dtype)
    return values.round().to(dtype=reference.dtype)


def _translate(
    frames: torch.Tensor,
    *,
    dx: int = 0,
    dy: int = 0,
) -> torch.Tensor:
    if dx == 0 and dy == 0:
        return frames.clone()
    values = frames.permute(0, 3, 1, 2)
    height, width = int(values.shape[-2]), int(values.shape[-1])
    left = max(dx, 0)
    right = max(-dx, 0)
    top = max(dy, 0)
    bottom = max(-dy, 0)
    padded = F.pad(
        values,
        (left, right, top, bottom),
        mode="replicate",
    )
    start_x = max(-dx, 0)
    start_y = max(-dy, 0)
    return (
        padded[
            :,
            :,
            start_y : start_y + height,
            start_x : start_x + width,
        ]
        .permute(0, 2, 3, 1)
        .contiguous()
    )


def _scale(frames: torch.Tensor, factor: float) -> torch.Tensor:
    if factor <= 0:
        raise ValueError("RGB scale factor must be positive.")
    values = frames.permute(0, 3, 1, 2).float()
    height, width = int(values.shape[-2]), int(values.shape[-1])
    scaled_height = max(1, round(height * float(factor)))
    scaled_width = max(1, round(width * float(factor)))
    scaled = F.interpolate(
        values,
        size=(scaled_height, scaled_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    if factor < 1.0:
        pad_y = height - scaled_height
        pad_x = width - scaled_width
        scaled = F.pad(
            scaled,
            (
                pad_x // 2,
                pad_x - pad_x // 2,
                pad_y // 2,
                pad_y - pad_y // 2,
            ),
            mode="replicate",
        )
    else:
        start_y = (scaled_height - height) // 2
        start_x = (scaled_width - width) // 2
        scaled = scaled[
            :,
            :,
            start_y : start_y + height,
            start_x : start_x + width,
        ]
    return _restore_frame_dtype(
        scaled.permute(0, 2, 3, 1),
        frames,
    ).contiguous()


def _preprocess_resized_video(
    frames: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            f"Expected resized [T,H,W,3] frames, got {tuple(frames.shape)}."
        )
    values = frames.permute(0, 3, 1, 2).float()
    values = values / 127.5 - 1.0
    return values.to(dtype=dtype).permute(1, 0, 2, 3).contiguous()


def _apply_operation(
    frames: torch.Tensor,
    condition: RGBCondition,
) -> torch.Tensor:
    if condition.operation == "identity":
        return frames.clone()
    if condition.operation == "brightness":
        return _restore_frame_dtype(
            frames.float().mul(float(condition.value)),
            frames,
        )
    if condition.operation == "contrast":
        values = frames.float()
        mean = values.mean(dim=(1, 2), keepdim=True)
        return _restore_frame_dtype(
            (values - mean) * float(condition.value) + mean,
            frames,
        )
    if condition.operation == "translate":
        displacement = int(round(float(condition.value)))
        return _translate(
            frames,
            dx=displacement if condition.axis == "x" else 0,
            dy=displacement if condition.axis == "y" else 0,
        )
    if condition.operation == "scale":
        return _scale(frames, float(condition.value))
    raise ValueError(f"Unsupported RGB operation `{condition.operation}`.")


def apply_rgb_condition(
    frames: torch.Tensor,
    condition: RGBCondition,
) -> torch.Tensor:
    if condition.scope == "all":
        return _apply_operation(frames, condition)
    if condition.scope != "endpoint":
        raise ValueError(f"Unsupported RGB condition scope `{condition.scope}`.")
    result = frames.clone()
    result[ENDPOINT_FRAME_START:ENDPOINT_FRAME_STOP] = _apply_operation(
        frames[ENDPOINT_FRAME_START:ENDPOINT_FRAME_STOP],
        condition,
    )
    return result


def _descriptor(
    pooled: torch.Tensor,
    *,
    stride: int,
    time_index: int,
) -> torch.Tensor:
    if pooled.ndim != 5:
        raise ValueError("Pooled condition tensor must be [N,T,C,G,G].")
    if time_index < 2 * stride or time_index >= pooled.shape[1]:
        raise IndexError("Descriptor time index is outside the pooled clip.")
    flattened = pooled.reshape(pooled.shape[0], pooled.shape[1], -1)
    return torch.cat(
        (
            flattened[:, time_index - 2 * stride],
            flattened[:, time_index - stride],
            flattened[:, time_index],
        ),
        dim=1,
    ).contiguous()


def _prefix_change(
    left: torch.Tensor,
    right: torch.Tensor,
) -> tuple[list[bool], list[bool]]:
    comparison = left != right
    level = [bool(value) for value in comparison.squeeze(0).tolist()]
    prefix = [
        bool(comparison[:, : depth + 1].any().item())
        for depth in range(comparison.shape[1])
    ]
    return level, prefix


def _quantized_prefixes(
    codes: torch.Tensor,
    centers: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    if codes.ndim != 2 or codes.shape[1] != len(centers):
        raise ValueError("Codes and RQ centers have inconsistent depth.")
    quantized = torch.zeros(
        (codes.shape[0], centers[0].shape[1]),
        device=codes.device,
        dtype=torch.float32,
    )
    prefixes = []
    for level, level_centers in enumerate(centers):
        level_centers = level_centers.to(
            device=codes.device,
            dtype=torch.float32,
        )
        quantized = quantized + level_centers[codes[:, level]]
        prefixes.append(quantized.clone())
    return tuple(prefixes)


def _condition_payload(condition: RGBCondition) -> dict[str, Any]:
    return {
        "name": condition.name,
        "category": condition.category,
        "scope": condition.scope,
        "operation": condition.operation,
        "value": condition.value,
        "axis": condition.axis,
    }


def _encode_conditions(
    *,
    frames: torch.Tensor,
    conditions: Sequence[RGBCondition],
    vae: Any,
    device: torch.device,
    dtype: torch.dtype,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    resized = _resize_float(frames, image_height, image_width)
    videos = [
        _preprocess_resized_video(
            apply_rgb_condition(resized, condition),
            dtype=dtype,
        )
        for condition in conditions
    ]
    with torch.inference_mode():
        latent = vae.encode(videos, device=device, tiled=False)
    if (
        latent.ndim != 5
        or latent.shape[0] != len(conditions)
        or latent.shape[1] != 48
        or latent.shape[2] != EXPECTED_LATENT_TICKS
    ):
        raise RuntimeError(
            f"Unexpected visual-probe latent shape {tuple(latent.shape)}."
        )
    condition_count, channels, ticks, height, width = latent.shape
    pooled = F.adaptive_avg_pool2d(
        latent.permute(0, 2, 1, 3, 4).reshape(
            condition_count * ticks,
            channels,
            height,
            width,
        ),
        output_size=(4, 4),
    )
    return (
        pooled.reshape(condition_count, ticks, channels, 4, 4)
        .float()
        .cpu()
        .contiguous()
    )


def _perplexity(counts: torch.Tensor) -> float:
    total = int(counts.sum().item())
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0].double() / float(total)
    entropy = -(probabilities * probabilities.log()).sum()
    return float(entropy.exp().item())


def _summarize_measurements(
    *,
    clips: Sequence[VisualClip],
    artifacts: dict[str, FrozenRQArtifact],
    measurements: list[dict[str, Any]],
    reproduction: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for family in sorted(artifacts):
        family_rows = [
            row for row in measurements if row["family"] == family
        ]
        natural = [float(row["natural_displacement_mse"]) for row in family_rows]
        natural_mean = sum(natural) / max(len(natural), 1)
        levels = len(artifacts[family].centers)
        natural_quantized_mean = [
            sum(
                float(
                    row["natural_quantized_prefix_displacement_mse"][level]
                )
                for row in family_rows
            )
            / max(len(family_rows), 1)
            for level in range(levels)
        ]
        for condition in RGB_CONDITIONS:
            selected = [
                row
                for row in family_rows
                if row["condition"] == condition.name
            ]
            if not selected:
                continue
            displacement = sum(
                float(row["descriptor_displacement_mse"])
                for row in selected
            ) / len(selected)
            quantized_displacement = [
                sum(
                    float(
                        row["quantized_prefix_displacement_mse"][level]
                    )
                    for row in selected
                )
                / len(selected)
                for level in range(levels)
            ]
            rows.append(
                {
                    "family": family,
                    **_condition_payload(condition),
                    "samples": len(selected),
                    "mean_normalized_descriptor_displacement_mse": (
                        displacement
                    ),
                    "relative_to_natural_next_displacement": (
                        displacement / natural_mean
                        if natural_mean > 0
                        else None
                    ),
                    "mean_quantized_prefix_displacement_mse": (
                        quantized_displacement
                    ),
                    "quantized_prefix_relative_to_natural_next": [
                        (
                            quantized_displacement[level]
                            / natural_quantized_mean[level]
                            if natural_quantized_mean[level] > 0
                            else None
                        )
                        for level in range(levels)
                    ],
                    "level_change_fraction": [
                        sum(
                            int(row["level_changed"][level])
                            for row in selected
                        )
                        / len(selected)
                        for level in range(levels)
                    ],
                    "prefix_change_fraction": [
                        sum(
                            int(row["prefix_changed"][level])
                            for row in selected
                        )
                        / len(selected)
                        for level in range(levels)
                    ],
                }
            )

    direction_rows = []
    by_name = {
        condition.name: index
        for index, condition in enumerate(RGB_CONDITIONS)
    }
    for family, artifact in sorted(artifacts.items()):
        family_rows = [
            row for row in measurements if row["family"] == family
        ]
        rows_by_clip = {row["clip_id"]: row for row in family_rows}
        levels = len(artifact.centers)
        for left_name, right_name in OPPOSITE_CONDITION_PAIRS:
            selected = []
            for clip in clips:
                row = rows_by_clip.get(clip.clip_id)
                if row is None:
                    continue
                left = torch.tensor(
                    row["condition_codes"][by_name[left_name]]
                )
                right = torch.tensor(
                    row["condition_codes"][by_name[right_name]]
                )
                selected.append(left != right)
            if not selected:
                continue
            comparison = torch.stack(selected)
            direction_rows.append(
                {
                    "family": family,
                    "left_condition": left_name,
                    "right_condition": right_name,
                    "samples": int(comparison.shape[0]),
                    "level_distinct_fraction": [
                        float(comparison[:, level].float().mean().item())
                        for level in range(levels)
                    ],
                    "prefix_distinct_fraction": [
                        float(
                            comparison[:, : level + 1]
                            .any(dim=1)
                            .float()
                            .mean()
                            .item()
                        )
                        for level in range(levels)
                    ],
                }
            )

    natural_rows = []
    usage_rows = []
    for family, artifact in sorted(artifacts.items()):
        selected_by_clip = {
            row["clip_id"]: row
            for row in measurements
            if row["family"] == family
        }
        selected = list(selected_by_clip.values())
        levels = len(artifact.centers)
        natural_rows.append(
            {
                "family": family,
                "samples": len(selected),
                "mean_normalized_descriptor_displacement_mse": sum(
                    float(row["natural_displacement_mse"])
                    for row in selected
                )
                / max(len(selected), 1),
                "mean_quantized_prefix_displacement_mse": [
                    sum(
                        float(
                            row[
                                "natural_quantized_prefix_displacement_mse"
                            ][level]
                        )
                        for row in selected
                    )
                    / max(len(selected), 1)
                    for level in range(levels)
                ],
                "level_change_fraction": [
                    sum(
                        int(row["natural_level_changed"][level])
                        for row in selected
                    )
                    / max(len(selected), 1)
                    for level in range(levels)
                ],
                "prefix_change_fraction": [
                    sum(
                        int(row["natural_prefix_changed"][level])
                        for row in selected
                    )
                    / max(len(selected), 1)
                    for level in range(levels)
                ],
            }
        )
        identity_codes = torch.tensor(
            [row["condition_codes"][0] for row in selected],
            dtype=torch.long,
        )
        k = int(artifact.centers[0].shape[0])
        for level in range(levels):
            counts = torch.bincount(
                identity_codes[:, level],
                minlength=k,
            )
            usage_rows.append(
                {
                    "family": family,
                    "level": level + 1,
                    "samples": int(identity_codes.shape[0]),
                    "active_codes": int((counts > 0).sum().item()),
                    "capacity": k,
                    "perplexity": _perplexity(counts),
                    "perplexity_fraction": _perplexity(counts) / k,
                    "counts": counts.tolist(),
                }
            )

    reproduction_rows = []
    for family in sorted(artifacts):
        selected = [
            row for row in reproduction if row["family"] == family
        ]
        if not selected:
            continue
        levels = len(artifacts[family].centers)
        reproduction_rows.append(
            {
                "family": family,
                "samples": len(selected),
                "mean_pooled_mse": sum(
                    float(row["pooled_mse"]) for row in selected
                )
                / len(selected),
                "maximum_pooled_absolute_error": max(
                    float(row["pooled_max_abs"]) for row in selected
                ),
                "level_code_match_fraction": [
                    sum(
                        int(row["level_code_match"][level])
                        for row in selected
                    )
                    / len(selected)
                    for level in range(levels)
                ],
                "full_prefix_match_fraction": sum(
                    int(all(row["level_code_match"])) for row in selected
                )
                / len(selected),
            }
        )

    return {
        "condition_rows": rows,
        "direction_rows": direction_rows,
        "natural_next_rows": natural_rows,
        "identity_usage_rows": usage_rows,
        "droid_reproduction_rows": reproduction_rows,
    }


def run_visual_perturbation_probe(
    *,
    clips: Sequence[VisualClip],
    artifacts: dict[str, FrozenRQArtifact],
    vae: Any,
    device: str,
    dtype: str = "bfloat16",
    image_height: int = 224,
    image_width: int = 224,
    center_block_size: int = 1024,
) -> dict[str, Any]:
    if not clips:
        raise ValueError("Visual perturbation probe requires clips.")
    if set(artifacts) != {artifact.family for artifact in artifacts.values()}:
        raise ValueError(
            "Visual perturbation artifact labels must equal family names."
        )
    if image_height % 16 or image_width % 16:
        raise ValueError("Wan probe resolution must be divisible by 16.")
    target_device = torch.device(device)
    target_dtype = _torch_dtype(dtype)
    centers = {
        family: tuple(
            center.to(device=target_device, dtype=torch.float32)
            for center in artifact.centers
        )
        for family, artifact in artifacts.items()
    }
    measurements: list[dict[str, Any]] = []
    reproduction: list[dict[str, Any]] = []

    for clip in clips:
        pooled = _encode_conditions(
            frames=clip.frames,
            conditions=RGB_CONDITIONS,
            vae=vae,
            device=target_device,
            dtype=target_dtype,
            image_height=image_height,
            image_width=image_width,
        )
        for family, artifact in sorted(artifacts.items()):
            stride = artifact.descriptor.stride
            raw = _descriptor(
                pooled,
                stride=stride,
                time_index=TARGET_LATENT_TICK,
            )
            normalized = artifact.normalization.normalize(raw).to(
                device=target_device,
                dtype=torch.float32,
            )
            codes, _, _ = encode_residual_quantizer(
                normalized,
                centers[family],
                center_block_size=center_block_size,
            )
            quantized_prefixes = _quantized_prefixes(
                codes,
                centers[family],
            )
            natural_raw = _descriptor(
                pooled[:1],
                stride=stride,
                time_index=TARGET_LATENT_TICK + 1,
            )
            natural_normalized = artifact.normalization.normalize(
                natural_raw
            ).to(device=target_device, dtype=torch.float32)
            natural_codes, _, _ = encode_residual_quantizer(
                natural_normalized,
                centers[family],
                center_block_size=center_block_size,
            )
            natural_quantized_prefixes = _quantized_prefixes(
                natural_codes,
                centers[family],
            )
            natural_level, natural_prefix = _prefix_change(
                codes[:1],
                natural_codes,
            )
            identity = normalized[:1]
            identity_quantized_prefixes = tuple(
                prefix[:1] for prefix in quantized_prefixes
            )
            natural_quantized_displacement = [
                float(
                    (
                        natural_quantized_prefixes[level]
                        - identity_quantized_prefixes[level]
                    )
                    .square()
                    .mean()
                    .item()
                )
                for level in range(len(centers[family]))
            ]
            condition_rows = []
            for index, condition in enumerate(RGB_CONDITIONS):
                level_changed, prefix_changed = _prefix_change(
                    codes[:1],
                    codes[index : index + 1],
                )
                condition_rows.append(
                    {
                        "condition": condition.name,
                        "descriptor_displacement_mse": float(
                            (
                                normalized[index : index + 1] - identity
                            )
                            .square()
                            .mean()
                            .item()
                        ),
                        "quantized_prefix_displacement_mse": [
                            float(
                                (
                                    quantized_prefixes[level][
                                        index : index + 1
                                    ]
                                    - identity_quantized_prefixes[level]
                                )
                                .square()
                                .mean()
                                .item()
                            )
                            for level in range(len(centers[family]))
                        ],
                        "level_changed": level_changed,
                        "prefix_changed": prefix_changed,
                    }
                )
            measurements.append(
                {
                    "clip_id": clip.clip_id,
                    "family": family,
                    "source": clip.source,
                    "split": clip.split,
                    "natural_displacement_mse": float(
                        (natural_normalized - identity)
                        .square()
                        .mean()
                        .item()
                    ),
                    "natural_quantized_prefix_displacement_mse": (
                        natural_quantized_displacement
                    ),
                    "natural_level_changed": natural_level,
                    "natural_prefix_changed": natural_prefix,
                    "condition_codes": codes.detach().cpu().tolist(),
                    "conditions": condition_rows,
                }
            )
            for row in condition_rows:
                measurements.append(
                    {
                        "clip_id": clip.clip_id,
                        "family": family,
                        "source": clip.source,
                        "split": clip.split,
                        "natural_displacement_mse": float(
                            (natural_normalized - identity)
                            .square()
                            .mean()
                            .item()
                        ),
                        "natural_quantized_prefix_displacement_mse": (
                            natural_quantized_displacement
                        ),
                        "natural_level_changed": natural_level,
                        "natural_prefix_changed": natural_prefix,
                        "condition_codes": codes.detach().cpu().tolist(),
                        **row,
                    }
                )

            if clip.canonical_pooled is not None:
                canonical = clip.canonical_pooled[
                    :EXPECTED_LATENT_TICKS
                ].float()
                canonical_raw = _descriptor(
                    canonical.unsqueeze(0),
                    stride=stride,
                    time_index=TARGET_LATENT_TICK,
                )
                canonical_normalized = (
                    artifact.normalization.normalize(canonical_raw).to(
                        device=target_device,
                        dtype=torch.float32,
                    )
                )
                canonical_codes, _, _ = encode_residual_quantizer(
                    canonical_normalized,
                    centers[family],
                    center_block_size=center_block_size,
                )
                reproduction.append(
                    {
                        "clip_id": clip.clip_id,
                        "family": family,
                        "pooled_mse": float(
                            (pooled[0] - canonical).square().mean().item()
                        ),
                        "pooled_max_abs": float(
                            (pooled[0] - canonical).abs().max().item()
                        ),
                        "level_code_match": (
                            canonical_codes == codes[:1]
                        )
                        .squeeze(0)
                        .cpu()
                        .tolist(),
                    }
                )

    condition_measurements = [
        row for row in measurements if "condition" in row
    ]
    clip_measurements = [
        row for row in measurements if "condition" not in row
    ]
    summaries = _summarize_measurements(
        clips=clips,
        artifacts=artifacts,
        measurements=condition_measurements,
        reproduction=reproduction,
    )
    return {
        **summaries,
        "clips": [
            {
                "clip_id": clip.clip_id,
                "source": clip.source,
                "split": clip.split,
                "scene_id": clip.scene_id,
                "task_id": clip.task_id,
                "metadata": clip.metadata,
            }
            for clip in clips
        ],
        "sample_measurements": clip_measurements,
    }


def _select_droid_plans(
    pooled_manifest: EpisodeManifest,
    *,
    split: str,
    max_samples: int,
) -> tuple[EpisodeRecord, ...]:
    if split not in {"val", "test"}:
        raise ValueError("DROID visual probe split must be val/test.")
    candidates = [
        record
        for record in pooled_manifest
        if record.split == split
        and record.num_steps >= EXPECTED_LATENT_TICKS
        and int(record.metadata["source_range"][1])
        - int(record.metadata["source_range"][0])
        >= CLIP_FRAMES
    ]
    candidates.sort(
        key=lambda record: (
            _stable_score(f"visual-perturbation-v1|{record.key}"),
            record.key,
        )
    )
    selected: list[EpisodeRecord] = []
    seen_scenes: set[str] = set()
    for record in candidates:
        scene = record.scene_id or record.episode_id
        if scene in seen_scenes:
            continue
        selected.append(record)
        seen_scenes.add(scene)
        if len(selected) == max_samples:
            return tuple(selected)
    for record in candidates:
        if record in selected:
            continue
        selected.append(record)
        if len(selected) == max_samples:
            return tuple(selected)
    if not selected:
        raise ValueError(f"No DROID visual clips are available for {split}.")
    return tuple(selected)


def _load_pooled_references(
    records: Sequence[EpisodeRecord],
    camera: str,
) -> dict[str, torch.Tensor]:
    by_shard: dict[Path, set[str]] = {}
    for record in records:
        path = Path(str(record.metadata["pooled_shard"]))
        by_shard.setdefault(path, set()).add(record.episode_id)
    references: dict[str, torch.Tensor] = {}
    for path, episode_ids in sorted(by_shard.items()):
        payload = load_torch_payload(path, map_location="cpu")
        for episode_payload in payload.get("episodes", ()):
            episode_id = str(episode_payload["episode_id"])
            if episode_id not in episode_ids:
                continue
            episode = PooledFeatureEpisode.from_payload(episode_payload)
            if camera not in episode.camera_ids:
                raise ValueError(
                    f"Pooled episode `{episode_id}` lacks camera `{camera}`."
                )
            camera_index = episode.camera_ids.index(camera)
            references[episode_id] = episode.pooled_g4[
                :EXPECTED_LATENT_TICKS,
                camera_index,
            ].float()
    missing = sorted(
        {record.episode_id for record in records} - set(references)
    )
    if missing:
        raise RuntimeError(
            f"DROID visual pooled references are missing: {missing[:8]}."
        )
    return references


def _droid_clip_plan(
    records: Sequence[EpisodeRecord],
) -> list[dict[str, Any]]:
    return [
        {
            "clip_id": record.episode_id,
            "split": record.split,
            "scene_id": record.scene_id,
            "task_ids": list(record.task_ids),
            "parent_manifest_key": record.metadata["parent_manifest_key"],
            "source_shard": record.metadata["source_shard"],
            "source_range": record.metadata["source_range"],
            "pooled_shard": record.metadata["pooled_shard"],
        }
        for record in records
    ]


def _load_droid_clips(
    *,
    records: Sequence[EpisodeRecord],
    source_manifest: EpisodeManifest,
    data_dir: str | Path,
    camera: str,
) -> list[VisualClip]:
    source_by_key = {record.key: record for record in source_manifest}
    requests: dict[str, set[int]] = {}
    for record in records:
        parent_key = str(record.metadata["parent_manifest_key"])
        if parent_key not in source_by_key:
            raise KeyError(
                f"DROID pooled record references unknown `{parent_key}`."
            )
        start = int(record.metadata["source_range"][0])
        requests.setdefault(parent_key, set()).update(
            range(start, start + CLIP_FRAMES)
        )
    decoded = read_manifest_droid_rlds_frames(
        data_dir,
        source_manifest,
        requests,
        camera=camera,
    )
    references = _load_pooled_references(records, camera)
    clips = []
    for record in records:
        parent_key = str(record.metadata["parent_manifest_key"])
        start = int(record.metadata["source_range"][0])
        frames = torch.stack(
            [
                decoded[(parent_key, index)]
                for index in range(start, start + CLIP_FRAMES)
            ]
        )
        clips.append(
            VisualClip(
                clip_id=record.episode_id,
                source="droid",
                split=str(record.split),
                scene_id=record.scene_id,
                task_id=record.task_ids[0] if record.task_ids else None,
                frames=frames,
                canonical_pooled=references[record.episode_id],
                metadata={
                    "parent_manifest_key": parent_key,
                    "source_range": [start, start + CLIP_FRAMES],
                    "source_shard": record.metadata["source_shard"],
                },
            )
        )
    return clips


def _select_libero_plans(
    root: str | Path,
    *,
    suites: Sequence[str],
    max_samples: int,
) -> tuple[tuple[Path, str, int], ...]:
    root = Path(root)
    candidates: list[tuple[Path, str, int]] = []
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("LIBERO visual probe requires h5py.") from exc
    for suite in suites:
        for path in sorted((root / suite).glob("*.hdf5")):
            with h5py.File(path, "r") as handle:
                demos = sorted(handle["data"], key=lambda value: int(value.split("_")[-1]))
                eligible = [
                    demo
                    for demo in demos
                    if int(
                        handle[f"data/{demo}/obs/eye_in_hand_rgb"].shape[0]
                    )
                    >= CLIP_FRAMES
                ]
                if not eligible:
                    continue
                demo = eligible[0]
                actions = torch.from_numpy(
                    handle[f"data/{demo}/actions"][:, :3]
                ).float()
                peak = int(actions.norm(dim=1).argmax().item())
                start = min(
                    max(peak - (CLIP_FRAMES - 5), 0),
                    int(actions.shape[0]) - CLIP_FRAMES,
                )
                candidates.append((path, demo, start))
    candidates.sort(
        key=lambda value: (
            _stable_score(
                f"libero-visual-v1|{value[0]}|{value[1]}|{value[2]}"
            ),
            str(value[0]),
        )
    )
    if not candidates:
        raise ValueError("No eligible LIBERO visual clips were found.")
    return tuple(candidates[:max_samples])


def _libero_clip_plan(
    plans: Sequence[tuple[Path, str, int]],
    root: str | Path,
) -> list[dict[str, Any]]:
    root = Path(root)
    return [
        {
            "path": str(path.resolve()),
            "relative_path": str(path.relative_to(root)),
            "demo": demo,
            "start": start,
            "stop": start + CLIP_FRAMES,
            "bytes": path.stat().st_size,
        }
        for path, demo, start in plans
    ]


def _load_libero_clips(
    plans: Sequence[tuple[Path, str, int]],
) -> list[VisualClip]:
    import h5py

    clips = []
    for path, demo, start in plans:
        with h5py.File(path, "r") as handle:
            frames = torch.from_numpy(
                handle[
                    f"data/{demo}/obs/eye_in_hand_rgb"
                ][start : start + CLIP_FRAMES]
            ).to(torch.uint8)
        suite = path.parent.name
        task = path.stem.removesuffix("_demo")
        clips.append(
            VisualClip(
                clip_id=f"{suite}:{path.stem}:{demo}:{start}",
                source="libero",
                split=suite,
                scene_id=task.split("_", 2)[0],
                task_id=task,
                frames=frames,
                metadata={
                    "path": str(path.resolve()),
                    "demo": demo,
                    "start": start,
                    "stop": start + CLIP_FRAMES,
                    "camera": "eye_in_hand_rgb",
                },
            )
        )
    return clips


def probe_rgb_visual_perturbations(
    *,
    source: str,
    artifacts: dict[str, str | Path],
    output_dir: str | Path,
    vae_path: str | Path,
    fastwam_src: str | Path,
    device: str,
    dtype: str = "bfloat16",
    image_height: int = 224,
    image_width: int = 224,
    max_samples: int = 24,
    droid_pooled_manifest: str | Path | None = None,
    droid_source_manifest: str | Path | None = None,
    droid_data_dir: str | Path | None = None,
    droid_split: str = "test",
    droid_camera: str = "wrist_image_left",
    libero_root: str | Path | None = None,
    libero_suites: Sequence[str] = (
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
    ),
    center_block_size: int = 1024,
    resume: bool = True,
) -> dict[str, Any]:
    if source not in {"droid", "libero"}:
        raise ValueError("Visual probe source must be droid or libero.")
    if max_samples <= 0 or center_block_size <= 0:
        raise ValueError("Visual probe sample and block sizes must be positive.")
    artifact_paths = {
        str(label): Path(path) for label, path in sorted(artifacts.items())
    }
    loaded_artifacts = {
        label: FrozenRQArtifact.load(path)
        for label, path in artifact_paths.items()
    }
    if set(loaded_artifacts) != {
        artifact.family for artifact in loaded_artifacts.values()
    }:
        raise ValueError(
            "Visual perturbation labels must match artifact family names."
        )
    if any(
        artifact.descriptor.camera_ids != (droid_camera,)
        for artifact in loaded_artifacts.values()
    ):
        raise ValueError(
            "Visual perturbation artifacts must use the requested wrist camera."
        )

    if source == "droid":
        if not all(
            value is not None
            for value in (
                droid_pooled_manifest,
                droid_source_manifest,
                droid_data_dir,
            )
        ):
            raise ValueError("DROID visual probe paths are incomplete.")
        pooled_manifest_path = Path(str(droid_pooled_manifest))
        source_manifest_path = Path(str(droid_source_manifest))
        pooled_manifest = EpisodeManifest.read_jsonl(pooled_manifest_path)
        source_manifest = EpisodeManifest.read_jsonl(source_manifest_path)
        selected = _select_droid_plans(
            pooled_manifest,
            split=droid_split,
            max_samples=max_samples,
        )
        sample_plan = _droid_clip_plan(selected)
        source_contract = {
            "source": "droid",
            "split": droid_split,
            "camera": droid_camera,
            "pooled_manifest": {
                "path": str(pooled_manifest_path.resolve()),
                "sha256": file_sha256(pooled_manifest_path),
                "fingerprint": pooled_manifest.fingerprint(),
            },
            "source_manifest": {
                "path": str(source_manifest_path.resolve()),
                "sha256": file_sha256(source_manifest_path),
                "fingerprint": source_manifest.fingerprint(),
            },
            "data_dir": str(Path(str(droid_data_dir)).resolve()),
            "samples": sample_plan,
        }
    else:
        if libero_root is None:
            raise ValueError("LIBERO visual probe requires a dataset root.")
        plans = _select_libero_plans(
            libero_root,
            suites=libero_suites,
            max_samples=max_samples,
        )
        sample_plan = _libero_clip_plan(plans, libero_root)
        source_contract = {
            "source": "libero",
            "root": str(Path(libero_root).resolve()),
            "suites": list(libero_suites),
            "samples": sample_plan,
        }

    vae_path = Path(vae_path)
    implementation_sha256 = {
        "visual_perturbations": file_sha256(Path(__file__)),
        "wan_probe_export": file_sha256(
            Path(__file__).with_name("wan_probe_export.py")
        ),
    }
    contract_payload = {
        "schema": VISUAL_PERTURBATION_CONTRACT_SCHEMA,
        "source": source_contract,
        "artifacts": {
            label: {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for label, path in artifact_paths.items()
        },
        "vae": {
            "path": str(vae_path.resolve()),
            "sha256": file_sha256(vae_path),
            "bytes": vae_path.stat().st_size,
        },
        "fastwam_src": str(Path(fastwam_src).resolve()),
        "device": device,
        "dtype": dtype,
        "image_height": image_height,
        "image_width": image_width,
        "target_latent_tick": TARGET_LATENT_TICK,
        "clip_frames": CLIP_FRAMES,
        "endpoint_frame_range": [
            ENDPOINT_FRAME_START,
            ENDPOINT_FRAME_STOP,
        ],
        "conditions": [
            _condition_payload(condition) for condition in RGB_CONDITIONS
        ],
        "center_block_size": center_block_size,
        "implementation_sha256": implementation_sha256,
    }
    contract_hash = _canonical_hash(contract_payload)
    contract = {**contract_payload, "contract_hash": contract_hash}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    report_path = output_dir / "visual_perturbation_report.json"
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != contract:
            raise RuntimeError("Existing visual perturbation contract differs.")
        if not resume:
            raise FileExistsError(
                f"Visual perturbation contract exists at `{contract_path}`."
            )
    else:
        write_json_report(contract_path, contract)
    if resume and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("contract_hash") != contract_hash:
            raise RuntimeError(
                "Visual perturbation report contract hash is invalid."
            )
        return report
    if report_path.exists():
        raise FileExistsError(
            f"Visual perturbation report exists at `{report_path}`."
        )

    if source == "droid":
        clips = _load_droid_clips(
            records=selected,
            source_manifest=source_manifest,
            data_dir=str(droid_data_dir),
            camera=droid_camera,
        )
    else:
        clips = _load_libero_clips(plans)

    vae_config = WanProbeExportConfig(
        data_dir=str(
            droid_data_dir if source == "droid" else libero_root
        ),
        output_dir=str(output_dir),
        vae_path=str(vae_path),
        fastwam_src=str(fastwam_src),
        max_episodes=1,
        cameras=(droid_camera,),
        image_height=image_height,
        image_width=image_width,
        device=device,
        dtype=dtype,
        resume=resume,
        hash_source_shards=False,
    )
    vae = _load_wan_vae(vae_config)
    result = run_visual_perturbation_probe(
        clips=clips,
        artifacts=loaded_artifacts,
        vae=vae,
        device=device,
        dtype=dtype,
        image_height=image_height,
        image_width=image_width,
        center_block_size=center_block_size,
    )
    report = {
        "schema": VISUAL_PERTURBATION_REPORT_SCHEMA,
        "contract_hash": contract_hash,
        "source": source,
        "conditions": [
            _condition_payload(condition) for condition in RGB_CONDITIONS
        ],
        "interpretation": [
            "Photometric rows measure nuisance sensitivity, not desired motion.",
            "Uniform geometry is a camera/global-scene reference; endpoint "
            "geometry is a synthetic motion intervention.",
            "Code changes show frozen representation sensitivity and do not "
            "prove object-level causal semantics.",
            "LIBERO uses the DROID-trained wrist codebooks as an out-of-domain "
            "stress test; low coverage is not a LIBERO-trained result.",
        ],
        **result,
    }
    write_json_report(report_path, report)
    return report
