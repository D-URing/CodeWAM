from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from codewam.codebook_eval.shards import file_sha256
from codewam.codebook_eval.streaming import (
    FrozenRQArtifact,
    encode_residual_quantizer,
)


DEFAULT_CODE_FAMILIES = ("Q2", "Q3", "Q5")
CHART_IDENTITY_KEYS = (
    "manifest_fingerprint",
    "dataset_revision",
    "wan_model_id",
    "wan_revision",
    "preprocess_revision",
    "source_checksums",
)


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FrozenArtifactChart:
    name: str
    artifacts: tuple[FrozenRQArtifact, ...]
    artifact_sha256: tuple[str, ...]
    artifact_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Frozen artifact chart name must not be empty.")
        if not self.artifacts:
            raise ValueError("A frozen artifact chart needs at least one family.")
        if len(self.artifacts) != len(self.artifact_sha256) or len(
            self.artifacts
        ) != len(self.artifact_paths):
            raise ValueError("Frozen artifact paths and hashes are not family-aligned.")
        families = tuple(artifact.family for artifact in self.artifacts)
        if len(set(families)) != len(families):
            raise ValueError(f"Chart `{self.name}` contains duplicate families.")

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(artifact.family for artifact in self.artifacts)

    def compact_identity(self) -> dict[str, Any]:
        rows = []
        for artifact, sha256, path in zip(
            self.artifacts,
            self.artifact_sha256,
            self.artifact_paths,
        ):
            metadata = artifact.metadata
            rows.append(
                {
                    "family": artifact.family,
                    "file_name": Path(path).name,
                    "sha256": sha256,
                    "config_hash": metadata["config_hash"],
                    "descriptor": {
                        "stride": artifact.descriptor.stride,
                        "pool": artifact.descriptor.pool,
                        "max_gap_factor": artifact.descriptor.max_gap_factor,
                        "camera_ids": (
                            None
                            if artifact.descriptor.camera_ids is None
                            else list(artifact.descriptor.camera_ids)
                        ),
                    },
                    "levels": len(artifact.centers),
                    "codebook_sizes": [
                        int(centers.shape[0]) for centers in artifact.centers
                    ],
                    "descriptor_dim": artifact.normalization.dim,
                }
            )
        reference = self.artifacts[0].metadata
        source_checksums = reference["source_checksums"]
        return {
            "name": self.name,
            "families": rows,
            "training_provenance": {
                key: reference[key]
                for key in CHART_IDENTITY_KEYS
                if key != "source_checksums"
            },
            "source_checksums_count": len(source_checksums),
            "source_checksums_hash": _canonical_hash(source_checksums),
        }


def load_frozen_artifact_chart(
    name: str,
    paths: Mapping[str, str | Path],
    *,
    families: Sequence[str] = DEFAULT_CODE_FAMILIES,
) -> FrozenArtifactChart:
    ordered_families = tuple(str(value) for value in families)
    if set(paths) != set(ordered_families):
        raise ValueError(
            f"Artifact paths {sorted(paths)} do not match {list(ordered_families)}."
        )
    resolved = tuple(Path(paths[family]).resolve() for family in ordered_families)
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen RQ artifacts: {missing}.")
    artifacts = tuple(FrozenRQArtifact.load(path) for path in resolved)
    if tuple(artifact.family for artifact in artifacts) != ordered_families:
        raise ValueError("Frozen artifact file/family order does not match its label.")
    reference = {
        key: artifacts[0].metadata[key] for key in CHART_IDENTITY_KEYS
    }
    for artifact in artifacts[1:]:
        identity = {
            key: artifact.metadata[key] for key in CHART_IDENTITY_KEYS
        }
        if identity != reference:
            changed = [
                key
                for key in CHART_IDENTITY_KEYS
                if identity[key] != reference[key]
            ]
            raise ValueError(
                f"Chart `{name}` mixes artifact provenance; "
                f"{artifact.family} differs in {changed}."
            )
    level_counts = {len(artifact.centers) for artifact in artifacts}
    if len(level_counts) != 1:
        raise ValueError("All chart families must use one RQ depth.")
    return FrozenArtifactChart(
        name=name,
        artifacts=artifacts,
        artifact_sha256=tuple(file_sha256(path) for path in resolved),
        artifact_paths=tuple(str(path) for path in resolved),
    )


@dataclass(frozen=True)
class FrozenCodeAssignment:
    code_ids: torch.Tensor
    available: torch.Tensor
    descriptor_source_indices: torch.Tensor
    families: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.code_ids.ndim != 3 or self.code_ids.dtype != torch.long:
            raise ValueError("Assigned code IDs must be long [T,F,L].")
        if (
            self.available.dtype != torch.bool
            or tuple(self.available.shape) != tuple(self.code_ids.shape[:2])
        ):
            raise ValueError("Assigned availability must be bool [T,F].")
        if (
            self.descriptor_source_indices.dtype != torch.long
            or tuple(self.descriptor_source_indices.shape)
            != (*self.code_ids.shape[:2], 3)
        ):
            raise ValueError(
                "Descriptor source indices must be long [T,F,3]."
            )
        if self.code_ids.shape[1] != len(self.families):
            raise ValueError("Assigned families do not match the code tensor.")
        unavailable = ~self.available
        if unavailable.any() and not torch.all(
            self.code_ids[unavailable] == -1
        ):
            raise ValueError("Unavailable code families must retain ID -1.")
        if unavailable.any() and not torch.all(
            self.descriptor_source_indices[unavailable] == -1
        ):
            raise ValueError(
                "Unavailable code families must retain source index -1."
            )


class FrozenCausalCodeAssigner:
    """Deterministic causal Q2/Q3/Q5 assignment with no codebook updates."""

    def __init__(
        self,
        chart: FrozenArtifactChart,
        *,
        center_block_size: int = 1024,
    ):
        if center_block_size <= 0:
            raise ValueError("Assignment center block size must be positive.")
        self.chart = chart
        self.center_block_size = int(center_block_size)
        self.levels = len(chart.artifacts[0].centers)

    @torch.no_grad()
    def assign(
        self,
        latents: torch.Tensor,
        *,
        latent_source_indices: torch.Tensor,
        camera_ids: Sequence[str],
        latent_valid: torch.Tensor | None = None,
        timestamps: torch.Tensor | None = None,
    ) -> FrozenCodeAssignment:
        if latents.ndim != 5:
            raise ValueError(
                f"Latents must be [T,V,C,H,W], got {tuple(latents.shape)}."
            )
        ticks, views, _, height, width = latents.shape
        if (
            latent_source_indices.dtype != torch.long
            or tuple(latent_source_indices.shape) != (ticks,)
        ):
            raise ValueError("Latent source indices must be long [T].")
        if ticks > 1 and not torch.all(
            latent_source_indices[1:] > latent_source_indices[:-1]
        ):
            raise ValueError("Latent source indices must be strictly increasing.")
        ordered_cameras = tuple(str(value) for value in camera_ids)
        if len(ordered_cameras) != views or len(set(ordered_cameras)) != views:
            raise ValueError("Latent camera IDs must be unique and view-aligned.")
        if latent_valid is None:
            latent_valid = torch.ones(
                (ticks, views),
                dtype=torch.bool,
                device=latents.device,
            )
        if (
            latent_valid.dtype != torch.bool
            or tuple(latent_valid.shape) != (ticks, views)
        ):
            raise ValueError("Latent validity must be bool [T,V].")
        if timestamps is None:
            timestamps = latent_source_indices.to(dtype=torch.float64)
        if tuple(timestamps.shape) != (ticks,) or not torch.isfinite(
            timestamps
        ).all():
            raise ValueError("Latent timestamps must be finite [T].")
        if ticks > 1 and not torch.all(timestamps[1:] > timestamps[:-1]):
            raise ValueError("Latent timestamps must be strictly increasing.")

        device = latents.device
        families = self.chart.families
        code_ids = torch.full(
            (ticks, len(families), self.levels),
            -1,
            dtype=torch.long,
            device=device,
        )
        available = torch.zeros(
            (ticks, len(families)),
            dtype=torch.bool,
            device=device,
        )
        source_indices = torch.full(
            (ticks, len(families), 3),
            -1,
            dtype=torch.long,
            device=device,
        )
        cadence = (
            torch.median(timestamps[1:] - timestamps[:-1])
            if ticks > 1
            else timestamps.new_tensor(1.0)
        )

        pooled_cache: dict[tuple[int, tuple[int, ...]], torch.Tensor] = {}
        for family_index, artifact in enumerate(self.chart.artifacts):
            spec = artifact.descriptor
            selected_cameras = (
                ordered_cameras
                if spec.camera_ids is None
                else spec.camera_ids
            )
            missing = [
                camera
                for camera in selected_cameras
                if camera not in ordered_cameras
            ]
            if missing:
                raise ValueError(
                    f"Latents lack {artifact.family} cameras {missing}; "
                    f"available={ordered_cameras}."
                )
            view_indices = tuple(
                ordered_cameras.index(camera) for camera in selected_cameras
            )
            cache_key = (spec.pool, view_indices)
            features = pooled_cache.get(cache_key)
            if features is None:
                selected = latents[:, view_indices].float()
                flat = selected.reshape(
                    ticks * len(view_indices),
                    selected.shape[2],
                    height,
                    width,
                )
                pooled = F.adaptive_avg_pool2d(
                    flat,
                    output_size=(spec.pool, spec.pool),
                )
                features = pooled.reshape(ticks, -1).contiguous()
                pooled_cache[cache_key] = features

            stride = spec.stride
            if ticks <= 2 * stride:
                continue
            current = torch.arange(
                2 * stride,
                ticks,
                dtype=torch.long,
                device=device,
            )
            selected_valid = latent_valid[:, view_indices]
            valid = (
                selected_valid[current - 2 * stride].all(dim=1)
                & selected_valid[current - stride].all(dim=1)
                & selected_valid[current].all(dim=1)
            )
            if spec.max_gap_factor is not None:
                maximum_gap = (
                    cadence * stride * float(spec.max_gap_factor)
                )
                valid &= (
                    timestamps[current - stride]
                    - timestamps[current - 2 * stride]
                    <= maximum_gap
                ) & (
                    timestamps[current] - timestamps[current - stride]
                    <= maximum_gap
                )
            current = current[valid]
            if current.numel() == 0:
                continue
            descriptor = torch.cat(
                (
                    features[current - 2 * stride],
                    features[current - stride],
                    features[current],
                ),
                dim=1,
            )
            if descriptor.shape[1] != artifact.normalization.dim:
                raise ValueError(
                    f"{artifact.family} descriptor dim {descriptor.shape[1]} "
                    f"does not match artifact dim {artifact.normalization.dim}."
                )
            normalized = artifact.normalization.normalize(descriptor)
            codes, _, _ = encode_residual_quantizer(
                normalized,
                artifact.centers,
                center_block_size=self.center_block_size,
            )
            code_ids[current, family_index] = codes
            available[current, family_index] = True
            source_indices[current, family_index] = torch.stack(
                (
                    latent_source_indices[current - 2 * stride],
                    latent_source_indices[current - stride],
                    latent_source_indices[current],
                ),
                dim=1,
            )
        return FrozenCodeAssignment(
            code_ids=code_ids,
            available=available,
            descriptor_source_indices=source_indices,
            families=families,
        )
