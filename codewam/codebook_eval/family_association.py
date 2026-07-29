from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import torch

from codewam.data.droid_manifest import write_json_report

from .association import (
    _episode_factory,
    _resolve_device,
    _selected_features,
    _spatial_moments,
)
from .manifest import EpisodeManifest
from .shards import (
    PooledFeatureEpisode,
    expand_shard_paths,
    file_sha256,
)
from .streaming import FrozenRQArtifact, encode_residual_quantizer


FAMILY_ASSOCIATION_CONTRACT_SCHEMA = (
    "codewam.rq-family-association-contract.v1"
)
FAMILY_ASSOCIATION_REPORT_SCHEMA = (
    "codewam.rq-family-association-report.v1"
)


def _target_definitions(future_offset: int) -> dict[str, str]:
    return {
        "current_action": (
            "Action recorded at the shared current latent tick. Every code "
            "input contains only visual states up to this tick."
        ),
        "common_future_proprio_change": (
            f"Proprioception {future_offset} latent tick(s) in the future "
            "minus current proprioception, shared by every family model."
        ),
        "common_future_latent_moment_change": (
            f"Selected Wan latent spatial moments {future_offset} latent "
            "tick(s) in the future minus current moments, shared by every "
            "family model."
        ),
    }


def _validate_family_artifacts(
    artifacts: dict[str, FrozenRQArtifact],
) -> tuple[tuple[str, ...], int]:
    if len(artifacts) < 2:
        raise ValueError("Family association requires at least two artifacts.")
    labels = tuple(
        sorted(
            artifacts,
            key=lambda label: (
                artifacts[label].descriptor.stride,
                label,
            ),
        )
    )
    reference = artifacts[labels[0]]
    reference_cameras = reference.descriptor.camera_ids
    levels = len(reference.centers)
    strides = []
    for label in labels:
        artifact = artifacts[label]
        descriptor = artifact.descriptor
        if descriptor.pool != reference.descriptor.pool:
            raise ValueError("Family artifacts must use the same spatial pool.")
        if descriptor.camera_ids != reference_cameras:
            raise ValueError("Family artifacts must use the same camera ids.")
        if len(artifact.centers) != levels:
            raise ValueError("Family artifacts must use the same RQ depth.")
        strides.append(descriptor.stride)
    if len(strides) != len(set(strides)):
        raise ValueError("Family artifacts must use distinct temporal strides.")
    return labels, levels


def _episode_aligned_probe_values(
    episode: PooledFeatureEpisode,
    artifacts: dict[str, FrozenRQArtifact],
    *,
    future_offset: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    if future_offset <= 0:
        raise ValueError("Family association future offset must be positive.")
    if episode.action is None or episode.proprio is None:
        raise ValueError(
            f"Episode `{episode.episode_id}` lacks action or proprio targets."
        )
    labels, _ = _validate_family_artifacts(artifacts)
    reference = artifacts[labels[0]].descriptor
    maximum_stride = max(
        artifacts[label].descriptor.stride for label in labels
    )
    if episode.ticks <= 2 * maximum_stride + future_offset:
        return {}, {}

    pooled, valid_mask = _selected_features(episode, reference)
    features = pooled.reshape(episode.ticks, -1)
    moments = _spatial_moments(pooled)
    current = torch.arange(
        2 * maximum_stride,
        episode.ticks - future_offset,
        dtype=torch.long,
        device=features.device,
    )
    valid_mask = valid_mask.to(device=features.device)
    valid = valid_mask[current + future_offset].all(dim=1)
    timestamps = episode.timestamps.to(device=features.device)

    for label in labels:
        spec = artifacts[label].descriptor
        stride = spec.stride
        valid &= (
            valid_mask[current - 2 * stride].all(dim=1)
            & valid_mask[current - stride].all(dim=1)
            & valid_mask[current].all(dim=1)
        )
        if spec.max_gap_factor is not None and episode.ticks > 1:
            cadence = torch.median(timestamps[1:] - timestamps[:-1])
            maximum_gap = cadence * stride * float(spec.max_gap_factor)
            valid &= (
                timestamps[current - stride]
                - timestamps[current - 2 * stride]
                <= maximum_gap
            )
            valid &= (
                timestamps[current] - timestamps[current - stride]
                <= maximum_gap
            )

    gap_factors = [
        artifacts[label].descriptor.max_gap_factor for label in labels
    ]
    if episode.ticks > 1 and all(value is not None for value in gap_factors):
        cadence = torch.median(timestamps[1:] - timestamps[:-1])
        future_gap = (
            cadence
            * future_offset
            * min(float(value) for value in gap_factors if value is not None)
        )
        valid &= (
            timestamps[current + future_offset] - timestamps[current]
            <= future_gap
        )
    current = current[valid]
    if current.numel() == 0:
        return {}, {}

    vectors = {}
    for label in labels:
        stride = artifacts[label].descriptor.stride
        vectors[label] = torch.cat(
            (
                features[current - 2 * stride],
                features[current - stride],
                features[current],
            ),
            dim=1,
        ).contiguous()
    targets = {
        "current_action": episode.action[current].float().contiguous(),
        "common_future_proprio_change": (
            episode.proprio[current + future_offset].float()
            - episode.proprio[current].float()
        ).contiguous(),
        "common_future_latent_moment_change": (
            moments[current + future_offset] - moments[current]
        ).contiguous(),
    }
    return vectors, targets


def _iter_aligned_probe_batches(
    episode_factory: Callable[[], Iterable[PooledFeatureEpisode]],
    artifacts: dict[str, FrozenRQArtifact],
    *,
    future_offset: int,
    batch_size: int,
) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
    labels, _ = _validate_family_artifacts(artifacts)
    vector_parts: dict[str, list[torch.Tensor]] = {
        label: [] for label in labels
    }
    target_parts: dict[str, list[torch.Tensor]] = {}
    pending = 0

    def emit() -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        nonlocal vector_parts, target_parts, pending
        result = (
            {
                label: torch.cat(parts, dim=0).contiguous()
                for label, parts in vector_parts.items()
            },
            {
                name: torch.cat(parts, dim=0).contiguous()
                for name, parts in target_parts.items()
            },
        )
        vector_parts = {label: [] for label in labels}
        target_parts = {}
        pending = 0
        return result

    for episode in episode_factory():
        vectors, targets = _episode_aligned_probe_values(
            episode,
            artifacts,
            future_offset=future_offset,
        )
        if not vectors:
            continue
        rows = next(iter(vectors.values())).shape[0]
        offset = 0
        while offset < rows:
            take = min(batch_size - pending, rows - offset)
            for label in labels:
                vector_parts[label].append(
                    vectors[label][offset : offset + take]
                )
            for name, values in targets.items():
                target_parts.setdefault(name, []).append(
                    values[offset : offset + take]
                )
            pending += take
            offset += take
            if pending == batch_size:
                yield emit()
    if pending:
        yield emit()


def _prefix_keys(
    codes: torch.Tensor,
    *,
    k: int,
    depth: int,
) -> torch.Tensor:
    codes = codes.detach().long().cpu()
    if codes.ndim != 2 or depth <= 0 or depth > codes.shape[1]:
        raise ValueError("Code matrix does not contain the requested prefix.")
    keys = torch.zeros(codes.shape[0], dtype=torch.long)
    for level in range(depth):
        keys = keys * int(k) + codes[:, level]
    return keys


class _AdditiveCodeStatistics:
    def __init__(
        self,
        capacities: dict[str, int],
        target_dimensions: dict[str, int],
    ) -> None:
        if len(capacities) < 2:
            raise ValueError("Additive statistics require two code families.")
        self.labels = tuple(capacities)
        self.capacities = dict(capacities)
        self.target_dimensions = dict(target_dimensions)
        self.total = 0
        self.counts = {
            label: torch.zeros(capacity, dtype=torch.long)
            for label, capacity in capacities.items()
        }
        self.cross_counts = {
            (left, right): torch.zeros(
                (capacities[left], capacities[right]),
                dtype=torch.long,
            )
            for left, right in itertools.combinations(self.labels, 2)
        }
        self.global_sums = {
            name: torch.zeros(dimension, dtype=torch.float64)
            for name, dimension in target_dimensions.items()
        }
        self.global_square_sums = {
            name: torch.zeros(dimension, dtype=torch.float64)
            for name, dimension in target_dimensions.items()
        }
        self.target_sums = {
            name: {
                label: torch.zeros(
                    (capacities[label], dimension),
                    dtype=torch.float64,
                )
                for label in self.labels
            }
            for name, dimension in target_dimensions.items()
        }

    def update(
        self,
        keys: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> None:
        if set(keys) != set(self.labels):
            raise ValueError("Additive keys do not match code families.")
        if set(targets) != set(self.target_dimensions):
            raise ValueError("Additive targets do not match target schema.")
        rows = next(iter(keys.values())).shape[0]
        if rows <= 0 or any(value.shape != (rows,) for value in keys.values()):
            raise ValueError("Additive keys must be nonempty aligned vectors.")
        ones = torch.ones(rows, dtype=torch.long)
        cpu_keys = {
            label: value.detach().long().cpu()
            for label, value in keys.items()
        }
        for label, key in cpu_keys.items():
            if int(key.min()) < 0 or int(key.max()) >= self.capacities[label]:
                raise ValueError(f"Additive key is out of range for `{label}`.")
            self.counts[label].index_add_(0, key, ones)
        for left, right in itertools.combinations(self.labels, 2):
            flat = (
                cpu_keys[left] * self.capacities[right]
                + cpu_keys[right]
            )
            pair_counts = torch.bincount(
                flat,
                minlength=self.capacities[left] * self.capacities[right],
            ).reshape(self.capacities[left], self.capacities[right])
            self.cross_counts[(left, right)] += pair_counts

        for name, values in targets.items():
            values = values.detach().double().cpu()
            expected = (rows, self.target_dimensions[name])
            if tuple(values.shape) != expected:
                raise ValueError(
                    f"Target `{name}` must be {expected}, got "
                    f"{tuple(values.shape)}."
                )
            self.global_sums[name] += values.sum(dim=0)
            self.global_square_sums[name] += values.square().sum(dim=0)
            for label, key in cpu_keys.items():
                self.target_sums[name][label].index_add_(0, key, values)
        self.total += int(rows)

    def global_statistics(
        self,
        target: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.total <= 0:
            raise ValueError("Cannot finalize empty additive statistics.")
        mean = self.global_sums[target] / float(self.total)
        variance = (
            self.global_square_sums[target] / float(self.total)
            - mean.square()
        ).clamp_min_(0.0)
        effective = variance > 1e-10
        if not effective.any():
            raise ValueError(f"Joint target `{target}` has no train variance.")
        return mean.float(), variance.float(), effective

    def fit(
        self,
        subset: tuple[str, ...],
        *,
        ridge: float,
        device: torch.device,
    ) -> dict[str, dict[str, torch.Tensor]]:
        if (
            not subset
            or len(subset) != len(set(subset))
            or any(label not in self.labels for label in subset)
        ):
            raise ValueError("Invalid additive family subset.")
        if ridge <= 0:
            raise ValueError("Additive ridge must be positive.")
        offsets = {}
        total_features = 0
        for label in subset:
            offsets[label] = total_features
            total_features += self.capacities[label]
        matrix = torch.zeros(
            (total_features, total_features),
            dtype=torch.float32,
            device=device,
        )
        for label in subset:
            start = offsets[label]
            capacity = self.capacities[label]
            diagonal = self.counts[label].float().to(device)
            matrix[
                start : start + capacity,
                start : start + capacity,
            ].diagonal().copy_(diagonal)
        for left, right in itertools.combinations(subset, 2):
            pair = (left, right)
            transpose = False
            if pair not in self.cross_counts:
                pair = (right, left)
                transpose = True
            counts = self.cross_counts[pair]
            if transpose:
                counts = counts.T
            left_start = offsets[left]
            right_start = offsets[right]
            block = counts.float().to(device)
            matrix[
                left_start : left_start + self.capacities[left],
                right_start : right_start + self.capacities[right],
            ] = block
            matrix[
                right_start : right_start + self.capacities[right],
                left_start : left_start + self.capacities[left],
            ] = block.T
        matrix.diagonal().add_(float(ridge))

        target_names = tuple(sorted(self.target_dimensions))
        target_slices = {}
        total_targets = 0
        for name in target_names:
            target_slices[name] = slice(
                total_targets,
                total_targets + self.target_dimensions[name],
            )
            total_targets += self.target_dimensions[name]
        right_hand_side = torch.zeros(
            (total_features, total_targets),
            dtype=torch.float32,
            device=device,
        )
        for name in target_names:
            mean, _, _ = self.global_statistics(name)
            target_slice = target_slices[name]
            for label in subset:
                start = offsets[label]
                capacity = self.capacities[label]
                centered = (
                    self.target_sums[name][label].float()
                    - self.counts[label].float().unsqueeze(1)
                    * mean.unsqueeze(0)
                )
                right_hand_side[
                    start : start + capacity,
                    target_slice,
                ] = centered.to(device)
        coefficients = torch.linalg.solve(matrix, right_hand_side).cpu()
        return {
            name: {
                label: coefficients[
                    offsets[label] : offsets[label]
                    + self.capacities[label],
                    target_slices[name],
                ].contiguous()
                for label in subset
            }
            for name in target_names
        }


@dataclass
class _AdditiveRegressionAccumulator:
    dimension: int
    effective_dimensions: int
    family_count: int
    vectors: int = 0
    raw_sse: float = 0.0
    baseline_raw_sse: float = 0.0
    normalized_sse: float = 0.0
    baseline_normalized_sse: float = 0.0
    all_features_seen: int = 0
    any_feature_seen: int = 0

    def update(
        self,
        *,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        seen: torch.Tensor,
        global_mean: torch.Tensor,
        variance: torch.Tensor,
        effective: torch.Tensor,
    ) -> None:
        targets = targets.detach().float().cpu()
        predictions = predictions.detach().float().cpu()
        seen = seen.detach().bool().cpu()
        error = (targets - predictions).square()
        baseline_error = (targets - global_mean.unsqueeze(0)).square()
        self.vectors += int(targets.shape[0])
        self.raw_sse += float(error.sum().item())
        self.baseline_raw_sse += float(baseline_error.sum().item())
        self.normalized_sse += float(
            (error[:, effective] / variance[effective]).sum().item()
        )
        self.baseline_normalized_sse += float(
            (
                baseline_error[:, effective]
                / variance[effective]
            ).sum().item()
        )
        self.all_features_seen += int(seen.all(dim=1).sum().item())
        self.any_feature_seen += int(seen.any(dim=1).sum().item())

    def row(self) -> dict[str, Any]:
        if self.vectors <= 0:
            raise ValueError("Cannot finalize an empty additive split.")
        raw_denominator = float(self.vectors * self.dimension)
        normalized_denominator = float(
            self.vectors * self.effective_dimensions
        )
        normalized_mse = self.normalized_sse / normalized_denominator
        baseline_normalized_mse = (
            self.baseline_normalized_sse / normalized_denominator
        )
        return {
            "vectors": self.vectors,
            "target_dimension": self.dimension,
            "effective_target_dimensions": self.effective_dimensions,
            "raw_mse": self.raw_sse / raw_denominator,
            "global_baseline_raw_mse": (
                self.baseline_raw_sse / raw_denominator
            ),
            "normalized_mse": normalized_mse,
            "global_baseline_normalized_mse": baseline_normalized_mse,
            "normalized_mse_reduction": (
                1.0
                - normalized_mse
                / max(baseline_normalized_mse, 1e-12)
            ),
            "all_family_code_coverage": (
                self.all_features_seen / float(self.vectors)
            ),
            "any_family_code_coverage": (
                self.any_feature_seen / float(self.vectors)
            ),
        }


def _family_subsets(labels: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        subset
        for size in range(1, len(labels) + 1)
        for subset in itertools.combinations(labels, size)
    )


def _validate_depth_profiles(
    profiles: dict[str, dict[str, int]] | None,
    *,
    labels: tuple[str, ...],
    levels: int,
    artifacts: dict[str, FrozenRQArtifact],
    max_pair_cells: int,
) -> dict[str, dict[str, int]]:
    result = {}
    for name, configured_depths in (profiles or {}).items():
        if not name:
            raise ValueError("Depth profile names must be nonempty.")
        if set(configured_depths) != set(labels):
            raise ValueError(
                f"Depth profile `{name}` must configure every family."
            )
        depths = {
            label: int(configured_depths[label]) for label in labels
        }
        invalid = [
            label
            for label, depth in depths.items()
            if depth <= 0 or depth > levels
        ]
        if invalid:
            raise ValueError(
                f"Depth profile `{name}` has invalid families {invalid}."
            )
        capacities = {
            label: int(artifacts[label].centers[0].shape[0])
            ** depths[label]
            for label in labels
        }
        largest_pair = max(
            capacities[left] * capacities[right]
            for left, right in itertools.combinations(labels, 2)
        )
        if largest_pair > max_pair_cells:
            raise ValueError(
                f"Depth profile `{name}` needs {largest_pair:,} pair cells, "
                f"above max_pair_cells={max_pair_cells:,}."
            )
        result[str(name)] = depths
    return result


def _encode_family_codes(
    vectors: dict[str, torch.Tensor],
    artifacts: dict[str, FrozenRQArtifact],
    centers: dict[str, tuple[torch.Tensor, ...]],
    *,
    device: torch.device,
    center_block_size: int,
) -> dict[str, torch.Tensor]:
    result = {}
    for label, values in vectors.items():
        normalized = artifacts[label].normalization.normalize(values).to(
            device=device,
            dtype=torch.float32,
        )
        codes, _, _ = encode_residual_quantizer(
            normalized,
            centers[label],
            center_block_size=center_block_size,
        )
        result[label] = codes.cpu()
    return result


def _predict_additive(
    keys: dict[str, torch.Tensor],
    coefficients: dict[str, torch.Tensor],
    global_mean: torch.Tensor,
    counts: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = tuple(coefficients)
    rows = next(iter(keys.values())).shape[0]
    predictions = global_mean.expand(rows, -1).clone()
    seen = []
    for label in labels:
        predictions += coefficients[label][keys[label]]
        seen.append(counts[label][keys[label]] > 0)
    return predictions, torch.stack(seen, dim=1)


def _write_contract(
    path: Path,
    contract: dict[str, Any],
    *,
    resume: bool,
) -> None:
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != contract:
            raise RuntimeError(
                f"Existing family association contract differs from `{path}`."
            )
        if not resume:
            raise FileExistsError(
                f"Family association contract exists at `{path}`."
            )
        return
    write_json_report(path, contract)


def probe_codebook_family_contributions(
    *,
    manifest_path: str | Path,
    pooled_shards: Iterable[str | Path],
    artifacts: dict[str, str | Path],
    output_dir: str | Path,
    splits: tuple[str, ...] = ("val", "test"),
    future_offset: int = 1,
    ridge: float = 8.0,
    max_pair_cells: int = 2_000_000,
    depth_profiles: dict[str, dict[str, int]] | None = None,
    device: str = "auto",
    cpu_threads: int = 4,
    batch_size: int = 8192,
    center_block_size: int = 1024,
    resume: bool = True,
) -> dict[str, Any]:
    if (
        cpu_threads <= 0
        or batch_size <= 0
        or center_block_size <= 0
        or future_offset <= 0
        or ridge <= 0
        or max_pair_cells <= 0
    ):
        raise ValueError("Family association numeric settings must be positive.")
    if not splits or any(split not in {"val", "test"} for split in splits):
        raise ValueError("Family association splits must be val/test.")
    if len(splits) != len(set(splits)):
        raise ValueError("Family association splits must be unique.")
    if len(artifacts) < 2 or any(not label for label in artifacts):
        raise ValueError(
            "Family association artifacts must have two or more labels."
        )
    torch.set_num_threads(int(cpu_threads))

    manifest_path = Path(manifest_path)
    manifest = EpisodeManifest.read_jsonl(manifest_path)
    manifest.assert_group_isolation("scene")
    manifest_fingerprint = manifest.fingerprint()
    shard_paths = tuple(expand_shard_paths(pooled_shards))
    shard_checksums = [file_sha256(path) for path in shard_paths]
    artifact_paths = {
        str(label): Path(path) for label, path in artifacts.items()
    }
    loaded_artifacts = {
        label: FrozenRQArtifact.load(path)
        for label, path in artifact_paths.items()
    }
    labels, levels = _validate_family_artifacts(loaded_artifacts)
    loaded_artifacts = {
        label: loaded_artifacts[label] for label in labels
    }
    for label, artifact in loaded_artifacts.items():
        expected = {
            "manifest_fingerprint": manifest_fingerprint,
            "source_checksums": shard_checksums,
        }
        mismatches = [
            key
            for key, value in expected.items()
            if artifact.metadata.get(key) != value
        ]
        if mismatches:
            raise RuntimeError(
                f"Family artifact `{label}` differs in {mismatches}."
            )
    for depth in range(1, levels + 1):
        capacities = {
            label: int(loaded_artifacts[label].centers[0].shape[0])
            ** depth
            for label in labels
        }
        largest_pair = max(
            capacities[left] * capacities[right]
            for left, right in itertools.combinations(labels, 2)
        )
        if largest_pair > max_pair_cells:
            raise ValueError(
                f"RQ prefix L{depth} needs {largest_pair:,} pair cells, "
                f"above max_pair_cells={max_pair_cells:,}."
            )
    selected_profiles = _validate_depth_profiles(
        depth_profiles,
        labels=labels,
        levels=levels,
        artifacts=loaded_artifacts,
        max_pair_cells=max_pair_cells,
    )

    expected_by_split = {
        split: {
            record.episode_id
            for record in manifest
            if record.split == split
        }
        for split in ("train", *splits)
    }
    empty = [
        split
        for split, identifiers in expected_by_split.items()
        if not identifiers
    ]
    if empty:
        raise ValueError(
            f"Family association manifest has empty splits {empty}."
        )

    target_definitions = _target_definitions(future_offset)
    implementation_sha256 = {
        "family_association": file_sha256(Path(__file__)),
        "association": file_sha256(Path(__file__).with_name("association.py")),
        "shards": file_sha256(Path(__file__).with_name("shards.py")),
        "streaming": file_sha256(Path(__file__).with_name("streaming.py")),
    }
    contract_payload = {
        "schema": FAMILY_ASSOCIATION_CONTRACT_SCHEMA,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": file_sha256(manifest_path),
            "fingerprint": manifest_fingerprint,
        },
        "pooled_shards": [
            {"path": str(path), "sha256": checksum}
            for path, checksum in zip(shard_paths, shard_checksums)
        ],
        "artifacts": {
            label: {
                "path": str(artifact_paths[label].resolve()),
                "sha256": file_sha256(artifact_paths[label]),
                "family": loaded_artifacts[label].family,
                "stride": loaded_artifacts[label].descriptor.stride,
            }
            for label in labels
        },
        "splits": list(splits),
        "future_offset": future_offset,
        "ridge": ridge,
        "max_pair_cells": max_pair_cells,
        "depth_profiles": selected_profiles,
        "device": device,
        "cpu_threads": cpu_threads,
        "batch_size": batch_size,
        "center_block_size": center_block_size,
        "target_definitions": target_definitions,
        "implementation_sha256": implementation_sha256,
    }
    contract_hash = hashlib.sha256(
        json.dumps(
            contract_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    contract = {**contract_payload, "contract_hash": contract_hash}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    report_path = output_dir / "family_association_report.json"
    _write_contract(contract_path, contract, resume=resume)
    if resume and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("contract_hash") != contract_hash:
            raise RuntimeError(
                "Family association report contract hash is invalid."
            )
        return report
    if report_path.exists():
        raise FileExistsError(
            f"Family association report exists at `{report_path}`."
        )

    target_device = _resolve_device(device)
    centers = {
        label: tuple(
            center.to(device=target_device, dtype=torch.float32)
            for center in loaded_artifacts[label].centers
        )
        for label in labels
    }
    train_factory = _episode_factory(
        shard_paths,
        "train",
        expected_by_split["train"],
    )
    statistics: dict[int, _AdditiveCodeStatistics] = {}
    profile_statistics: dict[str, _AdditiveCodeStatistics] = {}
    train_vectors = 0
    for vectors, targets in _iter_aligned_probe_batches(
        train_factory,
        loaded_artifacts,
        future_offset=future_offset,
        batch_size=batch_size,
    ):
        codes = _encode_family_codes(
            vectors,
            loaded_artifacts,
            centers,
            device=target_device,
            center_block_size=center_block_size,
        )
        if not statistics:
            target_dimensions = {
                name: int(values.shape[1])
                for name, values in targets.items()
            }
            statistics = {
                depth: _AdditiveCodeStatistics(
                    {
                        label: int(
                            loaded_artifacts[label].centers[0].shape[0]
                        )
                        ** depth
                        for label in labels
                    },
                    target_dimensions,
                )
                for depth in range(1, levels + 1)
            }
            profile_statistics = {
                name: _AdditiveCodeStatistics(
                    {
                        label: int(
                            loaded_artifacts[label].centers[0].shape[0]
                        )
                        ** depths[label]
                        for label in labels
                    },
                    target_dimensions,
                )
                for name, depths in selected_profiles.items()
            }
        for depth, stats in statistics.items():
            stats.update(
                {
                    label: _prefix_keys(
                        codes[label],
                        k=int(
                            loaded_artifacts[label].centers[0].shape[0]
                        ),
                        depth=depth,
                    )
                    for label in labels
                },
                targets,
            )
        for name, stats in profile_statistics.items():
            depths = selected_profiles[name]
            stats.update(
                {
                    label: _prefix_keys(
                        codes[label],
                        k=int(
                            loaded_artifacts[label].centers[0].shape[0]
                        ),
                        depth=depths[label],
                    )
                    for label in labels
                },
                targets,
            )
        train_vectors += int(next(iter(vectors.values())).shape[0])
    if train_vectors <= 0 or not statistics:
        raise ValueError("Family association train stream is incomplete.")
    observed_targets = set(
        next(iter(statistics.values())).target_dimensions
    )
    if observed_targets != set(target_definitions):
        raise ValueError("Family association train targets are incomplete.")

    subsets = _family_subsets(labels)
    fitted = {
        depth: {
            subset: stats.fit(
                subset,
                ridge=ridge,
                device=target_device,
            )
            for subset in subsets
        }
        for depth, stats in statistics.items()
    }
    fitted_profiles = {
        name: stats.fit(
            labels,
            ridge=ridge,
            device=target_device,
        )
        for name, stats in profile_statistics.items()
    }
    rows = []
    summary_rows = []
    profile_rows = []
    for split in splits:
        accumulators = {}
        profile_accumulators = {}
        split_factory = _episode_factory(
            shard_paths,
            split,
            expected_by_split[split],
        )
        for vectors, targets in _iter_aligned_probe_batches(
            split_factory,
            loaded_artifacts,
            future_offset=future_offset,
            batch_size=batch_size,
        ):
            codes = _encode_family_codes(
                vectors,
                loaded_artifacts,
                centers,
                device=target_device,
                center_block_size=center_block_size,
            )
            for depth, stats in statistics.items():
                keys = {
                    label: _prefix_keys(
                        codes[label],
                        k=int(
                            loaded_artifacts[label].centers[0].shape[0]
                        ),
                        depth=depth,
                    )
                    for label in labels
                }
                for target, values in targets.items():
                    mean, variance, effective = stats.global_statistics(target)
                    for subset in subsets:
                        coefficients = fitted[depth][subset][target]
                        predictions, seen = _predict_additive(
                            keys,
                            coefficients,
                            mean,
                            stats.counts,
                        )
                        key = (target, depth, subset)
                        accumulator = accumulators.setdefault(
                            key,
                            _AdditiveRegressionAccumulator(
                                dimension=stats.target_dimensions[target],
                                effective_dimensions=int(
                                    effective.sum().item()
                                ),
                                family_count=len(subset),
                            ),
                        )
                        accumulator.update(
                            targets=values,
                            predictions=predictions,
                            seen=seen,
                            global_mean=mean,
                            variance=variance,
                            effective=effective,
                        )
            for name, stats in profile_statistics.items():
                depths = selected_profiles[name]
                keys = {
                    label: _prefix_keys(
                        codes[label],
                        k=int(
                            loaded_artifacts[label].centers[0].shape[0]
                        ),
                        depth=depths[label],
                    )
                    for label in labels
                }
                for target, values in targets.items():
                    mean, variance, effective = stats.global_statistics(target)
                    predictions, seen = _predict_additive(
                        keys,
                        fitted_profiles[name][target],
                        mean,
                        stats.counts,
                    )
                    key = (target, name)
                    accumulator = profile_accumulators.setdefault(
                        key,
                        _AdditiveRegressionAccumulator(
                            dimension=stats.target_dimensions[target],
                            effective_dimensions=int(
                                effective.sum().item()
                            ),
                            family_count=len(labels),
                        ),
                    )
                    accumulator.update(
                        targets=values,
                        predictions=predictions,
                        seen=seen,
                        global_mean=mean,
                        variance=variance,
                        effective=effective,
                    )
        for (target, depth, subset), accumulator in sorted(
            accumulators.items()
        ):
            rows.append(
                {
                    "split": split,
                    "target": target,
                    "prefix_depth": depth,
                    "families": list(subset),
                    "family_count": len(subset),
                    "model": "+".join(subset),
                    "ridge": ridge,
                    "train_vectors": train_vectors,
                    **accumulator.row(),
                }
            )
        for (target, name), accumulator in sorted(
            profile_accumulators.items()
        ):
            depths = selected_profiles[name]
            profile_rows.append(
                {
                    "split": split,
                    "target": target,
                    "profile": name,
                    "depths_by_family": depths,
                    "model": "+".join(
                        f"{label}-L{depths[label]}" for label in labels
                    ),
                    "families": list(labels),
                    "family_count": len(labels),
                    "ridge": ridge,
                    "train_vectors": train_vectors,
                    **accumulator.row(),
                }
            )

        for target in sorted(target_definitions):
            for depth in range(1, levels + 1):
                matching = [
                    row
                    for row in rows
                    if row["split"] == split
                    and row["target"] == target
                    and row["prefix_depth"] == depth
                ]
                full = next(
                    row for row in matching if tuple(row["families"]) == labels
                )
                singles = [
                    row for row in matching if row["family_count"] == 1
                ]
                best_single = max(
                    singles,
                    key=lambda row: (
                        row["normalized_mse_reduction"],
                        row["model"],
                    ),
                )
                leave_one_out = {
                    label: next(
                        row
                        for row in matching
                        if tuple(row["families"])
                        == tuple(value for value in labels if value != label)
                    )
                    for label in labels
                }
                summary_rows.append(
                    {
                        "split": split,
                        "target": target,
                        "prefix_depth": depth,
                        "full_model": full["model"],
                        "full_normalized_mse_reduction": (
                            full["normalized_mse_reduction"]
                        ),
                        "best_single_model": best_single["model"],
                        "best_single_normalized_mse_reduction": (
                            best_single["normalized_mse_reduction"]
                        ),
                        "full_gain_over_best_single": (
                            full["normalized_mse_reduction"]
                            - best_single["normalized_mse_reduction"]
                        ),
                        "incremental_gain_by_family": {
                            label: (
                                full["normalized_mse_reduction"]
                                - leave_one_out[label][
                                    "normalized_mse_reduction"
                                ]
                            )
                            for label in labels
                        },
                        "full_all_family_code_coverage": (
                            full["all_family_code_coverage"]
                        ),
                    }
                )

    report = {
        "schema": FAMILY_ASSOCIATION_REPORT_SCHEMA,
        "contract_hash": contract_hash,
        "manifest_fingerprint": manifest_fingerprint,
        "families": list(labels),
        "target_definitions": target_definitions,
        "rows": rows,
        "summary_rows": summary_rows,
        "profile_rows": profile_rows,
    }
    write_json_report(report_path, report)
    return report
