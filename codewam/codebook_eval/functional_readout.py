from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import torch

from codewam.data.droid_manifest import write_json_report

from .association import _resolve_device, _selected_features
from .family_association import (
    _prefix_keys,
    _validate_family_artifacts,
)
from .manifest import EpisodeManifest
from .seed_stability import (
    _training_contract_identity,
    _validate_runs,
)
from .shards import (
    PooledFeatureEpisode,
    expand_shard_paths,
    file_sha256,
    iter_pooled_feature_episodes,
)
from .streaming import (
    FrozenRQArtifact,
    NormalizationStats,
    RunningMoments,
    encode_residual_quantizer,
)


FUNCTIONAL_READOUT_CONTRACT_SCHEMA = (
    "codewam.rq-functional-readout-contract.v1"
)
FUNCTIONAL_READOUT_REPORT_SCHEMA = (
    "codewam.rq-functional-readout-report.v1"
)


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class _FunctionalBatch:
    base: torch.Tensor
    family_vectors: dict[str, torch.Tensor]
    action: torch.Tensor


def _episode_functional_values(
    episode: PooledFeatureEpisode,
    artifacts: dict[str, FrozenRQArtifact],
) -> _FunctionalBatch | None:
    if episode.action is None or episode.proprio is None:
        raise ValueError(
            f"Episode `{episode.episode_id}` lacks action or proprio."
        )
    labels, _ = _validate_family_artifacts(artifacts)
    reference = artifacts[labels[0]].descriptor
    maximum_stride = max(
        artifacts[label].descriptor.stride for label in labels
    )
    if episode.ticks <= 2 * maximum_stride:
        return None

    pooled, valid_mask = _selected_features(episode, reference)
    features = pooled.reshape(episode.ticks, -1)
    current = torch.arange(
        2 * maximum_stride,
        episode.ticks,
        dtype=torch.long,
        device=features.device,
    )
    valid_mask = valid_mask.to(device=features.device)
    valid = valid_mask[current].all(dim=1)
    timestamps = episode.timestamps.to(device=features.device)
    offsets = set()
    for label in labels:
        spec = artifacts[label].descriptor
        stride = spec.stride
        family_offsets = (-2 * stride, -stride, 0)
        offsets.update(family_offsets)
        for offset in family_offsets:
            valid &= valid_mask[current + offset].all(dim=1)
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
    current = current[valid]
    if current.numel() == 0:
        return None

    ordered_offsets = tuple(sorted(offsets))
    visual = torch.cat(
        [features[current + offset] for offset in ordered_offsets],
        dim=1,
    )
    proprio = episode.proprio[current].float()
    base = torch.cat((proprio, visual.float()), dim=1).contiguous()
    family_vectors = {}
    for label in labels:
        stride = artifacts[label].descriptor.stride
        family_vectors[label] = torch.cat(
            (
                features[current - 2 * stride],
                features[current - stride],
                features[current],
            ),
            dim=1,
        ).contiguous()
    return _FunctionalBatch(
        base=base,
        family_vectors=family_vectors,
        action=episode.action[current].float().contiguous(),
    )


def _iter_functional_batches(
    episode_factory: Callable[[], Iterable[PooledFeatureEpisode]],
    artifacts: dict[str, FrozenRQArtifact],
    *,
    batch_size: int,
) -> Iterator[_FunctionalBatch]:
    labels, _ = _validate_family_artifacts(artifacts)
    base_parts: list[torch.Tensor] = []
    vector_parts: dict[str, list[torch.Tensor]] = {
        label: [] for label in labels
    }
    action_parts: list[torch.Tensor] = []
    pending = 0

    def emit() -> _FunctionalBatch:
        nonlocal base_parts, vector_parts, action_parts, pending
        batch = _FunctionalBatch(
            base=torch.cat(base_parts, dim=0).contiguous(),
            family_vectors={
                label: torch.cat(parts, dim=0).contiguous()
                for label, parts in vector_parts.items()
            },
            action=torch.cat(action_parts, dim=0).contiguous(),
        )
        base_parts = []
        vector_parts = {label: [] for label in labels}
        action_parts = []
        pending = 0
        return batch

    for episode in episode_factory():
        values = _episode_functional_values(episode, artifacts)
        if values is None:
            continue
        rows = int(values.base.shape[0])
        offset = 0
        while offset < rows:
            take = min(batch_size - pending, rows - offset)
            base_parts.append(values.base[offset : offset + take])
            for label in labels:
                vector_parts[label].append(
                    values.family_vectors[label][offset : offset + take]
                )
            action_parts.append(values.action[offset : offset + take])
            pending += take
            offset += take
            if pending == batch_size:
                yield emit()
    if pending:
        yield emit()


def _scene_train_subset(
    manifest: EpisodeManifest,
    *,
    fraction: float,
    seed: int,
) -> tuple[set[str], tuple[str, ...]]:
    if not 0 < fraction <= 1:
        raise ValueError("Train fraction must be in (0,1].")
    train_records = [record for record in manifest if record.split == "train"]
    scenes = sorted(
        {record.scene_id for record in train_records},
        key=lambda scene: (
            hashlib.sha256(f"{seed}:{scene}".encode("utf-8")).hexdigest(),
            scene,
        ),
    )
    if not scenes:
        raise ValueError("Functional readout manifest has no train scenes.")
    selected_count = min(
        len(scenes),
        max(1, int(math.ceil(float(fraction) * len(scenes)))),
    )
    selected_scenes = tuple(sorted(scenes[:selected_count]))
    selected = set(selected_scenes)
    episode_ids = {
        record.episode_id
        for record in train_records
        if record.scene_id in selected
    }
    if not episode_ids:
        raise ValueError("Functional readout train subset is empty.")
    return episode_ids, selected_scenes


def _subset_episode_factory(
    shard_paths: tuple[Path, ...],
    *,
    split: str,
    expected_episode_ids: set[str],
) -> Callable[[], Iterator[PooledFeatureEpisode]]:
    def episodes() -> Iterator[PooledFeatureEpisode]:
        seen: set[str] = set()
        for episode in iter_pooled_feature_episodes(
            shard_paths,
            split=split,
        ):
            if episode.episode_id not in expected_episode_ids:
                continue
            if episode.episode_id in seen:
                raise ValueError(
                    f"Duplicate functional episode `{episode.episode_id}`."
                )
            seen.add(episode.episode_id)
            yield episode
        missing = sorted(expected_episode_ids - seen)
        if missing:
            raise ValueError(
                f"Functional `{split}` episodes are missing: {missing[:8]}."
            )

    return episodes


class _ContinuousStatistics:
    def __init__(
        self,
        *,
        dimension: int,
        target_dimension: int,
        device: torch.device,
    ) -> None:
        self.dimension = int(dimension)
        self.target_dimension = int(target_dimension)
        self.device = device
        self.count = 0
        self.sum_x = torch.zeros(
            self.dimension,
            dtype=torch.float64,
            device=device,
        )
        self.sum_y = torch.zeros(
            self.target_dimension,
            dtype=torch.float64,
            device=device,
        )
        self.sum_y2 = torch.zeros_like(self.sum_y)
        self.xtx = torch.zeros(
            (self.dimension, self.dimension),
            dtype=torch.float64,
            device=device,
        )
        self.xty = torch.zeros(
            (self.dimension, self.target_dimension),
            dtype=torch.float64,
            device=device,
        )

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x = x.to(device=self.device, dtype=torch.float64)
        y = y.to(device=self.device, dtype=torch.float64)
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError("Continuous feature dimension changed.")
        if y.shape != (x.shape[0], self.target_dimension):
            raise ValueError("Functional action target dimension changed.")
        self.count += int(x.shape[0])
        self.sum_x += x.sum(dim=0)
        self.sum_y += y.sum(dim=0)
        self.sum_y2 += y.square().sum(dim=0)
        self.xtx.addmm_(x.T, x)
        self.xty.addmm_(x.T, y)

    def centered(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.count <= 0:
            raise ValueError("Cannot finalize empty continuous statistics.")
        count = float(self.count)
        mean_x = self.sum_x / count
        mean_y = self.sum_y / count
        covariance = self.xtx / count - torch.outer(mean_x, mean_x)
        cross = self.xty / count - torch.outer(mean_x, mean_y)
        variance_y = (
            self.sum_y2 / count - mean_y.square()
        ).clamp_min_(0.0)
        return covariance, cross, mean_x, mean_y, variance_y


@dataclass(frozen=True)
class _CodeTransform:
    labels: tuple[str, ...]
    capacities: dict[str, int]
    offsets: dict[str, int]
    active_full_indices: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    def dense(
        self,
        keys: dict[str, torch.Tensor],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        rows = next(iter(keys.values())).shape[0]
        total = sum(self.capacities.values())
        values = torch.zeros(
            (rows, total),
            dtype=torch.float64,
            device=device,
        )
        row_indices = torch.arange(rows, device=device)
        for label in self.labels:
            key = keys[label].to(device=device, dtype=torch.long)
            values[row_indices, self.offsets[label] + key] = 1.0
        values = values[:, self.active_full_indices.to(device=device)]
        return (
            values
            - self.mean.to(device=device).unsqueeze(0)
        ) / self.std.to(device=device).unsqueeze(0)


class _CodeStatistics:
    def __init__(
        self,
        *,
        capacities: dict[str, int],
        continuous_dimension: int,
        target_dimension: int,
        device: torch.device,
    ) -> None:
        self.labels = tuple(capacities)
        self.capacities = dict(capacities)
        self.continuous_dimension = int(continuous_dimension)
        self.target_dimension = int(target_dimension)
        self.device = device
        self.count = 0
        self.counts = {
            label: torch.zeros(
                capacity,
                dtype=torch.float64,
                device=device,
            )
            for label, capacity in capacities.items()
        }
        self.x_sums = {
            label: torch.zeros(
                (capacity, self.continuous_dimension),
                dtype=torch.float64,
                device=device,
            )
            for label, capacity in capacities.items()
        }
        self.y_sums = {
            label: torch.zeros(
                (capacity, self.target_dimension),
                dtype=torch.float64,
                device=device,
            )
            for label, capacity in capacities.items()
        }
        self.cross_counts = {
            (left, right): torch.zeros(
                (capacities[left], capacities[right]),
                dtype=torch.float64,
                device=device,
            )
            for index, left in enumerate(self.labels)
            for right in self.labels[index + 1 :]
        }

    def update(
        self,
        *,
        keys: dict[str, torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        if set(keys) != set(self.labels):
            raise ValueError("Functional code keys do not match profile.")
        x = x.to(device=self.device, dtype=torch.float64)
        y = y.to(device=self.device, dtype=torch.float64)
        rows = int(x.shape[0])
        if y.shape != (rows, self.target_dimension):
            raise ValueError("Functional code target dimension changed.")
        device_keys = {
            label: key.to(device=self.device, dtype=torch.long)
            for label, key in keys.items()
        }
        ones = torch.ones(rows, dtype=torch.float64, device=self.device)
        for label, key in device_keys.items():
            if (
                key.shape != (rows,)
                or int(key.min()) < 0
                or int(key.max()) >= self.capacities[label]
            ):
                raise ValueError(f"Invalid functional keys for `{label}`.")
            self.counts[label].index_add_(0, key, ones)
            self.x_sums[label].index_add_(0, key, x)
            self.y_sums[label].index_add_(0, key, y)
        for left, right in self.cross_counts:
            flat = (
                device_keys[left] * self.capacities[right]
                + device_keys[right]
            )
            self.cross_counts[(left, right)] += torch.bincount(
                flat,
                minlength=(
                    self.capacities[left] * self.capacities[right]
                ),
            ).reshape(
                self.capacities[left],
                self.capacities[right],
            ).double()
        self.count += rows

    def centered(
        self,
        continuous: _ContinuousStatistics,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        _CodeTransform,
    ]:
        if self.count != continuous.count or self.count <= 0:
            raise ValueError("Continuous/code statistics count mismatch.")
        offsets = {}
        total = 0
        for label in self.labels:
            offsets[label] = total
            total += self.capacities[label]
        raw_ctc = torch.zeros(
            (total, total),
            dtype=torch.float64,
            device=self.device,
        )
        for label in self.labels:
            start = offsets[label]
            capacity = self.capacities[label]
            raw_ctc[
                start : start + capacity,
                start : start + capacity,
            ].diagonal().copy_(self.counts[label])
        for (left, right), values in self.cross_counts.items():
            left_start = offsets[left]
            right_start = offsets[right]
            raw_ctc[
                left_start : left_start + self.capacities[left],
                right_start : right_start + self.capacities[right],
            ] = values
            raw_ctc[
                right_start : right_start + self.capacities[right],
                left_start : left_start + self.capacities[left],
            ] = values.T

        counts = torch.cat([self.counts[label] for label in self.labels])
        raw_ctx = torch.cat(
            [self.x_sums[label] for label in self.labels],
            dim=0,
        )
        raw_cty = torch.cat(
            [self.y_sums[label] for label in self.labels],
            dim=0,
        )
        count = float(self.count)
        probabilities = counts / count
        active = (counts > 0) & (counts < self.count)
        active_indices = active.nonzero(as_tuple=False).flatten()
        if active_indices.numel() <= len(self.labels):
            raise ValueError("Functional code profile has no active capacity.")
        probabilities = probabilities[active_indices]
        standard_deviation = (
            probabilities * (1.0 - probabilities)
        ).clamp_min_(1e-12).sqrt()
        raw_ctc = raw_ctc[
            active_indices[:, None],
            active_indices[None, :],
        ]
        raw_ctx = raw_ctx[active_indices]
        raw_cty = raw_cty[active_indices]
        _, _, mean_x, mean_y, _ = continuous.centered()

        covariance_c = (
            raw_ctc / count
            - torch.outer(probabilities, probabilities)
        )
        covariance_c = covariance_c / torch.outer(
            standard_deviation,
            standard_deviation,
        )
        covariance_xc = (
            raw_ctx.T / count
            - torch.outer(mean_x, probabilities)
        ) / standard_deviation.unsqueeze(0)
        cross_cy = (
            raw_cty / count
            - torch.outer(probabilities, mean_y)
        ) / standard_deviation.unsqueeze(1)
        transform = _CodeTransform(
            labels=self.labels,
            capacities=self.capacities,
            offsets=offsets,
            active_full_indices=active_indices.detach().cpu(),
            mean=probabilities.detach().cpu(),
            std=standard_deviation.detach().cpu(),
        )
        return covariance_c, covariance_xc, cross_cy, transform


@dataclass(frozen=True)
class _ReadoutModel:
    name: str
    x_indices: torch.Tensor
    beta_x: torch.Tensor
    mean_x: torch.Tensor
    mean_y: torch.Tensor
    beta_c: torch.Tensor | None = None
    code_transform: _CodeTransform | None = None

    def predict(
        self,
        x: torch.Tensor,
        *,
        keys: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        device = self.beta_x.device
        x = x.to(device=device, dtype=torch.float64)
        indices = self.x_indices.to(device=device)
        prediction = self.mean_y.to(device=device).unsqueeze(0)
        prediction = prediction + (
            x[:, indices]
            - self.mean_x.to(device=device)[indices].unsqueeze(0)
        ) @ self.beta_x
        if self.beta_c is not None:
            if self.code_transform is None or keys is None:
                raise ValueError("Code readout requires functional keys.")
            code = self.code_transform.dense(keys, device=device)
            prediction = prediction + code @ self.beta_c
        return prediction


def _solve(
    matrix: torch.Tensor,
    right_hand_side: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    if alpha <= 0:
        raise ValueError("Functional ridge alpha must be positive.")
    regularized = matrix.clone()
    regularized.diagonal().add_(float(alpha))
    factor, info = torch.linalg.cholesky_ex(regularized)
    if int(info.max().item()) != 0:
        return torch.linalg.solve(regularized, right_hand_side)
    return torch.cholesky_solve(right_hand_side, factor)


def _fit_continuous_model(
    statistics: _ContinuousStatistics,
    *,
    name: str,
    x_indices: torch.Tensor,
    alpha: float,
) -> _ReadoutModel:
    covariance, cross, mean_x, mean_y, _ = statistics.centered()
    indices = x_indices.to(device=statistics.device, dtype=torch.long)
    matrix = covariance[indices[:, None], indices[None, :]]
    beta = _solve(matrix, cross[indices], alpha=alpha)
    return _ReadoutModel(
        name=name,
        x_indices=indices.detach().cpu(),
        beta_x=beta,
        mean_x=mean_x.detach().cpu(),
        mean_y=mean_y.detach().cpu(),
    )


def _fit_code_model(
    continuous: _ContinuousStatistics,
    code: _CodeStatistics,
    *,
    name: str,
    x_indices: torch.Tensor,
    alpha: float,
) -> _ReadoutModel:
    covariance_x, cross_x, mean_x, mean_y, _ = continuous.centered()
    covariance_c, covariance_xc, cross_c, transform = code.centered(
        continuous
    )
    indices = x_indices.to(device=continuous.device, dtype=torch.long)
    selected_x = covariance_x[
        indices[:, None],
        indices[None, :],
    ]
    selected_cross = covariance_xc[indices]
    matrix = torch.cat(
        (
            torch.cat((selected_x, selected_cross), dim=1),
            torch.cat((selected_cross.T, covariance_c), dim=1),
        ),
        dim=0,
    )
    right_hand_side = torch.cat((cross_x[indices], cross_c), dim=0)
    beta = _solve(matrix, right_hand_side, alpha=alpha)
    split = int(indices.numel())
    return _ReadoutModel(
        name=name,
        x_indices=indices.detach().cpu(),
        beta_x=beta[:split],
        beta_c=beta[split:],
        mean_x=mean_x.detach().cpu(),
        mean_y=mean_y.detach().cpu(),
        code_transform=transform,
    )


@dataclass
class _ActionAccumulator:
    target_dimension: int
    effective_dimensions: int
    vectors: int = 0
    raw_sse: float = 0.0
    baseline_raw_sse: float = 0.0
    normalized_sse: float = 0.0
    baseline_normalized_sse: float = 0.0

    def update(
        self,
        *,
        target: torch.Tensor,
        prediction: torch.Tensor,
        mean: torch.Tensor,
        variance: torch.Tensor,
    ) -> None:
        target = target.detach().double().cpu()
        prediction = prediction.detach().double().cpu()
        mean = mean.detach().double().cpu()
        variance = variance.detach().double().cpu()
        effective = variance > 1e-10
        error = (target - prediction).square()
        baseline = (target - mean.unsqueeze(0)).square()
        self.vectors += int(target.shape[0])
        self.raw_sse += float(error.sum().item())
        self.baseline_raw_sse += float(baseline.sum().item())
        self.normalized_sse += float(
            (error[:, effective] / variance[effective]).sum().item()
        )
        self.baseline_normalized_sse += float(
            (baseline[:, effective] / variance[effective]).sum().item()
        )

    def row(self) -> dict[str, Any]:
        if self.vectors <= 0:
            raise ValueError("Cannot finalize empty functional metrics.")
        raw_denominator = float(self.vectors * self.target_dimension)
        normalized_mse = self.normalized_sse / float(
            self.vectors * self.effective_dimensions
        )
        baseline_normalized_mse = self.baseline_normalized_sse / float(
            self.vectors * self.effective_dimensions
        )
        return {
            "vectors": self.vectors,
            "target_dimension": self.target_dimension,
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
        }


def _encode_keys(
    family_vectors: dict[str, torch.Tensor],
    artifacts: dict[str, FrozenRQArtifact],
    centers: dict[str, tuple[torch.Tensor, ...]],
    profile: dict[str, int],
    *,
    device: torch.device,
    center_block_size: int,
) -> dict[str, torch.Tensor]:
    result = {}
    for label, values in family_vectors.items():
        normalized = artifacts[label].normalization.normalize(values).to(
            device=device,
            dtype=torch.float32,
        )
        codes, _, _ = encode_residual_quantizer(
            normalized,
            centers[label],
            center_block_size=center_block_size,
        )
        result[label] = _prefix_keys(
            codes,
            k=int(centers[label][0].shape[0]),
            depth=profile[label],
        )
    return result


def _evaluate_continuous_candidates(
    *,
    episode_factory: Callable[[], Iterable[PooledFeatureEpisode]],
    artifacts: dict[str, FrozenRQArtifact],
    normalization: NormalizationStats,
    models: dict[float, _ReadoutModel],
    variance_y: torch.Tensor,
    batch_size: int,
) -> list[dict[str, Any]]:
    accumulators = {
        alpha: _ActionAccumulator(
            target_dimension=int(variance_y.numel()),
            effective_dimensions=int((variance_y > 1e-10).sum().item()),
        )
        for alpha in models
    }
    for batch in _iter_functional_batches(
        episode_factory,
        artifacts,
        batch_size=batch_size,
    ):
        x = normalization.normalize(batch.base)
        for alpha, model in models.items():
            accumulators[alpha].update(
                target=batch.action,
                prediction=model.predict(x),
                mean=model.mean_y,
                variance=variance_y,
            )
    return [
        {"alpha": alpha, **accumulators[alpha].row()}
        for alpha in sorted(models)
    ]


def _evaluate_models(
    *,
    split: str,
    episode_factory: Callable[[], Iterable[PooledFeatureEpisode]],
    artifacts_by_run: dict[str, dict[str, FrozenRQArtifact]],
    centers_by_run: dict[str, dict[str, tuple[torch.Tensor, ...]]],
    normalization: NormalizationStats,
    profile: dict[str, int],
    models: dict[str, _ReadoutModel],
    variance_y: torch.Tensor,
    device: torch.device,
    batch_size: int,
    center_block_size: int,
) -> list[dict[str, Any]]:
    reference_run = next(iter(artifacts_by_run))
    reference = artifacts_by_run[reference_run]
    accumulators = {
        name: _ActionAccumulator(
            target_dimension=int(variance_y.numel()),
            effective_dimensions=int((variance_y > 1e-10).sum().item()),
        )
        for name in models
    }
    for batch in _iter_functional_batches(
        episode_factory,
        reference,
        batch_size=batch_size,
    ):
        x = normalization.normalize(batch.base)
        keys_by_run = {
            run: _encode_keys(
                batch.family_vectors,
                artifacts,
                centers_by_run[run],
                profile,
                device=device,
                center_block_size=center_block_size,
            )
            for run, artifacts in artifacts_by_run.items()
        }
        for name, model in models.items():
            run = name.split("@", 1)[1] if "@" in name else None
            prediction = model.predict(
                x,
                keys=None if run is None else keys_by_run[run],
            )
            accumulators[name].update(
                target=batch.action,
                prediction=prediction,
                mean=model.mean_y,
                variance=variance_y,
            )
    rows = []
    for name, accumulator in sorted(accumulators.items()):
        run = name.split("@", 1)[1] if "@" in name else None
        model = name.split("@", 1)[0]
        rows.append(
            {
                "split": split,
                "model": model,
                "run": run,
                **accumulator.row(),
            }
        )
    return rows


def probe_codebook_functional_readout(
    *,
    manifest_path: str | Path,
    pooled_shards: Iterable[str | Path],
    runs: dict[str, dict[str, str | Path]],
    output_dir: str | Path,
    code_depths: dict[str, int],
    train_fraction: float = 1.0,
    subset_seed: int = 20260729,
    alpha_candidates: tuple[float, ...] = (
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
    ),
    device: str = "auto",
    cpu_threads: int = 4,
    batch_size: int = 4096,
    center_block_size: int = 1024,
    resume: bool = True,
) -> dict[str, Any]:
    if (
        cpu_threads <= 0
        or batch_size <= 0
        or center_block_size <= 0
    ):
        raise ValueError("Functional readout numeric settings must be positive.")
    if (
        not alpha_candidates
        or any(alpha <= 0 for alpha in alpha_candidates)
        or len(set(alpha_candidates)) != len(alpha_candidates)
    ):
        raise ValueError("Functional alpha candidates must be unique/positive.")
    torch.set_num_threads(int(cpu_threads))
    target_device = _resolve_device(device)

    manifest_path = Path(manifest_path)
    manifest = EpisodeManifest.read_jsonl(manifest_path)
    manifest.assert_group_isolation("scene")
    manifest_fingerprint = manifest.fingerprint()
    shard_paths = tuple(expand_shard_paths(pooled_shards))
    shard_checksums = [file_sha256(path) for path in shard_paths]
    run_paths = {
        run: {
            family: Path(path)
            for family, path in sorted(artifacts.items())
        }
        for run, artifacts in sorted(runs.items())
    }
    loaded_runs = {
        run: {
            family: FrozenRQArtifact.load(path)
            for family, path in paths.items()
        }
        for run, paths in run_paths.items()
    }
    training_contracts = {
        run: {
            family: _training_contract_identity(
                run_paths[run][family],
                artifact,
            )
            for family, artifact in artifacts.items()
        }
        for run, artifacts in loaded_runs.items()
    }
    run_labels, family_labels, levels = _validate_runs(loaded_runs)
    if set(code_depths) != set(family_labels):
        raise ValueError("Code depth profile must configure every family.")
    profile = {
        family: int(code_depths[family]) for family in family_labels
    }
    invalid_depths = [
        family
        for family, depth in profile.items()
        if depth <= 0 or depth > levels
    ]
    if invalid_depths:
        raise ValueError(
            f"Invalid functional code depths for {invalid_depths}."
        )
    run_seeds = {}
    for run in run_labels:
        seeds = {
            int(contract["seed"])
            for contract in training_contracts[run].values()
        }
        if len(seeds) != 1:
            raise ValueError(f"Functional run `{run}` mixes seeds.")
        run_seeds[run] = next(iter(seeds))
    if len(set(run_seeds.values())) != len(run_seeds):
        raise ValueError("Functional runs must use distinct training seeds.")
    for run, artifacts in loaded_runs.items():
        for family, artifact in artifacts.items():
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
                    f"Functional artifact `{run}/{family}` differs in "
                    f"{mismatches}."
                )

    train_ids, selected_scenes = _scene_train_subset(
        manifest,
        fraction=train_fraction,
        seed=subset_seed,
    )
    expected_by_split = {
        split: {
            record.episode_id
            for record in manifest
            if record.split == split
        }
        for split in ("val", "test")
    }
    reference_run = run_labels[0]
    reference_artifacts = loaded_runs[reference_run]
    labels, _ = _validate_family_artifacts(reference_artifacts)
    strides = {
        label: reference_artifacts[label].descriptor.stride
        for label in labels
    }
    state_offsets = sorted(
        {
            offset
            for stride in strides.values()
            for offset in (-2 * stride, -stride, 0)
        }
    )
    code_capacities = {
        family: int(
            loaded_runs[reference_run][family].centers[0].shape[0]
        )
        ** profile[family]
        for family in family_labels
    }

    sample_episode = next(
        iter_pooled_feature_episodes(
            shard_paths,
            split="train",
        )
    )
    if sample_episode.proprio is None or sample_episode.action is None:
        raise ValueError("Functional readout requires proprio/action targets.")
    proprio_dimension = int(sample_episode.proprio.shape[1])
    action_dimension = int(sample_episode.action.shape[1])

    contract_payload = {
        "schema": FUNCTIONAL_READOUT_CONTRACT_SCHEMA,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": file_sha256(manifest_path),
            "fingerprint": manifest_fingerprint,
        },
        "pooled_shards": [
            {"path": str(path), "sha256": checksum}
            for path, checksum in zip(shard_paths, shard_checksums)
        ],
        "runs": {
            run: {
                family: {
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                    "training_contract": training_contracts[run][family],
                }
                for family, path in paths.items()
            }
            for run, paths in run_paths.items()
        },
        "run_seeds": run_seeds,
        "train_subset": {
            "fraction": train_fraction,
            "selection_seed": subset_seed,
            "episodes": len(train_ids),
            "scenes": len(selected_scenes),
            "scene_fingerprint": hashlib.sha256(
                "\n".join(selected_scenes).encode("utf-8")
            ).hexdigest(),
        },
        "feature_contract": {
            "camera_ids": list(
                reference_artifacts[labels[0]].descriptor.camera_ids or ()
            ),
            "pool": reference_artifacts[labels[0]].descriptor.pool,
            "continuous_state_offsets": state_offsets,
            "continuous_state_source": (
                "unquantized pooled Wan states at the union of Q2/Q3/Q5 "
                "causal offsets"
            ),
            "context": "current proprioception",
            "code_depths": profile,
            "code_capacities": code_capacities,
        },
        "target": {
            "name": "current_action",
            "dimension": action_dimension,
            "definition": (
                "The DROID action recorded at the current latent tick; no "
                "future observation or target enters inputs."
            ),
        },
        "models": {
            "P0": "current proprioception",
            "P1": "current proprioception + continuous H",
            "P2": "current proprioception + frozen C",
            "P3": "current proprioception + continuous H + frozen C",
        },
        "alpha_selection": (
            "P1-only validation normalized MSE reduction; selected alpha is "
            "reused unchanged for every codebook seed and model."
        ),
        "alpha_candidates": list(alpha_candidates),
        "device": device,
        "cpu_threads": cpu_threads,
        "batch_size": batch_size,
        "center_block_size": center_block_size,
        "implementation_sha256": {
            "functional_readout": file_sha256(Path(__file__)),
            "family_association": file_sha256(
                Path(__file__).with_name("family_association.py")
            ),
            "seed_stability": file_sha256(
                Path(__file__).with_name("seed_stability.py")
            ),
            "streaming": file_sha256(
                Path(__file__).with_name("streaming.py")
            ),
        },
    }
    contract_hash = _canonical_hash(contract_payload)
    contract = {**contract_payload, "contract_hash": contract_hash}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    report_path = output_dir / "functional_readout_report.json"
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != contract:
            raise RuntimeError("Existing functional readout contract differs.")
        if not resume:
            raise FileExistsError(
                f"Functional contract exists at `{contract_path}`."
            )
    else:
        write_json_report(contract_path, contract)
    if resume and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("contract_hash") != contract_hash:
            raise RuntimeError("Functional report contract hash is invalid.")
        return report
    if report_path.exists():
        raise FileExistsError(
            f"Functional report exists at `{report_path}`."
        )

    train_factory = _subset_episode_factory(
        shard_paths,
        split="train",
        expected_episode_ids=train_ids,
    )
    moments = RunningMoments()
    for batch in _iter_functional_batches(
        train_factory,
        reference_artifacts,
        batch_size=batch_size,
    ):
        moments.update(batch.base.to(device=target_device))
    normalization = moments.finalize()

    centers_by_run = {
        run: {
            family: tuple(
                center.to(device=target_device, dtype=torch.float32)
                for center in artifact.centers
            )
            for family, artifact in artifacts.items()
        }
        for run, artifacts in loaded_runs.items()
    }
    continuous = _ContinuousStatistics(
        dimension=normalization.dim,
        target_dimension=action_dimension,
        device=target_device,
    )
    code_statistics = {
        run: _CodeStatistics(
            capacities=code_capacities,
            continuous_dimension=normalization.dim,
            target_dimension=action_dimension,
            device=target_device,
        )
        for run in run_labels
    }
    for batch in _iter_functional_batches(
        train_factory,
        reference_artifacts,
        batch_size=batch_size,
    ):
        x = normalization.normalize(batch.base).to(
            device=target_device,
            dtype=torch.float32,
        )
        continuous.update(x, batch.action)
        for run in run_labels:
            keys = _encode_keys(
                batch.family_vectors,
                loaded_runs[run],
                centers_by_run[run],
                profile,
                device=target_device,
                center_block_size=center_block_size,
            )
            code_statistics[run].update(
                keys=keys,
                x=x,
                y=batch.action,
            )

    all_indices = torch.arange(normalization.dim, dtype=torch.long)
    proprio_indices = torch.arange(proprio_dimension, dtype=torch.long)
    alpha_models = {
        alpha: _fit_continuous_model(
            continuous,
            name=f"P1@alpha={alpha:g}",
            x_indices=all_indices,
            alpha=alpha,
        )
        for alpha in alpha_candidates
    }
    _, _, _, mean_y, variance_y = continuous.centered()
    val_factory = _subset_episode_factory(
        shard_paths,
        split="val",
        expected_episode_ids=expected_by_split["val"],
    )
    alpha_rows = _evaluate_continuous_candidates(
        episode_factory=val_factory,
        artifacts=reference_artifacts,
        normalization=normalization,
        models=alpha_models,
        variance_y=variance_y,
        batch_size=batch_size,
    )
    selected_alpha_row = max(
        alpha_rows,
        key=lambda row: (
            row["normalized_mse_reduction"],
            -float(row["alpha"]),
        ),
    )
    selected_alpha = float(selected_alpha_row["alpha"])

    models = {
        "P0": _fit_continuous_model(
            continuous,
            name="P0",
            x_indices=proprio_indices,
            alpha=selected_alpha,
        ),
        "P1": _fit_continuous_model(
            continuous,
            name="P1",
            x_indices=all_indices,
            alpha=selected_alpha,
        ),
    }
    for run in run_labels:
        models[f"P2@{run}"] = _fit_code_model(
            continuous,
            code_statistics[run],
            name=f"P2@{run}",
            x_indices=proprio_indices,
            alpha=selected_alpha,
        )
        models[f"P3@{run}"] = _fit_code_model(
            continuous,
            code_statistics[run],
            name=f"P3@{run}",
            x_indices=all_indices,
            alpha=selected_alpha,
        )

    rows = []
    for split in ("val", "test"):
        factory = _subset_episode_factory(
            shard_paths,
            split=split,
            expected_episode_ids=expected_by_split[split],
        )
        rows.extend(
            _evaluate_models(
                split=split,
                episode_factory=factory,
                artifacts_by_run={
                    run: loaded_runs[run] for run in run_labels
                },
                centers_by_run=centers_by_run,
                normalization=normalization,
                profile=profile,
                models=models,
                variance_y=variance_y,
                device=target_device,
                batch_size=batch_size,
                center_block_size=center_block_size,
            )
        )

    summaries = []
    for split in ("val", "test"):
        p1 = next(
            row
            for row in rows
            if row["split"] == split and row["model"] == "P1"
        )
        seed_rows = []
        for run in run_labels:
            p2 = next(
                row
                for row in rows
                if row["split"] == split
                and row["model"] == "P2"
                and row["run"] == run
            )
            p3 = next(
                row
                for row in rows
                if row["split"] == split
                and row["model"] == "P3"
                and row["run"] == run
            )
            seed_rows.append(
                {
                    "run": run,
                    "seed": run_seeds[run],
                    "p2_reduction": p2["normalized_mse_reduction"],
                    "p3_reduction": p3["normalized_mse_reduction"],
                    "p3_minus_p1": (
                        p3["normalized_mse_reduction"]
                        - p1["normalized_mse_reduction"]
                    ),
                }
            )
        increments = [row["p3_minus_p1"] for row in seed_rows]
        status = (
            "consistent_positive"
            if min(increments) > 0
            else "consistent_nonpositive"
            if max(increments) <= 0
            else "mixed"
        )
        summaries.append(
            {
                "split": split,
                "p1_reduction": p1["normalized_mse_reduction"],
                "seed_rows": seed_rows,
                "increment_status": status,
                "minimum_p3_minus_p1": min(increments),
                "maximum_p3_minus_p1": max(increments),
                "range_p3_minus_p1": max(increments) - min(increments),
            }
        )

    for run, paths in run_paths.items():
        for family, path in paths.items():
            expected_hash = contract["runs"][run][family]["sha256"]
            if file_sha256(path) != expected_hash:
                raise RuntimeError(
                    f"Frozen artifact `{run}/{family}` changed during probe."
                )
    report = {
        "schema": FUNCTIONAL_READOUT_REPORT_SCHEMA,
        "contract_hash": contract_hash,
        "manifest_fingerprint": manifest_fingerprint,
        "run_seeds": run_seeds,
        "train_fraction": train_fraction,
        "train_scenes": len(selected_scenes),
        "train_episodes": len(train_ids),
        "train_vectors": continuous.count,
        "continuous_dimension": normalization.dim,
        "proprio_dimension": proprio_dimension,
        "code_depths": profile,
        "code_capacities": code_capacities,
        "selected_alpha": selected_alpha,
        "alpha_selection_rows": alpha_rows,
        "rows": rows,
        "summaries": summaries,
        "interpretation": [
            "P3-P1 isolates the held-out contribution of frozen categorical "
            "codes beyond the exact continuous states from which they were "
            "computed.",
            "The closed-form readout has no optimization seed; differences "
            "across runs come from independently trained codebooks.",
            "This is a visual-proprio action screen without language or a "
            "policy backbone, not a closed-loop control result.",
        ],
    }
    write_json_report(report_path, report)
    return report
