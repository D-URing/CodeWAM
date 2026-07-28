from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageDraw

from .io import ensure_dir, save_json, write_rows_tsv
from .kmeans_diagnostics import (
    DiagnosticKMeansConfig,
    DiagnosticKMeansResult,
    adjusted_rand_index,
    fit_diagnostic_kmeans,
    fit_diagnostic_rq,
    usage_summary,
)
from .shards import (
    PooledFeatureEpisode,
    atomic_torch_save,
    iter_pooled_feature_episodes,
)
from .streaming import assign_nearest


LATENT_PROBE_SCHEMA = "codewam.wan-latent-probe-report.v1"


@dataclass(frozen=True)
class LatentProbeConfig:
    pooled_shards: tuple[str, ...]
    output_dir: str
    cameras: tuple[str, ...]
    pools: tuple[int, ...] = (1, 2, 4)
    strides: tuple[int, ...] = (2, 3, 5)
    k_values: tuple[int, ...] = (8, 16, 32)
    seeds: tuple[int, ...] = (0, 1, 2)
    tolerances: tuple[float, ...] = (1e-3, 1e-4, 1e-5)
    patiences: tuple[int, ...] = (2, 3)
    sweep_stride: int = 3
    selected_pool: int = 2
    selected_k: int = 16
    default_tolerance: float = 1e-4
    default_patience: int = 3
    max_iters: int = 50
    min_iters: int = 3
    rq_levels: int = 3
    device: str = "auto"
    chunk_size: int = 8192

    def __post_init__(self) -> None:
        if not self.pooled_shards:
            raise ValueError("At least one pooled shard path is required.")
        if not self.cameras:
            raise ValueError("At least one camera is required.")
        if any(value not in {1, 2, 4} for value in self.pools):
            raise ValueError(f"Probe pools must be drawn from 1, 2, 4; got {self.pools}.")
        if any(value <= 0 for value in self.strides):
            raise ValueError("Probe strides must be positive.")
        if self.sweep_stride not in self.strides:
            raise ValueError("`sweep_stride` must be one of `strides`.")
        if self.selected_pool not in self.pools:
            raise ValueError("`selected_pool` must be one of `pools`.")
        if self.selected_k not in self.k_values:
            raise ValueError("`selected_k` must be one of `k_values`.")


@dataclass(frozen=True)
class ProbeSamples:
    vectors: torch.Tensor
    episode_ids: tuple[str, ...]
    time_indices: torch.Tensor
    splits: tuple[str, ...]
    camera: str
    pool: int
    representation: str
    stride: int
    image_motion: torch.Tensor
    proprio_motion: torch.Tensor
    action_magnitude: torch.Tensor
    previous_thumbnails: torch.Tensor
    current_thumbnails: torch.Tensor

    def __post_init__(self) -> None:
        count = int(self.vectors.shape[0])
        if self.vectors.ndim != 2 or count == 0:
            raise ValueError(f"Probe vectors must be nonempty [N,D], got {self.vectors.shape}.")
        if len(self.episode_ids) != count or len(self.splits) != count:
            raise ValueError("Probe vector metadata does not align with N.")
        for name, value in (
            ("time_indices", self.time_indices),
            ("image_motion", self.image_motion),
            ("proprio_motion", self.proprio_motion),
            ("action_magnitude", self.action_magnitude),
        ):
            if tuple(value.shape) != (count,):
                raise ValueError(f"`{name}` must be [N], got {tuple(value.shape)}.")
        expected_thumbnail = (count, 3)
        if self.previous_thumbnails.shape[:2] != expected_thumbnail:
            raise ValueError("Previous thumbnails must be [N,3,H,W].")
        if self.current_thumbnails.shape != self.previous_thumbnails.shape:
            raise ValueError("Previous/current thumbnail shapes must match.")

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])

    def select(self, split: str) -> ProbeSamples:
        indices = torch.tensor(
            [index for index, value in enumerate(self.splits) if value == split],
            dtype=torch.long,
        )
        if indices.numel() == 0:
            raise ValueError(
                f"No `{split}` samples for {self.camera} {self.representation} Q{self.stride}."
            )
        return ProbeSamples(
            vectors=self.vectors[indices],
            episode_ids=tuple(self.episode_ids[index] for index in indices.tolist()),
            time_indices=self.time_indices[indices],
            splits=tuple(self.splits[index] for index in indices.tolist()),
            camera=self.camera,
            pool=self.pool,
            representation=self.representation,
            stride=self.stride,
            image_motion=self.image_motion[indices],
            proprio_motion=self.proprio_motion[indices],
            action_magnitude=self.action_magnitude[indices],
            previous_thumbnails=self.previous_thumbnails[indices],
            current_thumbnails=self.current_thumbnails[indices],
        )


def _resolve_device(spec: str) -> torch.device:
    if str(spec).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device `{spec}` was requested but CUDA is unavailable.")
    return device


def _camera_index(episode: PooledFeatureEpisode, camera: str) -> int:
    try:
        return episode.camera_ids.index(camera)
    except ValueError as exc:
        raise ValueError(
            f"Episode `{episode.episode_id}` has cameras {episode.camera_ids}, not `{camera}`."
        ) from exc


def _episode_thumbnails(
    episode: PooledFeatureEpisode,
    camera_index: int,
) -> torch.Tensor:
    thumbnails = episode.metadata.get("probe_thumbnails")
    if not isinstance(thumbnails, torch.Tensor):
        raise ValueError(
            f"Episode `{episode.episode_id}` lacks tensor metadata `probe_thumbnails`."
        )
    if thumbnails.ndim != 5 or thumbnails.shape[:2] != (
        episode.ticks,
        episode.views,
    ):
        raise ValueError(
            f"Bad thumbnail shape in `{episode.episode_id}`: {tuple(thumbnails.shape)}."
        )
    return thumbnails[:, camera_index].to(dtype=torch.uint8)


def _motion_scalars(
    episode: PooledFeatureEpisode,
    thumbnails: torch.Tensor,
    current: torch.Tensor,
    previous: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_motion = (
        thumbnails[current].float().sub(thumbnails[previous].float()).abs().mean(
            dim=(1, 2, 3)
        )
        / 255.0
    )
    if episode.proprio is None:
        proprio_motion = torch.full_like(image_motion, float("nan"))
    else:
        proprio = episode.proprio.float()
        proprio_motion = (
            proprio[current].sub(proprio[previous]).square().mean(dim=1).sqrt()
        )
    if episode.action is None:
        action_magnitude = torch.full_like(image_motion, float("nan"))
    else:
        action = episode.action.float()
        action_magnitude = torch.stack(
            [
                action[int(start) + 1 : int(end) + 1].square().mean().sqrt()
                for start, end in zip(previous.tolist(), current.tolist())
            ]
        )
    return image_motion, proprio_motion, action_magnitude


def build_probe_samples(
    episodes: Sequence[PooledFeatureEpisode],
    camera: str,
    pool: int,
    representation: str,
    stride: int,
) -> ProbeSamples:
    if representation not in {"absolute", "residual", "descriptor"}:
        raise ValueError(f"Unsupported probe representation `{representation}`.")
    if int(stride) <= 0:
        raise ValueError("Probe stride must be positive.")

    vector_parts: list[torch.Tensor] = []
    episode_ids: list[str] = []
    time_parts: list[torch.Tensor] = []
    splits: list[str] = []
    image_motion_parts: list[torch.Tensor] = []
    proprio_motion_parts: list[torch.Tensor] = []
    action_magnitude_parts: list[torch.Tensor] = []
    previous_thumbnail_parts: list[torch.Tensor] = []
    current_thumbnail_parts: list[torch.Tensor] = []

    for episode in episodes:
        view = _camera_index(episode, camera)
        start = 2 * stride if representation in {"absolute", "descriptor"} else stride
        if episode.ticks <= start:
            continue
        features = episode.pooled(pool)[:, view].float().flatten(1)
        current = torch.arange(start, episode.ticks, dtype=torch.long)
        previous = current - stride
        valid = episode.valid_mask[:, view]
        if representation == "descriptor":
            valid_current = (
                valid[current]
                & valid[current - stride]
                & valid[current - 2 * stride]
            )
        else:
            valid_current = valid[current] & valid[previous]
        current = current[valid_current]
        previous = previous[valid_current]
        if current.numel() == 0:
            continue

        if representation == "absolute":
            vectors = features[current]
        elif representation == "residual":
            vectors = features[current] - features[previous]
        else:
            vectors = torch.cat(
                [
                    features[current - 2 * stride],
                    features[current - stride],
                    features[current],
                ],
                dim=1,
            )
        thumbnails = _episode_thumbnails(episode, view)
        image_motion, proprio_motion, action_magnitude = _motion_scalars(
            episode,
            thumbnails,
            current,
            previous,
        )
        vector_parts.append(vectors)
        episode_ids.extend([episode.episode_id] * current.numel())
        time_parts.append(current)
        splits.extend([episode.split] * current.numel())
        image_motion_parts.append(image_motion)
        proprio_motion_parts.append(proprio_motion)
        action_magnitude_parts.append(action_magnitude)
        previous_thumbnail_parts.append(thumbnails[previous])
        current_thumbnail_parts.append(thumbnails[current])

    if not vector_parts:
        raise ValueError(
            f"No samples for camera={camera}, pool={pool}, "
            f"representation={representation}, stride={stride}."
        )
    return ProbeSamples(
        vectors=torch.cat(vector_parts).contiguous(),
        episode_ids=tuple(episode_ids),
        time_indices=torch.cat(time_parts),
        splits=tuple(splits),
        camera=camera,
        pool=pool,
        representation=representation,
        stride=stride,
        image_motion=torch.cat(image_motion_parts),
        proprio_motion=torch.cat(proprio_motion_parts),
        action_magnitude=torch.cat(action_magnitude_parts),
        previous_thumbnails=torch.cat(previous_thumbnail_parts),
        current_thumbnails=torch.cat(current_thumbnail_parts),
    )


def _median(values: torch.Tensor) -> float:
    finite = values.detach().float()
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        return float("nan")
    return float(finite.median().item())


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    sorted_values = values[order]
    starts = torch.ones(values.numel(), dtype=torch.bool, device=values.device)
    starts[1:] = sorted_values[1:] != sorted_values[:-1]
    group = starts.cumsum(dim=0) - 1
    positions = torch.arange(
        values.numel(),
        dtype=torch.float64,
        device=values.device,
    )
    counts = torch.bincount(group).double()
    sums = torch.zeros_like(counts)
    sums.index_add_(0, group, positions)
    average = sums / counts
    ranks = torch.empty_like(values, dtype=torch.float64)
    ranks[order] = average[group]
    return ranks


def spearman_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().double().flatten()
    b = b.detach().double().flatten()
    valid = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[valid], b[valid]
    if a.numel() < 3:
        return float("nan")
    rank_a = _rank(a)
    rank_b = _rank(b)
    rank_a = rank_a - rank_a.mean()
    rank_b = rank_b - rank_b.mean()
    denominator = rank_a.square().sum().sqrt() * rank_b.square().sum().sqrt()
    if float(denominator.item()) <= 0:
        return float("nan")
    return float((rank_a * rank_b).sum().div(denominator).item())


def _rms_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).square().mean(dim=1).sqrt()


def _effective_rank(
    vectors: torch.Tensor,
    device: torch.device,
) -> tuple[float, int, float]:
    values = vectors.float()
    values = values - values.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(values.to(device=device))
    eigenvalues = singular_values.double().square()
    total = eigenvalues.sum().clamp_min(1e-12)
    effective = float((total.square() / eigenvalues.square().sum().clamp_min(1e-12)).item())
    cumulative = torch.cumsum(eigenvalues.sort(descending=True).values, dim=0) / total
    rank95 = int(
        torch.searchsorted(
            cumulative,
            torch.tensor(0.95, device=device, dtype=cumulative.dtype),
        ).item()
    ) + 1
    variance = float(values.square().mean().item())
    return effective, rank95, variance


def state_metrics(
    episodes: Sequence[PooledFeatureEpisode],
    cameras: Sequence[str],
    pools: Sequence[int],
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for camera, pool in itertools.product(cameras, pools):
        vectors = []
        episode_count = 0
        for episode in episodes:
            view = _camera_index(episode, camera)
            valid = episode.valid_mask[:, view]
            values = episode.pooled(pool)[:, view].float().flatten(1)[valid]
            if values.numel():
                vectors.append(values)
                episode_count += 1
        all_vectors = torch.cat(vectors)
        effective, rank95, variance = _effective_rank(all_vectors, device=device)
        rows.append(
            {
                "camera": camera,
                "pool": int(pool),
                "episodes": episode_count,
                "ticks": int(all_vectors.shape[0]),
                "dimension": int(all_vectors.shape[1]),
                "mean_feature_variance": variance,
                "effective_rank": effective,
                "rank_95_percent": rank95,
                "effective_rank_fraction": effective
                / float(min(all_vectors.shape[0] - 1, all_vectors.shape[1])),
            }
        )
    return rows


def _cross_episode_distances(
    episode_vectors: Sequence[torch.Tensor],
    limit_per_pair: int = 256,
) -> torch.Tensor:
    distances: list[torch.Tensor] = []
    for first, second in zip(episode_vectors, episode_vectors[1:] + episode_vectors[:1]):
        count = min(first.shape[0], second.shape[0], int(limit_per_pair))
        if count:
            distances.append(_rms_distance(first[:count], second.flip(0)[:count]))
    return torch.cat(distances) if distances else torch.empty(0)


def motion_metrics(
    episodes: Sequence[PooledFeatureEpisode],
    cameras: Sequence[str],
    pools: Sequence[int],
    strides: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for camera, pool, stride in itertools.product(cameras, pools, strides):
        residual_norms: list[torch.Tensor] = []
        image_motions: list[torch.Tensor] = []
        proprio_motions: list[torch.Tensor] = []
        action_magnitudes: list[torch.Tensor] = []
        adjacent_distances: list[torch.Tensor] = []
        stride_distances: list[torch.Tensor] = []
        far_distances: list[torch.Tensor] = []
        episode_vectors: list[torch.Tensor] = []
        episode_image_correlations: list[float] = []
        episode_proprio_correlations: list[float] = []

        for episode in episodes:
            view = _camera_index(episode, camera)
            features = episode.pooled(pool)[:, view].float().flatten(1)
            episode_vectors.append(features)
            thumbnails = _episode_thumbnails(episode, view)
            if episode.ticks > 1:
                adjacent_distances.append(_rms_distance(features[1:], features[:-1]))
            if episode.ticks <= stride:
                continue
            current = torch.arange(stride, episode.ticks)
            previous = current - stride
            latent_motion = _rms_distance(features[current], features[previous])
            image_motion, proprio_motion, action_magnitude = _motion_scalars(
                episode,
                thumbnails,
                current,
                previous,
            )
            residual_norms.append(latent_motion)
            image_motions.append(image_motion)
            proprio_motions.append(proprio_motion)
            action_magnitudes.append(action_magnitude)
            image_correlation = spearman_correlation(latent_motion, image_motion)
            proprio_correlation = spearman_correlation(latent_motion, proprio_motion)
            if math.isfinite(image_correlation):
                episode_image_correlations.append(image_correlation)
            if math.isfinite(proprio_correlation):
                episode_proprio_correlations.append(proprio_correlation)
            stride_distances.append(latent_motion)
            far = 2 * stride
            if episode.ticks > far:
                far_distances.append(_rms_distance(features[far:], features[:-far]))

        residual = torch.cat(residual_norms)
        image = torch.cat(image_motions)
        proprio = torch.cat(proprio_motions)
        action = torch.cat(action_magnitudes)
        low_threshold = torch.quantile(image, 0.25)
        high_threshold = torch.quantile(image, 0.75)
        low = residual[image <= low_threshold]
        high = residual[image >= high_threshold]
        low_median = _median(low)
        high_median = _median(high)
        cross = _cross_episode_distances(episode_vectors)
        adjacent = torch.cat(adjacent_distances)
        stride_values = torch.cat(stride_distances)
        far_values = torch.cat(far_distances) if far_distances else torch.empty(0)
        adjacent_median = _median(adjacent)
        stride_median = _median(stride_values)
        far_median = _median(far_values)
        cross_median = _median(cross)
        rows.append(
            {
                "camera": camera,
                "pool": int(pool),
                "stride": int(stride),
                "pairs": int(residual.numel()),
                "latent_residual_rms_median": _median(residual),
                "image_l1_median": _median(image),
                "proprio_rms_median": _median(proprio),
                "action_rms_median": _median(action),
                "spearman_latent_image": spearman_correlation(residual, image),
                "spearman_latent_proprio": spearman_correlation(residual, proprio),
                "spearman_latent_action": spearman_correlation(residual, action),
                "median_episode_spearman_latent_image": (
                    float(torch.tensor(episode_image_correlations).median().item())
                    if episode_image_correlations
                    else float("nan")
                ),
                "positive_episode_fraction_latent_image": (
                    sum(value > 0 for value in episode_image_correlations)
                    / len(episode_image_correlations)
                    if episode_image_correlations
                    else float("nan")
                ),
                "median_episode_spearman_latent_proprio": (
                    float(torch.tensor(episode_proprio_correlations).median().item())
                    if episode_proprio_correlations
                    else float("nan")
                ),
                "low_image_motion_latent_median": low_median,
                "high_image_motion_latent_median": high_median,
                "motion_separation_ratio": high_median / max(low_median, 1e-12),
                "adjacent_distance_median": adjacent_median,
                "stride_distance_median": stride_median,
                "far_distance_median": far_median,
                "cross_episode_distance_median": cross_median,
                "distance_ordered": bool(
                    adjacent_median <= stride_median <= far_median <= cross_median
                ),
            }
        )
    return rows


def _standardized_splits(
    samples: ProbeSamples,
) -> tuple[
    dict[str, ProbeSamples],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    sample_splits = {name: samples.select(name) for name in ("train", "val", "test")}
    mean = sample_splits["train"].vectors.float().mean(dim=0)
    std = (
        sample_splits["train"]
        .vectors.float()
        .var(dim=0, unbiased=False)
        .sqrt()
        .clamp_min(1e-6)
    )
    normalized = {
        name: (value.vectors.float() - mean) / std
        for name, value in sample_splits.items()
    }
    return sample_splits, normalized, mean, std


def _evaluate_centers(
    values: torch.Tensor,
    centers: torch.Tensor,
    device: torch.device,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device_centers = centers.to(device=device, dtype=torch.float32)
    all_codes: list[torch.Tensor] = []
    all_distances: list[torch.Tensor] = []
    for start in range(0, values.shape[0], int(chunk_size)):
        code, distance = assign_nearest(
            values[start : start + int(chunk_size)].to(
                device=device,
                dtype=torch.float32,
            ),
            device_centers,
        )
        all_codes.append(code.cpu())
        all_distances.append(distance.cpu())
    return torch.cat(all_codes), torch.cat(all_distances)


def _kmeans_config(
    config: LatentProbeConfig,
    k: int,
    seed: int,
    tolerance: float,
    patience: int,
) -> DiagnosticKMeansConfig:
    return DiagnosticKMeansConfig(
        k=int(k),
        max_iters=config.max_iters,
        min_iters=config.min_iters,
        tol=float(tolerance),
        patience=int(patience),
        seed=int(seed),
        chunk_size=config.chunk_size,
        device=config.device,
    )


def _kmeans_row(
    run_id: str,
    run_type: str,
    samples: ProbeSamples,
    normalized: dict[str, torch.Tensor],
    result: DiagnosticKMeansResult,
    test_codes: torch.Tensor,
    test_distances: torch.Tensor,
    k: int,
    seed: int,
    tolerance: float,
    patience: int,
) -> dict[str, Any]:
    usage = usage_summary(test_codes, k=k)
    final = result.history[-1]
    return {
        "run_id": run_id,
        "run_type": run_type,
        "camera": samples.camera,
        "pool": samples.pool,
        "stride": samples.stride,
        "representation": samples.representation,
        "k": int(k),
        "seed": int(seed),
        "tolerance": float(tolerance),
        "patience": int(patience),
        "dimension": samples.dimension,
        "train_vectors": int(normalized["train"].shape[0]),
        "validation_vectors": int(normalized["val"].shape[0]),
        "test_vectors": int(normalized["test"].shape[0]),
        "iterations": result.iterations,
        "converged": result.converged,
        "stop_reason": result.stop_reason,
        "train_mse_per_dimension": result.train_inertia / samples.dimension,
        "validation_mse_per_dimension": (
            None
            if result.validation_inertia is None
            else result.validation_inertia / samples.dimension
        ),
        "test_mse_per_dimension": float(test_distances.mean().item())
        / samples.dimension,
        "final_relative_improvement": final.relative_improvement,
        "final_center_shift": final.relative_center_shift,
        "final_assignment_change": final.assignment_change,
        "final_empty_clusters": final.empty_clusters,
        **usage,
    }


def _iteration_rows(
    run_id: str,
    result: DiagnosticKMeansResult,
    dimension: int,
) -> list[dict[str, Any]]:
    rows = []
    for item in result.history:
        row = {"run_id": run_id, **item.to_dict()}
        row["train_mse_per_dimension"] = item.train_inertia / dimension
        row["validation_mse_per_dimension"] = (
            None
            if item.validation_inertia is None
            else item.validation_inertia / dimension
        )
        rows.append(row)
    return rows


def _pairwise_stability(
    assignments: dict[tuple[Any, ...], list[tuple[int, torch.Tensor]]],
    label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, values in assignments.items():
        pair_scores = []
        for (seed_a, codes_a), (seed_b, codes_b) in itertools.combinations(values, 2):
            pair_scores.append(adjusted_rand_index(codes_a, codes_b))
            rows.append(
                {
                    "kind": label,
                    "group": "|".join(str(value) for value in key),
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "ari": pair_scores[-1],
                }
            )
        if pair_scores:
            rows.append(
                {
                    "kind": f"{label}_mean",
                    "group": "|".join(str(value) for value in key),
                    "seed_a": "",
                    "seed_b": "",
                    "ari": sum(pair_scores) / len(pair_scores),
                }
            )
    return rows


def _cluster_montage(
    path: Path,
    samples: ProbeSamples,
    codes: torch.Tensor,
    distances: torch.Tensor,
    max_clusters: int = 12,
) -> None:
    counts = torch.bincount(codes, minlength=int(codes.max().item()) + 1)
    clusters = torch.argsort(counts, descending=True)
    clusters = clusters[counts[clusters] > 0][: int(max_clusters)]
    if clusters.numel() == 0:
        return
    tile = int(samples.current_thumbnails.shape[-1])
    label_height = 18
    canvas = Image.new(
        "RGB",
        (tile * 3, int(clusters.numel()) * (tile + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row, cluster in enumerate(clusters.tolist()):
        members = torch.nonzero(codes == cluster, as_tuple=False).flatten()
        index = int(members[torch.argmin(distances[members])].item())
        previous = samples.previous_thumbnails[index]
        current = samples.current_thumbnails[index]
        difference = (
            current.float().sub(previous.float()).abs().mul(3.0).clamp(0, 255).byte()
        )
        y = row * (tile + label_height)
        for column, tensor in enumerate((previous, current, difference)):
            image = Image.fromarray(tensor.permute(1, 2, 0).numpy(), mode="RGB")
            canvas.paste(image, (column * tile, y))
        draw.text(
            (2, y + tile + 2),
            f"cluster {cluster} n={int(counts[cluster])} t={int(samples.time_indices[index])}",
            fill="black",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _run_kmeans_sweeps(
    episodes: Sequence[PooledFeatureEpisode],
    config: LatentProbeConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    device = _resolve_device(config.device)
    rows: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []
    assignments: dict[tuple[Any, ...], list[tuple[int, torch.Tensor]]] = {}

    for camera, pool in itertools.product(
        config.cameras,
        config.pools,
    ):
        samples = build_probe_samples(
            episodes,
            camera=camera,
            pool=pool,
            representation="descriptor",
            stride=config.sweep_stride,
        )
        _, normalized, _, _ = _standardized_splits(samples)
        for k, seed in itertools.product(config.k_values, config.seeds):
            run_id = (
                f"capacity_{camera}_g{pool}_q{config.sweep_stride}_k{k}_seed{seed}"
            )
            result = fit_diagnostic_kmeans(
                normalized["train"],
                normalized["val"],
                _kmeans_config(
                    config,
                    k=k,
                    seed=seed,
                    tolerance=config.default_tolerance,
                    patience=config.default_patience,
                ),
            )
            test_codes, test_distances = _evaluate_centers(
                normalized["test"],
                result.centers,
                device=device,
                chunk_size=config.chunk_size,
            )
            rows.append(
                _kmeans_row(
                    run_id,
                    "capacity_pool",
                    samples,
                    normalized,
                    result,
                    test_codes,
                    test_distances,
                    k,
                    seed,
                    config.default_tolerance,
                    config.default_patience,
                )
            )
            iterations.extend(_iteration_rows(run_id, result, samples.dimension))
            key = (
                "capacity_pool",
                camera,
                pool,
                config.sweep_stride,
                k,
                config.default_tolerance,
                config.default_patience,
            )
            assignments.setdefault(key, []).append((seed, test_codes))
        print(
            f"Capacity sweep complete: camera={camera}, pool={pool}, "
            f"fits={len(config.k_values) * len(config.seeds)}",
            flush=True,
        )

    for camera in config.cameras:
        samples = build_probe_samples(
            episodes,
            camera=camera,
            pool=config.selected_pool,
            representation="descriptor",
            stride=config.sweep_stride,
        )
        _, normalized, _, _ = _standardized_splits(samples)
        for tolerance, patience, seed in itertools.product(
            config.tolerances,
            config.patiences,
            config.seeds,
        ):
            run_id = (
                f"stop_{camera}_g{config.selected_pool}_q{config.sweep_stride}"
                f"_k{config.selected_k}_tol{tolerance:g}_p{patience}_seed{seed}"
            )
            result = fit_diagnostic_kmeans(
                normalized["train"],
                normalized["val"],
                _kmeans_config(
                    config,
                    k=config.selected_k,
                    seed=seed,
                    tolerance=tolerance,
                    patience=patience,
                ),
            )
            test_codes, test_distances = _evaluate_centers(
                normalized["test"],
                result.centers,
                device=device,
                chunk_size=config.chunk_size,
            )
            rows.append(
                _kmeans_row(
                    run_id,
                    "early_stop",
                    samples,
                    normalized,
                    result,
                    test_codes,
                    test_distances,
                    config.selected_k,
                    seed,
                    tolerance,
                    patience,
                )
            )
            iterations.extend(_iteration_rows(run_id, result, samples.dimension))
            key = (
                "early_stop",
                camera,
                config.selected_pool,
                config.sweep_stride,
                config.selected_k,
                tolerance,
                patience,
            )
            assignments.setdefault(key, []).append((seed, test_codes))
        print(
            f"Early-stop sweep complete: camera={camera}, "
            f"fits={len(config.tolerances) * len(config.patiences) * len(config.seeds)}",
            flush=True,
        )

    return rows, iterations, _pairwise_stability(assignments, "kmeans")


def _capacity_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row["run_type"] != "capacity_pool":
            continue
        key = (str(row["camera"]), int(row["pool"]), int(row["k"]))
        grouped.setdefault(key, []).append(row)
    summaries = []
    for (camera, pool, k), group in sorted(grouped.items()):
        summaries.append(
            {
                "camera": camera,
                "pool": pool,
                "k": k,
                "seeds": len(group),
                "mean_iterations": sum(float(row["iterations"]) for row in group)
                / len(group),
                "mean_validation_mse_per_dimension": sum(
                    float(row["validation_mse_per_dimension"]) for row in group
                )
                / len(group),
                "mean_test_mse_per_dimension": sum(
                    float(row["test_mse_per_dimension"]) for row in group
                )
                / len(group),
                "mean_test_dead_fraction": sum(
                    float(row["dead_fraction"]) for row in group
                )
                / len(group),
                "mean_test_perplexity_fraction": sum(
                    float(row["perplexity_fraction"]) for row in group
                )
                / len(group),
                "mean_test_maximum_cluster_fraction": sum(
                    float(row["maximum_cluster_fraction"]) for row in group
                )
                / len(group),
            }
        )
    return summaries


def _reductions(values: Sequence[float]) -> list[float]:
    return [
        1.0 - float(after) / max(float(before), 1e-12)
        for before, after in zip(values, values[1:])
    ]


def _run_rq(
    episodes: Sequence[PooledFeatureEpisode],
    config: LatentProbeConfig,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device = _resolve_device(config.device)
    rows: list[dict[str, Any]] = []
    assignments: dict[tuple[Any, ...], list[tuple[int, torch.Tensor]]] = {}
    artifact_dir = ensure_dir(output_dir / "artifacts")
    montage_dir = ensure_dir(output_dir / "montages")

    for camera, stride in itertools.product(
        config.cameras,
        config.strides,
    ):
        samples = build_probe_samples(
            episodes,
            camera=camera,
            pool=config.selected_pool,
            representation="descriptor",
            stride=stride,
        )
        split_samples, normalized, mean, std = _standardized_splits(samples)
        for seed in config.seeds:
            result = fit_diagnostic_rq(
                normalized["train"],
                normalized["val"],
                normalized["test"],
                _kmeans_config(
                    config,
                    k=config.selected_k,
                    seed=seed,
                    tolerance=config.default_tolerance,
                    patience=config.default_patience,
                ),
                levels=config.rq_levels,
            )
            assert result.validation_residual_mse is not None
            assert result.test_residual_mse is not None
            assert result.test_codes is not None
            train_reductions = _reductions(result.train_residual_mse)
            validation_reductions = _reductions(result.validation_residual_mse)
            test_reductions = _reductions(result.test_residual_mse)
            row: dict[str, Any] = {
                "run_id": (
                    f"rq_{camera}_g{config.selected_pool}_q{stride}"
                    f"_k{config.selected_k}_seed{seed}"
                ),
                "camera": camera,
                "pool": config.selected_pool,
                "stride": stride,
                "k": config.selected_k,
                "levels": config.rq_levels,
                "seed": seed,
                "dimension": samples.dimension,
                "train_vectors": int(normalized["train"].shape[0]),
                "validation_vectors": int(normalized["val"].shape[0]),
                "test_vectors": int(normalized["test"].shape[0]),
                "train_initial_mse": result.train_residual_mse[0],
                "validation_initial_mse": result.validation_residual_mse[0],
                "test_initial_mse": result.test_residual_mse[0],
                "train_final_mse": result.train_residual_mse[-1],
                "validation_final_mse": result.validation_residual_mse[-1],
                "test_final_mse": result.test_residual_mse[-1],
                "train_total_reduction": 1.0
                - result.train_residual_mse[-1]
                / max(result.train_residual_mse[0], 1e-12),
                "validation_total_reduction": 1.0
                - result.validation_residual_mse[-1]
                / max(result.validation_residual_mse[0], 1e-12),
                "test_total_reduction": 1.0
                - result.test_residual_mse[-1]
                / max(result.test_residual_mse[0], 1e-12),
            }
            for level in range(config.rq_levels):
                prefix = f"level{level + 1}"
                row[f"{prefix}_iterations"] = result.levels[level].iterations
                row[f"{prefix}_stop_reason"] = result.levels[level].stop_reason
                row[f"{prefix}_train_reduction"] = train_reductions[level]
                row[f"{prefix}_validation_reduction"] = validation_reductions[level]
                row[f"{prefix}_test_reduction"] = test_reductions[level]
                row.update(
                    {
                        f"{prefix}_{key}": value
                        for key, value in usage_summary(
                            result.test_codes[:, level],
                            k=config.selected_k,
                        ).items()
                    }
                )
            rows.append(row)
            key = (
                camera,
                config.selected_pool,
                stride,
                config.selected_k,
                config.rq_levels,
            )
            for level in range(config.rq_levels):
                assignments.setdefault((*key, level + 1), []).append(
                    (seed, result.test_codes[:, level])
                )

            centers = [level.centers for level in result.levels]
            atomic_torch_save(
                {
                    "schema": "codewam.latent-probe-rq-artifact.v1",
                    "camera": camera,
                    "pool": config.selected_pool,
                    "stride": stride,
                    "k": config.selected_k,
                    "levels": config.rq_levels,
                    "seed": seed,
                    "normalization_mean": mean,
                    "normalization_std": std,
                    "centers": centers,
                    "metrics": row,
                },
                artifact_dir / f"{row['run_id']}.pt",
            )
            if seed == config.seeds[0]:
                level1_codes, level1_distances = _evaluate_centers(
                    normalized["test"],
                    centers[0],
                    device=device,
                    chunk_size=config.chunk_size,
                )
                _cluster_montage(
                    montage_dir / f"{camera}_q{stride}_level1.png",
                    split_samples["test"],
                    level1_codes,
                    level1_distances,
                )
        print(
            f"RQ sweep complete: camera={camera}, stride={stride}, "
            f"fits={len(config.seeds)}",
            flush=True,
        )
    return rows, _pairwise_stability(assignments, "rq_level")


def _recommend_early_stop(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    recommendations = []
    cameras = sorted({str(row["camera"]) for row in rows if row["run_type"] == "early_stop"})
    for camera in cameras:
        candidates: dict[tuple[float, int], list[dict[str, Any]]] = {}
        for row in rows:
            if row["run_type"] != "early_stop" or row["camera"] != camera:
                continue
            key = (float(row["tolerance"]), int(row["patience"]))
            candidates.setdefault(key, []).append(row)
        summaries = []
        for (tolerance, patience), group in candidates.items():
            summaries.append(
                {
                    "camera": camera,
                    "tolerance": tolerance,
                    "patience": patience,
                    "mean_iterations": sum(float(row["iterations"]) for row in group)
                    / len(group),
                    "mean_validation_mse": sum(
                        float(row["validation_mse_per_dimension"]) for row in group
                    )
                    / len(group),
                    "mean_test_mse": sum(
                        float(row["test_mse_per_dimension"]) for row in group
                    )
                    / len(group),
                }
            )
        best_validation = min(row["mean_validation_mse"] for row in summaries)
        eligible = [
            row
            for row in summaries
            if row["mean_validation_mse"] <= best_validation * 1.005
        ]
        selected = min(
            eligible,
            key=lambda row: (row["mean_iterations"], row["tolerance"], row["patience"]),
        )
        recommendations.append(selected)
    return recommendations


def _write_markdown_report(
    path: Path,
    episodes: Sequence[PooledFeatureEpisode],
    state_rows: Sequence[dict[str, Any]],
    motion_rows: Sequence[dict[str, Any]],
    capacity_rows: Sequence[dict[str, Any]],
    kmeans_run_count: int,
    rq_rows: Sequence[dict[str, Any]],
    recommendations: Sequence[dict[str, Any]],
    config: LatentProbeConfig,
) -> None:
    split_counts = {
        split: sum(episode.split == split for episode in episodes)
        for split in ("train", "val", "test")
    }
    lines = [
        "# Wan latent small-sample probe",
        "",
        f"- Episodes: {len(episodes)} ({split_counts})",
        f"- Cameras: {', '.join(config.cameras)}",
        f"- Pools: {list(config.pools)}; temporal families: {list(config.strides)}",
        "- Scope: visible scene motion and robot-state correlation. DROID has no object masks, "
        "so this report does not claim object-only motion isolation.",
        "",
        "## Absolute latent state",
        "",
        "| camera | g | ticks | dim | effective rank | rank95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in state_rows:
        lines.append(
            f"| {row['camera']} | {row['pool']} | {row['ticks']} | {row['dimension']} | "
            f"{row['effective_rank']:.2f} | {row['rank_95_percent']} |"
        )
    lines.extend(
        [
            "",
            "## Residual-motion alignment",
            "",
            "| camera | g | s | pooled rho | episode rho | positive episodes | latent-proprio rho | high/low motion |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in motion_rows:
        if row["pool"] != config.selected_pool:
            continue
        lines.append(
            f"| {row['camera']} | {row['pool']} | {row['stride']} | "
            f"{row['spearman_latent_image']:.3f} | "
            f"{row['median_episode_spearman_latent_image']:.3f} | "
            f"{row['positive_episode_fraction_latent_image']:.3f} | "
            f"{row['median_episode_spearman_latent_proprio']:.3f} | "
            f"{row['motion_separation_ratio']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Capacity and pooling sweep",
            "",
            "| camera | g | K | val MSE/dim | test MSE/dim | dead fraction | perplexity/K | max cluster |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in capacity_rows:
        lines.append(
            f"| {row['camera']} | {row['pool']} | {row['k']} | "
            f"{row['mean_validation_mse_per_dimension']:.6f} | "
            f"{row['mean_test_mse_per_dimension']:.6f} | "
            f"{row['mean_test_dead_fraction']:.3f} | "
            f"{row['mean_test_perplexity_fraction']:.3f} | "
            f"{row['mean_test_maximum_cluster_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## K-Means stopping recommendation",
            "",
            "The recommendation is the fastest setting whose mean validation distortion is "
            "within 0.5% of the best tested setting. It is a P0 engineering choice, not a "
            "research-level capacity decision.",
            "",
            "| camera | tolerance | patience | mean iterations | validation MSE/dim | test MSE/dim |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in recommendations:
        lines.append(
            f"| {row['camera']} | {row['tolerance']:.1e} | {row['patience']} | "
            f"{row['mean_iterations']:.2f} | {row['mean_validation_mse']:.6f} | "
            f"{row['mean_test_mse']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## RQ-3 held-out behavior",
            "",
            "| camera | Q | seed | test total reduction | L1 | L2 | L3 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rq_rows:
        lines.append(
            f"| {row['camera']} | {row['stride']} | {row['seed']} | "
            f"{row['test_total_reduction']:.3f} | "
            f"{row['level1_test_reduction']:.3f} | "
            f"{row['level2_test_reduction']:.3f} | "
            f"{row['level3_test_reduction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This run can reject a broken representation or unstable clustering setup. "
            "It cannot choose the final K from DROID-100, prove object semantics without masks, "
            "or replace DROID-10k held-out retrieval/action probes.",
            "",
            f"Detailed runs: {kmeans_run_count} K-Means fits and {len(rq_rows)} RQ fits.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_latent_probe(config: LatentProbeConfig) -> dict[str, Any]:
    output_dir = ensure_dir(config.output_dir)
    episodes = list(iter_pooled_feature_episodes(config.pooled_shards))
    if not episodes:
        raise ValueError("No pooled episodes were loaded.")
    ids = [episode.episode_id for episode in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Probe pooled shards contain duplicate episode ids.")
    available_cameras = set.intersection(*(set(episode.camera_ids) for episode in episodes))
    missing = sorted(set(config.cameras) - available_cameras)
    if missing:
        raise ValueError(f"Probe cameras absent from one or more episodes: {missing}.")
    split_counts = {
        split: sum(episode.split == split for episode in episodes)
        for split in ("train", "val", "test")
    }
    if any(count == 0 for count in split_counts.values()):
        raise ValueError(f"Probe needs episode-level train/val/test splits, got {split_counts}.")

    print(
        f"Probe loaded {len(episodes)} episodes with splits {split_counts}.",
        flush=True,
    )
    device = _resolve_device(config.device)
    print("Computing latent state and motion diagnostics.", flush=True)
    state_rows = state_metrics(episodes, config.cameras, config.pools, device=device)
    motion_rows = motion_metrics(
        episodes,
        config.cameras,
        config.pools,
        config.strides,
    )
    print("Running K-Means capacity and early-stop sweeps.", flush=True)
    kmeans_rows, iteration_rows, kmeans_stability = _run_kmeans_sweeps(
        episodes,
        config,
    )
    capacity_rows = _capacity_summary(kmeans_rows)
    print("Running three-level residual-quantization sweeps.", flush=True)
    rq_rows, rq_stability = _run_rq(episodes, config, output_dir)
    stability_rows = [*kmeans_stability, *rq_stability]
    recommendations = _recommend_early_stop(kmeans_rows)

    write_rows_tsv(output_dir / "state_metrics.tsv", list(state_rows))
    write_rows_tsv(output_dir / "motion_metrics.tsv", list(motion_rows))
    write_rows_tsv(output_dir / "kmeans_runs.tsv", list(kmeans_rows))
    write_rows_tsv(output_dir / "capacity_summary.tsv", list(capacity_rows))
    write_rows_tsv(output_dir / "kmeans_iterations.tsv", list(iteration_rows))
    write_rows_tsv(output_dir / "rq_runs.tsv", list(rq_rows))
    write_rows_tsv(output_dir / "seed_stability.tsv", list(stability_rows))
    write_rows_tsv(
        output_dir / "early_stop_recommendations.tsv",
        list(recommendations),
    )
    payload = {
        "schema": LATENT_PROBE_SCHEMA,
        "config": asdict(config),
        "episode_counts": split_counts,
        "state_metrics": state_rows,
        "motion_metrics": motion_rows,
        "kmeans_runs": kmeans_rows,
        "capacity_summary": capacity_rows,
        "rq_runs": rq_rows,
        "seed_stability": stability_rows,
        "early_stop_recommendations": recommendations,
    }
    save_json(output_dir / "report.json", payload)
    _write_markdown_report(
        output_dir / "report.md",
        episodes,
        state_rows,
        motion_rows,
        capacity_rows,
        len(kmeans_rows),
        rq_rows,
        recommendations,
        config,
    )
    return payload


def probe_config_from_mapping(mapping: dict[str, Any]) -> LatentProbeConfig:
    values = dict(mapping)
    for key in (
        "pooled_shards",
        "cameras",
        "pools",
        "strides",
        "k_values",
        "seeds",
        "tolerances",
        "patiences",
    ):
        if key in values:
            values[key] = tuple(values[key])
    return LatentProbeConfig(**values)
