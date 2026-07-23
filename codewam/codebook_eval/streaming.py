from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from .manifest import SplitName
from .shards import PooledFeatureEpisode, atomic_torch_save, load_torch_payload


TensorBatchFactory = Callable[[], Iterable[torch.Tensor]]
EpisodeFactory = Callable[[], Iterable[PooledFeatureEpisode]]
KMEANS_CHECKPOINT_SCHEMA = "codewam.streaming-kmeans.v1"
RQ_CHECKPOINT_SCHEMA = "codewam.streaming-rq.v1"
FROZEN_RQ_SCHEMA = "codewam.frozen-rq.v1"
REQUIRED_ARTIFACT_METADATA = frozenset(
    {
        "manifest_fingerprint",
        "dataset_revision",
        "wan_model_id",
        "wan_revision",
        "preprocess_revision",
        "config_hash",
        "source_checksums",
    }
)


def _distributed_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _is_primary_rank() -> bool:
    return not _distributed_ready() or torch.distributed.get_rank() == 0


def _resolve_device(spec: str | torch.device) -> torch.device:
    if isinstance(spec, torch.device):
        return spec
    lowered = str(spec).lower()
    if lowered == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


@dataclass(frozen=True)
class CausalDescriptorSpec:
    stride: int
    pool: int = 4
    max_gap_factor: float | None = 1.5

    def __post_init__(self) -> None:
        if int(self.stride) <= 0:
            raise ValueError(f"`stride` must be positive, got {self.stride}.")
        if int(self.pool) not in {1, 2, 4}:
            raise ValueError(f"`pool` must be one of 1, 2, 4; got {self.pool}.")
        if self.max_gap_factor is not None and float(self.max_gap_factor) < 1.0:
            raise ValueError("`max_gap_factor` must be >= 1 or None.")

    @property
    def family(self) -> str:
        return f"Q{self.stride}"

    @property
    def offsets(self) -> tuple[int, int, int]:
        return (-2 * self.stride, -self.stride, 0)


@dataclass(frozen=True)
class CausalDescriptorBatch:
    spec: CausalDescriptorSpec
    vectors: torch.Tensor
    episode_ids: tuple[str, ...]
    time_indices: torch.Tensor
    timestamps: torch.Tensor
    splits: tuple[str, ...]

    def __post_init__(self) -> None:
        size = int(self.vectors.shape[0])
        if self.vectors.ndim != 2:
            raise ValueError(f"`vectors` must be [N,D], got {tuple(self.vectors.shape)}.")
        if len(self.episode_ids) != size or len(self.splits) != size:
            raise ValueError("Descriptor metadata length does not match the vector batch.")
        if self.time_indices.shape != (size,) or self.timestamps.shape != (size,):
            raise ValueError("Descriptor index metadata must be one-dimensional and batch-aligned.")


@dataclass(frozen=True)
class CausalDescriptorSource:
    episode_factory: EpisodeFactory
    spec: CausalDescriptorSpec
    batch_size: int = 8192
    split: SplitName | None = None

    def __post_init__(self) -> None:
        if int(self.batch_size) <= 0:
            raise ValueError(f"`batch_size` must be positive, got {self.batch_size}.")

    def __iter__(self) -> Iterator[CausalDescriptorBatch]:
        vector_parts: list[torch.Tensor] = []
        episode_ids: list[str] = []
        time_parts: list[torch.Tensor] = []
        timestamp_parts: list[torch.Tensor] = []
        splits: list[str] = []
        pending = 0

        def emit() -> CausalDescriptorBatch:
            nonlocal vector_parts, episode_ids, time_parts, timestamp_parts, splits, pending
            batch = CausalDescriptorBatch(
                spec=self.spec,
                vectors=torch.cat(vector_parts, dim=0).contiguous(),
                episode_ids=tuple(episode_ids),
                time_indices=torch.cat(time_parts, dim=0),
                timestamps=torch.cat(timestamp_parts, dim=0),
                splits=tuple(splits),
            )
            vector_parts = []
            episode_ids = []
            time_parts = []
            timestamp_parts = []
            splits = []
            pending = 0
            return batch

        for episode in self.episode_factory():
            if self.split is not None and episode.split != self.split:
                continue
            stride = self.spec.stride
            if episode.ticks <= 2 * stride:
                continue

            pooled = episode.pooled(self.spec.pool)
            features = pooled.reshape(episode.ticks, -1)
            current_indices = torch.arange(
                2 * stride,
                episode.ticks,
                dtype=torch.long,
                device=features.device,
            )
            valid_mask = episode.valid_mask.to(device=features.device)
            valid = (
                valid_mask[current_indices - 2 * stride].all(dim=1)
                & valid_mask[current_indices - stride].all(dim=1)
                & valid_mask[current_indices].all(dim=1)
            )

            timestamps = episode.timestamps.to(device=features.device)
            if self.spec.max_gap_factor is not None and episode.ticks > 1:
                cadence = torch.median(timestamps[1:] - timestamps[:-1])
                maximum_gap = cadence * stride * float(self.spec.max_gap_factor)
                first_gap = timestamps[current_indices - stride] - timestamps[
                    current_indices - 2 * stride
                ]
                second_gap = timestamps[current_indices] - timestamps[current_indices - stride]
                valid &= (first_gap <= maximum_gap) & (second_gap <= maximum_gap)
            current_indices = current_indices[valid]

            offset = 0
            while offset < current_indices.numel():
                take = min(self.batch_size - pending, current_indices.numel() - offset)
                index = current_indices[offset : offset + take]
                vectors = torch.cat(
                    [
                        features[index - 2 * stride],
                        features[index - stride],
                        features[index],
                    ],
                    dim=1,
                )
                vector_parts.append(vectors)
                episode_ids.extend([episode.episode_id] * take)
                time_parts.append(index.detach().cpu())
                timestamp_parts.append(timestamps[index].detach().cpu())
                splits.extend([episode.split] * take)
                pending += take
                offset += take
                if pending == self.batch_size:
                    yield emit()

        if pending:
            yield emit()

    def vector_batch_factory(
        self,
        normalization: NormalizationStats | None = None,
        device: str | torch.device = "cpu",
    ) -> TensorBatchFactory:
        target_device = _resolve_device(device)

        def batches() -> Iterator[torch.Tensor]:
            for batch in self:
                vectors = batch.vectors
                if normalization is not None:
                    vectors = normalization.normalize(vectors)
                yield vectors.to(device=target_device, dtype=torch.float32)

        return batches


@dataclass(frozen=True)
class NormalizationStats:
    count: int
    mean: torch.Tensor
    std: torch.Tensor

    def __post_init__(self) -> None:
        if int(self.count) <= 0:
            raise ValueError(f"`count` must be positive, got {self.count}.")
        if self.mean.ndim != 1 or self.std.shape != self.mean.shape:
            raise ValueError("Normalization mean/std must be same-shaped one-dimensional tensors.")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all():
            raise ValueError("Normalization mean/std must be finite.")
        if not torch.all(self.std > 0):
            raise ValueError("Normalization std must be strictly positive.")

    @property
    def dim(self) -> int:
        return int(self.mean.numel())

    def normalize(self, vectors: torch.Tensor) -> torch.Tensor:
        if vectors.shape[-1] != self.dim:
            raise ValueError(
                f"Normalization dim mismatch: vectors={vectors.shape[-1]}, stats={self.dim}."
            )
        mean = self.mean.to(device=vectors.device, dtype=torch.float32)
        std = self.std.to(device=vectors.device, dtype=torch.float32)
        return (vectors.float() - mean) / std

    def to_payload(self) -> dict[str, Any]:
        return {
            "count": int(self.count),
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> NormalizationStats:
        return cls(
            count=int(payload["count"]),
            mean=payload["mean"].float().cpu(),
            std=payload["std"].float().cpu(),
        )


@dataclass
class RunningMoments:
    count: int = 0
    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None

    def update(self, vectors: torch.Tensor) -> None:
        if vectors.ndim != 2:
            raise ValueError(f"`vectors` must be [N,D], got {tuple(vectors.shape)}.")
        if vectors.shape[0] == 0:
            return
        values = vectors.detach().float()
        if not torch.isfinite(values).all():
            raise ValueError("Running moments cannot consume NaN or Inf.")
        batch_count = int(values.shape[0])
        batch_mean_device = values.mean(dim=0)
        batch_m2_device = (values - batch_mean_device).square().sum(dim=0)
        batch_mean = batch_mean_device.double().cpu()
        batch_m2 = batch_m2_device.double().cpu()
        self._merge_values(batch_count, batch_mean, batch_m2)

    def _merge_values(self, count: int, mean: torch.Tensor, m2: torch.Tensor) -> None:
        if count <= 0:
            return
        if self.count == 0:
            self.count = int(count)
            self.mean = mean.clone()
            self.m2 = m2.clone()
            return
        if self.mean is None or self.m2 is None:
            raise RuntimeError("RunningMoments state is inconsistent.")
        if mean.shape != self.mean.shape:
            raise ValueError(f"Moment dimension mismatch: {mean.shape} vs {self.mean.shape}.")

        total = self.count + int(count)
        delta = mean - self.mean
        self.mean = self.mean + delta * (float(count) / float(total))
        self.m2 = self.m2 + m2 + delta.square() * (
            float(self.count) * float(count) / float(total)
        )
        self.count = total

    def merge(self, other: RunningMoments) -> None:
        if other.count == 0:
            return
        if other.mean is None or other.m2 is None:
            raise RuntimeError("Other RunningMoments state is inconsistent.")
        self._merge_values(other.count, other.mean, other.m2)

    def merge_distributed(self) -> None:
        if not _distributed_ready():
            return
        gathered: list[dict[str, Any] | None] = [
            None for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather_object(gathered, self.to_payload())
        merged = RunningMoments()
        for payload in gathered:
            if payload is not None:
                merged.merge(RunningMoments.from_payload(payload))
        self.count, self.mean, self.m2 = merged.count, merged.mean, merged.m2

    def finalize(self, eps: float = 1e-6) -> NormalizationStats:
        if self.count <= 0 or self.mean is None or self.m2 is None:
            raise ValueError("Cannot finalize empty running moments.")
        variance = (self.m2 / float(self.count)).clamp_min(float(eps) ** 2)
        return NormalizationStats(
            count=self.count,
            mean=self.mean.float(),
            std=variance.sqrt().float(),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "count": int(self.count),
            "mean": None if self.mean is None else self.mean.clone(),
            "m2": None if self.m2 is None else self.m2.clone(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RunningMoments:
        return cls(
            count=int(payload["count"]),
            mean=payload.get("mean"),
            m2=payload.get("m2"),
        )


def fit_normalization(
    source: CausalDescriptorSource,
    require_train_split: bool = True,
) -> NormalizationStats:
    if require_train_split and source.split != "train":
        raise ValueError("Normalization may only be fit from an explicit train-only source.")
    moments = RunningMoments()
    for batch in source:
        moments.update(batch.vectors)
    moments.merge_distributed()
    return moments.finalize()


@dataclass
class UniformReservoir:
    max_samples: int
    seed: int = 0
    samples: torch.Tensor | None = None
    seen: int = 0
    _generator: torch.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.max_samples) <= 0:
            raise ValueError(f"`max_samples` must be positive, got {self.max_samples}.")
        self._generator = torch.Generator(device="cpu").manual_seed(int(self.seed))

    def update(self, vectors: torch.Tensor) -> None:
        if vectors.ndim != 2:
            raise ValueError(f"`vectors` must be [N,D], got {tuple(vectors.shape)}.")
        values = vectors.detach().float().cpu()
        if values.shape[0] == 0:
            return
        if self.samples is not None and values.shape[1] != self.samples.shape[1]:
            raise ValueError("Reservoir vector dimension changed within one stream.")

        offset = 0
        if self.samples is None:
            take = min(self.max_samples, int(values.shape[0]))
            self.samples = values[:take].clone().contiguous()
            self.seen = take
            offset = take
        elif self.samples.shape[0] < self.max_samples:
            take = min(
                self.max_samples - int(self.samples.shape[0]),
                int(values.shape[0]),
            )
            self.samples = torch.cat([self.samples, values[:take]], dim=0).contiguous()
            self.seen += take
            offset = take

        remaining = int(values.shape[0]) - offset
        if remaining <= 0:
            return
        if self.samples is None or self.samples.shape[0] != self.max_samples:
            raise RuntimeError("Reservoir fill state is inconsistent.")

        # Vectorized Algorithm R. If multiple stream items select the same slot,
        # the latest item wins, matching sequential reservoir sampling.
        global_indices = torch.arange(
            self.seen,
            self.seen + remaining,
            dtype=torch.float64,
        )
        random_values = torch.rand(
            remaining,
            generator=self._generator,
            dtype=torch.float64,
        )
        slots = torch.floor(random_values * (global_indices + 1.0)).long()
        positions = torch.arange(remaining, dtype=torch.long)
        eligible = slots < self.max_samples
        if eligible.any():
            selected_slots = slots[eligible]
            selected_positions = positions[eligible]
            last_position = torch.full(
                (self.max_samples,),
                -1,
                dtype=torch.long,
            )
            last_position.scatter_reduce_(
                0,
                selected_slots,
                selected_positions,
                reduce="amax",
                include_self=True,
            )
            changed_slots = torch.nonzero(last_position >= 0, as_tuple=False).flatten()
            self.samples[changed_slots] = values[
                offset + last_position[changed_slots]
            ]
        self.seen += remaining

    def result(self) -> torch.Tensor:
        if self.samples is None or self.samples.shape[0] == 0:
            raise ValueError("Cannot read an empty reservoir.")
        return self.samples


def build_reservoir(
    batch_factory: TensorBatchFactory,
    max_samples: int,
    seed: int,
) -> torch.Tensor:
    reservoir = UniformReservoir(max_samples=max_samples, seed=seed)
    for batch in batch_factory():
        reservoir.update(batch)
    return reservoir.result()


def assign_nearest(
    vectors: torch.Tensor,
    centers: torch.Tensor,
    center_block_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    if vectors.ndim != 2 or centers.ndim != 2:
        raise ValueError("Vectors and centers must both be two-dimensional.")
    if vectors.shape[1] != centers.shape[1]:
        raise ValueError(f"Vector/center dim mismatch: {vectors.shape[1]} vs {centers.shape[1]}.")
    if int(center_block_size) <= 0:
        raise ValueError("`center_block_size` must be positive.")

    values = vectors.float()
    centers = centers.to(device=values.device, dtype=torch.float32)
    value_norm = values.square().sum(dim=1, keepdim=True)
    best_distances = torch.full(
        (values.shape[0],),
        float("inf"),
        dtype=torch.float32,
        device=values.device,
    )
    best_codes = torch.zeros(values.shape[0], dtype=torch.long, device=values.device)
    for start in range(0, centers.shape[0], int(center_block_size)):
        block = centers[start : start + int(center_block_size)]
        distances = (
            value_norm
            - 2.0 * values @ block.t()
            + block.square().sum(dim=1).unsqueeze(0)
        ).clamp_min_(0.0)
        block_distances, block_codes = distances.min(dim=1)
        improve = block_distances < best_distances
        best_distances[improve] = block_distances[improve]
        best_codes[improve] = block_codes[improve] + start
    return best_codes, best_distances


def kmeans_plus_plus(
    vectors: torch.Tensor,
    k: int,
    seed: int = 0,
    distance_chunk_size: int = 8192,
) -> torch.Tensor:
    values = vectors.detach().float().cpu()
    if values.ndim != 2:
        raise ValueError(f"`vectors` must be [N,D], got {tuple(values.shape)}.")
    if values.shape[0] < int(k):
        raise ValueError(f"Need at least K samples, got N={values.shape[0]}, K={k}.")
    if not torch.isfinite(values).all():
        raise ValueError("K-Means++ initialization vectors must be finite.")
    if int(distance_chunk_size) <= 0:
        raise ValueError("`distance_chunk_size` must be positive.")

    def distance_to(center: torch.Tensor) -> torch.Tensor:
        distances = torch.empty(values.shape[0], dtype=torch.float32)
        for start in range(0, values.shape[0], int(distance_chunk_size)):
            chunk = values[start : start + int(distance_chunk_size)]
            distances[start : start + chunk.shape[0]] = (
                chunk - center
            ).square().sum(dim=1)
        return distances

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    first = int(torch.randint(values.shape[0], (1,), generator=generator).item())
    selected = [first]
    closest = distance_to(values[first])

    while len(selected) < int(k):
        total = closest.sum()
        if not torch.isfinite(total) or float(total.item()) <= 0.0:
            selected_set = set(selected)
            next_index = next(index for index in range(values.shape[0]) if index not in selected_set)
        else:
            probabilities = closest / total
            next_index = int(torch.multinomial(probabilities, 1, generator=generator).item())
        selected.append(next_index)
        distance = distance_to(values[next_index])
        closest = torch.minimum(closest, distance)
    return values[torch.tensor(selected, dtype=torch.long)].contiguous()


@dataclass(frozen=True)
class StreamingKMeansConfig:
    k: int
    max_iters: int = 50
    tol: float = 1e-5
    seed: int = 0
    reservoir_size: int = 100_000
    initialization_chunk_size: int = 8192
    center_block_size: int = 1024
    device: str = "auto"

    def __post_init__(self) -> None:
        if int(self.k) <= 1:
            raise ValueError(f"`k` must be > 1, got {self.k}.")
        if int(self.max_iters) <= 0:
            raise ValueError(f"`max_iters` must be positive, got {self.max_iters}.")
        if float(self.tol) < 0:
            raise ValueError(f"`tol` must be non-negative, got {self.tol}.")
        if int(self.reservoir_size) < int(self.k):
            raise ValueError("`reservoir_size` must be at least K.")
        if int(self.initialization_chunk_size) <= 0:
            raise ValueError("`initialization_chunk_size` must be positive.")


@dataclass(frozen=True)
class StreamingKMeansResult:
    centers: torch.Tensor
    counts: torch.Tensor
    inertia: float
    history: tuple[float, ...]
    iterations: int
    converged: bool


def _merge_hardest(
    scores: torch.Tensor | None,
    vectors: torch.Tensor | None,
    new_scores: torch.Tensor,
    new_vectors: torch.Tensor,
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scores is None or vectors is None:
        candidate_scores = new_scores.detach()
        candidate_vectors = new_vectors.detach()
    else:
        candidate_scores = torch.cat([scores, new_scores.detach()], dim=0)
        candidate_vectors = torch.cat([vectors, new_vectors.detach()], dim=0)
    keep = min(int(limit), int(candidate_scores.numel()))
    selected = torch.topk(candidate_scores, k=keep, largest=True, sorted=True).indices
    return candidate_scores[selected], candidate_vectors[selected]


def _gather_hardest(
    scores: torch.Tensor,
    vectors: torch.Tensor,
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _distributed_ready():
        return scores, vectors
    dimension = vectors.shape[1]
    padded_scores = torch.full(
        (limit,),
        -float("inf"),
        device=scores.device,
        dtype=scores.dtype,
    )
    padded_vectors = torch.zeros(
        (limit, dimension),
        device=vectors.device,
        dtype=vectors.dtype,
    )
    padded_scores[: scores.numel()] = scores
    padded_vectors[: vectors.shape[0]] = vectors
    gathered_scores = [torch.empty_like(padded_scores) for _ in range(torch.distributed.get_world_size())]
    gathered_vectors = [
        torch.empty_like(padded_vectors) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(gathered_scores, padded_scores)
    torch.distributed.all_gather(gathered_vectors, padded_vectors)
    all_scores = torch.cat(gathered_scores)
    all_vectors = torch.cat(gathered_vectors)
    valid = torch.isfinite(all_scores)
    all_scores = all_scores[valid]
    all_vectors = all_vectors[valid]
    keep = min(int(limit), int(all_scores.numel()))
    selected = torch.topk(all_scores, k=keep, largest=True, sorted=True).indices
    return all_scores[selected], all_vectors[selected]


class StreamingKMeans:
    def __init__(self, config: StreamingKMeansConfig) -> None:
        self.config = config

    def _load_checkpoint(
        self,
        path: Path,
        device: torch.device,
    ) -> tuple[torch.Tensor, int, float | None, list[float], bool]:
        payload = load_torch_payload(path, map_location="cpu")
        if payload.get("schema") != KMEANS_CHECKPOINT_SCHEMA:
            raise ValueError(f"Unsupported K-Means checkpoint schema in {path}.")
        if int(payload["k"]) != self.config.k:
            raise ValueError(
                f"Checkpoint K={payload['k']} does not match configured K={self.config.k}."
            )
        return (
            payload["centers"].to(device=device, dtype=torch.float32),
            int(payload["next_iteration"]),
            payload.get("previous_inertia"),
            [float(value) for value in payload.get("history", ())],
            bool(payload.get("converged", False)),
        )

    def _save_checkpoint(
        self,
        path: Path,
        centers: torch.Tensor,
        next_iteration: int,
        previous_inertia: float | None,
        history: Sequence[float],
        converged: bool,
    ) -> None:
        if not _is_primary_rank():
            return
        atomic_torch_save(
            {
                "schema": KMEANS_CHECKPOINT_SCHEMA,
                "k": self.config.k,
                "centers": centers.detach().cpu(),
                "next_iteration": int(next_iteration),
                "previous_inertia": previous_inertia,
                "history": [float(value) for value in history],
                "converged": bool(converged),
            },
            path,
        )

    @staticmethod
    def _reduce_statistics(
        sums: torch.Tensor,
        counts: torch.Tensor,
        inertia_sum: torch.Tensor,
        sample_count: torch.Tensor,
    ) -> None:
        if not _distributed_ready():
            return
        torch.distributed.all_reduce(sums)
        torch.distributed.all_reduce(counts)
        torch.distributed.all_reduce(inertia_sum)
        torch.distributed.all_reduce(sample_count)

    def _evaluate(
        self,
        batch_factory: TensorBatchFactory,
        centers: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, float]:
        counts = torch.zeros(self.config.k, device=device, dtype=torch.float32)
        inertia_sum = torch.zeros((), device=device, dtype=torch.float32)
        sample_count = torch.zeros((), device=device, dtype=torch.float32)
        for batch in batch_factory():
            values = batch.to(device=device, dtype=torch.float32)
            if values.ndim != 2 or values.shape[1] != centers.shape[1]:
                raise ValueError("K-Means stream changed vector shape during evaluation.")
            if not torch.isfinite(values).all():
                raise ValueError("K-Means evaluation stream contains NaN or Inf.")
            codes, distances = assign_nearest(
                values,
                centers,
                center_block_size=self.config.center_block_size,
            )
            counts += torch.bincount(codes, minlength=self.config.k).float()
            inertia_sum += distances.sum()
            sample_count += values.shape[0]
        if _distributed_ready():
            torch.distributed.all_reduce(counts)
            torch.distributed.all_reduce(inertia_sum)
            torch.distributed.all_reduce(sample_count)
        if float(sample_count.item()) <= 0:
            raise ValueError("K-Means stream yielded no vectors.")
        return counts.cpu(), float((inertia_sum / sample_count).item())

    def fit(
        self,
        batch_factory: TensorBatchFactory,
        initial_centers: torch.Tensor | None = None,
        initialization_sample: torch.Tensor | None = None,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
    ) -> StreamingKMeansResult:
        device = _resolve_device(self.config.device)
        checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
        history: list[float] = []
        previous_inertia: float | None = None
        start_iteration = 0
        converged = False

        if resume and checkpoint is not None and checkpoint.exists():
            (
                centers,
                start_iteration,
                previous_inertia,
                history,
                converged,
            ) = self._load_checkpoint(checkpoint, device=device)
        else:
            if initial_centers is not None:
                centers = initial_centers.detach().to(device=device, dtype=torch.float32)
            else:
                if initialization_sample is None:
                    if _distributed_ready():
                        raise ValueError(
                            "Distributed K-Means requires a shared initialization sample "
                            "or explicit initial centers."
                        )
                    initialization_sample = build_reservoir(
                        batch_factory,
                        max_samples=self.config.reservoir_size,
                        seed=self.config.seed,
                    )
                centers = kmeans_plus_plus(
                    initialization_sample,
                    k=self.config.k,
                    seed=self.config.seed,
                    distance_chunk_size=self.config.initialization_chunk_size,
                ).to(device=device)
            if centers.ndim != 2 or centers.shape[0] != self.config.k:
                raise ValueError(
                    f"Initial centers must be [K,D], got {tuple(centers.shape)}."
                )
            if not torch.isfinite(centers).all():
                raise ValueError("Initial centers must be finite.")

        iteration_range = (
            range(start_iteration, self.config.max_iters)
            if not converged
            else range(0)
        )
        for iteration in iteration_range:
            sums = torch.zeros_like(centers)
            counts = torch.zeros(self.config.k, device=device, dtype=torch.float32)
            inertia_sum = torch.zeros((), device=device, dtype=torch.float32)
            sample_count = torch.zeros((), device=device, dtype=torch.float32)
            hardest_scores: torch.Tensor | None = None
            hardest_vectors: torch.Tensor | None = None

            for batch in batch_factory():
                values = batch.to(device=device, dtype=torch.float32)
                if values.ndim != 2 or values.shape[1] != centers.shape[1]:
                    raise ValueError(
                        "K-Means stream changed vector dimension: "
                        f"expected {centers.shape[1]}, got {tuple(values.shape)}."
                    )
                if values.shape[0] == 0:
                    continue
                if not torch.isfinite(values).all():
                    raise ValueError("K-Means stream contains NaN or Inf.")
                codes, distances = assign_nearest(
                    values,
                    centers,
                    center_block_size=self.config.center_block_size,
                )
                sums.index_add_(0, codes, values)
                counts += torch.bincount(codes, minlength=self.config.k).float()
                inertia_sum += distances.sum()
                sample_count += values.shape[0]
                hardest_scores, hardest_vectors = _merge_hardest(
                    hardest_scores,
                    hardest_vectors,
                    distances,
                    values,
                    limit=self.config.k,
                )

            if hardest_scores is None or hardest_vectors is None:
                raise ValueError("K-Means stream yielded no vectors.")
            hardest_scores, hardest_vectors = _gather_hardest(
                hardest_scores,
                hardest_vectors,
                limit=self.config.k,
            )
            self._reduce_statistics(sums, counts, inertia_sum, sample_count)
            if int(sample_count.item()) < self.config.k:
                raise ValueError(
                    f"K-Means needs at least K vectors, got N={int(sample_count.item())}, "
                    f"K={self.config.k}."
                )

            new_centers = centers.clone()
            nonempty = counts > 0
            new_centers[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
            empty_indices = torch.nonzero(~nonempty, as_tuple=False).flatten()
            if empty_indices.numel():
                if hardest_vectors.shape[0] < empty_indices.numel():
                    raise RuntimeError("Not enough global hardest samples to refill empty centers.")
                new_centers[empty_indices] = hardest_vectors[: empty_indices.numel()]

            mean_inertia = float((inertia_sum / sample_count).item())
            history.append(mean_inertia)
            improvement = None
            if previous_inertia is not None:
                improvement = abs(previous_inertia - mean_inertia) / max(
                    abs(previous_inertia),
                    1e-12,
                )
            centers = new_centers
            previous_inertia = mean_inertia
            converged = (
                improvement is not None
                and improvement < self.config.tol
                and empty_indices.numel() == 0
            )
            if checkpoint is not None:
                self._save_checkpoint(
                    checkpoint,
                    centers=centers,
                    next_iteration=iteration + 1,
                    previous_inertia=previous_inertia,
                    history=history,
                    converged=converged,
                )
            if converged:
                break

        final_counts, final_inertia = self._evaluate(batch_factory, centers, device=device)
        return StreamingKMeansResult(
            centers=centers.detach().cpu(),
            counts=final_counts,
            inertia=final_inertia,
            history=tuple(history),
            iterations=len(history),
            converged=converged,
        )


def residual_batch_factory(
    batch_factory: TensorBatchFactory,
    centers: Sequence[torch.Tensor],
    center_block_size: int = 1024,
) -> TensorBatchFactory:
    frozen_centers = tuple(center.detach().float().cpu() for center in centers)

    def batches() -> Iterator[torch.Tensor]:
        cached_device: torch.device | None = None
        device_centers: list[torch.Tensor] = []
        for batch in batch_factory():
            residual = batch.float()
            if cached_device != residual.device:
                cached_device = residual.device
                device_centers = [
                    center.to(device=residual.device, dtype=torch.float32)
                    for center in frozen_centers
                ]
            for center in device_centers:
                codes, _ = assign_nearest(
                    residual,
                    center,
                    center_block_size=center_block_size,
                )
                residual = residual - center[codes]
            yield residual

    return batches


def _stream_mean_square(batch_factory: TensorBatchFactory) -> tuple[float, int]:
    total = None
    vector_count = None
    dimension = None
    for batch in batch_factory():
        values = batch.float()
        if total is None:
            total = torch.zeros((), device=values.device, dtype=torch.float32)
            vector_count = torch.zeros((), device=values.device, dtype=torch.float32)
            dimension = int(values.shape[1])
        if values.ndim != 2 or values.shape[1] != dimension:
            raise ValueError("RQ stream changed vector shape while computing its baseline.")
        total += values.square().sum()
        vector_count += values.shape[0]
    if total is None or vector_count is None or dimension is None:
        raise ValueError("RQ stream yielded no vectors.")
    if _distributed_ready():
        torch.distributed.all_reduce(total)
        torch.distributed.all_reduce(vector_count)
    elements = int(round(float(vector_count.item()))) * dimension
    return float((total / (vector_count * dimension)).item()), elements


@dataclass(frozen=True)
class StreamingRQResult:
    centers: tuple[torch.Tensor, ...]
    residual_mse: tuple[float, ...]
    iterations_per_level: tuple[int, ...]


class StreamingRQTrainer:
    def __init__(self, kmeans_config: StreamingKMeansConfig, levels: int = 3) -> None:
        if int(levels) <= 0:
            raise ValueError(f"`levels` must be positive, got {levels}.")
        self.kmeans_config = kmeans_config
        self.levels = int(levels)

    def fit(
        self,
        batch_factory: TensorBatchFactory,
        initial_centers: Sequence[torch.Tensor] | None = None,
        checkpoint_dir: str | Path | None = None,
        resume: bool = False,
    ) -> StreamingRQResult:
        checkpoint_root = None if checkpoint_dir is None else Path(checkpoint_dir)
        state_path = None if checkpoint_root is None else checkpoint_root / "rq_state.pt"
        centers: list[torch.Tensor] = []
        residual_mse: list[float] = []
        iterations_per_level: list[int] = []

        if resume and state_path is not None and state_path.exists():
            payload = load_torch_payload(state_path, map_location="cpu")
            if payload.get("schema") != RQ_CHECKPOINT_SCHEMA:
                raise ValueError(f"Unsupported RQ checkpoint schema in {state_path}.")
            if int(payload["k"]) != self.kmeans_config.k:
                raise ValueError("RQ checkpoint K does not match the current configuration.")
            centers = [center.float().cpu() for center in payload.get("centers", ())]
            residual_mse = [float(value) for value in payload.get("residual_mse", ())]
            iterations_per_level = [
                int(value) for value in payload.get("iterations_per_level", ())
            ]
            if len(centers) > self.levels:
                raise ValueError(
                    f"RQ checkpoint has {len(centers)} levels, configured for {self.levels}."
                )

        if not residual_mse:
            baseline_mse, _ = _stream_mean_square(batch_factory)
            residual_mse.append(baseline_mse)

        for level in range(len(centers), self.levels):
            level_config = replace(
                self.kmeans_config,
                seed=self.kmeans_config.seed + level * 9973,
            )
            level_factory = residual_batch_factory(
                batch_factory,
                centers=centers,
                center_block_size=level_config.center_block_size,
            )
            level_checkpoint = (
                None
                if checkpoint_root is None
                else checkpoint_root / f"level_{level + 1}_kmeans.pt"
            )
            level_initial = None
            if initial_centers is not None and level < len(initial_centers):
                level_initial = initial_centers[level]
            result = StreamingKMeans(level_config).fit(
                level_factory,
                initial_centers=level_initial,
                checkpoint_path=level_checkpoint,
                resume=resume,
            )
            centers.append(result.centers)
            residual_mse.append(result.inertia / float(result.centers.shape[1]))
            iterations_per_level.append(result.iterations)
            if state_path is not None and _is_primary_rank():
                atomic_torch_save(
                    {
                        "schema": RQ_CHECKPOINT_SCHEMA,
                        "k": self.kmeans_config.k,
                        "levels": self.levels,
                        "centers": centers,
                        "residual_mse": residual_mse,
                        "iterations_per_level": iterations_per_level,
                    },
                    state_path,
                )

        return StreamingRQResult(
            centers=tuple(centers),
            residual_mse=tuple(residual_mse),
            iterations_per_level=tuple(iterations_per_level),
        )


def encode_residual_quantizer(
    vectors: torch.Tensor,
    centers: Sequence[torch.Tensor],
    center_block_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = vectors.float()
    residual = values.clone()
    quantized = torch.zeros_like(values)
    all_codes: list[torch.Tensor] = []
    for level_centers in centers:
        level_centers = level_centers.to(device=values.device, dtype=torch.float32)
        codes, _ = assign_nearest(
            residual,
            level_centers,
            center_block_size=center_block_size,
        )
        contribution = level_centers[codes]
        quantized += contribution
        residual -= contribution
        all_codes.append(codes)
    if not all_codes:
        raise ValueError("At least one RQ level is required.")
    return torch.stack(all_codes, dim=1), quantized, residual


@dataclass(frozen=True)
class FrozenRQArtifact:
    family: str
    descriptor: CausalDescriptorSpec
    normalization: NormalizationStats
    centers: tuple[torch.Tensor, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.family:
            raise ValueError("`family` must not be empty.")
        if self.family != self.descriptor.family:
            raise ValueError(
                f"Artifact family `{self.family}` does not match descriptor "
                f"`{self.descriptor.family}`."
            )
        if not self.centers:
            raise ValueError("A frozen RQ artifact must contain at least one level.")
        missing_metadata = sorted(REQUIRED_ARTIFACT_METADATA - self.metadata.keys())
        if missing_metadata:
            raise ValueError(f"Missing frozen artifact metadata fields: {missing_metadata}.")
        expected_dim = self.normalization.dim
        k = int(self.centers[0].shape[0])
        for level, center in enumerate(self.centers, start=1):
            if center.ndim != 2 or center.shape != (k, expected_dim):
                raise ValueError(
                    f"RQ center shape mismatch at level {level}: "
                    f"expected {(k, expected_dim)}, got {tuple(center.shape)}."
                )

    def save(self, path: str | Path) -> None:
        atomic_torch_save(
            {
                "schema": FROZEN_RQ_SCHEMA,
                "family": self.family,
                "descriptor": {
                    "stride": self.descriptor.stride,
                    "pool": self.descriptor.pool,
                    "max_gap_factor": self.descriptor.max_gap_factor,
                },
                "normalization": self.normalization.to_payload(),
                "centers": [center.detach().float().cpu() for center in self.centers],
                "metadata": dict(self.metadata),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> FrozenRQArtifact:
        payload = load_torch_payload(path, map_location="cpu")
        if payload.get("schema") != FROZEN_RQ_SCHEMA:
            raise ValueError(f"Unsupported frozen RQ artifact schema in {path}.")
        descriptor = payload["descriptor"]
        return cls(
            family=str(payload["family"]),
            descriptor=CausalDescriptorSpec(
                stride=int(descriptor["stride"]),
                pool=int(descriptor["pool"]),
                max_gap_factor=descriptor.get("max_gap_factor"),
            ),
            normalization=NormalizationStats.from_payload(payload["normalization"]),
            centers=tuple(center.float().cpu() for center in payload["centers"]),
            metadata=dict(payload.get("metadata", {})),
        )
