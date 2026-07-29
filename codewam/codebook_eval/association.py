from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import torch

from codewam.data.droid_manifest import write_json_report

from .manifest import EpisodeManifest
from .shards import (
    PooledFeatureEpisode,
    expand_shard_paths,
    file_sha256,
    iter_pooled_feature_episodes,
)
from .streaming import (
    CausalDescriptorSpec,
    FrozenRQArtifact,
    encode_residual_quantizer,
)


ASSOCIATION_CONTRACT_SCHEMA = "codewam.rq-association-contract.v1"
ASSOCIATION_REPORT_SCHEMA = "codewam.rq-association-report.v1"
TARGET_DEFINITIONS = {
    "current_action": (
        "Action recorded at the current latent tick. The code input contains "
        "only visual states up to this tick."
    ),
    "future_proprio_change": (
        "Proprioception one family stride in the future minus current "
        "proprioception."
    ),
    "future_latent_moment_change": (
        "One-family-stride future minus current selected Wan latent spatial "
        "moments [mean, x, y, radius-squared] per view and channel."
    ),
}


def _resolve_device(value: str) -> torch.device:
    if str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device `{value}` is unavailable.")
    return device


def _episode_factory(
    shard_paths: tuple[Path, ...],
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
                raise ValueError(
                    f"Association episode `{episode.episode_id}` is absent "
                    f"from the `{split}` manifest."
                )
            if episode.episode_id in seen:
                raise ValueError(
                    f"Duplicate association episode `{episode.episode_id}`."
                )
            seen.add(episode.episode_id)
            yield episode
        missing = sorted(expected_episode_ids - seen)
        if missing:
            raise ValueError(
                f"Association `{split}` episodes are missing: {missing[:8]}."
            )

    return episodes


def _selected_features(
    episode: PooledFeatureEpisode,
    spec: CausalDescriptorSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    pooled = episode.pooled(spec.pool)
    valid_mask = episode.valid_mask
    if spec.camera_ids is not None:
        missing = [
            camera
            for camera in spec.camera_ids
            if camera not in episode.camera_ids
        ]
        if missing:
            raise ValueError(
                f"Episode `{episode.episode_id}` lacks cameras {missing}."
            )
        view_indices = [
            episode.camera_ids.index(camera)
            for camera in spec.camera_ids
        ]
        pooled = pooled[:, view_indices]
        valid_mask = valid_mask[:, view_indices]
    return pooled, valid_mask


def _spatial_moments(features: torch.Tensor) -> torch.Tensor:
    grid = int(features.shape[-1])
    coordinates = torch.linspace(
        -1.0,
        1.0,
        grid,
        dtype=torch.float32,
        device=features.device,
    )
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    basis = torch.stack(
        (
            torch.ones_like(xx),
            xx,
            yy,
            xx.square() + yy.square(),
        )
    ) / float(grid * grid)
    moments = torch.einsum(
        "tvcij,mij->tvcm",
        features.float(),
        basis,
    )
    return moments.reshape(features.shape[0], -1)


def _episode_probe_values(
    episode: PooledFeatureEpisode,
    spec: CausalDescriptorSpec,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if episode.action is None or episode.proprio is None:
        raise ValueError(
            f"Episode `{episode.episode_id}` lacks action or proprio targets."
        )
    stride = spec.stride
    if episode.ticks <= 3 * stride:
        return torch.empty((0, 0)), {}

    pooled, valid_mask = _selected_features(episode, spec)
    features = pooled.reshape(episode.ticks, -1)
    moments = _spatial_moments(pooled)
    current = torch.arange(
        2 * stride,
        episode.ticks - stride,
        dtype=torch.long,
        device=features.device,
    )
    valid_mask = valid_mask.to(device=features.device)
    valid = (
        valid_mask[current - 2 * stride].all(dim=1)
        & valid_mask[current - stride].all(dim=1)
        & valid_mask[current].all(dim=1)
        & valid_mask[current + stride].all(dim=1)
    )

    timestamps = episode.timestamps.to(device=features.device)
    if spec.max_gap_factor is not None and episode.ticks > 1:
        cadence = torch.median(timestamps[1:] - timestamps[:-1])
        history_gap = cadence * stride * float(spec.max_gap_factor)
        future_gap = cadence * stride * float(spec.max_gap_factor)
        valid &= (
            timestamps[current - stride]
            - timestamps[current - 2 * stride]
            <= history_gap
        )
        valid &= (
            timestamps[current] - timestamps[current - stride]
            <= history_gap
        )
        valid &= (
            timestamps[current + stride] - timestamps[current]
            <= future_gap
        )
    current = current[valid]
    if current.numel() == 0:
        return torch.empty((0, features.shape[1] * 3)), {}

    vectors = torch.cat(
        (
            features[current - 2 * stride],
            features[current - stride],
            features[current],
        ),
        dim=1,
    )
    targets = {
        "current_action": episode.action[current].float(),
        "future_proprio_change": (
            episode.proprio[current + stride].float()
            - episode.proprio[current].float()
        ),
        "future_latent_moment_change": (
            moments[current + stride] - moments[current]
        ),
    }
    return vectors.contiguous(), {
        name: value.contiguous() for name, value in targets.items()
    }


def _iter_probe_batches(
    episode_factory: Callable[[], Iterable[PooledFeatureEpisode]],
    spec: CausalDescriptorSpec,
    batch_size: int,
) -> Iterator[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    vector_parts: list[torch.Tensor] = []
    target_parts: dict[str, list[torch.Tensor]] = {}
    pending = 0

    def emit() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        nonlocal vector_parts, target_parts, pending
        result = (
            torch.cat(vector_parts, dim=0).contiguous(),
            {
                name: torch.cat(parts, dim=0).contiguous()
                for name, parts in target_parts.items()
            },
        )
        vector_parts = []
        target_parts = {}
        pending = 0
        return result

    for episode in episode_factory():
        vectors, targets = _episode_probe_values(episode, spec)
        offset = 0
        while offset < vectors.shape[0]:
            take = min(batch_size - pending, vectors.shape[0] - offset)
            vector_parts.append(vectors[offset : offset + take])
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


class _ConditionalMeans:
    def __init__(self, *, k: int, levels: int, dimension: int) -> None:
        self.k = int(k)
        self.levels = int(levels)
        self.dimension = int(dimension)
        self.counts = [
            torch.zeros(self.k ** (level + 1), dtype=torch.long)
            for level in range(self.levels)
        ]
        self.sums = [
            torch.zeros(
                (self.k ** (level + 1), self.dimension),
                dtype=torch.float32,
            )
            for level in range(self.levels)
        ]
        self.total = 0
        self.global_sum = torch.zeros(self.dimension, dtype=torch.float64)
        self.global_square_sum = torch.zeros(
            self.dimension,
            dtype=torch.float64,
        )

    def _keys(self, codes: torch.Tensor) -> list[torch.Tensor]:
        codes = codes.detach().long().cpu()
        if codes.ndim != 2 or codes.shape[1] != self.levels:
            raise ValueError("Code matrix does not match conditional table.")
        keys = []
        key = torch.zeros(codes.shape[0], dtype=torch.long)
        for level in range(self.levels):
            key = key * self.k + codes[:, level]
            keys.append(key.clone())
        return keys

    def update(self, codes: torch.Tensor, targets: torch.Tensor) -> None:
        targets = targets.detach().float().cpu()
        if targets.ndim != 2 or targets.shape[1] != self.dimension:
            raise ValueError("Target matrix does not match conditional table.")
        keys = self._keys(codes)
        ones = torch.ones(targets.shape[0], dtype=torch.long)
        for level, key in enumerate(keys):
            self.counts[level].index_add_(0, key, ones)
            self.sums[level].index_add_(0, key, targets)
        self.total += int(targets.shape[0])
        self.global_sum += targets.double().sum(dim=0)
        self.global_square_sum += targets.double().square().sum(dim=0)

    def global_statistics(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.total <= 0:
            raise ValueError("Cannot finalize an empty conditional table.")
        mean = self.global_sum / float(self.total)
        variance = (
            self.global_square_sum / float(self.total) - mean.square()
        ).clamp_min_(0.0)
        effective = variance > 1e-10
        if not effective.any():
            raise ValueError("Association target has no train variance.")
        return mean.float(), variance.float(), effective

    def predict(
        self,
        codes: torch.Tensor,
        *,
        depth: int,
        min_train_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if depth <= 0 or depth > self.levels:
            raise ValueError(f"Invalid RQ prefix depth `{depth}`.")
        keys = self._keys(codes)
        mean, _, _ = self.global_statistics()
        predictions = mean.expand(codes.shape[0], -1).clone()
        chosen_depth = torch.zeros(codes.shape[0], dtype=torch.long)
        for level in range(depth):
            key = keys[level]
            counts = self.counts[level][key]
            usable = counts >= int(min_train_count)
            if usable.any():
                predictions[usable] = (
                    self.sums[level][key[usable]]
                    / counts[usable].float().unsqueeze(1)
                )
                chosen_depth[usable] = level + 1
        return predictions, chosen_depth


@dataclass
class _RegressionAccumulator:
    dimension: int
    effective_dimensions: int
    depth: int
    vectors: int = 0
    raw_sse: float = 0.0
    baseline_raw_sse: float = 0.0
    normalized_sse: float = 0.0
    baseline_normalized_sse: float = 0.0
    exact_matches: int = 0
    any_matches: int = 0
    backoff_depth_counts: list[int] | None = None

    def __post_init__(self) -> None:
        if self.backoff_depth_counts is None:
            self.backoff_depth_counts = [0 for _ in range(self.depth + 1)]

    def update(
        self,
        *,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        chosen_depth: torch.Tensor,
        global_mean: torch.Tensor,
        variance: torch.Tensor,
        effective: torch.Tensor,
    ) -> None:
        targets = targets.detach().float().cpu()
        predictions = predictions.detach().float().cpu()
        error = (targets - predictions).square()
        baseline_error = (
            targets - global_mean.unsqueeze(0)
        ).square()
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
        self.exact_matches += int((chosen_depth == self.depth).sum().item())
        self.any_matches += int((chosen_depth > 0).sum().item())
        if self.backoff_depth_counts is None:
            raise RuntimeError("Backoff counts were not initialized.")
        counts = torch.bincount(
            chosen_depth,
            minlength=self.depth + 1,
        )
        for index, value in enumerate(counts.tolist()):
            self.backoff_depth_counts[index] += int(value)

    def row(self) -> dict[str, Any]:
        if self.vectors <= 0:
            raise ValueError("Cannot finalize an empty association split.")
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
            "exact_prefix_coverage": self.exact_matches / float(self.vectors),
            "any_code_coverage": self.any_matches / float(self.vectors),
            "backoff_depth_counts": self.backoff_depth_counts,
        }


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
                f"Existing association contract differs from `{path}`."
            )
        if not resume:
            raise FileExistsError(f"Association contract exists at `{path}`.")
        return
    write_json_report(path, contract)


def probe_frozen_codebook_associations(
    *,
    manifest_path: str | Path,
    pooled_shards: Iterable[str | Path],
    artifacts: dict[str, str | Path],
    output_dir: str | Path,
    splits: tuple[str, ...] = ("val", "test"),
    device: str = "auto",
    cpu_threads: int = 4,
    batch_size: int = 8192,
    center_block_size: int = 1024,
    min_train_count: int = 8,
    resume: bool = True,
) -> dict[str, Any]:
    if not artifacts or any(not label for label in artifacts):
        raise ValueError("Association artifact labels must be nonempty and unique.")
    if (
        cpu_threads <= 0
        or batch_size <= 0
        or center_block_size <= 0
        or min_train_count <= 0
    ):
        raise ValueError(
            "Association thread, batch, block and count values must be positive."
        )
    if not splits or any(split not in {"val", "test"} for split in splits):
        raise ValueError("Association splits must be val/test.")
    if len(splits) != len(set(splits)):
        raise ValueError("Association splits must be unique.")
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
        for label, path in sorted(artifact_paths.items())
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
                f"Association artifact `{label}` differs in {mismatches}."
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
        split for split, identifiers in expected_by_split.items()
        if not identifiers
    ]
    if empty:
        raise ValueError(f"Association manifest has empty splits {empty}.")

    implementation_sha256 = {
        "association": file_sha256(Path(__file__)),
        "shards": file_sha256(Path(__file__).with_name("shards.py")),
        "streaming": file_sha256(Path(__file__).with_name("streaming.py")),
    }
    contract_payload = {
        "schema": ASSOCIATION_CONTRACT_SCHEMA,
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
            }
            for label in sorted(artifact_paths)
        },
        "splits": list(splits),
        "device": device,
        "cpu_threads": cpu_threads,
        "batch_size": batch_size,
        "center_block_size": center_block_size,
        "min_train_count": min_train_count,
        "target_definitions": TARGET_DEFINITIONS,
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
    report_path = output_dir / "association_report.json"
    _write_contract(contract_path, contract, resume=resume)
    if resume and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("contract_hash") != contract_hash:
            raise RuntimeError("Association report contract hash is invalid.")
        return report
    if report_path.exists():
        raise FileExistsError(f"Association report exists at `{report_path}`.")

    target_device = _resolve_device(device)
    rows = []
    for label, artifact in loaded_artifacts.items():
        centers = tuple(
            center.to(device=target_device, dtype=torch.float32)
            for center in artifact.centers
        )
        k = int(centers[0].shape[0])
        levels = len(centers)
        tables: dict[str, _ConditionalMeans] = {}
        train_vectors = 0
        train_factory = _episode_factory(
            shard_paths,
            "train",
            expected_by_split["train"],
        )
        for vectors, targets in _iter_probe_batches(
            train_factory,
            artifact.descriptor,
            batch_size,
        ):
            normalized = artifact.normalization.normalize(vectors).to(
                device=target_device,
                dtype=torch.float32,
            )
            codes, _, _ = encode_residual_quantizer(
                normalized,
                centers,
                center_block_size=center_block_size,
            )
            codes = codes.cpu()
            for name, values in targets.items():
                if name not in tables:
                    tables[name] = _ConditionalMeans(
                        k=k,
                        levels=levels,
                        dimension=int(values.shape[1]),
                    )
                tables[name].update(codes, values)
            train_vectors += int(vectors.shape[0])
        if train_vectors <= 0 or set(tables) != set(TARGET_DEFINITIONS):
            raise ValueError(f"Association train stream is incomplete for `{label}`.")

        for split in splits:
            accumulators: dict[
                tuple[str, int],
                _RegressionAccumulator,
            ] = {}
            split_factory = _episode_factory(
                shard_paths,
                split,
                expected_by_split[split],
            )
            for vectors, targets in _iter_probe_batches(
                split_factory,
                artifact.descriptor,
                batch_size,
            ):
                normalized = artifact.normalization.normalize(vectors).to(
                    device=target_device,
                    dtype=torch.float32,
                )
                codes, _, _ = encode_residual_quantizer(
                    normalized,
                    centers,
                    center_block_size=center_block_size,
                )
                codes = codes.cpu()
                for name, values in targets.items():
                    table = tables[name]
                    mean, variance, effective = table.global_statistics()
                    for depth in range(1, levels + 1):
                        predictions, chosen_depth = table.predict(
                            codes,
                            depth=depth,
                            min_train_count=min_train_count,
                        )
                        key = (name, depth)
                        accumulator = accumulators.setdefault(
                            key,
                            _RegressionAccumulator(
                                dimension=table.dimension,
                                effective_dimensions=int(
                                    effective.sum().item()
                                ),
                                depth=depth,
                            ),
                        )
                        accumulator.update(
                            targets=values,
                            predictions=predictions,
                            chosen_depth=chosen_depth,
                            global_mean=mean,
                            variance=variance,
                            effective=effective,
                        )
            for (name, depth), accumulator in sorted(accumulators.items()):
                rows.append(
                    {
                        "label": label,
                        "family": artifact.family,
                        "stride": artifact.descriptor.stride,
                        "pool": artifact.descriptor.pool,
                        "camera_ids": (
                            None
                            if artifact.descriptor.camera_ids is None
                            else list(artifact.descriptor.camera_ids)
                        ),
                        "k": k,
                        "levels": levels,
                        "split": split,
                        "target": name,
                        "prefix_depth": depth,
                        "min_train_count": min_train_count,
                        "train_vectors": train_vectors,
                        **accumulator.row(),
                    }
                )

    report = {
        "schema": ASSOCIATION_REPORT_SCHEMA,
        "contract_hash": contract_hash,
        "manifest_fingerprint": manifest_fingerprint,
        "target_definitions": TARGET_DEFINITIONS,
        "rows": rows,
    }
    write_json_report(report_path, report)
    return report
