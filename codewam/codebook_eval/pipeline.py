from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from omegaconf import DictConfig, OmegaConf

from .io import ensure_dir, save_json
from .manifest import EpisodeManifest, EpisodeRecord
from .shards import (
    PooledFeatureEpisode,
    atomic_torch_save,
    expand_shard_paths,
    file_sha256,
    iter_pooled_feature_episodes,
    load_torch_payload,
    write_pooled_feature_shard,
)
from .streaming import (
    CausalDescriptorSource,
    CausalDescriptorSpec,
    FrozenRQArtifact,
    NormalizationStats,
    StreamingKMeansConfig,
    StreamingRQTrainer,
    fit_normalization,
)


NORMALIZATION_SCHEMA = "codewam.normalization.v1"


def _distributed_ready() -> bool:
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    )


def _distributed_rank() -> int:
    return torch.distributed.get_rank() if _distributed_ready() else 0


def _distributed_world_size() -> int:
    return torch.distributed.get_world_size() if _distributed_ready() else 1


def _is_primary_rank() -> bool:
    return _distributed_rank() == 0


def _distributed_backend() -> str:
    if not _distributed_ready():
        return "none"
    return str(torch.distributed.get_backend())


def _broadcast_from_primary(value: Any) -> Any:
    if not _distributed_ready():
        return value
    payload = [value if _is_primary_rank() else None]
    torch.distributed.broadcast_object_list(payload, src=0)
    return payload[0]


def _runtime_device(configured: str) -> str:
    if not _distributed_ready():
        return configured
    lowered = str(configured).lower()
    if lowered == "auto" or lowered.startswith("cuda"):
        if not torch.cuda.is_available():
            if lowered.startswith("cuda"):
                raise RuntimeError("Distributed CUDA training requested unavailable CUDA.")
            return "cpu"
        local_rank = int(os.environ.get("LOCAL_RANK", _distributed_rank()))
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"
    return configured


def partition_shard_paths(
    paths: Sequence[Path],
    world_size: int,
) -> tuple[tuple[Path, ...], ...]:
    if int(world_size) <= 0:
        raise ValueError("Shard partition world size must be positive.")
    if len(paths) < int(world_size):
        raise ValueError(
            f"Need at least one pooled shard per rank, got "
            f"{len(paths)} shards and {world_size} ranks."
        )
    loads = [0 for _ in range(int(world_size))]
    assignments: list[list[Path]] = [
        [] for _ in range(int(world_size))
    ]
    for path in sorted(
        paths,
        key=lambda value: (-value.stat().st_size, str(value)),
    ):
        rank = min(
            range(int(world_size)),
            key=lambda value: (loads[value], len(assignments[value]), value),
        )
        assignments[rank].append(path)
        loads[rank] += path.stat().st_size
    return tuple(
        tuple(sorted(rank_paths))
        for rank_paths in assignments
    )


def _validate_distributed_episode_partition(
    shard_paths: tuple[Path, ...],
    *,
    split: str,
    expected_episode_ids: set[str],
) -> set[str]:
    local_ids = [
        episode.episode_id
        for episode in iter_pooled_feature_episodes(
            shard_paths,
            split=split,
        )
    ]
    if len(local_ids) != len(set(local_ids)):
        raise ValueError("Duplicate episode ids within one distributed shard partition.")
    gathered: list[list[str] | None] = [
        None for _ in range(_distributed_world_size())
    ]
    torch.distributed.all_gather_object(gathered, local_ids)
    observed: set[str] = set()
    duplicates: set[str] = set()
    for rank_ids in gathered:
        if rank_ids is None:
            raise RuntimeError("Distributed episode id gathering failed.")
        for episode_id in rank_ids:
            if episode_id in observed:
                duplicates.add(episode_id)
            observed.add(episode_id)
    if duplicates:
        raise ValueError(
            "Episodes occur in more than one distributed partition: "
            f"{sorted(duplicates)[:8]}."
        )
    missing = sorted(expected_episode_ids - observed)
    unexpected = sorted(observed - expected_episode_ids)
    if missing or unexpected:
        raise ValueError(
            "Distributed pooled shards differ from the train manifest: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}."
        )
    return set(local_ids)


def _plain_config(config: DictConfig) -> dict[str, Any]:
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("The streaming codebook config must be a mapping.")
    return payload


def _config_hash(config: DictConfig) -> str:
    canonical = json.dumps(
        _plain_config(config),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source_checksums(paths: Sequence[Path], configured: Any) -> list[str]:
    observed = [file_sha256(path) for path in paths]
    if configured:
        expected = [str(value) for value in configured]
        if len(expected) != len(paths):
            raise ValueError(
                "Configured source_checksums must align one-to-one with pooled shards."
            )
        mismatches = [
            str(path)
            for path, expected_value, observed_value in zip(
                paths,
                expected,
                observed,
            )
            if expected_value != observed_value
        ]
        if mismatches:
            raise RuntimeError(
                "Configured pooled shard checksums do not match disk: "
                f"{mismatches[:8]}."
            )
    return observed


def _write_contract(path: Path, contract: dict[str, Any], resume: bool) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError(
                f"Existing run contract differs from the requested run: {path}. "
                "Use a new output directory."
            )
        if not resume:
            raise FileExistsError(
                f"Run contract already exists at {path}; enable resume or use a new output."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        mode = (path.stat().st_mode & 0o777) if path.exists() else 0o644
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifacts_equal(
    existing: FrozenRQArtifact,
    expected: FrozenRQArtifact,
) -> bool:
    return (
        existing.family == expected.family
        and existing.descriptor == expected.descriptor
        and existing.metadata == expected.metadata
        and existing.normalization.count == expected.normalization.count
        and torch.equal(
            existing.normalization.mean,
            expected.normalization.mean,
        )
        and torch.equal(
            existing.normalization.std,
            expected.normalization.std,
        )
        and len(existing.centers) == len(expected.centers)
        and all(
            torch.equal(existing_center, expected_center)
            for existing_center, expected_center in zip(
                existing.centers,
                expected.centers,
            )
        )
    )


def _write_frozen_artifact(
    path: Path,
    artifact: FrozenRQArtifact,
    resume: bool,
) -> None:
    if path.exists():
        if not resume:
            raise FileExistsError(
                f"Frozen RQ artifact already exists at {path}; "
                "enable resume or use a new output."
            )
        existing = FrozenRQArtifact.load(path)
        if not _artifacts_equal(existing, artifact):
            raise RuntimeError(
                f"Existing frozen RQ artifact differs from resumed state: {path}."
            )
        return
    artifact.save(path)


def _normalization_path(family_dir: Path) -> Path:
    return family_dir / "normalization.pt"


def _load_or_fit_normalization(
    family_dir: Path,
    source: CausalDescriptorSource,
    contract_hash: str,
    resume: bool,
) -> NormalizationStats:
    path = _normalization_path(family_dir)
    if resume and path.exists():
        payload = load_torch_payload(path, map_location="cpu")
        if payload.get("schema") != NORMALIZATION_SCHEMA:
            raise ValueError(f"Unsupported normalization schema in {path}.")
        if payload.get("contract_hash") != contract_hash:
            raise ValueError(f"Normalization contract mismatch in {path}.")
        return NormalizationStats.from_payload(payload["stats"])

    stats = fit_normalization(source, require_train_split=True)
    if _is_primary_rank():
        atomic_torch_save(
            {
                "schema": NORMALIZATION_SCHEMA,
                "contract_hash": contract_hash,
                "stats": stats.to_payload(),
            },
            path,
        )
    if _distributed_ready():
        torch.distributed.barrier()
    return stats


def _manifest_context(
    config: DictConfig,
    dataset_name: str,
) -> tuple[str, set[str] | None]:
    input_config = config.get("input", {})
    manifest_path = input_config.get("manifest", None)
    if manifest_path:
        manifest = EpisodeManifest.read_jsonl(manifest_path)
        manifest.assert_group_isolation(str(input_config.get("group_by", "scene")))
        train_ids = {
            record.episode_id
            for record in manifest
            if record.dataset == dataset_name and record.split == "train"
        }
        if not train_ids:
            raise ValueError(
                f"Manifest has no train episodes for dataset `{dataset_name}`."
            )
        return manifest.fingerprint(), train_ids

    fingerprint = str(config.get("metadata", {}).get("manifest_fingerprint", ""))
    if not fingerprint:
        raise ValueError("Provide `input.manifest` or `metadata.manifest_fingerprint`.")
    return fingerprint, None


def _episode_factory(
    shard_paths: tuple[Path, ...],
    split: str,
    expected_episode_ids: set[str] | None,
):
    def episodes() -> Iterator[PooledFeatureEpisode]:
        seen: set[str] = set()
        for episode in iter_pooled_feature_episodes(shard_paths, split=split):
            if expected_episode_ids is not None and episode.episode_id not in expected_episode_ids:
                raise ValueError(
                    f"Shard episode `{episode.episode_id}` is absent from the train manifest."
                )
            if episode.episode_id in seen:
                raise ValueError(f"Duplicate episode `{episode.episode_id}` across pooled shards.")
            seen.add(episode.episode_id)
            yield episode
        if expected_episode_ids is not None:
            missing = sorted(expected_episode_ids - seen)
            if missing:
                raise ValueError(f"Manifest train episodes are missing from shards: {missing[:8]}.")

    return episodes


def train_streaming_codebooks(config_path: str | Path) -> list[dict[str, Any]]:
    config_path = Path(config_path)
    config = OmegaConf.load(config_path)
    input_config = config.get("input", {})
    training = config.get("training", {})
    descriptor_config = config.get("descriptor", {})
    metadata_config = config.get("metadata", {})
    cpu_threads = int(training.get("cpu_threads", 4))
    if cpu_threads <= 0:
        raise ValueError("`training.cpu_threads` must be positive.")
    torch.set_num_threads(cpu_threads)

    split = str(input_config.get("split", "train"))
    if split != "train":
        raise ValueError("Frozen RQ artifacts may only be fit from the train split.")
    patterns = input_config.get("pooled_shards", ())
    if not patterns:
        raise ValueError("`input.pooled_shards` must contain at least one path or glob.")
    shard_paths = tuple(expand_shard_paths(patterns))
    dataset_name = str(metadata_config.get("dataset", ""))
    if not dataset_name:
        raise ValueError("`metadata.dataset` must not be empty.")

    manifest_fingerprint, expected_episode_ids = _manifest_context(config, dataset_name)
    configured_checksums = metadata_config.get("source_checksums", ())
    source_checksums = _broadcast_from_primary(
        _source_checksums(shard_paths, configured_checksums)
        if _is_primary_rank()
        else None
    )
    if not isinstance(source_checksums, list):
        raise RuntimeError("Distributed source checksum broadcast failed.")
    config_hash = _config_hash(config)
    implementation_sha256 = {
        "pipeline": file_sha256(Path(__file__)),
        "streaming": file_sha256(Path(__file__).with_name("streaming.py")),
    }
    output_dir = ensure_dir(config.get("output_dir", "runs/codebook_eval/streaming"))
    resume = bool(training.get("resume", True))
    world_size = _distributed_world_size()
    rank = _distributed_rank()
    shard_assignments = partition_shard_paths(shard_paths, world_size)
    local_shard_paths = shard_assignments[rank]
    if _distributed_ready():
        if expected_episode_ids is None:
            raise ValueError(
                "Distributed RQ training requires an explicit episode manifest."
            )
        local_episode_ids = _validate_distributed_episode_partition(
            local_shard_paths,
            split=split,
            expected_episode_ids=expected_episode_ids,
        )
    else:
        local_episode_ids = expected_episode_ids

    strides = [int(value) for value in descriptor_config.get("strides", (2, 3, 5))]
    if len(strides) != len(set(strides)):
        raise ValueError(f"Descriptor strides must be unique, got {strides}.")
    configured_camera_ids = descriptor_config.get("camera_ids", None)
    camera_ids = (
        None
        if configured_camera_ids is None
        else tuple(str(value) for value in configured_camera_ids)
    )
    batch_size = int(training.get("batch_size", 8192))
    levels = int(training.get("levels", 3))
    configured_device = str(training.get("device", "auto"))
    kmeans_config = StreamingKMeansConfig(
        k=int(training.get("k", 32)),
        max_iters=int(training.get("max_iters", 50)),
        tol=float(training.get("tol", 1e-5)),
        patience=int(training.get("patience", 2)),
        seed=int(training.get("seed", 0)),
        reservoir_size=int(training.get("reservoir_size", 100_000)),
        initialization_chunk_size=int(training.get("initialization_chunk_size", 8192)),
        center_block_size=int(training.get("center_block_size", 1024)),
        device=_runtime_device(configured_device),
    )
    episode_factory = _episode_factory(
        local_shard_paths,
        split=split,
        expected_episode_ids=local_episode_ids,
    )
    initialization_episode_factory = _episode_factory(
        shard_paths,
        split=split,
        expected_episode_ids=expected_episode_ids,
    )

    artifact_metadata = {
        "manifest_fingerprint": manifest_fingerprint,
        "dataset_revision": str(metadata_config.get("dataset_revision", "")),
        "wan_model_id": str(metadata_config.get("wan_model_id", "")),
        "wan_revision": str(metadata_config.get("wan_revision", "")),
        "preprocess_revision": str(metadata_config.get("preprocess_revision", "")),
        "config_hash": config_hash,
        "source_checksums": source_checksums,
        "dataset": dataset_name,
        "implementation_sha256": implementation_sha256,
        "distributed_world_size": world_size,
        "distributed_backend": _distributed_backend(),
        "initialization_policy": (
            "rank0-global-reservoir-kmeans++-v1"
            if world_size > 1
            else "single-stream-reservoir-kmeans++-v1"
        ),
    }
    empty_metadata = [
        key
        for key, value in artifact_metadata.items()
        if key not in {"source_checksums"} and value == ""
    ]
    if empty_metadata:
        raise ValueError(f"Empty artifact metadata fields: {empty_metadata}.")

    rows: list[dict[str, Any]] = []
    for stride in strides:
        spec = CausalDescriptorSpec(
            stride=stride,
            pool=int(descriptor_config.get("pool", 4)),
            max_gap_factor=descriptor_config.get("max_gap_factor", 1.5),
            camera_ids=camera_ids,
        )
        family_dir = ensure_dir(output_dir / spec.family)
        contract = {
            "schema": "codewam.family-run-contract.v1",
            "family": spec.family,
            "stride": spec.stride,
            "pool": spec.pool,
            "max_gap_factor": spec.max_gap_factor,
            "camera_ids": (
                None
                if spec.camera_ids is None
                else list(spec.camera_ids)
            ),
            "k": kmeans_config.k,
            "levels": levels,
            "batch_size": batch_size,
            "tol": kmeans_config.tol,
            "patience": kmeans_config.patience,
            "seed": kmeans_config.seed,
            "reservoir_size": kmeans_config.reservoir_size,
            "initialization_chunk_size": kmeans_config.initialization_chunk_size,
            "center_block_size": kmeans_config.center_block_size,
            "device": configured_device,
            "cpu_threads": cpu_threads,
            "distributed_world_size": world_size,
            "distributed_backend": artifact_metadata[
                "distributed_backend"
            ],
            "shard_assignments": [
                [str(path) for path in rank_paths]
                for rank_paths in shard_assignments
            ],
            "initialization_policy": artifact_metadata[
                "initialization_policy"
            ],
            "manifest_fingerprint": manifest_fingerprint,
            "source_checksums": source_checksums,
            "shards": [str(path) for path in shard_paths],
            "implementation_sha256": implementation_sha256,
        }
        contract_text = json.dumps(contract, sort_keys=True, separators=(",", ":"))
        contract_hash = hashlib.sha256(contract_text.encode("utf-8")).hexdigest()
        contract_path = family_dir / "contract.json"
        if _is_primary_rank():
            _write_contract(contract_path, contract, resume=resume)
        if _distributed_ready():
            torch.distributed.barrier()
            if not _is_primary_rank():
                _write_contract(contract_path, contract, resume=True)

        source = CausalDescriptorSource(
            episode_factory=episode_factory,
            spec=spec,
            batch_size=batch_size,
            split="train",
        )
        normalization = _load_or_fit_normalization(
            family_dir,
            source=source,
            contract_hash=contract_hash,
            resume=resume,
        )
        batch_factory = source.vector_batch_factory(
            normalization=normalization,
            device=kmeans_config.device,
        )
        initialization_source = CausalDescriptorSource(
            episode_factory=initialization_episode_factory,
            spec=spec,
            batch_size=batch_size,
            split="train",
        )
        initialization_batch_factory = (
            initialization_source.vector_batch_factory(
                normalization=normalization,
                device=kmeans_config.device,
            )
            if _is_primary_rank()
            else lambda: iter(())
        )
        rq_result = StreamingRQTrainer(kmeans_config, levels=levels).fit(
            batch_factory,
            initialization_batch_factory=initialization_batch_factory,
            checkpoint_dir=family_dir / "checkpoints",
            resume=resume,
        )
        artifact = FrozenRQArtifact(
            family=spec.family,
            descriptor=spec,
            normalization=normalization,
            centers=rq_result.centers,
            metadata=artifact_metadata,
        )
        artifact_path = family_dir / "codebook.pt"
        if _is_primary_rank():
            _write_frozen_artifact(
                artifact_path,
                artifact,
                resume=resume,
            )

        reductions = [
            1.0 - after / max(before, 1e-12)
            for before, after in zip(rq_result.residual_mse, rq_result.residual_mse[1:])
        ]
        row = {
            "family": spec.family,
            "stride": spec.stride,
            "pool": spec.pool,
            "camera_ids": (
                None
                if spec.camera_ids is None
                else list(spec.camera_ids)
            ),
            "k": kmeans_config.k,
            "levels": levels,
            "normalization_count": normalization.count,
            "dim": normalization.dim,
            "residual_mse": list(rq_result.residual_mse),
            "residual_reduction_by_level": reductions,
            "residual_total_reduction": (
                1.0
                - rq_result.residual_mse[-1]
                / max(rq_result.residual_mse[0], 1e-12)
            ),
            "iterations_per_level": list(rq_result.iterations_per_level),
            "patience": kmeans_config.patience,
            "cpu_threads": cpu_threads,
            "distributed_world_size": world_size,
            "implementation_sha256": implementation_sha256,
            "artifact": str(artifact_path),
        }
        if _is_primary_rank():
            save_json(family_dir / "train_summary.json", row)
        if _distributed_ready():
            torch.distributed.barrier()
        rows.append(row)

    if _is_primary_rank():
        save_json(output_dir / "train_summary.json", rows)
    if _distributed_ready():
        torch.distributed.barrier()
    return rows


def create_synthetic_streaming_fixture(output_dir: str | Path) -> Path:
    output_dir = ensure_dir(output_dir)
    config_path = output_dir / "synthetic_streaming.yaml"
    if config_path.exists():
        return config_path
    shard_dir = ensure_dir(output_dir / "pooled")
    generator = torch.Generator().manual_seed(20260723)
    records: list[EpisodeRecord] = []
    episodes: list[PooledFeatureEpisode] = []

    for index in range(18):
        split = "train" if index < 12 else ("val" if index < 15 else "test")
        episode_id = f"synthetic-{index:03d}"
        ticks = 18 + index % 4
        phase = torch.randn((2, 3, 4, 4), generator=generator)
        velocity = 0.08 * torch.randn((2, 3, 4, 4), generator=generator)
        time = torch.arange(ticks, dtype=torch.float32).view(ticks, 1, 1, 1, 1)
        periodic = torch.sin(time / float(2 + index % 3))
        features = phase + time * velocity + 0.1 * periodic
        features += 0.01 * torch.randn(features.shape, generator=generator)
        episodes.append(
            PooledFeatureEpisode(
                episode_id=episode_id,
                split=split,
                timestamps=torch.arange(ticks, dtype=torch.float64) / 4.0,
                pooled_g4=features.half(),
                camera_ids=("exterior", "wrist"),
                action=torch.randn((ticks, 7), generator=generator),
                proprio=torch.randn((ticks, 7), generator=generator),
                metadata={"kind": "synthetic-streaming-smoke"},
            )
        )
        records.append(
            EpisodeRecord(
                dataset="synthetic",
                episode_id=episode_id,
                num_steps=ticks,
                source_uri=f"memory://{episode_id}",
                scene_id=f"scene-{index:03d}",
                task_ids=(f"task-{index % 3}",),
                camera_ids=("exterior", "wrist"),
                split=split,
            )
        )

    manifest = EpisodeManifest.from_records(records)
    manifest_path = output_dir / "manifest.jsonl"
    manifest.write_jsonl(manifest_path)
    shard_paths = []
    shard_metadata = {
        "dataset_revision": "synthetic-v1",
        "wan_model_id": "synthetic-wan",
        "wan_revision": "synthetic-wan-v1",
        "preprocess_revision": "synthetic-preprocess-v1",
        "source_checksums": ["generated"],
    }
    for shard_index, start in enumerate(range(0, len(episodes), 6)):
        path = shard_dir / f"shard_{shard_index:05d}.pt"
        write_pooled_feature_shard(
            path,
            episodes[start : start + 6],
            metadata=shard_metadata,
        )
        shard_paths.append(path)

    config = OmegaConf.create(
        {
            "output_dir": str(output_dir / "artifacts"),
            "input": {
                "pooled_shards": [str(path) for path in shard_paths],
                "manifest": str(manifest_path),
                "split": "train",
                "group_by": "scene",
            },
            "metadata": {
                "dataset": "synthetic",
                "dataset_revision": "synthetic-v1",
                "wan_model_id": "synthetic-wan",
                "wan_revision": "synthetic-wan-v1",
                "preprocess_revision": "synthetic-preprocess-v1",
            },
            "descriptor": {
                "strides": [2, 3, 5],
                "pool": 2,
                "max_gap_factor": 1.5,
            },
            "training": {
                "device": "auto",
                "batch_size": 64,
                "k": 8,
                "levels": 3,
                "max_iters": 5,
                "tol": 0.0,
                "seed": 17,
                "reservoir_size": 256,
                "initialization_chunk_size": 64,
                "center_block_size": 64,
                "resume": True,
            },
        }
    )
    OmegaConf.save(config, config_path)
    return config_path
