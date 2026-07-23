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
    if configured:
        checksums = [str(value) for value in configured]
        if len(checksums) != len(paths):
            raise ValueError("Configured source_checksums must align one-to-one with pooled shards.")
        return checksums
    return [file_sha256(path) for path in paths]


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
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
    atomic_torch_save(
        {
            "schema": NORMALIZATION_SCHEMA,
            "contract_hash": contract_hash,
            "stats": stats.to_payload(),
        },
        path,
    )
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
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        raise RuntimeError(
            "The canonical launcher does not yet partition pooled shards by rank. "
            "Use one process until rank-aware orchestration is implemented."
        )

    config_path = Path(config_path)
    config = OmegaConf.load(config_path)
    input_config = config.get("input", {})
    training = config.get("training", {})
    descriptor_config = config.get("descriptor", {})
    metadata_config = config.get("metadata", {})

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
    source_checksums = _source_checksums(shard_paths, configured_checksums)
    config_hash = _config_hash(config)
    output_dir = ensure_dir(config.get("output_dir", "runs/codebook_eval/streaming"))
    resume = bool(training.get("resume", True))

    strides = [int(value) for value in descriptor_config.get("strides", (2, 3, 5))]
    if len(strides) != len(set(strides)):
        raise ValueError(f"Descriptor strides must be unique, got {strides}.")
    batch_size = int(training.get("batch_size", 8192))
    levels = int(training.get("levels", 3))
    kmeans_config = StreamingKMeansConfig(
        k=int(training.get("k", 32)),
        max_iters=int(training.get("max_iters", 50)),
        tol=float(training.get("tol", 1e-5)),
        seed=int(training.get("seed", 0)),
        reservoir_size=int(training.get("reservoir_size", 100_000)),
        initialization_chunk_size=int(training.get("initialization_chunk_size", 8192)),
        center_block_size=int(training.get("center_block_size", 1024)),
        device=str(training.get("device", "auto")),
    )
    episode_factory = _episode_factory(
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
        )
        family_dir = ensure_dir(output_dir / spec.family)
        contract = {
            "schema": "codewam.family-run-contract.v1",
            "family": spec.family,
            "stride": spec.stride,
            "pool": spec.pool,
            "max_gap_factor": spec.max_gap_factor,
            "k": kmeans_config.k,
            "levels": levels,
            "batch_size": batch_size,
            "tol": kmeans_config.tol,
            "seed": kmeans_config.seed,
            "reservoir_size": kmeans_config.reservoir_size,
            "initialization_chunk_size": kmeans_config.initialization_chunk_size,
            "center_block_size": kmeans_config.center_block_size,
            "device": kmeans_config.device,
            "manifest_fingerprint": manifest_fingerprint,
            "source_checksums": source_checksums,
            "shards": [str(path) for path in shard_paths],
        }
        contract_text = json.dumps(contract, sort_keys=True, separators=(",", ":"))
        contract_hash = hashlib.sha256(contract_text.encode("utf-8")).hexdigest()
        _write_contract(family_dir / "contract.json", contract, resume=resume)

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
        rq_result = StreamingRQTrainer(kmeans_config, levels=levels).fit(
            batch_factory,
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
        artifact.save(artifact_path)

        reductions = [
            1.0 - after / max(before, 1e-12)
            for before, after in zip(rq_result.residual_mse, rq_result.residual_mse[1:])
        ]
        row = {
            "family": spec.family,
            "stride": spec.stride,
            "pool": spec.pool,
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
            "artifact": str(artifact_path),
        }
        save_json(family_dir / "train_summary.json", row)
        rows.append(row)

    save_json(output_dir / "train_summary.json", rows)
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
