from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from codewam.codebook_eval.manifest import EpisodeManifest, EpisodeRecord
from codewam.data.droid_manifest import write_json_report
from codewam.data.droid_rlds import (
    DroidRLDSSegment,
    DroidShardWork,
    iter_manifest_droid_rlds_episodes,
    plan_droid_rank_assignments,
)

from .shards import (
    POOLED_SHARD_SCHEMA,
    PooledFeatureEpisode,
    file_sha256,
    load_torch_payload,
    write_pooled_feature_shard,
)
from .wan_probe_export import (
    _load_wan_vae,
    _preprocess_video,
    _torch_dtype,
    latent_frame_indices,
)


DROID_POOLED_EXPORT_SCHEMA = "codewam.droid-pooled-export.v1"
DROID_POOLED_SEGMENT_SCHEMA = "codewam.droid-pooled-segment.v1"


@dataclass(frozen=True)
class DroidPooledExportConfig:
    source_manifest: str
    data_dir: str
    output_dir: str
    vae_path: str
    fastwam_src: str
    rank: int = 0
    world_size: int = 1
    cameras: tuple[str, ...] = (
        "exterior_image_1_left",
        "wrist_image_left",
    )
    nominal_fps: float = 15.0
    image_height: int = 224
    image_width: int = 224
    device: str = "cuda"
    dtype: str = "bfloat16"
    resume: bool = True
    contract_wait_seconds: int = 120

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("DROID pooled export world size must be positive.")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(
                f"Rank {self.rank} is invalid for world size {self.world_size}."
            )
        if not self.cameras or len(set(self.cameras)) != len(self.cameras):
            raise ValueError("DROID pooled export cameras must be nonempty and unique.")
        if self.nominal_fps <= 0:
            raise ValueError("DROID pooled export FPS must be positive.")
        if self.image_height % 16 or self.image_width % 16:
            raise ValueError("Wan input height and width must be divisible by 16.")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"Unsupported Wan dtype `{self.dtype}`.")
        if self.contract_wait_seconds <= 0:
            raise ValueError("Contract wait time must be positive.")


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contract_parameters(
    config: DroidPooledExportConfig,
    manifest: EpisodeManifest,
    manifest_sha256: str,
    vae_path: Path,
) -> dict[str, Any]:
    fastwam_src = Path(config.fastwam_src)
    implementation_paths = {
        "wan_vae": fastwam_src / "fastwam/models/wan22/wan_video_vae.py",
        "state_dict_converter": (
            fastwam_src
            / "fastwam/models/wan22/helpers/state_dict_converters.py"
        ),
    }
    missing_implementations = [
        str(path) for path in implementation_paths.values() if not path.is_file()
    ]
    if missing_implementations:
        raise FileNotFoundError(
            f"Missing FastWAM Wan implementation files: {missing_implementations}."
        )
    return {
        "schema": DROID_POOLED_EXPORT_SCHEMA,
        "source_manifest_fingerprint": manifest.fingerprint(),
        "source_manifest_sha256": manifest_sha256,
        "dataset_revisions": sorted({record.dataset for record in manifest}),
        "vae_model_id": "Wan-AI/Wan2.2-TI2V-5B",
        "vae_file_name": vae_path.name,
        "vae_file_bytes": vae_path.stat().st_size,
        "implementation_sha256": {
            name: file_sha256(path)
            for name, path in sorted(implementation_paths.items())
        },
        "codewam_exporter_sha256": file_sha256(Path(__file__)),
        "cameras": list(config.cameras),
        "nominal_fps": config.nominal_fps,
        "image_height": config.image_height,
        "image_width": config.image_width,
        "dtype": config.dtype,
        "pool_grid": 4,
        "segment_policy": "independent-half-open-keep-ranges-no-padding",
        "timestamp_policy": "absolute-source-step/nominal-fps",
        "preprocess_revision": (
            f"rgb-direct-bilinear-{config.image_height}x{config.image_width}"
            "-minus1-plus1-pooled-g4-v1"
        ),
    }


def _load_or_create_contract(
    config: DroidPooledExportConfig,
    manifest: EpisodeManifest,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    source_manifest_path = Path(config.source_manifest)
    vae_path = Path(config.vae_path)
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest `{source_manifest_path}`.")
    if not vae_path.is_file():
        raise FileNotFoundError(f"Missing Wan VAE checkpoint `{vae_path}`.")
    parameters = _contract_parameters(
        config,
        manifest,
        file_sha256(source_manifest_path),
        vae_path,
    )

    if contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        observed_parameters = {
            key: contract.get(key) for key in parameters
        }
        if observed_parameters != parameters:
            mismatches = [
                key
                for key, value in parameters.items()
                if observed_parameters.get(key) != value
            ]
            raise RuntimeError(
                f"Existing pooled export contract differs in {mismatches}."
            )
        contract_payload = {
            **parameters,
            "vae_sha256": contract.get("vae_sha256"),
        }
        if not contract_payload["vae_sha256"]:
            raise RuntimeError("Existing pooled export contract lacks VAE SHA-256.")
        if contract.get("contract_hash") != _canonical_hash(contract_payload):
            raise RuntimeError("Existing pooled export contract hash is invalid.")
        return contract

    if config.rank != 0:
        deadline = time.monotonic() + config.contract_wait_seconds
        while time.monotonic() < deadline and not contract_path.is_file():
            time.sleep(0.25)
        if not contract_path.is_file():
            raise TimeoutError(
                f"Rank {config.rank} timed out waiting for `{contract_path}`."
            )
        return _load_or_create_contract(config, manifest)

    contract_payload = {
        **parameters,
        "vae_sha256": file_sha256(vae_path),
    }
    contract = {
        **contract_payload,
        "contract_hash": _canonical_hash(contract_payload),
    }
    write_json_report(contract_path, contract)
    return contract


def _work_output_path(output_dir: Path, work: DroidShardWork) -> Path:
    indices = {
        int(record.metadata["rlds_shard_index"]) for record in work.records
    }
    if len(indices) != 1:
        raise ValueError(
            f"Source shard `{work.shard_name}` has inconsistent numeric indices."
        )
    return output_dir / "pooled" / f"droid-rlds-{next(iter(indices)):05d}.pt"


def _expected_segment_ids(work: DroidShardWork) -> tuple[str, ...]:
    identifiers = []
    for record in work.records:
        ranges = record.metadata.get("keep_ranges")
        if not isinstance(ranges, list):
            raise ValueError(f"Episode `{record.key}` has no keep ranges.")
        identifiers.extend(
            f"{record.episode_id}@{int(start)}:{int(stop)}"
            for start, stop in ranges
        )
    return tuple(identifiers)


def _validate_reused_work(
    path: Path,
    work: DroidShardWork,
    contract_hash: str,
) -> dict[str, Any]:
    payload = load_torch_payload(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema") != POOLED_SHARD_SCHEMA:
        raise RuntimeError(f"Unsupported pooled shard `{path}`.")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Pooled shard `{path}` has no metadata mapping.")
    expected_metadata = {
        "export_contract_hash": contract_hash,
        "source_shard_name": work.shard_name,
        "source_shard_bytes": work.source_bytes,
    }
    mismatches = [
        key
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            f"Reused pooled shard `{path}` differs in {mismatches}."
        )
    episodes = [
        PooledFeatureEpisode.from_payload(value)
        for value in payload.get("episodes", ())
    ]
    if tuple(episode.episode_id for episode in episodes) != (
        _expected_segment_ids(work)
    ):
        raise RuntimeError(f"Reused pooled shard `{path}` has wrong segment ids.")
    return {
        "source_shard": work.shard_name,
        "source_episodes": work.episodes,
        "segments": len(episodes),
        "ticks": sum(episode.ticks for episode in episodes),
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "status": "reused",
    }


def encode_droid_segment(
    segment: DroidRLDSSegment,
    vae: Any,
    config: DroidPooledExportConfig,
) -> PooledFeatureEpisode:
    if segment.split is None:
        raise ValueError(f"DROID segment `{segment.segment_id}` has no split.")
    dtype = _torch_dtype(config.dtype)
    camera_videos = [
        _preprocess_video(
            segment.frames[camera],
            height=config.image_height,
            width=config.image_width,
            dtype=dtype,
        )
        for camera in config.cameras
    ]
    with torch.inference_mode():
        latent = vae.encode(
            camera_videos,
            device=torch.device(config.device),
            tiled=False,
        )
    if latent.ndim != 5 or latent.shape[0] != len(config.cameras):
        raise RuntimeError(f"Unexpected Wan latent shape: {tuple(latent.shape)}.")
    views, channels, ticks, latent_height, latent_width = latent.shape
    if channels != 48:
        raise RuntimeError(f"Expected 48 Wan latent channels, got {channels}.")
    pooled = F.adaptive_avg_pool2d(
        latent.permute(0, 2, 1, 3, 4).reshape(
            views * ticks,
            channels,
            latent_height,
            latent_width,
        ),
        output_size=(4, 4),
    )
    pooled = (
        pooled.reshape(views, ticks, channels, 4, 4)
        .permute(1, 0, 2, 3, 4)
        .half()
        .cpu()
        .contiguous()
    )
    relative_indices = latent_frame_indices(segment.steps, ticks)
    absolute_indices = relative_indices + segment.start
    return PooledFeatureEpisode(
        episode_id=segment.segment_id,
        split=segment.split,
        timestamps=absolute_indices.double() / config.nominal_fps,
        pooled_g4=pooled,
        camera_ids=config.cameras,
        action=segment.action[relative_indices].half(),
        proprio=segment.proprio[relative_indices].half(),
        action_components={
            name: values[relative_indices].half()
            for name, values in segment.action_components.items()
        },
        metadata={
            "schema": DROID_POOLED_SEGMENT_SCHEMA,
            "parent_episode_id": segment.episode_id,
            "parent_manifest_key": segment.manifest_key,
            "source_shard": segment.source_shard,
            "record_index": segment.record_index,
            "range_index": segment.range_index,
            "source_range": [segment.start, segment.stop],
            "source_frames": segment.steps,
            "relative_latent_frame_indices": relative_indices,
            "absolute_latent_frame_indices": absolute_indices,
            "latent_shape": [channels, ticks, latent_height, latent_width],
            "source_image_shape": list(
                segment.frames[config.cameras[0]].shape[1:]
            ),
            "model_input_shape": [config.image_height, config.image_width],
            "timestamp_source": "absolute-source-step/nominal-fps",
            "nominal_fps": config.nominal_fps,
        },
    )


def _pooled_shard_metadata(
    contract: dict[str, Any],
    work: DroidShardWork,
) -> dict[str, Any]:
    checksums = {record.source_checksum for record in work.records}
    if len(checksums) != 1 or None in checksums:
        raise ValueError(
            f"Source shard `{work.shard_name}` has inconsistent checksums."
        )
    return {
        "dataset_revision": ",".join(contract["dataset_revisions"]),
        "wan_model_id": contract["vae_model_id"],
        "wan_revision": contract["vae_sha256"],
        "preprocess_revision": contract["preprocess_revision"],
        "source_checksums": sorted(checksums),
        "source_manifest_fingerprint": contract[
            "source_manifest_fingerprint"
        ],
        "export_contract_hash": contract["contract_hash"],
        "source_shard_name": work.shard_name,
        "source_shard_bytes": work.source_bytes,
    }


def export_droid_pooled_features(
    config: DroidPooledExportConfig,
) -> dict[str, Any]:
    source_manifest = EpisodeManifest.read_jsonl(config.source_manifest)
    contract = _load_or_create_contract(config, source_manifest)
    assignment = plan_droid_rank_assignments(
        source_manifest,
        config.world_size,
    )[config.rank]
    output_dir = Path(config.output_dir)
    (output_dir / "pooled").mkdir(parents=True, exist_ok=True)

    rows = []
    pending_work: dict[str, DroidShardWork] = {}
    completed_episode_keys: set[str] = set()
    for work in assignment.shards:
        path = _work_output_path(output_dir, work)
        if path.is_file() and config.resume:
            rows.append(
                _validate_reused_work(path, work, contract["contract_hash"])
            )
            completed_episode_keys.update(
                record.key for record in work.records
            )
        elif path.exists():
            raise FileExistsError(f"Pooled shard already exists: {path}.")
        else:
            pending_work[work.shard_name] = work

    episode_iterator = iter_manifest_droid_rlds_episodes(
        config.data_dir,
        source_manifest,
        rank=config.rank,
        world_size=config.world_size,
        cameras=config.cameras,
        completed_episode_keys=completed_episode_keys,
    )
    vae = None
    current_shard: str | None = None
    current_pooled: list[PooledFeatureEpisode] = []
    shard_started = time.monotonic()
    shard_peak_gib = 0.0

    def flush() -> None:
        nonlocal current_shard, current_pooled, shard_started, shard_peak_gib
        if current_shard is None:
            return
        work = pending_work[current_shard]
        path = _work_output_path(output_dir, work)
        info = write_pooled_feature_shard(
            path,
            current_pooled,
            metadata=_pooled_shard_metadata(contract, work),
        )
        rows.append(
            {
                "source_shard": current_shard,
                "source_episodes": work.episodes,
                "segments": info.episodes,
                "ticks": info.ticks,
                "path": str(path),
                "sha256": info.sha256,
                "bytes": path.stat().st_size,
                "elapsed_seconds": time.monotonic() - shard_started,
                "peak_cuda_memory_gib": shard_peak_gib,
                "status": "exported",
            }
        )
        print(
            f"Exported {current_shard}: episodes={work.episodes} "
            f"segments={info.episodes} ticks={info.ticks} "
            f"sha256={info.sha256[:12]}",
            flush=True,
        )
        current_shard = None
        current_pooled = []
        shard_peak_gib = 0.0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    started = time.monotonic()
    for episode in episode_iterator:
        if episode.source_shard != current_shard:
            flush()
            current_shard = episode.source_shard
            shard_started = time.monotonic()
            device = torch.device(config.device)
            if device.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("Wan export requested unavailable CUDA.")
                torch.cuda.reset_peak_memory_stats(device)
        if vae is None:
            vae = _load_wan_vae(config)
        for segment in episode.iter_eligible_segments():
            current_pooled.append(encode_droid_segment(segment, vae, config))
            device = torch.device(config.device)
            if device.type == "cuda":
                shard_peak_gib = max(
                    shard_peak_gib,
                    float(torch.cuda.max_memory_allocated(device) / 1024**3),
                )
    flush()

    report = {
        "schema": DROID_POOLED_EXPORT_SCHEMA,
        "contract_hash": contract["contract_hash"],
        "rank": config.rank,
        "world_size": config.world_size,
        "assignment": {
            "source_shards": len(assignment.shards),
            "source_episodes": assignment.episodes,
            "source_bytes": assignment.source_bytes,
        },
        "outputs": sorted(rows, key=lambda row: row["source_shard"]),
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path = (
        output_dir
        / f"rank-{config.rank:03d}-of-{config.world_size:03d}-report.json"
    )
    write_json_report(report_path, report)
    return report


def finalize_droid_pooled_export(
    source_manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source_manifest_path = Path(source_manifest_path)
    output_dir = Path(output_dir)
    contract_path = output_dir / "contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(f"Missing pooled export contract `{contract_path}`.")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    source_manifest = EpisodeManifest.read_jsonl(source_manifest_path)
    if source_manifest.fingerprint() != contract.get(
        "source_manifest_fingerprint"
    ):
        raise RuntimeError("Finalize source manifest differs from export contract.")

    source_by_key = {record.key: record for record in source_manifest}
    records: list[EpisodeRecord] = []
    file_rows = []
    for work in plan_droid_rank_assignments(source_manifest, 1)[0].shards:
        path = _work_output_path(output_dir, work)
        if not path.is_file():
            raise FileNotFoundError(f"Missing expected pooled shard `{path}`.")
        row = _validate_reused_work(path, work, contract["contract_hash"])
        file_rows.append(row)
        payload = load_torch_payload(path, map_location="cpu")
        for value in payload["episodes"]:
            episode = PooledFeatureEpisode.from_payload(value)
            parent_key = str(episode.metadata["parent_manifest_key"])
            parent = source_by_key[parent_key]
            records.append(
                EpisodeRecord(
                    dataset=parent.dataset,
                    episode_id=episode.episode_id,
                    num_steps=episode.ticks,
                    source_uri=f"{path}#{episode.episode_id}",
                    scene_id=parent.scene_id,
                    building_id=parent.building_id,
                    institution_id=parent.institution_id,
                    task_ids=parent.task_ids,
                    camera_ids=episode.camera_ids,
                    source_checksum=f"sha256:{row['sha256']}",
                    split=episode.split,
                    metadata={
                        "parent_episode_id": parent.episode_id,
                        "parent_manifest_key": parent.key,
                        "source_shard": work.shard_name,
                        "pooled_shard": str(path),
                        "source_range": episode.metadata["source_range"],
                        "source_frames": episode.metadata["source_frames"],
                        "latent_ticks": episode.ticks,
                        "export_contract_hash": contract["contract_hash"],
                    },
                )
            )

    pooled_manifest = EpisodeManifest.from_records(records)
    pooled_manifest.assert_group_isolation("scene")
    pooled_manifest_path = output_dir / "pooled_manifest.jsonl"
    pooled_manifest.write_jsonl(pooled_manifest_path)
    report = {
        "schema": DROID_POOLED_EXPORT_SCHEMA,
        "contract_hash": contract["contract_hash"],
        "source_manifest": {
            "path": str(source_manifest_path),
            "fingerprint": source_manifest.fingerprint(),
            "episodes": len(source_manifest),
        },
        "pooled_manifest": {
            "path": str(pooled_manifest_path),
            **pooled_manifest.stats(),
        },
        "pooled_shards": len(file_rows),
        "pooled_bytes": sum(row["bytes"] for row in file_rows),
        "latent_ticks": sum(row["ticks"] for row in file_rows),
        "files": file_rows,
    }
    write_json_report(output_dir / "export_summary.json", report)
    return report
