from __future__ import annotations

import ctypes
import gc
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch

from codewam.codebook_eval.manifest import EpisodeManifest, EpisodeRecord
from codewam.codebook_eval.shards import file_sha256
from codewam.codebook_eval.wan_probe_export import (
    _load_wan_vae,
    _preprocess_video,
    _torch_dtype,
    latent_frame_indices,
)

from .droid_endpoint import (
    DROID_ENDPOINT_AUDIT_SCHEMA,
    DROID_ENDPOINT_POLICY,
)
from .droid_rlds import (
    DroidRLDSSegment,
    DroidShardWork,
    iter_manifest_droid_rlds_episodes,
    plan_droid_rank_assignments,
)
from .frozen_assignment import (
    FrozenArtifactChart,
    FrozenCausalCodeAssigner,
    load_frozen_artifact_chart,
)
from .joint_cache import (
    JointEpisode,
    JointWindowConfig,
    build_joint_windows,
    create_joint_cache_contract,
    finalize_joint_cache,
    validate_joint_episode_shard,
    write_joint_cache_contract,
    write_joint_episode_shard,
)
from .roles import trajectory_role


JOINT_CACHE_EXPORT_REPORT_SCHEMA = "codewam.joint-cache-export-report.v2"
JOINT_CACHE_EXPORT_AUDIT_SCHEMA = "codewam.joint-cache-export-audit.v1"


@dataclass(frozen=True)
class JointCacheExportConfig:
    source_manifest: str
    data_dir: str
    output_dir: str
    endpoint_audit: str
    artifact_paths: dict[str, str]
    chart_name: str
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
    window: JointWindowConfig = JointWindowConfig()
    resume: bool = True
    max_source_shards: int | None = None

    def __post_init__(self) -> None:
        if self.world_size <= 0 or self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("Joint export rank/world size is invalid.")
        if not self.chart_name:
            raise ValueError("Joint export chart name must not be empty.")
        if not self.cameras or len(set(self.cameras)) != len(self.cameras):
            raise ValueError("Joint export cameras must be nonempty and unique.")
        if self.nominal_fps <= 0:
            raise ValueError("Joint export nominal FPS must be positive.")
        if self.image_height % 16 or self.image_width % 16:
            raise ValueError("Wan input dimensions must be divisible by 16.")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"Unsupported Wan dtype `{self.dtype}`.")
        if self.max_source_shards is not None and self.max_source_shards <= 0:
            raise ValueError("Maximum source shard count must be positive.")


def _release_process_memory(device: torch.device) -> None:
    gc.collect()
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    except (AttributeError, OSError):
        pass
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _resolve_fastwam_src(path: str | Path) -> Path:
    root = Path(path).resolve()
    candidates = (root, root / "src")
    for candidate in candidates:
        if (candidate / "fastwam").is_dir():
            return candidate
    raise FileNotFoundError(
        f"`{root}` is neither a FastWAM package root nor a repository "
        "containing src/fastwam."
    )


def _work_stem(work: DroidShardWork) -> str:
    indices = {
        int(record.metadata["rlds_shard_index"]) for record in work.records
    }
    if len(indices) != 1:
        raise ValueError(
            f"DROID source shard `{work.shard_name}` has inconsistent indices."
        )
    return f"droid-rlds-{next(iter(indices)):05d}"


def _work_metadata(work: DroidShardWork) -> dict[str, Any]:
    keys = [record.key for record in work.records]
    return {
        "source_shard_name": work.shard_name,
        "source_shard_bytes": work.source_bytes,
        "source_episode_keys_hash": _hash_strings(keys),
        "source_episode_count": len(keys),
    }


def _hash_strings(values: list[str]) -> str:
    encoded = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def _write_export_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_export_report(
    config: JointCacheExportConfig,
    *,
    contract_hash: str,
    planned_shards: int,
    selected_shards: int,
    rows: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    if len(rows) != selected_shards:
        raise RuntimeError(
            f"Rank {config.rank} completed {len(rows)} of "
            f"{selected_shards} selected source shards."
        )
    return {
        "schema": JOINT_CACHE_EXPORT_REPORT_SCHEMA,
        "contract_hash": contract_hash,
        "rank": config.rank,
        "world_size": config.world_size,
        "source_shards_planned": planned_shards,
        "source_shards_selected": selected_shards,
        "source_shards_exported_or_reused": len(rows),
        "selection_complete": selected_shards == planned_shards,
        "max_source_shards": config.max_source_shards,
        "episodes": sum(int(row["episodes"]) for row in rows),
        "windows": sum(int(row["windows"]) for row in rows),
        "elapsed_seconds": time.monotonic() - started,
        "completed_unix_seconds": time.time(),
        "outputs": sorted(rows, key=lambda row: row["source_shard"]),
    }


def _load_endpoint_audit(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != DROID_ENDPOINT_AUDIT_SCHEMA
        or report.get("endpoint_policy") != DROID_ENDPOINT_POLICY
        or report.get("verdict") != "pass"
    ):
        raise RuntimeError(
            f"Endpoint audit `{path}` is absent, incompatible or did not pass."
        )
    return report


def _create_contract(
    config: JointCacheExportConfig,
    manifest: EpisodeManifest,
    chart: FrozenArtifactChart,
) -> dict[str, Any]:
    source_manifest = Path(config.source_manifest)
    endpoint_audit = Path(config.endpoint_audit)
    vae_path = Path(config.vae_path)
    for path in (source_manifest, endpoint_audit, vae_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing joint export input `{path}`.")
    fastwam_src = _resolve_fastwam_src(config.fastwam_src)
    _load_endpoint_audit(endpoint_audit)
    preprocess_revision = (
        f"rgb-direct-bilinear-{config.image_height}x{config.image_width}"
        "-minus1-plus1-pooled-g4-v1"
    )
    dataset_revisions = sorted({record.dataset for record in manifest})
    if len(dataset_revisions) != 1:
        raise ValueError(
            f"Joint cache needs one dataset revision, got {dataset_revisions}."
        )
    implementation_paths = {
        "joint_cache_export": Path(__file__),
        "joint_cache": Path(__file__).with_name("joint_cache.py"),
        "frozen_assignment": Path(__file__).with_name(
            "frozen_assignment.py"
        ),
        "droid_rlds": Path(__file__).with_name("droid_rlds.py"),
        "wan_probe_export": (
            Path(__file__).parents[1]
            / "codebook_eval"
            / "wan_probe_export.py"
        ),
        "fastwam_wan_video_vae": (
            fastwam_src
            / "fastwam"
            / "models"
            / "wan22"
            / "wan_video_vae.py"
        ),
        "fastwam_wan_io": (
            fastwam_src
            / "fastwam"
            / "models"
            / "wan22"
            / "helpers"
            / "io.py"
        ),
        "fastwam_wan_state_dict_converter": (
            fastwam_src
            / "fastwam"
            / "models"
            / "wan22"
            / "helpers"
            / "state_dict_converters.py"
        ),
    }
    return create_joint_cache_contract(
        dataset_revision=dataset_revisions[0],
        source_manifest_fingerprint=manifest.fingerprint(),
        source_manifest_sha256=file_sha256(source_manifest),
        endpoint_audit_sha256=file_sha256(endpoint_audit),
        chart=chart,
        camera_ids=config.cameras,
        wan_model_id="Wan-AI/Wan2.2-TI2V-5B",
        wan_revision=file_sha256(vae_path),
        preprocess_revision=preprocess_revision,
        nominal_fps=config.nominal_fps,
        action_dim=7,
        proprio_dim=14,
        latent_channels=48,
        window=config.window,
        implementation_paths=implementation_paths,
    )


@torch.no_grad()
def encode_joint_segment(
    segment: DroidRLDSSegment,
    *,
    record: EpisodeRecord,
    vae: Any,
    chart: FrozenArtifactChart,
    assigner: FrozenCausalCodeAssigner,
    config: JointCacheExportConfig,
) -> JointEpisode:
    if segment.split is None:
        raise ValueError(f"DROID segment `{segment.segment_id}` has no split.")
    if segment.action_valid is None:
        raise ValueError(
            f"DROID segment `{segment.segment_id}` lacks RLDS terminal flags."
        )
    dtype = _torch_dtype(config.dtype)
    videos = [
        _preprocess_video(
            segment.frames[camera],
            height=config.image_height,
            width=config.image_width,
            dtype=dtype,
        )
        for camera in config.cameras
    ]
    latent = vae.encode(
        videos,
        device=torch.device(config.device),
        tiled=False,
    )
    if latent.ndim != 5 or latent.shape[0] != len(config.cameras):
        raise RuntimeError(f"Unexpected Wan latent shape: {tuple(latent.shape)}.")
    views, channels, ticks, _, _ = latent.shape
    if channels != 48:
        raise RuntimeError(f"Expected 48 Wan latent channels, got {channels}.")
    relative_indices = latent_frame_indices(segment.steps, ticks)
    absolute_indices = relative_indices + segment.start
    timestamps = absolute_indices.double() / config.nominal_fps
    latent_tvc = latent.permute(2, 0, 1, 3, 4).contiguous()
    latent_valid = torch.ones(
        (ticks, views),
        dtype=torch.bool,
        device=latent_tvc.device,
    )
    assignment = assigner.assign(
        latent_tvc,
        latent_source_indices=absolute_indices.to(latent_tvc.device),
        camera_ids=config.cameras,
        latent_valid=latent_valid,
        timestamps=timestamps.to(latent_tvc.device),
    )
    return JointEpisode(
        episode_id=segment.segment_id,
        parent_episode_id=segment.episode_id,
        manifest_key=record.key,
        range_index=segment.range_index,
        range_start=segment.start,
        range_stop=segment.stop,
        split=segment.split,
        chart_name=chart.name,
        role=trajectory_role(record),
        camera_ids=config.cameras,
        latents=latent_tvc.half().cpu(),
        latent_source_indices=absolute_indices.cpu(),
        latent_valid=latent_valid.cpu(),
        source_actions=segment.action.float().cpu(),
        source_proprio=segment.proprio.float().cpu(),
        source_action_valid=segment.action_valid.cpu(),
        code_ids=assignment.code_ids.cpu(),
        code_available=assignment.available.cpu(),
        descriptor_source_indices=(
            assignment.descriptor_source_indices.cpu()
        ),
        families=assignment.families,
        language_instruction=segment.language_instruction,
        metadata={
            "parent_manifest_key": record.key,
            "source_shard": segment.source_shard,
            "record_index": segment.record_index,
            "source_checksum": record.source_checksum,
            "source_range": [segment.start, segment.stop],
            "latent_shape": list(latent.shape[1:]),
            "nominal_fps": config.nominal_fps,
        },
    )


def _validate_reused_work(
    config: JointCacheExportConfig,
    work: DroidShardWork,
    *,
    contract_hash: str,
) -> dict[str, Any] | None:
    shard_path = (
        Path(config.output_dir)
        / "episode_shards"
        / f"{_work_stem(work)}.pt"
    )
    sidecar_path = shard_path.with_suffix(".index.json")
    if not shard_path.exists() and not sidecar_path.exists():
        return None
    if not config.resume:
        raise FileExistsError(f"Joint cache shard already exists: {shard_path}.")
    if not shard_path.is_file() or not sidecar_path.is_file():
        raise RuntimeError(f"Incomplete joint cache shard: {shard_path}.")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if (
        sidecar.get("contract_hash") != contract_hash
        or sidecar.get("metadata") != _work_metadata(work)
    ):
        raise RuntimeError(f"Reused joint cache shard differs: {shard_path}.")
    episodes = validate_joint_episode_shard(
        shard_path,
        contract_hash=contract_hash,
        expected_sha256=str(sidecar["episode_shard_sha256"]),
    )
    expected = {
        f"{record.episode_id}@{int(start)}:{int(stop)}"
        for record in work.records
        for start, stop in record.metadata["keep_ranges"]
    }
    if {episode.episode_id for episode in episodes} != expected:
        raise RuntimeError(f"Reused joint cache shard is incomplete: {shard_path}.")
    return {
        "source_shard": work.shard_name,
        "episodes": len(episodes),
        "windows": len(sidecar.get("windows", ())),
        "status": "reused",
        "path": str(shard_path),
        "sha256": str(sidecar["episode_shard_sha256"]),
    }


def export_joint_window_cache(
    config: JointCacheExportConfig,
) -> dict[str, Any]:
    manifest = EpisodeManifest.read_jsonl(config.source_manifest)
    chart = load_frozen_artifact_chart(
        config.chart_name,
        config.artifact_paths,
    )
    contract = _create_contract(config, manifest, chart)
    write_joint_cache_contract(config.output_dir, contract)
    assignment = plan_droid_rank_assignments(
        manifest,
        config.world_size,
    )[config.rank]
    planned_works = assignment.shards
    works = planned_works
    if config.max_source_shards is not None:
        works = works[: config.max_source_shards]
    pending: dict[str, DroidShardWork] = {}
    rows = []
    for work in works:
        reused = _validate_reused_work(
            config,
            work,
            contract_hash=contract["contract_hash"],
        )
        if reused is None:
            pending[work.shard_name] = work
        else:
            rows.append(reused)

    selected_manifest = EpisodeManifest.from_records(
        record
        for work in pending.values()
        for record in work.records
    )
    records_by_key = {record.key: record for record in selected_manifest}
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Joint cache export requested unavailable CUDA.")
        torch.cuda.set_device(device)
        torch.cuda.init()
    started = time.monotonic()
    if not pending:
        report = _build_export_report(
            config,
            contract_hash=contract["contract_hash"],
            planned_shards=len(planned_works),
            selected_shards=len(works),
            rows=rows,
            started=started,
        )
        report_path = (
            Path(config.output_dir)
            / f"rank-{config.rank:03d}-of-{config.world_size:03d}-report.json"
        )
        _write_export_report(report_path, report)
        return report
    vae = None
    vae_config = replace(
        config,
        fastwam_src=str(_resolve_fastwam_src(config.fastwam_src)),
    )
    assigner = FrozenCausalCodeAssigner(chart)
    current_shard: str | None = None
    current_episodes: list[JointEpisode] = []
    current_windows = []

    def flush() -> None:
        nonlocal current_shard, current_episodes, current_windows
        if current_shard is None:
            return
        work = pending[current_shard]
        sidecar = write_joint_episode_shard(
            config.output_dir,
            _work_stem(work),
            current_episodes,
            current_windows,
            contract_hash=contract["contract_hash"],
            metadata=_work_metadata(work),
        )
        rows.append(
            {
                "source_shard": current_shard,
                "episodes": len(current_episodes),
                "windows": len(current_windows),
                "status": "exported",
                "path": str(
                    Path(config.output_dir) / sidecar["episode_shard"]
                ),
                "sha256": sidecar["episode_shard_sha256"],
            }
        )
        print(
            f"Exported {current_shard}: episodes={len(current_episodes)} "
            f"windows={len(current_windows)}",
            flush=True,
        )
        current_shard = None
        current_episodes = []
        current_windows = []
        _release_process_memory(device)

    episode_iterator = iter_manifest_droid_rlds_episodes(
        config.data_dir,
        selected_manifest,
        rank=0,
        world_size=1,
        cameras=config.cameras,
    )
    for episode in episode_iterator:
        if episode.source_shard != current_shard:
            flush()
            current_shard = episode.source_shard
        if vae is None:
            vae = _load_wan_vae(vae_config)
            _release_process_memory(device)
        if episode.manifest_key is None:
            raise RuntimeError("Manifest-backed DROID episode lost its key.")
        record = records_by_key[episode.manifest_key]
        for segment in episode.iter_eligible_segments():
            joint_episode = encode_joint_segment(
                segment,
                record=record,
                vae=vae,
                chart=chart,
                assigner=assigner,
                config=config,
            )
            windows = build_joint_windows(
                joint_episode,
                config=config.window,
                artifact_sha256=chart.artifact_sha256,
            )
            current_episodes.append(joint_episode)
            current_windows.extend(windows)
    flush()
    report = _build_export_report(
        config,
        contract_hash=contract["contract_hash"],
        planned_shards=len(planned_works),
        selected_shards=len(works),
        rows=rows,
        started=started,
    )
    report_path = (
        Path(config.output_dir)
        / f"rank-{config.rank:03d}-of-{config.world_size:03d}-report.json"
    )
    _write_export_report(report_path, report)
    return report


def _audit_export_reports(
    output_dir: str | Path,
    *,
    allow_partial: bool,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    contract_path = output_dir / "contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"Missing joint cache contract `{contract_path}`."
        )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_hash = str(contract.get("contract_hash", ""))

    source_shards = {}
    for path in sorted((output_dir / "episode_shards").glob("*.index.json")):
        sidecar = json.loads(path.read_text(encoding="utf-8"))
        source_shard = str(
            sidecar.get("metadata", {}).get("source_shard_name", "")
        )
        if not source_shard:
            raise RuntimeError(
                f"Joint cache sidecar lacks source shard provenance: {path}."
            )
        if source_shard in source_shards:
            raise RuntimeError(
                f"Joint cache repeats source shard `{source_shard}`."
            )
        source_shards[source_shard] = path
    if not source_shards:
        raise FileNotFoundError("Joint cache has no shard sidecars.")

    groups: dict[int, dict[int, tuple[Path, dict[str, Any]]]] = {}
    for path in sorted(output_dir.glob("rank-*-of-*-report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("contract_hash") != contract_hash:
            continue
        if report.get("schema") != JOINT_CACHE_EXPORT_REPORT_SCHEMA:
            raise RuntimeError(
                f"Export report `{path}` predates the completeness contract; "
                "rerun the exporter to refresh rank reports."
            )
        rank = int(report.get("rank", -1))
        world_size = int(report.get("world_size", 0))
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise RuntimeError(f"Export report `{path}` has invalid rank metadata.")
        ranks = groups.setdefault(world_size, {})
        if rank in ranks:
            raise RuntimeError(
                f"Duplicate export report for rank {rank}/{world_size}."
            )
        ranks[rank] = (path, report)
    if not groups:
        raise FileNotFoundError(
            "Joint cache has no compatible rank export reports."
        )

    candidates = []
    failures = []
    for world_size, ranks in sorted(groups.items()):
        expected_ranks = set(range(world_size))
        if set(ranks) != expected_ranks:
            failures.append(
                f"world_size={world_size} missing ranks "
                f"{sorted(expected_ranks - set(ranks))}"
            )
            continue
        reported_sources: dict[str, tuple[int, dict[str, Any]]] = {}
        planned_total = 0
        selected_total = 0
        episode_total = 0
        window_total = 0
        reports = []
        group_complete = True
        invalid = None
        for rank in range(world_size):
            path, report = ranks[rank]
            planned = int(report.get("source_shards_planned", -1))
            selected = int(report.get("source_shards_selected", -1))
            completed = int(
                report.get("source_shards_exported_or_reused", -1)
            )
            outputs = report.get("outputs")
            if (
                planned < 0
                or selected < 0
                or selected > planned
                or completed != selected
                or not isinstance(outputs, list)
                or len(outputs) != selected
                or bool(report.get("selection_complete"))
                != (selected == planned)
            ):
                invalid = f"rank {rank}/{world_size} has inconsistent counts"
                break
            if (
                sum(int(row["episodes"]) for row in outputs)
                != int(report.get("episodes", -1))
                or sum(int(row["windows"]) for row in outputs)
                != int(report.get("windows", -1))
            ):
                invalid = f"rank {rank}/{world_size} has inconsistent totals"
                break
            for row in outputs:
                source_shard = str(row.get("source_shard", ""))
                if not source_shard or source_shard in reported_sources:
                    invalid = (
                        f"rank group {world_size} repeats or omits "
                        "a source shard"
                    )
                    break
                reported_sources[source_shard] = (rank, row)
            if invalid is not None:
                break
            planned_total += planned
            selected_total += selected
            episode_total += int(report["episodes"])
            window_total += int(report["windows"])
            group_complete &= bool(report["selection_complete"])
            reports.append(
                {
                    "path": path.name,
                    "sha256": file_sha256(path),
                    "rank": rank,
                }
            )
        if invalid is not None:
            failures.append(invalid)
            continue
        if set(reported_sources) != set(source_shards):
            failures.append(
                f"world_size={world_size} report/sidecar source sets differ"
            )
            continue
        if not group_complete and not allow_partial:
            failures.append(
                f"world_size={world_size} selected {selected_total} of "
                f"{planned_total} planned shards"
            )
            continue
        completed_at = max(
            float(report.get("completed_unix_seconds", 0.0))
            for _, report in ranks.values()
        )
        candidates.append(
            (
                completed_at,
                world_size,
                {
                    "schema": JOINT_CACHE_EXPORT_AUDIT_SCHEMA,
                    "contract_hash": contract_hash,
                    "status": "complete" if group_complete else "partial",
                    "world_size": world_size,
                    "source_shards_planned": planned_total,
                    "source_shards_selected": selected_total,
                    "episodes": episode_total,
                    "windows": window_total,
                    "reports": reports,
                },
            )
        )
    if not candidates:
        detail = "; ".join(failures) or "no coherent report group"
        raise RuntimeError(f"Joint cache export is incomplete: {detail}.")
    return max(candidates, key=lambda value: (value[0], value[1]))[2]


def finalize_exported_joint_cache(
    output_dir: str | Path,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    audit = _audit_export_reports(
        output_dir,
        allow_partial=allow_partial,
    )
    summary = finalize_joint_cache(output_dir, export_audit=audit)
    if (
        int(summary["episodes"]) != int(audit["episodes"])
        or int(summary["windows"]) != int(audit["windows"])
        or int(summary["episode_shards"])
        != int(audit["source_shards_selected"])
    ):
        raise RuntimeError(
            "Finalized joint cache totals differ from its rank reports."
        )
    return summary
