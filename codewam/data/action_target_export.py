from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from codewam.codebook_eval.manifest import EpisodeManifest, EpisodeRecord
from codewam.codebook_eval.shards import file_sha256

from .action_targets import (
    DroidActionTargetSegment,
    action_target_mapping_statistics,
    create_droid_action_target_contract,
    validate_action_targets_against_joint_episodes,
    validate_droid_action_target_contract,
    validate_droid_action_target_shard,
    write_droid_action_target_contract,
    write_droid_action_target_index,
    write_droid_action_target_shard,
)
from .droid_manifest import write_json_report
from .droid_rlds import (
    DroidRLDSActionEpisode,
    DroidShardWork,
    iter_manifest_droid_action_episodes,
    plan_droid_rank_assignments,
)
from .joint_cache import (
    JOINT_CACHE_SUMMARY_SCHEMA,
    validate_joint_cache_contract,
    validate_joint_episode_shard,
)


@dataclass(frozen=True)
class DroidActionTargetExportConfig:
    source_manifest: str | Path
    data_dir: str | Path
    joint_cache_dir: str | Path
    output_dir: str | Path
    rank: int = 0
    world_size: int = 1
    resume: bool = True

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("Action-target world size must be positive.")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("Action-target rank is outside its world size.")


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}.") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            rows.append(value)
    return tuple(rows)


def _record_ranges(record: EpisodeRecord) -> tuple[tuple[int, int], ...]:
    raw = record.metadata.get("keep_ranges", ())
    ranges = tuple((int(value[0]), int(value[1])) for value in raw)
    if not ranges:
        ranges = ((0, record.num_steps),)
    previous_stop = 0
    for start, stop in ranges:
        if start < previous_stop or stop <= start or stop > record.num_steps:
            raise ValueError(f"Invalid keep ranges for `{record.key}`.")
        previous_stop = stop
    return ranges


def _expected_segments(
    manifest: EpisodeManifest,
) -> dict[str, dict[str, Any]]:
    expected = {}
    for record in manifest:
        if record.split is None:
            raise ValueError(f"Source record `{record.key}` has no split.")
        for range_index, (start, stop) in enumerate(_record_ranges(record)):
            episode_id = f"{record.episode_id}@{start}:{stop}"
            if episode_id in expected:
                raise ValueError(f"Duplicate source segment `{episode_id}`.")
            expected[episode_id] = {
                "episode_id": episode_id,
                "parent_episode_id": record.episode_id,
                "manifest_key": record.key,
                "range_index": range_index,
                "range_start": start,
                "range_stop": stop,
                "source_steps": stop - start,
                "split": record.split,
                "source_shard": str(record.metadata["rlds_shard_name"]),
                "record_index": int(record.metadata["rlds_record_index"]),
            }
    return expected


def _load_context(
    source_manifest_path: Path,
    joint_cache_dir: Path,
) -> tuple[
    EpisodeManifest,
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    source_manifest = EpisodeManifest.read_jsonl(source_manifest_path)
    joint_contract = json.loads(
        (joint_cache_dir / "contract.json").read_text(encoding="utf-8")
    )
    validate_joint_cache_contract(joint_contract)
    joint_summary = json.loads(
        (joint_cache_dir / "summary.json").read_text(encoding="utf-8")
    )
    if (
        joint_summary.get("schema") != JOINT_CACHE_SUMMARY_SCHEMA
        or joint_summary.get("contract_hash") != joint_contract["contract_hash"]
    ):
        raise RuntimeError("Joint cache summary differs from its contract.")
    if (
        source_manifest.fingerprint()
        != joint_contract["source_manifest_fingerprint"]
        or file_sha256(source_manifest_path)
        != joint_contract["source_manifest_sha256"]
    ):
        raise RuntimeError("Source manifest differs from the joint cache contract.")
    expected = _expected_segments(source_manifest)
    joint_rows = {
        str(row["episode_id"]): row
        for row in _read_jsonl(joint_cache_dir / "episodes.jsonl")
    }
    if len(joint_rows) != int(joint_summary["episodes"]):
        raise RuntimeError("Joint cache episode index count changed.")
    if set(expected) != set(joint_rows):
        raise RuntimeError("Source keep ranges differ from joint cache segments.")
    for episode_id, row in expected.items():
        locator = joint_rows[episode_id]
        if (
            int(locator["source_steps"]) != row["source_steps"]
            or str(locator["split"]) != row["split"]
        ):
            raise RuntimeError(
                f"Source/joint segment metadata differs for `{episode_id}`."
            )
    return source_manifest, joint_contract, joint_summary, expected, joint_rows


def _create_contract(
    *,
    source_manifest_path: Path,
    joint_cache_dir: Path,
    joint_contract: Mapping[str, Any],
    source_manifest: EpisodeManifest,
) -> dict[str, Any]:
    here = Path(__file__)
    contract = create_droid_action_target_contract(
        joint_cache_contract_hash=str(joint_contract["contract_hash"]),
        joint_cache_summary_sha256=file_sha256(joint_cache_dir / "summary.json"),
        source_manifest_fingerprint=source_manifest.fingerprint(),
        source_manifest_sha256=file_sha256(source_manifest_path),
        dataset_revision=str(joint_contract["dataset_revision"]),
        implementation_sha256={
            "action_target_export": file_sha256(here),
            "action_targets": file_sha256(here.with_name("action_targets.py")),
            "droid_rlds": file_sha256(here.with_name("droid_rlds.py")),
        },
    )
    validate_droid_action_target_contract(contract)
    return contract


def _work_segment_ids(work: DroidShardWork) -> tuple[str, ...]:
    return tuple(
        f"{record.episode_id}@{start}:{stop}"
        for record in work.records
        for start, stop in _record_ranges(record)
    )


def _work_paths(
    *,
    output_dir: Path,
    work: DroidShardWork,
    joint_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[Path, str, str]:
    segment_ids = _work_segment_ids(work)
    joint_shards = {
        str(joint_rows[episode_id]["episode_shard"])
        for episode_id in segment_ids
    }
    joint_hashes = {
        str(joint_rows[episode_id]["episode_shard_sha256"])
        for episode_id in segment_ids
    }
    if len(joint_shards) != 1 or len(joint_hashes) != 1:
        raise RuntimeError(
            f"Source shard `{work.shard_name}` does not map to one joint shard."
        )
    joint_relative = next(iter(joint_shards))
    relative = str(Path("shards") / Path(joint_relative).name)
    return output_dir / relative, joint_relative, next(iter(joint_hashes))


def _segments_from_episode(
    episode: DroidRLDSActionEpisode,
) -> tuple[DroidActionTargetSegment, ...]:
    rows = []
    for range_index, start, stop in episode.eligible_ranges():
        rows.append(
            DroidActionTargetSegment(
                episode_id=f"{episode.episode_id}@{start}:{stop}",
                parent_episode_id=episode.episode_id,
                manifest_key=episode.manifest_key,
                range_index=range_index,
                range_start=start,
                range_stop=stop,
                split=episode.split,
                source_shard=episode.source_shard,
                record_index=episode.record_index,
                flat_action=episode.action[start:stop].clone().contiguous(),
                action_components={
                    name: value[start:stop].clone().contiguous()
                    for name, value in episode.action_components.items()
                },
                action_valid=episode.action_valid[start:stop].clone().contiguous(),
            )
        )
    return tuple(rows)


def _aggregate_mapping(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    combined = {
        "rows": 0,
        "values": 0,
        "exact_values": 0,
        "absolute_error_sum": 0.0,
        "squared_error_sum": 0.0,
        "max_abs_error": 0.0,
    }
    for row in rows:
        mapping = row["flat_action_mapping"]
        for name in ("rows", "values", "exact_values"):
            combined[name] += int(mapping[name])
        for name in ("absolute_error_sum", "squared_error_sum"):
            combined[name] += float(mapping[name])
        combined["max_abs_error"] = max(
            combined["max_abs_error"],
            float(mapping["max_abs_error"]),
        )
    values = combined["values"]
    if values <= 0:
        raise RuntimeError("Action-target mapping aggregate is empty.")
    combined["exact_fraction"] = combined["exact_values"] / values
    combined["mean_abs_error"] = combined["absolute_error_sum"] / values
    combined["rmse"] = (combined["squared_error_sum"] / values) ** 0.5
    combined["candidate"] = (
        "concat(action_dict.cartesian_position,"
        "action_dict.gripper_position)"
    )
    return combined


def _validate_work_output(
    *,
    path: Path,
    relative: str,
    work: DroidShardWork,
    joint_cache_dir: Path,
    joint_relative: str,
    joint_sha256: str,
    contract_hash: str,
) -> tuple[dict[str, Any], tuple[DroidActionTargetSegment, ...]]:
    segments, metadata = validate_droid_action_target_shard(
        path,
        contract_hash=contract_hash,
    )
    expected_ids = set(_work_segment_ids(work))
    if {segment.episode_id for segment in segments} != expected_ids:
        raise RuntimeError(
            f"Action-target output differs from `{work.shard_name}`."
        )
    if (
        metadata.get("source_shard") != work.shard_name
        or int(metadata.get("source_shard_bytes", -1)) != work.source_bytes
        or metadata.get("joint_episode_shard") != joint_relative
        or metadata.get("joint_episode_shard_sha256") != joint_sha256
    ):
        raise RuntimeError(
            f"Action-target metadata differs for `{work.shard_name}`."
        )
    joint_episodes = validate_joint_episode_shard(
        joint_cache_dir / joint_relative,
        contract_hash=str(metadata["joint_cache_contract_hash"]),
        expected_sha256=joint_sha256,
    )
    selected_joint = tuple(
        episode for episode in joint_episodes if episode.episode_id in expected_ids
    )
    alignment = validate_action_targets_against_joint_episodes(
        segments,
        selected_joint,
    )
    mapping = action_target_mapping_statistics(segments)
    row = {
        "source_shard": work.shard_name,
        "path": relative,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "segments": len(segments),
        "source_steps": sum(segment.source_steps for segment in segments),
        "joint_episode_shard": joint_relative,
        "joint_episode_shard_sha256": joint_sha256,
        "joint_alignment": alignment,
        "flat_action_mapping": mapping,
    }
    return row, segments


def export_droid_action_targets(
    config: DroidActionTargetExportConfig,
) -> dict[str, Any]:
    source_manifest_path = Path(config.source_manifest)
    joint_cache_dir = Path(config.joint_cache_dir)
    output_dir = Path(config.output_dir)
    (
        source_manifest,
        joint_contract,
        _,
        _,
        joint_rows,
    ) = _load_context(source_manifest_path, joint_cache_dir)
    contract = _create_contract(
        source_manifest_path=source_manifest_path,
        joint_cache_dir=joint_cache_dir,
        joint_contract=joint_contract,
        source_manifest=source_manifest,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_droid_action_target_contract(output_dir, contract)
    assignment = plan_droid_rank_assignments(
        source_manifest,
        config.world_size,
    )[config.rank]
    work_by_name = {work.shard_name: work for work in assignment.shards}
    completed_keys: set[str] = set()
    file_rows: list[dict[str, Any]] = []
    pending: dict[str, tuple[DroidShardWork, Path, str, str, str]] = {}
    for work in assignment.shards:
        path, joint_relative, joint_sha256 = _work_paths(
            output_dir=output_dir,
            work=work,
            joint_rows=joint_rows,
        )
        relative = str(path.relative_to(output_dir))
        if path.is_file() and config.resume:
            row, _ = _validate_work_output(
                path=path,
                relative=relative,
                work=work,
                joint_cache_dir=joint_cache_dir,
                joint_relative=joint_relative,
                joint_sha256=joint_sha256,
                contract_hash=str(contract["contract_hash"]),
            )
            row["status"] = "reused"
            file_rows.append(row)
            completed_keys.update(record.key for record in work.records)
        elif path.exists():
            raise FileExistsError(f"Action-target output exists: {path}.")
        else:
            pending[work.shard_name] = (
                work,
                path,
                relative,
                joint_relative,
                joint_sha256,
            )

    started = time.monotonic()
    iterator = iter_manifest_droid_action_episodes(
        config.data_dir,
        source_manifest,
        rank=config.rank,
        world_size=config.world_size,
        completed_episode_keys=completed_keys,
    )
    current_shard: str | None = None
    current_segments: list[DroidActionTargetSegment] = []

    def flush() -> None:
        nonlocal current_shard, current_segments
        if current_shard is None:
            return
        work, path, relative, joint_relative, joint_sha256 = pending[
            current_shard
        ]
        expected_ids = set(_work_segment_ids(work))
        if {segment.episode_id for segment in current_segments} != expected_ids:
            raise RuntimeError(
                f"Extractor did not cover source shard `{current_shard}`."
            )
        joint_episodes = validate_joint_episode_shard(
            joint_cache_dir / joint_relative,
            contract_hash=str(joint_contract["contract_hash"]),
            expected_sha256=joint_sha256,
        )
        selected_joint = tuple(
            episode
            for episode in joint_episodes
            if episode.episode_id in expected_ids
        )
        alignment = validate_action_targets_against_joint_episodes(
            current_segments,
            selected_joint,
        )
        mapping = action_target_mapping_statistics(current_segments)
        write_droid_action_target_shard(
            path,
            contract_hash=str(contract["contract_hash"]),
            segments=current_segments,
            metadata={
                "source_shard": work.shard_name,
                "source_shard_bytes": work.source_bytes,
                "source_checksums": sorted(
                    {str(record.source_checksum) for record in work.records}
                ),
                "joint_cache_contract_hash": joint_contract["contract_hash"],
                "joint_episode_shard": joint_relative,
                "joint_episode_shard_sha256": joint_sha256,
                "joint_alignment": alignment,
                "flat_action_mapping": mapping,
            },
        )
        row, _ = _validate_work_output(
            path=path,
            relative=relative,
            work=work,
            joint_cache_dir=joint_cache_dir,
            joint_relative=joint_relative,
            joint_sha256=joint_sha256,
            contract_hash=str(contract["contract_hash"]),
        )
        row["status"] = "exported"
        file_rows.append(row)
        print(
            f"Exported {current_shard}: segments={row['segments']} "
            f"steps={row['source_steps']} sha256={row['sha256'][:12]}",
            flush=True,
        )
        current_shard = None
        current_segments = []

    for episode in iterator:
        if episode.source_shard != current_shard:
            flush()
            if episode.source_shard not in pending:
                raise RuntimeError(
                    f"Unexpected pending source shard `{episode.source_shard}`."
                )
            current_shard = episode.source_shard
        current_segments.extend(_segments_from_episode(episode))
    flush()
    if set(work_by_name) != {row["source_shard"] for row in file_rows}:
        raise RuntimeError("Action-target rank did not cover its assignment.")
    report = {
        "schema": "codewam.droid-action-target-rank-report.v1",
        "contract_hash": contract["contract_hash"],
        "rank": config.rank,
        "world_size": config.world_size,
        "assignment": {
            "source_shards": len(assignment.shards),
            "source_episodes": assignment.episodes,
            "source_bytes": assignment.source_bytes,
        },
        "elapsed_seconds": time.monotonic() - started,
        "files": sorted(file_rows, key=lambda row: row["source_shard"]),
    }
    write_json_report(
        output_dir / f"rank-{config.rank:05d}.json",
        report,
    )
    return report


def finalize_droid_action_targets(
    *,
    source_manifest_path: str | Path,
    joint_cache_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source_manifest_path = Path(source_manifest_path)
    joint_cache_dir = Path(joint_cache_dir)
    output_dir = Path(output_dir)
    (
        source_manifest,
        joint_contract,
        _,
        expected,
        joint_rows,
    ) = _load_context(source_manifest_path, joint_cache_dir)
    contract = json.loads(
        (output_dir / "contract.json").read_text(encoding="utf-8")
    )
    expected_contract = _create_contract(
        source_manifest_path=source_manifest_path,
        joint_cache_dir=joint_cache_dir,
        joint_contract=joint_contract,
        source_manifest=source_manifest,
    )
    if contract != expected_contract:
        raise RuntimeError("Action-target export contract is stale or mismatched.")

    file_rows = []
    segment_rows = []
    found_ids: set[str] = set()
    for work in plan_droid_rank_assignments(source_manifest, 1)[0].shards:
        path, joint_relative, joint_sha256 = _work_paths(
            output_dir=output_dir,
            work=work,
            joint_rows=joint_rows,
        )
        relative = str(path.relative_to(output_dir))
        if not path.is_file():
            raise FileNotFoundError(f"Missing action-target shard `{path}`.")
        row, segments = _validate_work_output(
            path=path,
            relative=relative,
            work=work,
            joint_cache_dir=joint_cache_dir,
            joint_relative=joint_relative,
            joint_sha256=joint_sha256,
            contract_hash=str(contract["contract_hash"]),
        )
        file_rows.append(row)
        for offset, segment in enumerate(segments):
            if segment.episode_id in found_ids:
                raise RuntimeError(
                    f"Duplicate finalized segment `{segment.episode_id}`."
                )
            found_ids.add(segment.episode_id)
            segment_rows.append(
                {
                    **expected[segment.episode_id],
                    "shard": relative,
                    "offset": offset,
                }
            )
    if found_ids != set(expected):
        raise RuntimeError("Finalized action-target coverage is incomplete.")
    mapping = _aggregate_mapping(file_rows)
    return write_droid_action_target_index(
        output_dir,
        contract_hash=str(contract["contract_hash"]),
        file_rows=file_rows,
        segment_rows=segment_rows,
        mapping_statistics=mapping,
    )
