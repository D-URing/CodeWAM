from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from codewam.codebook_eval.manifest import (
    EpisodeManifest,
    EpisodeRecord,
    SplitName,
)

from .droid_rlds import DROID_CAMERA_KEYS


DROID_METADATA_SCHEMA = "codewam.droid-raw-metadata.v1"
DROID_RLDS_INDEX_SCHEMA = "codewam.droid-rlds-shard-index.v1"
DROID_DATASET_REVISION = "droid-1.0.1"
PATH_MARKER = "/r2d2-data-full/"
DEFAULT_SPLIT_FRACTIONS: dict[SplitName, float] = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}


@dataclass(frozen=True)
class DroidManifestBuildResult:
    manifest: EpisodeManifest
    report: dict[str, Any]


@dataclass(frozen=True)
class DroidBalancedSampleResult:
    manifest: EpisodeManifest
    report: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(mode="rt", encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}.")
            yield payload


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    mode = (path.stat().st_mode & 0o777) if path.exists() else 0o644
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def write_json_report(path: str | Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        Path(path),
        json.dumps(dict(payload), sort_keys=True, indent=2) + "\n",
    )


def canonical_droid_episode_path(value: str) -> str:
    value = str(value)
    if PATH_MARKER in value:
        value = value.split(PATH_MARKER, 1)[1]
    value = value.strip("/")
    for suffix in ("/trajectory.h5", "/recordings/MP4"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    parts = value.split("/")
    if len(parts) < 4 or parts[1] not in {"success", "failure"}:
        raise ValueError(f"Cannot canonicalize DROID episode path `{value}`.")
    return value


def _validate_keep_ranges(value: object, episode_path: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list):
        raise ValueError(f"DROID episode `{episode_path}` has invalid keep ranges.")
    ranges: list[tuple[int, int]] = []
    previous_end = -1
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"Invalid keep range for `{episode_path}`: {item!r}.")
        start, end = (int(part) for part in item)
        if start < 0 or end <= start or start < previous_end:
            raise ValueError(f"Invalid or overlapping keep range for `{episode_path}`: {item}.")
        ranges.append((start, end))
        previous_end = end
    return tuple(ranges)


def _load_keep_ranges(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DROID keep-ranges file must contain a JSON object.")
    rows: dict[str, dict[str, Any]] = {}
    for episode_key, raw_ranges in payload.items():
        try:
            recording_folder, file_path = str(episode_key).split("--", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid DROID episode key `{episode_key}`.") from exc
        recording_path = canonical_droid_episode_path(recording_folder)
        source_path = canonical_droid_episode_path(file_path)
        if recording_path != source_path:
            raise ValueError(
                f"DROID recording/file path mismatch in episode key `{episode_key}`."
            )
        if source_path in rows:
            raise ValueError(f"Duplicate DROID keep-ranges path `{source_path}`.")
        ranges = _validate_keep_ranges(raw_ranges, source_path)
        rows[source_path] = {
            "episode_key": str(episode_key),
            "ranges": ranges,
            "eligible_steps": sum(end - start for start, end in ranges),
        }
    return rows


def _group_jsonl_by_episode_path(
    path: Path,
    expected_schema: str,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    count = 0
    for payload in _iter_jsonl(path):
        if payload.get("schema") != expected_schema:
            raise ValueError(
                f"Unexpected schema `{payload.get('schema')}` in {path}; "
                f"expected `{expected_schema}`."
            )
        if payload.get("dataset_revision") != DROID_DATASET_REVISION:
            raise ValueError(
                f"Unexpected DROID revision `{payload.get('dataset_revision')}` in {path}."
            )
        episode_path = canonical_droid_episode_path(str(payload["episode_path"]))
        grouped[episode_path].append(payload)
        count += 1
    return dict(grouped), count


def _load_language_annotations(path: Path | None) -> dict[str, tuple[str, ...]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DROID language annotations must contain a JSON object.")
    annotations: dict[str, tuple[str, ...]] = {}
    for episode_id, row in payload.items():
        if not isinstance(row, dict):
            raise ValueError(f"Invalid language annotations for `{episode_id}`.")
        values = tuple(
            normalized
            for key in sorted(row)
            if (normalized := " ".join(str(row[key]).split()))
        )
        annotations[str(episode_id)] = values
    return annotations


def _load_gcs_object_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    objects: dict[str, dict[str, Any]] = {}
    current_name: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("gs://") and stripped.endswith(":"):
            current_name = stripped[:-1].rsplit("/", 1)[-1]
            objects.setdefault(current_name, {})
        elif current_name is not None and stripped.startswith("Content-Length:"):
            objects[current_name]["bytes"] = int(stripped.split(":", 1)[1].strip())
        elif current_name is not None and stripped.startswith("Hash (crc32c):"):
            objects[current_name]["crc32c"] = stripped.split(":", 1)[1].strip()
    if not any("crc32c" in value for value in objects.values()):
        raise ValueError(f"No GCS CRC32C entries found in {path}.")
    return objects


def _task_texts(
    metadata: Mapping[str, Any],
    language_annotations: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    episode_id = str(metadata["episode_id"])
    values = list(language_annotations.get(episode_id, ()))
    current_task = " ".join(str(metadata.get("task", "")).split())
    if current_task:
        values.append(current_task)
    return tuple(dict.fromkeys(value for value in values if value))


def build_droid_manifest(
    *,
    metadata_index: str | Path,
    rlds_index: str | Path,
    keep_ranges: str | Path,
    language_annotations: str | Path | None = None,
    gcs_metadata: str | Path | None = None,
    successful_only: bool = True,
    exclude_quality_flags: bool = True,
) -> DroidManifestBuildResult:
    metadata_index = Path(metadata_index)
    rlds_index = Path(rlds_index)
    keep_ranges = Path(keep_ranges)
    language_path = Path(language_annotations) if language_annotations else None
    gcs_path = Path(gcs_metadata) if gcs_metadata else None

    metadata_by_path, metadata_objects = _group_jsonl_by_episode_path(
        metadata_index,
        DROID_METADATA_SCHEMA,
    )
    rlds_by_path, rlds_episodes = _group_jsonl_by_episode_path(
        rlds_index,
        DROID_RLDS_INDEX_SCHEMA,
    )
    keep_by_path = _load_keep_ranges(keep_ranges)
    annotations = _load_language_annotations(language_path)
    gcs_objects = _load_gcs_object_metadata(gcs_path)
    metadata_id_paths: dict[str, set[str]] = defaultdict(set)
    for episode_path, rows in metadata_by_path.items():
        for row in rows:
            metadata_id_paths[str(row["episode_id"])].add(episode_path)
    ambiguous_metadata_ids = {
        episode_id
        for episode_id, paths in metadata_id_paths.items()
        if len(paths) > 1
    }

    exclusion_counts: Counter[str] = Counter()
    records: list[EpisodeRecord] = []
    raw_length_deltas: Counter[int] = Counter()
    missing_shard_checksums = 0

    for episode_path, keep_row in sorted(keep_by_path.items()):
        metadata_rows = metadata_by_path.get(episode_path, ())
        if not metadata_rows:
            exclusion_counts["missing_raw_metadata"] += 1
            continue
        if len(metadata_rows) != 1:
            exclusion_counts["ambiguous_raw_metadata_path"] += 1
            continue
        rlds_rows = rlds_by_path.get(episode_path, ())
        if not rlds_rows:
            exclusion_counts["missing_rlds_index"] += 1
            continue
        if len(rlds_rows) != 1:
            exclusion_counts["ambiguous_rlds_path"] += 1
            continue

        metadata = metadata_rows[0]
        rlds = rlds_rows[0]
        if str(metadata["episode_id"]) in ambiguous_metadata_ids:
            exclusion_counts["ambiguous_raw_metadata_id"] += 1
            continue
        quality_flags = tuple(str(value) for value in metadata.get("quality_flags", ()))
        if exclude_quality_flags and quality_flags:
            exclusion_counts["raw_metadata_quality_flag"] += 1
            continue
        if successful_only and not bool(metadata["success"]):
            exclusion_counts["failure_episode"] += 1
            continue

        num_steps = int(rlds["num_steps"])
        ranges = tuple(keep_row["ranges"])
        if not ranges:
            exclusion_counts["no_eligible_keep_ranges"] += 1
            continue
        if num_steps <= 0 or ranges[-1][1] > num_steps:
            exclusion_counts["keep_range_exceeds_rlds_length"] += 1
            continue
        raw_length_deltas[num_steps - int(metadata["num_steps"])] += 1

        institution_id = str(metadata.get("institution_id", ""))
        building_id = str(metadata.get("building_id", ""))
        scene_id = str(metadata.get("scene_id", ""))
        if not institution_id or not building_id or not scene_id:
            exclusion_counts["missing_scene_identity"] += 1
            continue

        shard_name = str(rlds["shard_name"])
        shard_object = gcs_objects.get(shard_name, {})
        checksum = shard_object.get("crc32c")
        if gcs_objects and checksum is None:
            missing_shard_checksums += 1
        task_texts = _task_texts(metadata, annotations)
        primary_task = task_texts[0] if task_texts else ""
        camera_serials = dict(metadata.get("camera_ids", {}))
        records.append(
            EpisodeRecord(
                dataset=DROID_DATASET_REVISION,
                episode_id=str(metadata["episode_id"]),
                num_steps=num_steps,
                source_uri=str(rlds["file_path"]),
                scene_id=scene_id,
                building_id=building_id,
                institution_id=institution_id,
                task_ids=(primary_task,) if primary_task else (),
                camera_ids=DROID_CAMERA_KEYS,
                source_checksum=f"crc32c:{checksum}" if checksum else None,
                metadata={
                    "episode_path": episode_path,
                    "rlds_episode_key": str(rlds["episode_key"]),
                    "rlds_shard_index": int(rlds["shard_index"]),
                    "rlds_record_index": int(rlds["record_index"]),
                    "rlds_shard_name": shard_name,
                    "rlds_shard_bytes": shard_object.get("bytes"),
                    "recording_folderpath": str(rlds["recording_folderpath"]),
                    "collector_id": str(metadata.get("collector_id", "")),
                    "date": str(metadata.get("date", "")),
                    "timestamp": str(metadata.get("timestamp", "")),
                    "robot_serial": str(metadata.get("robot_serial", "")),
                    "r2d2_version": str(metadata.get("r2d2_version", "")),
                    "success": bool(metadata["success"]),
                    "metadata_success": bool(metadata.get("metadata_success", metadata["success"])),
                    "quality_flags": list(quality_flags),
                    "keep_ranges": [list(value) for value in ranges],
                    "eligible_steps": int(keep_row["eligible_steps"]),
                    "raw_num_steps": int(metadata["num_steps"]),
                    "task_texts": list(task_texts),
                    "camera_serials": camera_serials,
                    "raw_metadata_object": str(metadata["metadata_object"]),
                    "raw_metadata_sha256": str(metadata["metadata_sha256"]),
                },
            )
        )

    if missing_shard_checksums:
        raise ValueError(
            f"Missing GCS CRC32C metadata for {missing_shard_checksums} selected episodes."
        )

    manifest = EpisodeManifest.from_records(records)
    report = {
        "schema": "codewam.droid-manifest-build.v1",
        "dataset_revision": DROID_DATASET_REVISION,
        "filters": {
            "successful_only": successful_only,
            "exclude_quality_flags": exclude_quality_flags,
        },
        "inputs": {
            "metadata_index": {
                "path": str(metadata_index),
                "sha256": _sha256_file(metadata_index),
            },
            "rlds_index": {
                "path": str(rlds_index),
                "sha256": _sha256_file(rlds_index),
            },
            "keep_ranges": {
                "path": str(keep_ranges),
                "sha256": _sha256_file(keep_ranges),
            },
            "language_annotations": (
                {"path": str(language_path), "sha256": _sha256_file(language_path)}
                if language_path
                else None
            ),
            "gcs_metadata": (
                {"path": str(gcs_path), "sha256": _sha256_file(gcs_path)}
                if gcs_path
                else None
            ),
        },
        "source_counts": {
            "raw_metadata_objects": metadata_objects,
            "raw_metadata_paths": len(metadata_by_path),
            "ambiguous_raw_metadata_ids": len(ambiguous_metadata_ids),
            "rlds_episodes": rlds_episodes,
            "rlds_paths": len(rlds_by_path),
            "keep_range_episodes": len(keep_by_path),
            "language_annotated_ids": len(annotations),
        },
        "excluded": dict(sorted(exclusion_counts.items())),
        "raw_vs_rlds_length_mismatches": (
            len(manifest) - raw_length_deltas.get(0, 0)
        ),
        "raw_vs_rlds_length_delta_counts": {
            str(delta): count
            for delta, count in sorted(raw_length_deltas.items())
        },
        "manifest": {
            **manifest.stats(),
            "institutions": len({record.institution_id for record in manifest}),
            "buildings": len(
                {(record.institution_id, record.building_id) for record in manifest}
            ),
            "scenes": len(
                {
                    (record.institution_id, record.building_id, record.scene_id)
                    for record in manifest
                }
            ),
            "eligible_steps": sum(
                int(record.metadata["eligible_steps"]) for record in manifest
            ),
        },
    }
    return DroidManifestBuildResult(manifest=manifest, report=report)


def _stable_score(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def _allocate_split_targets(
    total_size: int,
    fractions: Mapping[SplitName, float],
) -> dict[SplitName, int]:
    if total_size <= 0:
        raise ValueError("Sample size must be positive.")
    if set(fractions) != {"train", "val", "test"}:
        raise ValueError("Split fractions must define train, val, and test.")
    if any(value < 0.0 for value in fractions.values()):
        raise ValueError("Split fractions must be non-negative.")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to one.")

    raw = {split: total_size * float(fractions[split]) for split in fractions}
    targets = {split: int(value) for split, value in raw.items()}
    remainder = total_size - sum(targets.values())
    order = sorted(
        fractions,
        key=lambda split: (-(raw[split] - targets[split]), split),
    )
    for split in order[:remainder]:
        targets[split] += 1
    return targets


def _record_balance_labels(record: EpisodeRecord) -> tuple[str, str]:
    collector = str(record.metadata.get("collector_id") or "_")
    task = record.task_ids[0] if record.task_ids else "_"
    return collector, task


def _balanced_group_targets(
    total: int,
    capacities: Mapping[str, int],
    *,
    salt: str,
) -> dict[str, int]:
    if total < 0:
        raise ValueError("Balanced target total must be non-negative.")
    normalized = {str(key): int(value) for key, value in capacities.items()}
    if any(value < 0 for value in normalized.values()):
        raise ValueError("Balanced target capacities must be non-negative.")
    if total > sum(normalized.values()):
        raise ValueError(
            f"Requested {total} balanced records but only "
            f"{sum(normalized.values())} are available."
        )

    targets = {key: 0 for key in normalized}
    for _ in range(total):
        eligible = [
            key for key, capacity in normalized.items() if targets[key] < capacity
        ]
        selected = min(
            eligible,
            key=lambda key: (
                targets[key],
                _stable_score(f"{salt}|quota|{key}"),
            ),
        )
        targets[selected] += 1
    return targets


def _sample_scene_pool(
    records: Iterable[EpisodeRecord],
    target: int,
    salt: str,
    split: SplitName,
    collector_counts: Counter[str],
    task_counts: Counter[str],
) -> list[EpisodeRecord]:
    candidates = list(records)
    if target > len(candidates):
        raise ValueError(
            f"Requested {target} `{split}` episodes but only {len(candidates)} are available."
        )
    if target == 0:
        return []

    by_scene: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for record in candidates:
        by_scene[record.group_key("scene")].append(record)
    for scene, scene_records in by_scene.items():
        scene_records.sort(
            key=lambda record: _stable_score(f"{salt}|{split}|{scene}|{record.key}")
        )

    scene_order = sorted(
        by_scene,
        key=lambda scene: _stable_score(f"{salt}|{split}|scene|{scene}"),
    )
    selected: list[EpisodeRecord] = []
    round_index = 0
    while len(selected) < target:
        progressed = False
        offset = round_index % len(scene_order)
        ordered_scenes = scene_order[offset:] + scene_order[:offset]
        for scene in ordered_scenes:
            remaining = by_scene[scene]
            if not remaining:
                continue
            best_index = min(
                range(len(remaining)),
                key=lambda index: (
                    collector_counts[_record_balance_labels(remaining[index])[0]],
                    task_counts[_record_balance_labels(remaining[index])[1]],
                    _stable_score(
                        f"{salt}|{split}|episode|{remaining[index].key}"
                    ),
                ),
            )
            record = remaining.pop(best_index)
            collector, task = _record_balance_labels(record)
            collector_counts[collector] += 1
            task_counts[task] += 1
            selected.append(record)
            progressed = True
            if len(selected) == target:
                break
        if not progressed:
            raise RuntimeError(f"Could not fill `{split}` sample target {target}.")
        round_index += 1
    return selected


def _sample_one_split(
    records: Iterable[EpisodeRecord],
    target: int,
    salt: str,
    split: SplitName,
) -> list[EpisodeRecord]:
    candidates = list(records)
    if target > len(candidates):
        raise ValueError(
            f"Requested {target} `{split}` episodes but only "
            f"{len(candidates)} are available."
        )

    by_institution: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for record in candidates:
        by_institution[str(record.institution_id or "_")].append(record)
    institution_targets = _balanced_group_targets(
        target,
        {key: len(value) for key, value in by_institution.items()},
        salt=f"{salt}|{split}|institution",
    )
    institution_order = sorted(
        by_institution,
        key=lambda institution: _stable_score(
            f"{salt}|{split}|institution|{institution}"
        ),
    )
    collector_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    selected: list[EpisodeRecord] = []
    for institution in institution_order:
        selected.extend(
            _sample_scene_pool(
                by_institution[institution],
                institution_targets[institution],
                f"{salt}|institution|{institution}",
                split,
                collector_counts,
                task_counts,
            )
        )
    return selected


def balanced_scene_sample(
    manifest: EpisodeManifest,
    total_size: int,
    *,
    salt: str = "codewam-droid-balanced-v1",
    split_fractions: Mapping[SplitName, float] = DEFAULT_SPLIT_FRACTIONS,
) -> EpisodeManifest:
    if not salt:
        raise ValueError("Sampling salt must not be empty.")
    manifest.assert_group_isolation("scene")
    targets = _allocate_split_targets(total_size, split_fractions)
    selected: list[EpisodeRecord] = []
    for split in ("train", "val", "test"):
        split_records = [record for record in manifest if record.split == split]
        selected.extend(
            _sample_one_split(
                split_records,
                targets[split],
                salt,
                split,
            )
        )
    sampled = EpisodeManifest.from_records(sorted(selected, key=lambda record: record.key))
    sampled.assert_group_isolation("scene")
    if len(sampled) != total_size:
        raise RuntimeError(f"Expected {total_size} sampled episodes, got {len(sampled)}.")
    return sampled


def shard_aware_balanced_sample(
    manifest: EpisodeManifest,
    total_size: int,
    *,
    salt: str = "codewam-droid-balanced-v1",
    split_fractions: Mapping[SplitName, float] = DEFAULT_SPLIT_FRACTIONS,
    candidate_multiplier: float = 1.0,
) -> DroidBalancedSampleResult:
    if candidate_multiplier < 1.0:
        raise ValueError("Candidate multiplier must be at least one.")
    manifest.assert_group_isolation("scene")
    targets = _allocate_split_targets(total_size, split_fractions)
    availability = Counter(record.split for record in manifest)
    for split, target in targets.items():
        if availability[split] < target:
            raise ValueError(
                f"Requested {target} `{split}` episodes but only "
                f"{availability[split]} are available."
            )
    candidate_targets = {
        split: min(
            availability[split],
            max(target, int(math.ceil(target * candidate_multiplier))),
        )
        for split, target in targets.items()
    }
    institution_availability: dict[SplitName, Counter[str]] = {
        split: Counter(
            str(record.institution_id or "_")
            for record in manifest
            if record.split == split
        )
        for split in ("train", "val", "test")
    }
    institution_candidate_targets: dict[SplitName, dict[str, int]] = {
        split: _balanced_group_targets(
            candidate_targets[split],
            institution_availability[split],
            salt=f"{salt}|{split}|candidate-institution",
        )
        for split in ("train", "val", "test")
    }
    cell_targets = {
        (split, institution): target
        for split, institution_targets in institution_candidate_targets.items()
        for institution, target in institution_targets.items()
        if target > 0
    }

    by_shard: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for record in manifest:
        shard_name = str(record.metadata.get("rlds_shard_name") or "")
        if not shard_name:
            raise ValueError(f"Episode `{record.key}` has no RLDS shard identity.")
        by_shard[shard_name].append(record)

    shard_cell_counts = {
        shard: Counter(
            (record.split, str(record.institution_id or "_"))
            for record in records
        )
        for shard, records in by_shard.items()
    }
    selected_shards: set[str] = set()
    candidate_cell_counts: Counter[tuple[SplitName, str]] = Counter()
    covered_scenes: set[str] = set()
    while any(
        candidate_cell_counts[cell] < target
        for cell, target in cell_targets.items()
    ):
        deficits = {
            cell: max(0, target - candidate_cell_counts[cell])
            for cell, target in cell_targets.items()
        }
        choices: list[tuple[tuple[Any, ...], str]] = []
        for shard, records in by_shard.items():
            if shard in selected_shards:
                continue
            counts = shard_cell_counts[shard]
            coverage_gain = sum(
                min(counts[cell], deficits[cell]) / cell_targets[cell]
                for cell in cell_targets
            )
            if coverage_gain <= 0.0:
                continue
            new_scenes = len(
                {
                    record.group_key("scene")
                    for record in records
                    if deficits.get(
                        (record.split, str(record.institution_id or "_")), 0
                    )
                    > 0
                    if record.group_key("scene") not in covered_scenes
                }
            )
            usable = sum(
                min(counts[cell], deficits[cell]) for cell in cell_targets
            )
            choices.append(
                (
                    (
                        -coverage_gain,
                        -new_scenes,
                        -usable,
                        _stable_score(f"{salt}|candidate-shard|{shard}"),
                    ),
                    shard,
                )
            )
        if not choices:
            raise RuntimeError(
                "Could not satisfy institution-aware shard candidate targets; "
                f"remaining deficits={dict(deficits)}."
            )
        _, selected = min(choices)
        selected_shards.add(selected)
        candidate_cell_counts.update(shard_cell_counts[selected])
        covered_scenes.update(
            record.group_key("scene") for record in by_shard[selected]
        )

    candidate_counts: Counter[SplitName] = Counter()
    institution_candidate_counts: dict[SplitName, dict[str, int]] = {
        split: {} for split in ("train", "val", "test")
    }
    for (split, institution), count in candidate_cell_counts.items():
        candidate_counts[split] += count
        institution_candidate_counts[split][institution] = count

    candidates = EpisodeManifest.from_records(
        record
        for record in manifest
        if str(record.metadata["rlds_shard_name"]) in selected_shards
    )
    sampled = balanced_scene_sample(
        candidates,
        total_size,
        salt=salt,
        split_fractions=split_fractions,
    )

    shard_bytes: dict[str, int | None] = {}
    for shard in selected_shards:
        values = {
            record.metadata.get("rlds_shard_bytes")
            for record in by_shard[shard]
        }
        if len(values) != 1:
            raise ValueError(f"Inconsistent source byte metadata for shard `{shard}`.")
        value = next(iter(values))
        shard_bytes[shard] = int(value) if value is not None else None
    selected_source_bytes = (
        sum(value for value in shard_bytes.values() if value is not None)
        if all(value is not None for value in shard_bytes.values())
        else None
    )
    report = {
        "schema": "codewam.droid-shard-aware-sample.v1",
        "sample_salt": salt,
        "candidate_multiplier": candidate_multiplier,
        "split_targets": targets,
        "candidate_targets": candidate_targets,
        "candidate_counts": {
            split: candidate_counts[split] for split in ("train", "val", "test")
        },
        "institution_candidate_targets": institution_candidate_targets,
        "institution_candidate_counts": {
            split: dict(sorted(institution_candidate_counts[split].items()))
            for split in ("train", "val", "test")
        },
        "source_shards_available": len(by_shard),
        "source_shards_selected": len(selected_shards),
        "selected_source_bytes": selected_source_bytes,
        "selected_shards": sorted(selected_shards),
        "sample": manifest_distribution(sampled),
    }
    return DroidBalancedSampleResult(manifest=sampled, report=report)


def manifest_distribution(manifest: EpisodeManifest) -> dict[str, Any]:
    split_counts = Counter(record.split or "unassigned" for record in manifest)
    scene_counts = Counter(record.group_key("scene") for record in manifest)
    institution_counts = Counter(
        str(record.institution_id or "_") for record in manifest
    )
    collector_counts = Counter(
        str(record.metadata.get("collector_id") or "_") for record in manifest
    )
    task_counts = Counter(record.task_ids[0] if record.task_ids else "_" for record in manifest)
    per_split: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        records = [record for record in manifest if record.split == split]
        per_split[split] = {
            "episodes": len(records),
            "scenes": len({record.group_key("scene") for record in records}),
            "buildings": len({record.group_key("building") for record in records}),
            "institutions": len(
                {record.group_key("institution") for record in records}
            ),
            "collectors": len(
                {str(record.metadata.get("collector_id") or "_") for record in records}
            ),
            "tasks": len(
                {record.task_ids[0] if record.task_ids else "_" for record in records}
            ),
        }
    return {
        **manifest.stats(),
        "split_counts": dict(sorted(split_counts.items())),
        "scenes": len(scene_counts),
        "scene_episode_min": min(scene_counts.values(), default=0),
        "scene_episode_max": max(scene_counts.values(), default=0),
        "institution_episode_counts": dict(sorted(institution_counts.items())),
        "collectors": len(collector_counts),
        "collector_episode_counts": dict(sorted(collector_counts.items())),
        "collector_episode_min": min(collector_counts.values(), default=0),
        "collector_episode_max": max(collector_counts.values(), default=0),
        "tasks": len(task_counts),
        "task_episode_min": min(task_counts.values(), default=0),
        "task_episode_max": max(task_counts.values(), default=0),
        "per_split": per_split,
        "temporal": droid_temporal_distribution(manifest),
    }


def _integer_quantiles(values: Sequence[int]) -> dict[str, int]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        name: ordered[int((len(ordered) - 1) * fraction)]
        for name, fraction in (
            ("min", 0.0),
            ("p25", 0.25),
            ("p50", 0.5),
            ("p75", 0.75),
            ("p90", 0.9),
            ("p95", 0.95),
            ("p99", 0.99),
            ("max", 1.0),
        )
    }


def droid_temporal_distribution(manifest: EpisodeManifest) -> dict[str, Any]:
    family_offsets = {"Q2": 4, "Q3": 6, "Q5": 10}
    range_counts: Counter[int] = Counter()
    segment_lengths: list[int] = []
    source_steps = 0
    eligible_steps = 0
    implicit_full_episode_ranges = 0
    family_segments: Counter[str] = Counter()
    family_episodes: Counter[str] = Counter()
    descriptor_ticks: Counter[str] = Counter()

    for record in manifest:
        source_steps += record.num_steps
        raw_ranges = record.metadata.get("keep_ranges")
        if raw_ranges is None:
            ranges = ((0, record.num_steps),)
            implicit_full_episode_ranges += 1
        else:
            ranges = _validate_keep_ranges(raw_ranges, record.key)
            if any(stop > record.num_steps for _, stop in ranges):
                raise ValueError(
                    f"Keep range exceeds source length for `{record.key}`."
                )
        observed_eligible_steps = sum(stop - start for start, stop in ranges)
        expected_eligible_steps = record.metadata.get("eligible_steps")
        if (
            expected_eligible_steps is not None
            and observed_eligible_steps != int(expected_eligible_steps)
        ):
            raise ValueError(
                f"Eligible-step metadata mismatch for `{record.key}`."
            )

        eligible_steps += observed_eligible_steps
        range_counts[len(ranges)] += 1
        episode_families: set[str] = set()
        for start, stop in ranges:
            segment_length = stop - start
            segment_lengths.append(segment_length)
            latent_ticks = 1 + (segment_length - 1) // 4
            for family, maximum_offset in family_offsets.items():
                available = max(0, latent_ticks - maximum_offset)
                descriptor_ticks[family] += available
                if available:
                    family_segments[family] += 1
                    episode_families.add(family)
        family_episodes.update(episode_families)

    return {
        "episodes": len(manifest),
        "source_steps": source_steps,
        "eligible_steps": eligible_steps,
        "eligible_step_fraction": (
            eligible_steps / source_steps if source_steps else 0.0
        ),
        "removed_idle_steps": source_steps - eligible_steps,
        "segments": len(segment_lengths),
        "ranges_per_episode": {
            str(count): episodes for count, episodes in sorted(range_counts.items())
        },
        "segment_length_steps": _integer_quantiles(segment_lengths),
        "implicit_full_episode_ranges": implicit_full_episode_ranges,
        "families": {
            family: {
                "maximum_latent_offset": maximum_offset,
                "minimum_source_steps": 4 * maximum_offset + 1,
                "eligible_segments": family_segments[family],
                "eligible_episodes": family_episodes[family],
                "descriptor_ticks": descriptor_ticks[family],
            }
            for family, maximum_offset in family_offsets.items()
        },
    }
