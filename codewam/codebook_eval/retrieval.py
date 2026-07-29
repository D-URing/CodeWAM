from __future__ import annotations

import hashlib
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from codewam.data.droid_manifest import write_json_report
from codewam.data.droid_rlds import read_manifest_droid_rlds_frames

from .manifest import EpisodeManifest, EpisodeRecord
from .shards import (
    POOLED_SHARD_SCHEMA,
    file_sha256,
    load_torch_payload,
)


HELDOUT_EVALUATION_SCHEMA = "codewam.heldout-rq-evaluation.v1"
RETRIEVAL_CONTRACT_SCHEMA = "codewam.codebook-retrieval-contract.v1"
RETRIEVAL_REPORT_SCHEMA = "codewam.codebook-retrieval-report.v1"


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in `{path}`.")
    return payload


def _resolve_pooled_shard(path_value: str, manifest_path: Path) -> Path:
    path = Path(path_value)
    if path.is_file():
        return path.resolve()
    fallback = manifest_path.parent / "pooled" / path.name
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(
        f"Missing pooled shard `{path}` and fallback `{fallback}`."
    )


def _load_evaluation_rows(
    report_paths: Sequence[Path],
    *,
    manifest_fingerprint: str,
    splits: tuple[str, ...],
    levels: tuple[int, ...],
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    list[dict[str, Any]],
]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    inputs = []
    all_families: set[str] = set()
    datasets: set[str] = set()
    for path in report_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing held-out report `{path}`.")
        payload = _read_json(path)
        if payload.get("schema") != HELDOUT_EVALUATION_SCHEMA:
            raise ValueError(f"Unsupported held-out report schema in `{path}`.")
        if payload.get("manifest_fingerprint") != manifest_fingerprint:
            raise RuntimeError(
                f"Held-out report `{path}` does not match the pooled manifest."
            )
        datasets.add(str(payload.get("dataset") or ""))
        report_families = {
            str(row.get("family") or "")
            for row in payload.get("rows", ())
        }
        if "" in report_families:
            raise ValueError(f"Held-out report `{path}` has an empty family.")
        all_families.update(report_families)
        inputs.append(
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "families": sorted(report_families),
            }
        )
        for row in payload.get("rows", ()):
            family = str(row["family"])
            split = str(row["split"])
            if split not in splits:
                continue
            stride = int(row["stride"])
            if family != f"Q{stride}":
                raise ValueError(
                    f"Held-out row family `{family}` disagrees with stride "
                    f"{stride}."
                )
            available_levels = {
                int(value["level"]): value
                for value in row.get("representatives", ())
            }
            missing_levels = sorted(set(levels) - set(available_levels))
            if missing_levels:
                raise ValueError(
                    f"Held-out row {family}/{split} lacks representative "
                    f"levels {missing_levels}."
                )
            key = (family, split)
            if key in selected:
                raise ValueError(
                    f"Duplicate held-out row for {family}/{split}."
                )
            selected[key] = row

    if len(datasets) != 1 or "" in datasets:
        raise ValueError(
            f"Held-out reports must share one nonempty dataset: {datasets}."
        )
    missing_rows = [
        (family, split)
        for family in sorted(all_families)
        for split in splits
        if (family, split) not in selected
    ]
    if missing_rows:
        raise ValueError(f"Missing held-out family/split rows: {missing_rows}.")
    return selected, inputs


def _load_needed_pooled_metadata(
    pooled_manifest_path: Path,
    records: dict[str, EpisodeRecord],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_shard: dict[Path, set[str]] = defaultdict(set)
    expected_checksums: dict[Path, set[str]] = defaultdict(set)
    for episode_id, record in records.items():
        shard = _resolve_pooled_shard(
            str(record.metadata.get("pooled_shard") or ""),
            pooled_manifest_path,
        )
        by_shard[shard].add(episode_id)
        if record.source_checksum:
            expected_checksums[shard].add(record.source_checksum)

    metadata_by_episode: dict[str, dict[str, Any]] = {}
    shard_rows = []
    for path in sorted(by_shard):
        actual_sha256 = file_sha256(path)
        expected = expected_checksums[path]
        if expected and expected != {f"sha256:{actual_sha256}"}:
            raise RuntimeError(
                f"Pooled shard checksum mismatch for `{path}`: {expected}."
            )
        payload = load_torch_payload(path, map_location="cpu")
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != POOLED_SHARD_SCHEMA
        ):
            raise ValueError(f"Unsupported pooled shard schema in `{path}`.")
        wanted = by_shard[path]
        for episode in payload.get("episodes", ()):
            episode_id = str(episode.get("episode_id") or "")
            if episode_id not in wanted:
                continue
            if episode_id in metadata_by_episode:
                raise ValueError(
                    f"Duplicate pooled episode payload `{episode_id}`."
                )
            metadata_by_episode[episode_id] = {
                "camera_ids": tuple(episode.get("camera_ids", ())),
                "metadata": dict(episode.get("metadata", {})),
            }
        missing = sorted(wanted - set(metadata_by_episode))
        if missing:
            raise RuntimeError(
                f"Pooled shard `{path}` is missing episodes {missing[:8]}."
            )
        shard_rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": actual_sha256,
                "selected_episodes": len(wanted),
            }
        )
    return metadata_by_episode, shard_rows


def _as_int_list(value: Any, *, name: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"`{name}` must be a sequence.")
    return [int(item) for item in value]


def _select_representative_samples(
    samples: Sequence[dict[str, Any]],
    *,
    pooled_by_id: dict[str, EpisodeRecord],
    limit: int,
    diversity_by: str,
) -> list[tuple[int, dict[str, Any]]]:
    chosen: list[tuple[int, dict[str, Any]]] = []
    seen_groups: set[str] = set()
    for source_rank, sample in enumerate(samples, start=1):
        episode_id = str(sample["episode_id"])
        try:
            pooled_record = pooled_by_id[episode_id]
        except KeyError as exc:
            raise KeyError(
                "Representative episode is absent from pooled manifest: "
                f"`{episode_id}`."
            ) from exc
        if diversity_by == "parent":
            group = str(
                pooled_record.metadata.get(
                    "parent_manifest_key",
                    pooled_record.key,
                )
            )
        elif diversity_by == "scene":
            group = pooled_record.group_key("scene")
        else:
            group = f"sample:{source_rank}"
        if group in seen_groups:
            continue
        seen_groups.add(group)
        chosen.append((source_rank, sample))
        if len(chosen) == limit:
            break
    return chosen


def _resolve_clips(
    *,
    selected_rows: dict[tuple[str, str], dict[str, Any]],
    levels: tuple[int, ...],
    representatives_per_code: int,
    pooled_manifest_path: Path,
    pooled_manifest: EpisodeManifest,
    source_manifest: EpisodeManifest,
    camera: str,
    diversity_by: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, set[int]],
    list[dict[str, Any]],
]:
    pooled_by_id = {record.episode_id: record for record in pooled_manifest}
    source_by_key = {record.key: record for record in source_manifest}

    selected_samples: dict[
        tuple[str, str, int, int],
        list[tuple[int, dict[str, Any]]],
    ] = {}
    sample_episode_ids: set[str] = set()
    for (family, split), row in selected_rows.items():
        representatives = {
            int(value["level"]): value
            for value in row["representatives"]
        }
        for level in levels:
            for code in representatives[level]["codes"]:
                chosen = _select_representative_samples(
                    code.get("samples", ()),
                    pooled_by_id=pooled_by_id,
                    limit=representatives_per_code,
                    diversity_by=diversity_by,
                )
                sample_episode_ids.update(
                    str(sample["episode_id"]) for _, sample in chosen
                )
                selected_samples[
                    (family, split, level, int(code["code"]))
                ] = chosen
    selected_pooled = {
        episode_id: pooled_by_id[episode_id]
        for episode_id in sample_episode_ids
    }
    pooled_payloads, pooled_shards = _load_needed_pooled_metadata(
        pooled_manifest_path,
        selected_pooled,
    )

    clips: list[dict[str, Any]] = []
    frame_requests: dict[str, set[int]] = defaultdict(set)
    for (family, split), row in sorted(selected_rows.items()):
        stride = int(row["stride"])
        k = int(row["k"])
        representatives = {
            int(value["level"]): value
            for value in row["representatives"]
        }
        for level in levels:
            code_rows = representatives[level]["codes"]
            observed_codes = [int(value["code"]) for value in code_rows]
            if observed_codes != list(range(k)):
                raise ValueError(
                    f"{family}/{split}/L{level} representative codes must be "
                    f"ordered 0..{k - 1}, got {observed_codes}."
                )
            for code_row in code_rows:
                code = int(code_row["code"])
                samples = selected_samples[(family, split, level, code)]
                for rank, (source_rank, sample) in enumerate(
                    samples,
                    start=1,
                ):
                    episode_id = str(sample["episode_id"])
                    pooled_record = selected_pooled[episode_id]
                    if pooled_record.split != split:
                        raise RuntimeError(
                            f"Representative `{episode_id}` has pooled split "
                            f"`{pooled_record.split}`, expected `{split}`."
                        )
                    payload = pooled_payloads[episode_id]
                    if camera not in payload["camera_ids"]:
                        raise ValueError(
                            f"Pooled episode `{episode_id}` lacks camera "
                            f"`{camera}`."
                        )
                    metadata = payload["metadata"]
                    parent_key = str(
                        metadata.get("parent_manifest_key") or ""
                    )
                    if (
                        parent_key
                        != pooled_record.metadata.get("parent_manifest_key")
                    ):
                        raise RuntimeError(
                            f"Pooled episode `{episode_id}` has inconsistent "
                            "parent provenance."
                        )
                    try:
                        source_record = source_by_key[parent_key]
                    except KeyError as exc:
                        raise KeyError(
                            f"Pooled episode `{episode_id}` references missing "
                            f"source record `{parent_key}`."
                        ) from exc
                    if camera not in source_record.camera_ids:
                        raise ValueError(
                            f"Source episode `{parent_key}` lacks camera "
                            f"`{camera}`."
                        )
                    if (
                        metadata.get("source_shard")
                        != source_record.metadata.get("rlds_shard_name")
                        or int(metadata.get("record_index", -1))
                        != int(
                            source_record.metadata.get(
                                "rlds_record_index",
                                -2,
                            )
                        )
                    ):
                        raise RuntimeError(
                            f"Pooled/source RLDS position mismatch for "
                            f"`{episode_id}`."
                        )

                    time_index = int(sample["time_index"])
                    latent_indices = _as_int_list(
                        metadata.get("absolute_latent_frame_indices"),
                        name="absolute_latent_frame_indices",
                    )
                    latent_offsets = (-2 * stride, -stride, 0)
                    tick_indices = [
                        time_index + offset for offset in latent_offsets
                    ]
                    if (
                        tick_indices[0] < 0
                        or tick_indices[-1] >= len(latent_indices)
                    ):
                        raise IndexError(
                            f"Representative `{episode_id}` time index "
                            f"{time_index} cannot provide Q{stride} history."
                        )
                    source_frames = [
                        latent_indices[index] for index in tick_indices
                    ]
                    source_range = _as_int_list(
                        metadata.get("source_range"),
                        name="source_range",
                    )
                    if len(source_range) != 2 or any(
                        frame < source_range[0] or frame >= source_range[1]
                        for frame in source_frames
                    ):
                        raise RuntimeError(
                            f"Representative `{episode_id}` escaped its source "
                            f"range {source_range}: {source_frames}."
                        )
                    if any(
                        frame < 0 or frame >= source_record.num_steps
                        for frame in source_frames
                    ):
                        raise RuntimeError(
                            f"Representative `{episode_id}` references source "
                            f"frames outside [0, {source_record.num_steps})."
                        )
                    nominal_fps = float(metadata.get("nominal_fps", 0.0))
                    if nominal_fps <= 0:
                        raise ValueError(
                            f"Pooled episode `{episode_id}` has invalid FPS."
                        )
                    timestamp = float(sample["timestamp"])
                    expected_timestamp = source_frames[-1] / nominal_fps
                    if abs(timestamp - expected_timestamp) > 1e-6:
                        raise RuntimeError(
                            f"Representative `{episode_id}` timestamp "
                            f"{timestamp} disagrees with source frame "
                            f"{source_frames[-1]} at {nominal_fps} FPS."
                        )
                    frame_requests[parent_key].update(source_frames)
                    clips.append(
                        {
                            "clip_id": (
                                f"{family}:{split}:L{level}:C{code}:R{rank}"
                            ),
                            "family": family,
                            "stride": stride,
                            "split": split,
                            "level": level,
                            "code": code,
                            "rank": rank,
                            "source_anchor_rank": source_rank,
                            "distance_mse": float(sample["distance_mse"]),
                            "segment_id": episode_id,
                            "latent_time_index": time_index,
                            "latent_tick_indices": tick_indices,
                            "timestamp": timestamp,
                            "parent_manifest_key": parent_key,
                            "source_shard": str(
                                source_record.metadata["rlds_shard_name"]
                            ),
                            "source_record_index": int(
                                source_record.metadata["rlds_record_index"]
                            ),
                            "source_frame_indices": source_frames,
                            "relative_seconds": [
                                (frame - source_frames[-1]) / nominal_fps
                                for frame in source_frames
                            ],
                            "nominal_fps": nominal_fps,
                            "task_ids": list(pooled_record.task_ids),
                            "scene_id": pooled_record.scene_id,
                            "institution_id": pooled_record.institution_id,
                        }
                    )
    return clips, frame_requests, pooled_shards


def _validate_frame(frame: torch.Tensor, key: tuple[str, int]) -> np.ndarray:
    if (
        frame.ndim != 3
        or int(frame.shape[-1]) != 3
        or frame.dtype != torch.uint8
    ):
        raise ValueError(
            f"RGB frame {key} must be uint8 [H,W,3], got "
            f"{tuple(frame.shape)} {frame.dtype}."
        )
    return frame.detach().cpu().numpy()


def _motion_metrics(frames: Sequence[np.ndarray]) -> dict[str, Any]:
    if len(frames) != 3:
        raise ValueError("A retrieval clip must contain exactly three frames.")
    if len({frame.shape for frame in frames}) != 1:
        raise ValueError("Retrieval clip frames must share one image shape.")

    values = [frame.astype(np.float32) for frame in frames]
    adjacent = [
        float(np.abs(after - before).mean() / 255.0)
        for before, after in zip(values, values[1:])
    ]
    difference = np.abs(values[-1] - values[0]).mean(axis=2)
    total = float(difference.sum())
    centroid = None
    if total > 0:
        height, width = difference.shape
        x_coordinates = np.linspace(0.0, 1.0, width, dtype=np.float32)
        y_coordinates = np.linspace(0.0, 1.0, height, dtype=np.float32)
        centroid = [
            float((difference.sum(axis=0) * x_coordinates).sum() / total),
            float((difference.sum(axis=1) * y_coordinates).sum() / total),
        ]
    return {
        "adjacent_mean_absolute_rgb_difference": adjacent,
        "first_last_mean_absolute_rgb_difference": float(
            difference.mean() / 255.0
        ),
        "first_last_changed_pixel_fraction_at_20": float(
            (difference >= 20.0).mean()
        ),
        "first_last_difference_centroid_xy": centroid,
    }


def _thumbnail(frame: np.ndarray, size: tuple[int, int]) -> Image.Image:
    image = Image.fromarray(frame, mode="RGB")
    return ImageOps.fit(
        image,
        size,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def _difference_thumbnail(
    first: np.ndarray,
    last: np.ndarray,
    *,
    size: tuple[int, int],
    gain: float,
) -> Image.Image:
    difference = np.clip(
        np.abs(last.astype(np.float32) - first.astype(np.float32)) * gain,
        0.0,
        255.0,
    ).astype(np.uint8)
    return _thumbnail(difference, size)


def _atomic_save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        image.save(temporary, format="PNG", compress_level=6)
        mode = (path.stat().st_mode & 0o777) if path.exists() else 0o644
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _render_montage(
    path: Path,
    *,
    family: str,
    split: str,
    level: int,
    k: int,
    representatives_per_code: int,
    clips: Sequence[dict[str, Any]],
    frames: dict[tuple[str, int], torch.Tensor],
    thumbnail_size: tuple[int, int],
    difference_gain: float,
) -> None:
    tile_width, tile_height = thumbnail_size
    label_width = 112
    group_gap = 8
    caption_height = 18
    row_gap = 4
    header_height = 62
    tiles_per_group = 4
    group_width = tile_width * tiles_per_group
    row_height = tile_height + caption_height + row_gap
    width = (
        label_width
        + representatives_per_code * group_width
        + max(0, representatives_per_code - 1) * group_gap
    )
    height = header_height + k * row_height
    canvas = Image.new("RGB", (width, height), "#f7f7f5")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (8, 6),
        f"{family} / {split} / RQ level {level}",
        fill="#111111",
        font=font,
    )
    draw.text(
        (8, 23),
        "Each example: exact RGB at t-2s, t-s, t; final tile is |delta| x gain",
        fill="#555555",
        font=font,
    )
    column_labels = ("t-2s", "t-s", "t", f"|delta| x{difference_gain:g}")
    for rank in range(representatives_per_code):
        group_x = label_width + rank * (group_width + group_gap)
        draw.text(
            (group_x + 3, 39),
            f"nearest {rank + 1}",
            fill="#222222",
            font=font,
        )
        for column, label in enumerate(column_labels):
            draw.text(
                (group_x + column * tile_width + 3, 50),
                label,
                fill="#666666",
                font=font,
            )
        if rank:
            draw.line(
                (group_x - group_gap // 2, 38, group_x - group_gap // 2, height),
                fill="#b8b8b4",
                width=1,
            )

    by_code_rank = {
        (int(clip["code"]), int(clip["rank"])): clip for clip in clips
    }
    for code in range(k):
        y = header_height + code * row_height
        draw.rectangle(
            (0, y, width, y + row_height - 1),
            fill="#ffffff" if code % 2 == 0 else "#f2f2ef",
        )
        draw.text((8, y + 8), f"code {code}", fill="#111111", font=font)
        available = sum(
            (code, rank) in by_code_rank
            for rank in range(1, representatives_per_code + 1)
        )
        draw.text(
            (8, y + 25),
            f"{available} examples",
            fill="#666666",
            font=font,
        )
        for rank in range(1, representatives_per_code + 1):
            group_x = label_width + (rank - 1) * (group_width + group_gap)
            clip = by_code_rank.get((code, rank))
            if clip is None:
                draw.rectangle(
                    (
                        group_x,
                        y,
                        group_x + group_width - 1,
                        y + tile_height - 1,
                    ),
                    fill="#deded9",
                )
                draw.text(
                    (group_x + 5, y + 5),
                    "no representative",
                    fill="#777777",
                    font=font,
                )
                continue
            frame_arrays = [
                _validate_frame(
                    frames[(clip["parent_manifest_key"], frame_index)],
                    (clip["parent_manifest_key"], frame_index),
                )
                for frame_index in clip["source_frame_indices"]
            ]
            images = [
                *[
                    _thumbnail(frame, thumbnail_size)
                    for frame in frame_arrays
                ],
                _difference_thumbnail(
                    frame_arrays[0],
                    frame_arrays[-1],
                    size=thumbnail_size,
                    gain=difference_gain,
                ),
            ]
            for column, image in enumerate(images):
                canvas.paste(image, (group_x + column * tile_width, y))
            draw.text(
                (group_x + 3, y + tile_height + 3),
                (
                    f"d={clip['distance_mse']:.3f} "
                    f"anchor={clip['source_anchor_rank']} "
                    f"frames={','.join(str(v) for v in clip['source_frame_indices'])}"
                ),
                fill="#333333",
                font=font,
            )
        draw.line(
            (0, y + row_height - 1, width, y + row_height - 1),
            fill="#d1d1cd",
            width=1,
        )
    _atomic_save_png(canvas, path)


def _validate_resumed_report(
    report_path: Path,
    *,
    contract_hash: str,
) -> dict[str, Any] | None:
    if not report_path.is_file():
        return None
    report = _read_json(report_path)
    if (
        report.get("schema") != RETRIEVAL_REPORT_SCHEMA
        or report.get("contract_hash") != contract_hash
    ):
        raise RuntimeError(
            f"Existing retrieval report differs from `{report_path}`."
        )
    for montage in report.get("montages", ()):
        path = Path(str(montage["path"]))
        if not path.is_file() or file_sha256(path) != montage["sha256"]:
            return None
    return report


def render_codebook_retrieval_montages(
    *,
    source_manifest_path: str | Path,
    pooled_manifest_path: str | Path,
    droid_data_dir: str | Path,
    evaluation_report_paths: Sequence[str | Path],
    output_dir: str | Path,
    camera: str = "wrist_image_left",
    splits: Iterable[str] = ("test",),
    levels: Iterable[int] = (1,),
    representatives_per_code: int = 3,
    thumbnail_size: tuple[int, int] = (128, 72),
    difference_gain: float = 3.0,
    diversity_by: str = "scene",
    resume: bool = True,
) -> dict[str, Any]:
    """Render exact three-frame RGB retrievals for frozen RQ centers."""

    source_manifest_path = Path(source_manifest_path)
    pooled_manifest_path = Path(pooled_manifest_path)
    droid_data_dir = Path(droid_data_dir)
    report_paths = tuple(Path(path) for path in evaluation_report_paths)
    output_dir = Path(output_dir)
    splits = tuple(str(value) for value in splits)
    levels = tuple(int(value) for value in levels)
    if not report_paths:
        raise ValueError("At least one held-out evaluation report is required.")
    if (
        not splits
        or len(set(splits)) != len(splits)
        or any(value not in {"val", "test"} for value in splits)
    ):
        raise ValueError("Retrieval splits must be unique val/test values.")
    if not levels or len(set(levels)) != len(levels) or min(levels) <= 0:
        raise ValueError("Retrieval levels must be unique positive integers.")
    if representatives_per_code <= 0:
        raise ValueError("`representatives_per_code` must be positive.")
    if (
        len(thumbnail_size) != 2
        or min(int(value) for value in thumbnail_size) <= 0
    ):
        raise ValueError("Thumbnail width and height must be positive.")
    if difference_gain <= 0:
        raise ValueError("`difference_gain` must be positive.")
    if diversity_by not in {"none", "parent", "scene"}:
        raise ValueError(
            "`diversity_by` must be one of none, parent, or scene."
        )
    if not source_manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing source manifest `{source_manifest_path}`."
        )
    if not pooled_manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing pooled manifest `{pooled_manifest_path}`."
        )
    dataset_info_path = droid_data_dir / "dataset_info.json"
    if not dataset_info_path.is_file():
        raise FileNotFoundError(
            f"Not a TFDS builder directory: {droid_data_dir}."
        )

    source_manifest = EpisodeManifest.read_jsonl(source_manifest_path)
    pooled_manifest = EpisodeManifest.read_jsonl(pooled_manifest_path)
    pooled_fingerprint = pooled_manifest.fingerprint()
    selected_rows, evaluation_inputs = _load_evaluation_rows(
        report_paths,
        manifest_fingerprint=pooled_fingerprint,
        splits=splits,
        levels=levels,
    )
    clips, frame_requests, pooled_shards = _resolve_clips(
        selected_rows=selected_rows,
        levels=levels,
        representatives_per_code=representatives_per_code,
        pooled_manifest_path=pooled_manifest_path,
        pooled_manifest=pooled_manifest,
        source_manifest=source_manifest,
        camera=camera,
        diversity_by=diversity_by,
    )
    source_by_key = {record.key: record for record in source_manifest}
    selected_source_records = {
        key: source_by_key[key] for key in sorted(frame_requests)
    }
    source_shards = {}
    for record in selected_source_records.values():
        name = str(record.metadata["rlds_shard_name"])
        source_shards[name] = {
            "name": name,
            "bytes": int(record.metadata["rlds_shard_bytes"]),
            "source_checksum": record.source_checksum,
        }

    contract = {
        "schema": RETRIEVAL_CONTRACT_SCHEMA,
        "source_manifest": {
            "path": str(source_manifest_path.resolve()),
            "sha256": file_sha256(source_manifest_path),
            "fingerprint": source_manifest.fingerprint(),
        },
        "pooled_manifest": {
            "path": str(pooled_manifest_path.resolve()),
            "sha256": file_sha256(pooled_manifest_path),
            "fingerprint": pooled_fingerprint,
        },
        "evaluation_reports": evaluation_inputs,
        "selected_pooled_shards": pooled_shards,
        "selected_source_shards": [
            source_shards[name] for name in sorted(source_shards)
        ],
        "droid_dataset": {
            "path": str(droid_data_dir.resolve()),
            "dataset_info_sha256": file_sha256(dataset_info_path),
        },
        "config": {
            "camera": camera,
            "splits": list(splits),
            "levels": list(levels),
            "representatives_per_code": representatives_per_code,
            "thumbnail_size": list(thumbnail_size),
            "difference_gain": float(difference_gain),
            "diversity_by": diversity_by,
        },
        "implementation_sha256": {
            "retrieval": file_sha256(Path(__file__)),
            "droid_rlds": file_sha256(
                Path(
                    __import__(
                        "codewam.data.droid_rlds",
                        fromlist=["__file__"],
                    ).__file__
                )
            ),
        },
    }
    contract["contract_hash"] = _canonical_hash(contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    report_path = output_dir / "retrieval_report.json"
    if contract_path.is_file():
        previous = _read_json(contract_path)
        if previous != contract:
            raise RuntimeError(
                f"Existing retrieval contract differs from `{contract_path}`."
            )
        if not resume:
            raise FileExistsError(
                f"Retrieval contract already exists at `{contract_path}`."
            )
        resumed = _validate_resumed_report(
            report_path,
            contract_hash=contract["contract_hash"],
        )
        if resumed is not None:
            return resumed
    else:
        write_json_report(contract_path, contract)

    source_subset = EpisodeManifest.from_records(
        selected_source_records[key] for key in sorted(selected_source_records)
    )
    frames = read_manifest_droid_rlds_frames(
        droid_data_dir,
        source_subset,
        frame_requests,
        camera=camera,
    )
    for clip in clips:
        arrays = [
            _validate_frame(
                frames[(clip["parent_manifest_key"], frame_index)],
                (clip["parent_manifest_key"], frame_index),
            )
            for frame_index in clip["source_frame_indices"]
        ]
        clip["rgb_motion"] = _motion_metrics(arrays)

    clips_by_group: dict[tuple[str, str, int], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for clip in clips:
        clips_by_group[
            (clip["family"], clip["split"], int(clip["level"]))
        ].append(clip)

    montage_rows = []
    montage_dir = output_dir / "montages"
    for (family, split), row in sorted(selected_rows.items()):
        for level in levels:
            group_clips = clips_by_group[(family, split, level)]
            path = montage_dir / f"{family}_{split}_L{level}.png"
            _render_montage(
                path,
                family=family,
                split=split,
                level=level,
                k=int(row["k"]),
                representatives_per_code=representatives_per_code,
                clips=group_clips,
                frames=frames,
                thumbnail_size=thumbnail_size,
                difference_gain=difference_gain,
            )
            by_code = []
            for code in range(int(row["k"])):
                code_clips = [
                    clip for clip in group_clips if int(clip["code"]) == code
                ]
                by_code.append(
                    {
                        "code": code,
                        "examples": len(code_clips),
                        "mean_distance_mse": (
                            None
                            if not code_clips
                            else sum(
                                float(clip["distance_mse"])
                                for clip in code_clips
                            )
                            / len(code_clips)
                        ),
                        "mean_first_last_rgb_difference": (
                            None
                            if not code_clips
                            else sum(
                                float(
                                    clip["rgb_motion"][
                                        "first_last_mean_absolute_rgb_difference"
                                    ]
                                )
                                for clip in code_clips
                            )
                            / len(code_clips)
                        ),
                    }
                )
            motion_values = [
                float(
                    clip["rgb_motion"][
                        "first_last_mean_absolute_rgb_difference"
                    ]
                )
                for clip in group_clips
            ]
            motion_eta_squared = None
            if motion_values:
                motion_mean = sum(motion_values) / len(motion_values)
                total_variation = sum(
                    (value - motion_mean) ** 2
                    for value in motion_values
                )
                between_code_variation = 0.0
                for code in range(int(row["k"])):
                    code_values = [
                        float(
                            clip["rgb_motion"][
                                "first_last_mean_absolute_rgb_difference"
                            ]
                        )
                        for clip in group_clips
                        if int(clip["code"]) == code
                    ]
                    if code_values:
                        code_mean = sum(code_values) / len(code_values)
                        between_code_variation += len(code_values) * (
                            code_mean - motion_mean
                        ) ** 2
                if total_variation > 0:
                    motion_eta_squared = (
                        between_code_variation / total_variation
                    )
            source_anchor_ranks = [
                int(clip["source_anchor_rank"]) for clip in group_clips
            ]
            montage_rows.append(
                {
                    "family": family,
                    "split": split,
                    "level": level,
                    "path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                    "selection_summary": {
                        "examples": len(group_clips),
                        "expected_examples": (
                            int(row["k"]) * representatives_per_code
                        ),
                        "codes_with_full_diversity": sum(
                            int(value["examples"])
                            == representatives_per_code
                            for value in by_code
                        ),
                        "median_source_anchor_rank": (
                            None
                            if not source_anchor_ranks
                            else statistics.median(source_anchor_ranks)
                        ),
                        "maximum_source_anchor_rank": (
                            None
                            if not source_anchor_ranks
                            else max(source_anchor_ranks)
                        ),
                        "mean_first_last_rgb_difference": (
                            None
                            if not motion_values
                            else sum(motion_values) / len(motion_values)
                        ),
                        (
                            "descriptive_motion_energy_eta_squared_"
                            "on_selected_anchors"
                        ): motion_eta_squared,
                    },
                    "codes": by_code,
                }
            )

    report = {
        "schema": RETRIEVAL_REPORT_SCHEMA,
        "contract_hash": contract["contract_hash"],
        "descriptor_semantics": {
            "cluster_input": "concat(z[t-2s], z[t-s], z[t])",
            "s": "family stride in pooled Wan latent ticks",
            "difference_tile": (
                "RGB |frame[t] - frame[t-2s]| multiplied only for display"
            ),
            "representative_selection": (
                f"nearest anchors with diversity_by={diversity_by}"
            ),
        },
        "interpretation_guardrails": [
            (
                "L1 rows retrieve observations nearest to first-stage RQ "
                "centers and are the primary standalone qualitative view."
            ),
            (
                "L2/L3 rows, when requested, retrieve residual-center use "
                "across different earlier prefixes; they are not standalone "
                "semantic state classes."
            ),
            (
                "RGB difference tiles are visual aids and are not codebook "
                "training inputs."
            ),
        ],
        "totals": {
            "families": sorted(
                {str(clip["family"]) for clip in clips}
            ),
            "splits": list(splits),
            "levels": list(levels),
            "clips": len(clips),
            "unique_source_episodes": len(selected_source_records),
            "unique_source_shards": len(source_shards),
            "unique_source_frames": len(frames),
            "montages": len(montage_rows),
        },
        "montages": montage_rows,
        "clips": clips,
    }
    write_json_report(report_path, report)
    return report
