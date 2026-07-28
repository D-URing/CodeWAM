from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch

from codewam.codebook_eval.manifest import EpisodeManifest, EpisodeRecord


DROID_CAMERA_KEYS = (
    "exterior_image_1_left",
    "exterior_image_2_left",
    "wrist_image_left",
)


def _validate_modalities(
    episode_id: str,
    frames: dict[str, torch.Tensor],
    action: torch.Tensor,
    proprio: torch.Tensor,
    action_components: dict[str, torch.Tensor] | None = None,
) -> int:
    if not frames:
        raise ValueError(f"DROID episode `{episode_id}` has no camera frames.")
    lengths = {int(value.shape[0]) for value in frames.values()}
    lengths.update({int(action.shape[0]), int(proprio.shape[0])})
    if len(lengths) != 1:
        raise ValueError(
            f"DROID episode `{episode_id}` has inconsistent sequence lengths: "
            f"{lengths}."
        )
    for camera, camera_frames in frames.items():
        if camera_frames.ndim != 4 or camera_frames.shape[-1] != 3:
            raise ValueError(
                f"Camera `{camera}` must be [T,H,W,3], "
                f"got {tuple(camera_frames.shape)}."
            )
        if camera_frames.dtype != torch.uint8:
            raise ValueError(f"Camera `{camera}` must contain uint8 RGB frames.")
    if action.ndim != 2 or proprio.ndim != 2:
        raise ValueError("DROID action and proprio tensors must both be [T,D].")
    length = next(iter(lengths))
    if length <= 0:
        raise ValueError(f"DROID episode `{episode_id}` must not be empty.")
    for name, values in (action_components or {}).items():
        if values.ndim != 2 or int(values.shape[0]) != length:
            raise ValueError(
                f"DROID action component `{name}` must be [T,D] with T={length}, "
                f"got {tuple(values.shape)}."
            )
    return length


def _validate_keep_ranges(
    ranges: Iterable[Sequence[int]],
    *,
    num_steps: int,
    episode_id: str,
) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    previous_stop = 0
    for value in ranges:
        if len(value) != 2:
            raise ValueError(
                f"DROID episode `{episode_id}` has malformed keep range `{value}`."
            )
        start, stop = int(value[0]), int(value[1])
        if start < previous_stop or stop <= start or stop > num_steps:
            raise ValueError(
                f"DROID episode `{episode_id}` has invalid keep range "
                f"`[{start}, {stop})` for {num_steps} steps."
            )
        normalized.append((start, stop))
        previous_stop = stop
    return tuple(normalized)


@dataclass(frozen=True)
class DroidRLDSSegment:
    episode_id: str
    range_index: int
    start: int
    stop: int
    frames: dict[str, torch.Tensor]
    action: torch.Tensor
    proprio: torch.Tensor
    language_instruction: str
    action_components: dict[str, torch.Tensor] = field(default_factory=dict)
    split: str | None = None
    manifest_key: str | None = None
    source_shard: str | None = None
    record_index: int | None = None

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("DROID segment episode id must not be empty.")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError(
                f"Invalid DROID segment interval [{self.start}, {self.stop})."
            )
        length = _validate_modalities(
            self.episode_id,
            self.frames,
            self.action,
            self.proprio,
            self.action_components,
        )
        if length != self.stop - self.start:
            raise ValueError(
                f"DROID segment `{self.segment_id}` contains {length} rows, "
                f"expected {self.stop - self.start}."
            )
        object.__setattr__(self, "action_components", dict(self.action_components))

    @property
    def segment_id(self) -> str:
        return f"{self.episode_id}@{self.start}:{self.stop}"

    @property
    def steps(self) -> int:
        return int(self.action.shape[0])


@dataclass(frozen=True)
class DroidRLDSEpisode:
    episode_id: str
    index: int
    frames: dict[str, torch.Tensor]
    action: torch.Tensor
    proprio: torch.Tensor
    language_instruction: str
    source_file: str
    recording_folder: str
    action_components: dict[str, torch.Tensor] = field(default_factory=dict)
    split: str | None = None
    keep_ranges: tuple[tuple[int, int], ...] = ()
    manifest_key: str | None = None
    source_shard: str | None = None
    record_index: int | None = None

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("DROID episode id must not be empty.")
        steps = _validate_modalities(
            self.episode_id,
            self.frames,
            self.action,
            self.proprio,
            self.action_components,
        )
        ranges = _validate_keep_ranges(
            self.keep_ranges,
            num_steps=steps,
            episode_id=self.episode_id,
        )
        object.__setattr__(self, "keep_ranges", ranges)
        object.__setattr__(self, "action_components", dict(self.action_components))
        if self.split is not None and self.split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported DROID split `{self.split}`.")

    @property
    def steps(self) -> int:
        return int(self.action.shape[0])

    def iter_eligible_segments(self) -> Iterator[DroidRLDSSegment]:
        ranges = self.keep_ranges or ((0, self.steps),)
        for range_index, (start, stop) in enumerate(ranges):
            yield DroidRLDSSegment(
                episode_id=self.episode_id,
                range_index=range_index,
                start=start,
                stop=stop,
                frames={
                    camera: values[start:stop]
                    for camera, values in self.frames.items()
                },
                action=self.action[start:stop],
                proprio=self.proprio[start:stop],
                language_instruction=self.language_instruction,
                action_components={
                    name: values[start:stop]
                    for name, values in self.action_components.items()
                },
                split=self.split,
                manifest_key=self.manifest_key,
                source_shard=self.source_shard,
                record_index=self.record_index,
            )


@dataclass(frozen=True)
class DroidShardWork:
    shard_name: str
    source_bytes: int
    records: tuple[EpisodeRecord, ...]

    def __post_init__(self) -> None:
        if not self.shard_name or Path(self.shard_name).name != self.shard_name:
            raise ValueError(f"Invalid DROID shard name `{self.shard_name}`.")
        if self.source_bytes <= 0:
            raise ValueError(
                f"DROID shard `{self.shard_name}` has invalid source size "
                f"{self.source_bytes}."
            )
        if not self.records:
            raise ValueError(f"DROID shard `{self.shard_name}` has no records.")

    @property
    def episodes(self) -> int:
        return len(self.records)


@dataclass(frozen=True)
class DroidRankAssignment:
    rank: int
    world_size: int
    shards: tuple[DroidShardWork, ...]

    @property
    def episodes(self) -> int:
        return sum(shard.episodes for shard in self.shards)

    @property
    def source_bytes(self) -> int:
        return sum(shard.source_bytes for shard in self.shards)


def _manifest_shard_work(manifest: EpisodeManifest) -> tuple[DroidShardWork, ...]:
    by_shard: dict[str, list[EpisodeRecord]] = defaultdict(list)
    shard_bytes: dict[str, int] = {}
    positions: set[tuple[str, int]] = set()
    for record in manifest:
        shard_name = str(record.metadata.get("rlds_shard_name") or "")
        try:
            record_index = int(record.metadata["rlds_record_index"])
            source_bytes = int(record.metadata["rlds_shard_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Episode `{record.key}` lacks exact DROID shard metadata."
            ) from exc
        if record_index < 0:
            raise ValueError(
                f"Episode `{record.key}` has negative RLDS record index."
            )
        position = (shard_name, record_index)
        if position in positions:
            raise ValueError(f"Duplicate DROID shard position {position}.")
        positions.add(position)
        previous_bytes = shard_bytes.setdefault(shard_name, source_bytes)
        if previous_bytes != source_bytes:
            raise ValueError(
                f"Inconsistent source bytes for DROID shard `{shard_name}`."
            )
        by_shard[shard_name].append(record)

    work = []
    for shard_name, records in by_shard.items():
        records.sort(key=lambda record: int(record.metadata["rlds_record_index"]))
        work.append(
            DroidShardWork(
                shard_name=shard_name,
                source_bytes=shard_bytes[shard_name],
                records=tuple(records),
            )
        )
    return tuple(sorted(work, key=lambda shard: shard.shard_name))


def plan_droid_rank_assignments(
    manifest: EpisodeManifest,
    world_size: int,
) -> tuple[DroidRankAssignment, ...]:
    if world_size <= 0:
        raise ValueError("DROID reader world size must be positive.")
    loads = [0] * world_size
    episode_counts = [0] * world_size
    assigned: list[list[DroidShardWork]] = [[] for _ in range(world_size)]
    work = sorted(
        _manifest_shard_work(manifest),
        key=lambda shard: (-shard.source_bytes, shard.shard_name),
    )
    for shard in work:
        rank = min(
            range(world_size),
            key=lambda value: (loads[value], episode_counts[value], value),
        )
        assigned[rank].append(shard)
        loads[rank] += shard.source_bytes
        episode_counts[rank] += shard.episodes
    return tuple(
        DroidRankAssignment(
            rank=rank,
            world_size=world_size,
            shards=tuple(sorted(assigned[rank], key=lambda shard: shard.shard_name)),
        )
        for rank in range(world_size)
    )


def _decode_text(value: object) -> str:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _episode_id(index: int, source_file: str, recording_folder: str) -> str:
    source = recording_folder or source_file or f"index-{index}"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"droid100-{index:05d}-{digest}"


def probe_split(index: int) -> str:
    """A deterministic 4/1/1 train/validation/test episode cycle."""

    remainder = int(index) % 6
    if remainder == 0:
        return "val"
    if remainder == 1:
        return "test"
    return "train"


def _disable_tensorflow_gpu(tf: object) -> None:
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as exc:
        raise RuntimeError(
            "TensorFlow initialized CUDA before the DROID reader could disable it. "
            "Construct the RLDS iterator before loading the Wan model."
        ) from exc


def _requested_cameras(cameras: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(str(value) for value in cameras)
    unknown = sorted(set(requested) - set(DROID_CAMERA_KEYS))
    if unknown:
        raise ValueError(f"Unsupported DROID cameras: {unknown}.")
    if not requested:
        raise ValueError("At least one DROID camera must be requested.")
    if len(set(requested)) != len(requested):
        raise ValueError(f"Duplicate DROID cameras requested: {requested}.")
    return requested


def _decode_manifest_record(
    *,
    features: Any,
    raw_record: Any,
    decoders: dict[str, Any],
    record: EpisodeRecord,
    cameras: tuple[str, ...],
) -> DroidRLDSEpisode:
    decoded = features.deserialize_example(raw_record, decoders=decoders)
    source_file = _decode_text(decoded["episode_metadata"]["file_path"].numpy())
    recording_folder = _decode_text(
        decoded["episode_metadata"]["recording_folderpath"].numpy()
    )
    expected_recording_folder = str(
        record.metadata.get("recording_folderpath") or ""
    )
    mismatches = []
    if source_file != record.source_uri:
        mismatches.append("source_uri")
    if recording_folder != expected_recording_folder:
        mismatches.append("recording_folderpath")
    if mismatches:
        raise RuntimeError(
            f"DROID manifest/source mismatch for `{record.key}`: {mismatches}."
        )

    step_rows = list(decoded["steps"].as_numpy_iterator())
    if len(step_rows) != record.num_steps:
        raise RuntimeError(
            f"DROID manifest/source mismatch for `{record.key}`: ['num_steps']."
        )

    frame_tensors = {
        camera: torch.from_numpy(
            np.stack(
                [row["observation"][camera] for row in step_rows],
                axis=0,
            )
        ).contiguous()
        for camera in cameras
    }
    action = torch.from_numpy(
        np.stack([row["action"] for row in step_rows], axis=0)
    ).float()
    action_components = {
        name: torch.from_numpy(
            np.stack([row["action_dict"][name] for row in step_rows], axis=0)
        ).float()
        for name in sorted(step_rows[0]["action_dict"])
    }
    proprio = torch.from_numpy(
        np.concatenate(
            [
                np.stack(
                    [row["observation"]["cartesian_position"] for row in step_rows],
                    axis=0,
                ),
                np.stack(
                    [row["observation"]["joint_position"] for row in step_rows],
                    axis=0,
                ),
                np.stack(
                    [row["observation"]["gripper_position"] for row in step_rows],
                    axis=0,
                ),
            ],
            axis=1,
        )
    ).float()
    instruction = _decode_text(step_rows[0]["language_instruction"])
    keep_ranges = _validate_keep_ranges(
        record.metadata.get("keep_ranges", ()),
        num_steps=record.num_steps,
        episode_id=record.episode_id,
    )
    expected_eligible_steps = record.metadata.get("eligible_steps")
    if expected_eligible_steps is not None and sum(
        stop - start for start, stop in keep_ranges
    ) != int(expected_eligible_steps):
        raise RuntimeError(
            f"DROID manifest eligible-step mismatch for `{record.key}`."
        )
    return DroidRLDSEpisode(
        episode_id=record.episode_id,
        index=int(record.metadata["rlds_record_index"]),
        frames=frame_tensors,
        action=action,
        proprio=proprio,
        language_instruction=instruction,
        source_file=source_file,
        recording_folder=recording_folder,
        action_components=action_components,
        split=record.split,
        keep_ranges=keep_ranges,
        manifest_key=record.key,
        source_shard=str(record.metadata["rlds_shard_name"]),
        record_index=int(record.metadata["rlds_record_index"]),
    )


def iter_manifest_droid_rlds_episodes(
    data_dir: str | Path,
    manifest: EpisodeManifest,
    *,
    rank: int = 0,
    world_size: int = 1,
    cameras: Sequence[str] = DROID_CAMERA_KEYS,
    completed_episode_keys: Iterable[str] = (),
) -> Iterator[DroidRLDSEpisode]:
    """Read exact manifest positions while assigning every source shard to one rank."""

    requested_cameras = _requested_cameras(cameras)
    if world_size <= 0:
        raise ValueError("DROID reader world size must be positive.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"DROID reader rank {rank} is invalid for world size {world_size}.")
    data_dir = Path(data_dir)
    if not (data_dir / "dataset_info.json").is_file():
        raise FileNotFoundError(f"Not a TFDS builder directory: {data_dir}.")
    assignment = plan_droid_rank_assignments(manifest, world_size)[rank]
    completed = {str(value) for value in completed_episode_keys}

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise RuntimeError(
            "Reading DROID RLDS requires `tensorflow-cpu` and `tensorflow-datasets`."
        ) from exc
    _disable_tensorflow_gpu(tf)
    builder = tfds.builder_from_directory(builder_dir=str(data_dir))
    skipped_cameras = set(DROID_CAMERA_KEYS) - set(requested_cameras)
    decoders = {
        "steps": {
            "observation": {
                camera: tfds.decode.SkipDecoding()
                for camera in sorted(skipped_cameras)
            }
        }
    }

    for shard in assignment.shards:
        pending = {
            int(record.metadata["rlds_record_index"]): record
            for record in shard.records
            if record.key not in completed and record.episode_id not in completed
        }
        if not pending:
            continue
        shard_path = data_dir / shard.shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing DROID shard `{shard_path}`.")
        if shard_path.stat().st_size != shard.source_bytes:
            raise RuntimeError(
                f"DROID shard `{shard_path}` has {shard_path.stat().st_size} bytes, "
                f"expected {shard.source_bytes}."
            )

        maximum_index = max(pending)
        found: set[int] = set()
        dataset = tf.data.TFRecordDataset(str(shard_path), num_parallel_reads=1)
        for record_index, raw_record in enumerate(dataset):
            if record_index > maximum_index:
                break
            record = pending.get(record_index)
            if record is None:
                continue
            found.add(record_index)
            yield _decode_manifest_record(
                features=builder.info.features,
                raw_record=raw_record,
                decoders=decoders,
                record=record,
                cameras=requested_cameras,
            )
            if len(found) == len(pending):
                break
        missing = sorted(set(pending) - found)
        if missing:
            raise RuntimeError(
                f"DROID shard `{shard.shard_name}` is missing manifest positions "
                f"{missing[:8]}."
            )


def iter_droid_rlds_episodes(
    data_dir: str | Path,
    cameras: Sequence[str] = DROID_CAMERA_KEYS,
    max_episodes: int | None = None,
    split: str = "train",
) -> Iterator[DroidRLDSEpisode]:
    """Read official DROID RLDS episodes without importing TensorFlow at module import."""

    requested_cameras = _requested_cameras(cameras)
    data_dir = Path(data_dir)
    if not (data_dir / "dataset_info.json").is_file():
        raise FileNotFoundError(f"Not a TFDS builder directory: {data_dir}.")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise RuntimeError(
            "Reading DROID RLDS requires `tensorflow-cpu` and `tensorflow-datasets`."
        ) from exc
    _disable_tensorflow_gpu(tf)

    builder = tfds.builder_from_directory(builder_dir=str(data_dir))
    dataset = builder.as_dataset(split=split, shuffle_files=False)
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)
    if max_episodes is not None:
        if int(max_episodes) <= 0:
            raise ValueError(f"`max_episodes` must be positive, got {max_episodes}.")
        dataset = dataset.take(int(max_episodes))

    for index, episode in enumerate(dataset):
        metadata = episode["episode_metadata"]
        source_file = _decode_text(metadata["file_path"].numpy())
        recording_folder = _decode_text(metadata["recording_folderpath"].numpy())
        step_rows = list(episode["steps"].as_numpy_iterator())
        if not step_rows:
            continue

        frame_tensors = {
            camera: torch.from_numpy(
                np.stack(
                    [row["observation"][camera] for row in step_rows],
                    axis=0,
                )
            ).contiguous()
            for camera in requested_cameras
        }
        action = torch.from_numpy(
            np.stack([row["action"] for row in step_rows], axis=0)
        ).float()
        action_components = {
            name: torch.from_numpy(
                np.stack([row["action_dict"][name] for row in step_rows], axis=0)
            ).float()
            for name in sorted(step_rows[0]["action_dict"])
        }
        proprio = torch.from_numpy(
            np.concatenate(
                [
                    np.stack(
                        [row["observation"]["cartesian_position"] for row in step_rows],
                        axis=0,
                    ),
                    np.stack(
                        [row["observation"]["joint_position"] for row in step_rows],
                        axis=0,
                    ),
                    np.stack(
                        [row["observation"]["gripper_position"] for row in step_rows],
                        axis=0,
                    ),
                ],
                axis=1,
            )
        ).float()
        instruction = _decode_text(step_rows[0]["language_instruction"])
        yield DroidRLDSEpisode(
            episode_id=_episode_id(index, source_file, recording_folder),
            index=index,
            frames=frame_tensors,
            action=action,
            proprio=proprio,
            language_instruction=instruction,
            source_file=source_file,
            recording_folder=recording_folder,
            action_components=action_components,
        )
