from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch


DROID_CAMERA_KEYS = (
    "exterior_image_1_left",
    "exterior_image_2_left",
    "wrist_image_left",
)


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

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("DROID episode id must not be empty.")
        if not self.frames:
            raise ValueError(f"DROID episode `{self.episode_id}` has no camera frames.")
        lengths = {int(value.shape[0]) for value in self.frames.values()}
        lengths.update({int(self.action.shape[0]), int(self.proprio.shape[0])})
        if len(lengths) != 1:
            raise ValueError(
                f"DROID episode `{self.episode_id}` has inconsistent sequence lengths: {lengths}."
            )
        for camera, frames in self.frames.items():
            if frames.ndim != 4 or frames.shape[-1] != 3:
                raise ValueError(
                    f"Camera `{camera}` must be [T,H,W,3], got {tuple(frames.shape)}."
                )
            if frames.dtype != torch.uint8:
                raise ValueError(f"Camera `{camera}` must contain uint8 RGB frames.")
        if self.action.ndim != 2 or self.proprio.ndim != 2:
            raise ValueError("DROID action and proprio tensors must both be [T,D].")

    @property
    def steps(self) -> int:
        return int(self.action.shape[0])


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


def iter_droid_rlds_episodes(
    data_dir: str | Path,
    cameras: Sequence[str] = DROID_CAMERA_KEYS,
    max_episodes: int | None = None,
    split: str = "train",
) -> Iterator[DroidRLDSEpisode]:
    """Read official DROID RLDS episodes without importing TensorFlow at module import."""

    requested_cameras = tuple(str(value) for value in cameras)
    unknown = sorted(set(requested_cameras) - set(DROID_CAMERA_KEYS))
    if unknown:
        raise ValueError(f"Unsupported DROID cameras: {unknown}.")
    if not requested_cameras:
        raise ValueError("At least one DROID camera must be requested.")
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
        )

