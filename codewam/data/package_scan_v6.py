from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


DEFAULT_CAMERA_KEYS = ("observation.images.top", "observation.images.wrist")


@dataclass(frozen=True)
class VideoRef:
    chunk_index: int
    file_index: int
    from_timestamp: float
    to_timestamp: float


@dataclass(frozen=True)
class EpisodeRef:
    episode_index: int
    from_index: int
    to_index: int
    length: int
    tasks: tuple[str, ...]
    videos: dict[str, VideoRef]


@dataclass(frozen=True)
class WindowRef:
    episode_index: int
    start_index: int


def _read_parquet_tree(path: Path) -> pa.Table:
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    return pa.concat_tables([pq.read_table(file) for file in files], promote_options="default")


def _as_float_array(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(table[column].to_pylist(), dtype=np.float32)


def _as_int_array(table: pa.Table, column: str) -> np.ndarray:
    return np.asarray(table[column].to_pylist(), dtype=np.int64)


def _normalize_camera_key(key: str) -> str:
    return key.removeprefix("observation.images.")


def _format_video_path(root: Path, template: str, camera_key: str, ref: VideoRef) -> Path:
    return root / template.format(
        video_key=camera_key,
        chunk_index=ref.chunk_index,
        file_index=ref.file_index,
    )


def _decode_video_frames(path: Path, timestamps: Sequence[float]) -> torch.Tensor:
    if not timestamps:
        raise ValueError("At least one timestamp is required.")
    if not path.exists():
        raise FileNotFoundError(path)

    targets = [(float(ts), i) for i, ts in enumerate(timestamps)]
    order = sorted(range(len(targets)), key=lambda i: targets[i][0])
    sorted_targets = [targets[i] for i in order]
    frames: list[np.ndarray | None] = [None] * len(targets)

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate is not None else 30.0
        preroll = 1.0 / max(fps, 1.0)
        seek_time = max(0.0, sorted_targets[0][0] - preroll)
        container.seek(int(seek_time / float(stream.time_base)), stream=stream, backward=True)

        target_pos = 0
        prev_time: float | None = None
        prev_frame: np.ndarray | None = None

        for frame in container.decode(stream):
            frame_time = float(frame.time if frame.time is not None else frame.pts * stream.time_base)
            frame_rgb = frame.to_ndarray(format="rgb24")

            while target_pos < len(sorted_targets) and frame_time >= sorted_targets[target_pos][0]:
                target_time, original_index = sorted_targets[target_pos]
                if prev_frame is not None and prev_time is not None:
                    use_prev = abs(prev_time - target_time) <= abs(frame_time - target_time)
                    frames[original_index] = prev_frame.copy() if use_prev else frame_rgb.copy()
                else:
                    frames[original_index] = frame_rgb.copy()
                target_pos += 1

            prev_time = frame_time
            prev_frame = frame_rgb
            if target_pos >= len(sorted_targets):
                break

        if target_pos < len(sorted_targets):
            if prev_frame is None:
                raise RuntimeError(f"No frames decoded from {path}")
            for _, original_index in sorted_targets[target_pos:]:
                frames[original_index] = prev_frame.copy()

    stacked = np.stack([frame for frame in frames if frame is not None], axis=0)
    if stacked.shape[0] != len(timestamps):
        raise RuntimeError(f"Decoded {stacked.shape[0]} frames, expected {len(timestamps)} from {path}")
    return torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()


class PackageScanV6Dataset(Dataset):
    """Lightweight local reader for the LeRobot v3 package_scan_v6 dataset.

    The dataset returns CodeWAM-friendly samples:
      video: [C, T_video, H, W], normalized to [-1, 1] by default
      action: [T_action, 7]
      proprio: [T_action, 7]
    """

    def __init__(
        self,
        root: str | Path = "package_scan_v6",
        camera_keys: Sequence[str] | None = None,
        num_frames: int = 33,
        global_sample_stride: int = 1,
        action_video_freq_ratio: int = 4,
        camera_size: Sequence[int] = (224, 224),
        concat_multi_camera: str = "horizontal",
        normalize_video: bool = True,
        window_start_stride: int = 1,
        max_windows: int | None = None,
        return_camera_stack: bool = False,
        shape_meta: Any | None = None,
        processor: Any | None = None,
        text_embedding_cache_dir: str | None = None,
        context_len: int = 128,
    ) -> None:
        self.root = Path(root)
        self.info_path = self.root / "meta" / "info.json"
        if not self.info_path.exists():
            raise FileNotFoundError(f"Missing Package Scan v6 metadata: {self.info_path}")
        self.info = json.loads(self.info_path.read_text())
        self.fps = int(self.info["fps"])
        self.video_path_template = str(self.info["video_path"])
        self.num_frames = int(num_frames)
        self.global_sample_stride = int(global_sample_stride)
        self.action_video_freq_ratio = int(action_video_freq_ratio)
        self.camera_size = (int(camera_size[0]), int(camera_size[1]))
        self.concat_multi_camera = concat_multi_camera
        self.normalize_video = bool(normalize_video)
        self.window_start_stride = int(window_start_stride)
        self.max_windows = None if max_windows is None else int(max_windows)
        self.return_camera_stack = bool(return_camera_stack)
        self.shape_meta = shape_meta
        self.processor = processor
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = int(context_len)

        if self.num_frames <= 1:
            raise ValueError("num_frames must be > 1.")
        if self.global_sample_stride <= 0:
            raise ValueError("global_sample_stride must be > 0.")
        if self.action_video_freq_ratio <= 0:
            raise ValueError("action_video_freq_ratio must be > 0.")
        if (self.num_frames - 1) % self.action_video_freq_ratio != 0:
            raise ValueError("num_frames - 1 must be divisible by action_video_freq_ratio.")
        if self.concat_multi_camera not in {"horizontal", "vertical", "none", None}:
            raise ValueError("concat_multi_camera must be one of: horizontal, vertical, none.")

        video_keys = [key for key, spec in self.info["features"].items() if spec["dtype"] == "video"]
        if camera_keys is None:
            preferred = [key for key in DEFAULT_CAMERA_KEYS if key in video_keys]
            self.camera_keys = preferred or video_keys
        else:
            self.camera_keys = list(camera_keys)
        missing = [key for key in self.camera_keys if key not in video_keys]
        if missing:
            raise ValueError(f"Unknown camera keys {missing}; available video keys: {video_keys}")

        self.tasks = self._load_tasks()
        self.episodes = self._load_episodes()
        self._load_frame_table()
        self.windows = self._build_windows()

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    @property
    def proprio_dim(self) -> int:
        return int(self.proprio.shape[1])

    @property
    def video_frames_per_sample(self) -> int:
        return len(range(0, self.num_frames, self.action_video_freq_ratio))

    def _load_tasks(self) -> dict[int, str]:
        table = pq.read_table(self.root / "meta" / "tasks.parquet")
        task_indices = table["task_index"].to_pylist()
        tasks = table["task"].to_pylist()
        return {int(idx): str(task) for idx, task in zip(task_indices, tasks)}

    def _load_episodes(self) -> dict[int, EpisodeRef]:
        table = _read_parquet_tree(self.root / "meta" / "episodes")
        refs: dict[int, EpisodeRef] = {}
        for row in sorted(table.to_pylist(), key=lambda item: int(item["episode_index"])):
            videos = {}
            for camera_key in self.camera_keys:
                prefix = f"videos/{camera_key}"
                videos[camera_key] = VideoRef(
                    chunk_index=int(row[f"{prefix}/chunk_index"]),
                    file_index=int(row[f"{prefix}/file_index"]),
                    from_timestamp=float(row[f"{prefix}/from_timestamp"]),
                    to_timestamp=float(row[f"{prefix}/to_timestamp"]),
                )
            episode_index = int(row["episode_index"])
            refs[episode_index] = EpisodeRef(
                episode_index=episode_index,
                from_index=int(row["dataset_from_index"]),
                to_index=int(row["dataset_to_index"]),
                length=int(row["length"]),
                tasks=tuple(row["tasks"] or ()),
                videos=videos,
            )
        return refs

    def _load_frame_table(self) -> None:
        table = _read_parquet_tree(self.root / "data")
        indices = _as_int_array(table, "index")
        order = np.argsort(indices)

        self.indices = indices[order]
        self.actions = _as_float_array(table, "action")[order]
        self.proprio = _as_float_array(table, "observation.state")[order]
        self.timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)[order]
        self.episode_indices = _as_int_array(table, "episode_index")[order]
        self.task_indices = _as_int_array(table, "task_index")[order]

        if np.array_equal(self.indices, np.arange(len(self.indices), dtype=np.int64)):
            self._index_is_position = True
        else:
            self._index_is_position = False

    def _positions_for_indices(self, indices: np.ndarray) -> np.ndarray:
        if self._index_is_position:
            return indices.astype(np.int64)
        positions = np.searchsorted(self.indices, indices)
        if np.any(positions >= len(self.indices)) or np.any(self.indices[positions] != indices):
            raise IndexError(f"Requested missing frame indices: {indices.tolist()}")
        return positions.astype(np.int64)

    def _build_windows(self) -> list[WindowRef]:
        windows: list[WindowRef] = []
        max_offset = (self.num_frames - 1) * self.global_sample_stride
        for episode in self.episodes.values():
            last_start = episode.to_index - 1 - max_offset
            if last_start < episode.from_index:
                continue
            for start in range(episode.from_index, last_start + 1, self.window_start_stride):
                windows.append(WindowRef(episode_index=episode.episode_index, start_index=start))
                if self.max_windows is not None and len(windows) >= self.max_windows:
                    return windows
        return windows

    def _frame_indices(self, start_index: int, count: int) -> np.ndarray:
        return start_index + np.arange(count, dtype=np.int64) * self.global_sample_stride

    def _load_camera_frames(
        self,
        episode: EpisodeRef,
        camera_key: str,
        frame_positions: np.ndarray,
    ) -> torch.Tensor:
        ref = episode.videos[camera_key]
        video_path = _format_video_path(self.root, self.video_path_template, camera_key, ref)
        timestamps = [ref.from_timestamp + float(self.timestamps[pos]) for pos in frame_positions]
        frames = _decode_video_frames(video_path, timestamps)
        frames = frames.to(dtype=torch.float32) / 255.0
        if tuple(frames.shape[-2:]) != self.camera_size:
            frames = F.interpolate(frames, size=self.camera_size, mode="bilinear", align_corners=False)
        return frames

    def _combine_cameras(self, camera_frames: list[torch.Tensor]) -> torch.Tensor:
        if len(camera_frames) == 1 or self.concat_multi_camera in {"none", None}:
            return camera_frames[0]
        if self.concat_multi_camera == "horizontal":
            return torch.cat(camera_frames, dim=-1)
        if self.concat_multi_camera == "vertical":
            return torch.cat(camera_frames, dim=-2)
        raise ValueError(f"Unsupported concat_multi_camera: {self.concat_multi_camera}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        window = self.windows[idx]
        episode = self.episodes[window.episode_index]

        obs_indices = self._frame_indices(window.start_index, self.num_frames)
        action_indices = self._frame_indices(window.start_index, self.num_frames - 1)
        video_indices = obs_indices[:: self.action_video_freq_ratio]

        obs_positions = self._positions_for_indices(obs_indices)
        action_positions = self._positions_for_indices(action_indices)
        video_positions = self._positions_for_indices(video_indices)

        camera_frames = [
            self._load_camera_frames(episode, camera_key, video_positions)
            for camera_key in self.camera_keys
        ]
        video = self._combine_cameras(camera_frames)
        if self.normalize_video:
            video = video * 2.0 - 1.0
        video = video.permute(1, 0, 2, 3).contiguous()

        task_index = int(self.task_indices[obs_positions[0]])
        prompt = self.tasks.get(task_index)
        if prompt is None:
            prompt = episode.tasks[0] if episode.tasks else ""

        sample: dict[str, Any] = {
            "video": video,
            "action": torch.from_numpy(self.actions[action_positions].copy()).float(),
            "proprio": torch.from_numpy(self.proprio[action_positions].copy()).float(),
            "prompt": prompt,
            "episode_index": int(window.episode_index),
            "start_index": int(window.start_index),
            "frame_index": torch.from_numpy(obs_indices.copy()).long(),
            "video_frame_index": torch.from_numpy(video_indices.copy()).long(),
            "task_index": task_index,
            "camera_keys": [_normalize_camera_key(key) for key in self.camera_keys],
            "image_is_pad": torch.zeros((len(video_indices),), dtype=torch.bool),
            "action_is_pad": torch.zeros((len(action_indices),), dtype=torch.bool),
            "proprio_is_pad": torch.zeros((len(action_indices),), dtype=torch.bool),
        }
        if self.return_camera_stack:
            stack = torch.stack(camera_frames, dim=0)
            if self.normalize_video:
                stack = stack * 2.0 - 1.0
            sample["video_by_camera"] = stack
        return sample

    def summary(self) -> dict[str, Any]:
        lengths = [episode.length for episode in self.episodes.values()]
        return {
            "root": str(self.root),
            "fps": self.fps,
            "episodes": len(self.episodes),
            "frames": int(len(self.indices)),
            "windows": len(self.windows),
            "num_frames": self.num_frames,
            "video_frames_per_sample": self.video_frames_per_sample,
            "action_dim": self.action_dim,
            "proprio_dim": self.proprio_dim,
            "camera_keys": list(self.camera_keys),
            "camera_size": list(self.camera_size),
            "concat_multi_camera": self.concat_multi_camera,
            "tasks": self.tasks,
            "episode_length_min": int(min(lengths)) if lengths else 0,
            "episode_length_max": int(max(lengths)) if lengths else 0,
        }
