from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
import torch.nn.functional as F

from .manifest import VALID_SPLITS, SplitName


POOLED_SHARD_SCHEMA = "codewam.pooled-feature-shard.v1"
REQUIRED_SHARD_METADATA = frozenset(
    {
        "dataset_revision",
        "wan_model_id",
        "wan_revision",
        "preprocess_revision",
        "source_checksums",
    }
)


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_torch_payload(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def file_sha256(path: str | Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def expand_shard_paths(patterns: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = [Path(value) for value in glob.glob(str(pattern))]
        paths.extend(matched or [Path(pattern)])
    unique = sorted({path.resolve() for path in paths})
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing pooled feature shards: {[str(path) for path in missing[:8]]}")
    return unique


@dataclass(frozen=True)
class PooledFeatureEpisode:
    episode_id: str
    split: SplitName
    timestamps: torch.Tensor
    pooled_g4: torch.Tensor
    camera_ids: tuple[str, ...]
    valid_mask: torch.Tensor | None = None
    action: torch.Tensor | None = None
    proprio: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("`episode_id` must not be empty.")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"Unsupported split `{self.split}`.")
        if self.timestamps.ndim != 1:
            raise ValueError(
                f"`timestamps` must be [T], got {tuple(self.timestamps.shape)}."
            )
        if self.pooled_g4.ndim != 5:
            raise ValueError(
                "`pooled_g4` must be [T,V,C,4,4], "
                f"got {tuple(self.pooled_g4.shape)}."
            )
        ticks, views, _, height, width = self.pooled_g4.shape
        if ticks != self.timestamps.numel():
            raise ValueError(
                f"Timestamp/feature length mismatch: {self.timestamps.numel()} vs {ticks}."
            )
        if (height, width) != (4, 4):
            raise ValueError(f"`pooled_g4` must use a 4x4 grid, got {height}x{width}.")
        if views != len(self.camera_ids):
            raise ValueError(
                f"Feature views ({views}) do not match camera ids ({len(self.camera_ids)})."
            )
        if len(set(self.camera_ids)) != len(self.camera_ids):
            raise ValueError(f"Duplicate camera ids in episode `{self.episode_id}`.")
        if not torch.isfinite(self.timestamps).all():
            raise ValueError(f"Timestamps must be finite in `{self.episode_id}`.")
        if not torch.isfinite(self.pooled_g4).all():
            raise ValueError(f"Pooled features must be finite in `{self.episode_id}`.")
        if ticks > 1 and not torch.all(self.timestamps[1:] > self.timestamps[:-1]):
            raise ValueError(f"Timestamps must be strictly increasing in `{self.episode_id}`.")

        valid_mask = self.valid_mask
        if valid_mask is None:
            valid_mask = torch.ones((ticks, views), dtype=torch.bool)
        if tuple(valid_mask.shape) != (ticks, views):
            raise ValueError(
                f"`valid_mask` must be [T,V]={ticks, views}, got {tuple(valid_mask.shape)}."
            )
        object.__setattr__(self, "camera_ids", tuple(str(value) for value in self.camera_ids))
        object.__setattr__(self, "valid_mask", valid_mask.to(dtype=torch.bool))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def ticks(self) -> int:
        return int(self.pooled_g4.shape[0])

    @property
    def views(self) -> int:
        return int(self.pooled_g4.shape[1])

    @property
    def channels(self) -> int:
        return int(self.pooled_g4.shape[2])

    def pooled(self, grid: int) -> torch.Tensor:
        if grid not in {1, 2, 4}:
            raise ValueError(f"`grid` must be one of 1, 2, 4; got {grid}.")
        if grid == 4:
            return self.pooled_g4
        ticks, views, channels, _, _ = self.pooled_g4.shape
        flat = self.pooled_g4.reshape(ticks * views, channels, 4, 4)
        pooled = F.avg_pool2d(flat, kernel_size=4 // grid, stride=4 // grid)
        return pooled.reshape(ticks, views, channels, grid, grid)

    def to_payload(self) -> dict[str, Any]:
        def cpu(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.detach().cpu()

        return {
            "episode_id": self.episode_id,
            "split": self.split,
            "timestamps": cpu(self.timestamps),
            "pooled_g4": cpu(self.pooled_g4),
            "camera_ids": list(self.camera_ids),
            "valid_mask": cpu(self.valid_mask),
            "action": cpu(self.action),
            "proprio": cpu(self.proprio),
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> PooledFeatureEpisode:
        return cls(
            episode_id=str(payload["episode_id"]),
            split=payload["split"],
            timestamps=payload["timestamps"],
            pooled_g4=payload["pooled_g4"],
            camera_ids=tuple(payload["camera_ids"]),
            valid_mask=payload.get("valid_mask"),
            action=payload.get("action"),
            proprio=payload.get("proprio"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class PooledShardInfo:
    path: Path
    sha256: str
    episodes: int
    ticks: int


def write_pooled_feature_shard(
    path: str | Path,
    episodes: Iterable[PooledFeatureEpisode],
    metadata: dict[str, Any],
) -> PooledShardInfo:
    path = Path(path)
    episode_list = list(episodes)
    if not episode_list:
        raise ValueError("A pooled feature shard must contain at least one episode.")
    keys = [episode.episode_id for episode in episode_list]
    if len(keys) != len(set(keys)):
        raise ValueError("A pooled feature shard cannot contain duplicate episode ids.")
    reference = episode_list[0]
    inconsistent = [
        episode.episode_id
        for episode in episode_list[1:]
        if episode.camera_ids != reference.camera_ids
        or episode.channels != reference.channels
        or episode.views != reference.views
    ]
    if inconsistent:
        raise ValueError(
            "All episodes in one shard must share camera order and feature shape; "
            f"inconsistent episodes: {inconsistent[:8]}."
        )
    missing_metadata = sorted(REQUIRED_SHARD_METADATA - metadata.keys())
    if missing_metadata:
        raise ValueError(f"Missing pooled shard metadata fields: {missing_metadata}.")

    payload = {
        "schema": POOLED_SHARD_SCHEMA,
        "metadata": dict(metadata),
        "episodes": [episode.to_payload() for episode in episode_list],
    }
    atomic_torch_save(payload, path)
    return PooledShardInfo(
        path=path,
        sha256=file_sha256(path),
        episodes=len(episode_list),
        ticks=sum(episode.ticks for episode in episode_list),
    )


def iter_pooled_feature_episodes(
    patterns: Iterable[str | Path],
    split: SplitName | None = None,
    map_location: str | torch.device = "cpu",
) -> Iterator[PooledFeatureEpisode]:
    if split is not None and split not in VALID_SPLITS:
        raise ValueError(f"Unsupported split `{split}`.")
    for path in expand_shard_paths(patterns):
        payload = load_torch_payload(path, map_location=map_location)
        if not isinstance(payload, dict) or payload.get("schema") != POOLED_SHARD_SCHEMA:
            raise ValueError(f"Unsupported pooled feature shard schema in {path}.")
        episodes = payload.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError(f"Shard `{path}` has no episode list.")
        for episode_payload in episodes:
            episode = PooledFeatureEpisode.from_payload(episode_payload)
            if split is None or episode.split == split:
                yield episode
