from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Literal


SplitName = Literal["train", "val", "test"]
GroupBy = Literal["scene", "building", "institution", "episode"]
VALID_SPLITS = frozenset({"train", "val", "test"})


def _stable_unit_interval(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) / float(2**64)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class EpisodeRecord:
    dataset: str
    episode_id: str
    num_steps: int
    source_uri: str
    scene_id: str | None = None
    building_id: str | None = None
    institution_id: str | None = None
    task_ids: tuple[str, ...] = ()
    camera_ids: tuple[str, ...] = ()
    source_checksum: str | None = None
    split: SplitName | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ValueError("`dataset` must not be empty.")
        if not self.episode_id:
            raise ValueError("`episode_id` must not be empty.")
        if int(self.num_steps) <= 0:
            raise ValueError(f"`num_steps` must be positive, got {self.num_steps}.")
        if not self.source_uri:
            raise ValueError("`source_uri` must not be empty.")
        if self.split is not None and self.split not in VALID_SPLITS:
            raise ValueError(f"Unsupported split `{self.split}`.")
        if len(set(self.camera_ids)) != len(self.camera_ids):
            raise ValueError(f"Duplicate camera ids in episode `{self.episode_id}`.")
        object.__setattr__(self, "num_steps", int(self.num_steps))
        object.__setattr__(self, "task_ids", tuple(str(value) for value in self.task_ids))
        object.__setattr__(self, "camera_ids", tuple(str(value) for value in self.camera_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def key(self) -> str:
        return f"{self.dataset}:{self.episode_id}"

    def group_key(self, group_by: GroupBy) -> str:
        prefix = [self.dataset]
        if group_by == "institution" and self.institution_id:
            return "/".join([*prefix, "institution", self.institution_id])
        if group_by in {"institution", "building"} and self.building_id:
            return "/".join(
                [
                    *prefix,
                    "building",
                    self.institution_id or "_",
                    self.building_id,
                ]
            )
        if group_by in {"institution", "building", "scene"} and self.scene_id:
            return "/".join(
                [
                    *prefix,
                    "scene",
                    self.institution_id or "_",
                    self.building_id or "_",
                    self.scene_id,
                ]
            )
        return "/".join([*prefix, "episode", self.episode_id])

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "episode_id": self.episode_id,
            "num_steps": self.num_steps,
            "source_uri": self.source_uri,
            "scene_id": self.scene_id,
            "building_id": self.building_id,
            "institution_id": self.institution_id,
            "task_ids": list(self.task_ids),
            "camera_ids": list(self.camera_ids),
            "source_checksum": self.source_checksum,
            "split": self.split,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EpisodeRecord:
        return cls(
            dataset=str(payload["dataset"]),
            episode_id=str(payload["episode_id"]),
            num_steps=int(payload["num_steps"]),
            source_uri=str(payload["source_uri"]),
            scene_id=payload.get("scene_id"),
            building_id=payload.get("building_id"),
            institution_id=payload.get("institution_id"),
            task_ids=tuple(payload.get("task_ids", ())),
            camera_ids=tuple(payload.get("camera_ids", ())),
            source_checksum=payload.get("source_checksum"),
            split=payload.get("split"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    group_by: GroupBy = "scene"
    stratify_by_task: bool = True
    salt: str = "codewam-v1"

    def __post_init__(self) -> None:
        fractions = (self.train_fraction, self.val_fraction, self.test_fraction)
        if any(value < 0.0 for value in fractions):
            raise ValueError(f"Split fractions must be non-negative, got {fractions}.")
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError(f"Split fractions must sum to 1, got {fractions}.")
        if self.group_by not in {"scene", "building", "institution", "episode"}:
            raise ValueError(f"Unsupported group_by `{self.group_by}`.")
        if not self.salt:
            raise ValueError("`salt` must not be empty.")


@dataclass(frozen=True)
class EpisodeManifest:
    records: tuple[EpisodeRecord, ...]

    def __post_init__(self) -> None:
        records = tuple(self.records)
        keys = [record.key for record in records]
        duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
        if duplicates:
            raise ValueError(f"Duplicate manifest episode keys: {duplicates[:8]}.")
        object.__setattr__(self, "records", records)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def fingerprint(self) -> str:
        canonical = "\n".join(
            json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
            for record in sorted(self.records, key=lambda item: item.key)
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def select(self, split: SplitName) -> EpisodeManifest:
        if split not in VALID_SPLITS:
            raise ValueError(f"Unsupported split `{split}`.")
        return EpisodeManifest(tuple(record for record in self.records if record.split == split))

    def assign_splits(self, config: SplitConfig = SplitConfig()) -> EpisodeManifest:
        grouped: dict[str, list[EpisodeRecord]] = defaultdict(list)
        for record in self.records:
            grouped[record.group_key(config.group_by)].append(record)

        assignments: dict[str, SplitName] = {}
        train_end = config.train_fraction
        val_end = train_end + config.val_fraction
        for group_key, group_records in sorted(grouped.items()):
            stratum = "_"
            if config.stratify_by_task:
                task_counts = Counter(
                    task_id
                    for record in group_records
                    for task_id in record.task_ids
                )
                if task_counts:
                    max_count = max(task_counts.values())
                    stratum = min(
                        task_id for task_id, count in task_counts.items() if count == max_count
                    )
            score = _stable_unit_interval(f"{config.salt}|{stratum}|{group_key}")
            if score < train_end:
                assignments[group_key] = "train"
            elif score < val_end:
                assignments[group_key] = "val"
            else:
                assignments[group_key] = "test"

        assigned = tuple(
            replace(record, split=assignments[record.group_key(config.group_by)])
            for record in self.records
        )
        manifest = EpisodeManifest(assigned)
        manifest.assert_group_isolation(config.group_by)
        return manifest

    def assert_group_isolation(self, group_by: GroupBy = "scene") -> None:
        group_splits: dict[str, set[str]] = defaultdict(set)
        for record in self.records:
            if record.split is None:
                raise ValueError(f"Episode `{record.key}` has no split assignment.")
            group_splits[record.group_key(group_by)].add(record.split)
        leaked = sorted(key for key, splits in group_splits.items() if len(splits) > 1)
        if leaked:
            raise ValueError(f"Groups span multiple splits: {leaked[:8]}.")

    def stats(self) -> dict[str, Any]:
        split_counts = Counter(record.split or "unassigned" for record in self.records)
        return {
            "episodes": len(self.records),
            "steps": sum(record.num_steps for record in self.records),
            "datasets": dict(sorted(Counter(record.dataset for record in self.records).items())),
            "splits": dict(sorted(split_counts.items())),
            "fingerprint": self.fingerprint(),
        }

    def write_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        lines = [
            json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":"))
            for record in sorted(self.records, key=lambda item: item.key)
        ]
        _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))

    @classmethod
    def read_jsonl(cls, path: str | Path) -> EpisodeManifest:
        path = Path(path)
        records: list[EpisodeRecord] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            records.append(EpisodeRecord.from_dict(payload))
        return cls(tuple(records))

    @classmethod
    def from_records(cls, records: Iterable[EpisodeRecord]) -> EpisodeManifest:
        return cls(tuple(records))
