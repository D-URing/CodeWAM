from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from codewam.codebook_eval.shards import (
    atomic_torch_save,
    file_sha256,
    load_torch_payload,
)

from .droid_endpoint import DROID_ENDPOINT_POLICY
from .droid_rlds import DROID_ACTION_COMPONENT_DIMS
from .joint_cache import JointEpisode


DROID_ACTION_TARGET_CACHE_SCHEMA = "codewam.droid-action-target-cache.v1"
DROID_ACTION_TARGET_SHARD_SCHEMA = "codewam.droid-action-target-shard.v1"
DROID_ACTION_TARGET_SUMMARY_SCHEMA = "codewam.droid-action-target-summary.v1"
DROID_ACTION_TARGET_INDEX_SCHEMA = "codewam.droid-action-target-index.v1"
DROID_ACTION_EXTRACTION_POLICY = "same-rlds-position/same-half-open-keep-range-v1"
DROID_FLAT_ACTION_COMPONENTS = (
    "cartesian_position",
    "gripper_position",
)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        mode = (path.stat().st_mode & 0o777) if path.exists() else 0o644
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
        mode = (path.stat().st_mode & 0o777) if path.exists() else 0o644
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_droid_action_target_contract(
    *,
    joint_cache_contract_hash: str,
    joint_cache_summary_sha256: str,
    source_manifest_fingerprint: str,
    source_manifest_sha256: str,
    dataset_revision: str,
    implementation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    required = (
        joint_cache_contract_hash,
        joint_cache_summary_sha256,
        source_manifest_fingerprint,
        source_manifest_sha256,
        dataset_revision,
    )
    if any(not str(value) for value in required):
        raise ValueError("Action-target provenance strings must be nonempty.")
    if not implementation_sha256:
        raise ValueError("Action-target implementation hashes must not be empty.")
    payload = {
        "schema": DROID_ACTION_TARGET_CACHE_SCHEMA,
        "joint_cache": {
            "contract_hash": str(joint_cache_contract_hash),
            "summary_sha256": str(joint_cache_summary_sha256),
        },
        "source_manifest": {
            "fingerprint": str(source_manifest_fingerprint),
            "sha256": str(source_manifest_sha256),
            "dataset_revision": str(dataset_revision),
        },
        "endpoint": {
            "policy": DROID_ENDPOINT_POLICY,
            "last_step_action_valid": False,
        },
        "extraction_policy": DROID_ACTION_EXTRACTION_POLICY,
        "storage_dtype": "float32",
        "sources": {
            "flat_action": {"feature": "action", "dim": 7},
            "action_dict": [
                {"name": name, "dim": width}
                for name, width in DROID_ACTION_COMPONENT_DIMS.items()
            ],
        },
        "target_selection": "unselected-controller-contract-required",
        "implementation_sha256": {
            str(name): str(value)
            for name, value in sorted(implementation_sha256.items())
        },
    }
    return {**payload, "contract_hash": _canonical_hash(payload)}


def validate_droid_action_target_contract(contract: Mapping[str, Any]) -> None:
    payload = {
        key: value for key, value in contract.items() if key != "contract_hash"
    }
    if (
        contract.get("schema") != DROID_ACTION_TARGET_CACHE_SCHEMA
        or contract.get("contract_hash") != _canonical_hash(payload)
    ):
        raise RuntimeError("DROID action-target contract is invalid.")
    source_rows = contract.get("sources", {}).get("action_dict")
    component_dims = {
        str(row.get("name")): int(row.get("dim", 0))
        for row in source_rows or ()
        if isinstance(row, dict)
    }
    if (
        contract.get("extraction_policy") != DROID_ACTION_EXTRACTION_POLICY
        or contract.get("storage_dtype") != "float32"
        or contract.get("endpoint", {}).get("policy") != DROID_ENDPOINT_POLICY
        or contract.get("endpoint", {}).get("last_step_action_valid") is not False
        or contract.get("sources", {}).get("flat_action", {}).get("dim") != 7
        or component_dims != DROID_ACTION_COMPONENT_DIMS
        or contract.get("target_selection")
        != "unselected-controller-contract-required"
    ):
        raise ValueError("DROID action-target contract fields are invalid.")


def write_droid_action_target_contract(
    output_dir: str | Path,
    contract: Mapping[str, Any],
) -> Path:
    validate_droid_action_target_contract(contract)
    path = Path(output_dir) / "contract.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != dict(contract):
            raise RuntimeError("Existing DROID action-target contract differs.")
        return path
    _atomic_json(path, contract)
    return path


@dataclass(frozen=True)
class DroidActionTargetSegment:
    episode_id: str
    parent_episode_id: str
    manifest_key: str
    range_index: int
    range_start: int
    range_stop: int
    split: str
    source_shard: str
    record_index: int
    flat_action: torch.Tensor
    action_components: dict[str, torch.Tensor]
    action_valid: torch.Tensor

    def __post_init__(self) -> None:
        if (
            not self.episode_id
            or not self.parent_episode_id
            or not self.manifest_key
            or not self.source_shard
        ):
            raise ValueError("Action-target segment identity must not be empty.")
        expected_id = (
            f"{self.parent_episode_id}@{self.range_start}:{self.range_stop}"
        )
        if self.episode_id != expected_id:
            raise ValueError("Action-target segment ID differs from its range.")
        if (
            self.range_index < 0
            or self.range_start < 0
            or self.range_stop <= self.range_start
            or self.record_index < 0
        ):
            raise ValueError("Action-target segment range metadata is invalid.")
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported action-target split `{self.split}`.")
        steps = self.range_stop - self.range_start
        if (
            self.flat_action.dtype != torch.float32
            or tuple(self.flat_action.shape) != (steps, 7)
            or not torch.isfinite(self.flat_action).all()
        ):
            raise ValueError("Flat action must be finite float32 [T,7].")
        components = dict(self.action_components)
        if set(components) != set(DROID_ACTION_COMPONENT_DIMS):
            raise ValueError("Action-target component schema changed.")
        for name, width in DROID_ACTION_COMPONENT_DIMS.items():
            value = components[name]
            if (
                value.dtype != torch.float32
                or tuple(value.shape) != (steps, width)
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"Action-target `{name}` must be finite float32 "
                    f"[{steps},{width}]."
                )
        if (
            self.action_valid.dtype != torch.bool
            or tuple(self.action_valid.shape) != (steps,)
        ):
            raise ValueError("Action-target validity must be bool [T].")
        object.__setattr__(self, "action_components", components)

    @property
    def source_steps(self) -> int:
        return self.range_stop - self.range_start

    def to_payload(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "parent_episode_id": self.parent_episode_id,
            "manifest_key": self.manifest_key,
            "range_index": self.range_index,
            "range_start": self.range_start,
            "range_stop": self.range_stop,
            "split": self.split,
            "source_shard": self.source_shard,
            "record_index": self.record_index,
            "flat_action": self.flat_action.detach().float().cpu().contiguous(),
            "action_components": {
                name: value.detach().float().cpu().contiguous()
                for name, value in self.action_components.items()
            },
            "action_valid": self.action_valid.detach().bool().cpu().contiguous(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DroidActionTargetSegment:
        return cls(
            episode_id=str(payload["episode_id"]),
            parent_episode_id=str(payload["parent_episode_id"]),
            manifest_key=str(payload["manifest_key"]),
            range_index=int(payload["range_index"]),
            range_start=int(payload["range_start"]),
            range_stop=int(payload["range_stop"]),
            split=str(payload["split"]),
            source_shard=str(payload["source_shard"]),
            record_index=int(payload["record_index"]),
            flat_action=payload["flat_action"],
            action_components=dict(payload["action_components"]),
            action_valid=payload["action_valid"],
        )


def action_target_mapping_statistics(
    segments: Sequence[DroidActionTargetSegment],
) -> dict[str, Any]:
    if not segments:
        raise ValueError("Action mapping statistics need at least one segment.")
    rows = 0
    exact_values = 0
    absolute_sum = 0.0
    squared_sum = 0.0
    maximum = 0.0
    for segment in segments:
        candidate = torch.cat(
            tuple(
                segment.action_components[name]
                for name in DROID_FLAT_ACTION_COMPONENTS
            ),
            dim=-1,
        )
        difference = (segment.flat_action.double() - candidate.double()).abs()
        rows += segment.source_steps
        exact_values += int((difference == 0).sum())
        absolute_sum += float(difference.sum())
        squared_sum += float(difference.square().sum())
        maximum = max(maximum, float(difference.max()))
    values = rows * 7
    return {
        "rows": rows,
        "values": values,
        "exact_values": exact_values,
        "absolute_error_sum": absolute_sum,
        "squared_error_sum": squared_sum,
        "max_abs_error": maximum,
    }


def validate_action_targets_against_joint_episodes(
    segments: Sequence[DroidActionTargetSegment],
    episodes: Sequence[JointEpisode],
) -> dict[str, Any]:
    target_by_id = {segment.episode_id: segment for segment in segments}
    joint_by_id = {episode.episode_id: episode for episode in episodes}
    if len(target_by_id) != len(segments) or len(joint_by_id) != len(episodes):
        raise RuntimeError("Action/joint shard contains duplicate segment IDs.")
    if set(target_by_id) != set(joint_by_id):
        raise RuntimeError("Action and joint shards contain different segments.")
    rows = 0
    invalid_rows = 0
    for episode_id in sorted(target_by_id):
        target = target_by_id[episode_id]
        joint = joint_by_id[episode_id]
        identity = (
            target.parent_episode_id == joint.parent_episode_id
            and target.manifest_key == joint.manifest_key
            and target.range_index == joint.range_index
            and target.range_start == joint.range_start
            and target.range_stop == joint.range_stop
            and target.split == joint.split
        )
        if not identity:
            raise RuntimeError(
                f"Action/joint identity differs for `{episode_id}`."
            )
        if not torch.equal(target.flat_action, joint.source_actions):
            raise RuntimeError(
                f"Action/joint flat values differ for `{episode_id}`."
            )
        if not torch.equal(target.action_valid, joint.source_action_valid):
            raise RuntimeError(
                f"Action/joint validity differs for `{episode_id}`."
            )
        rows += target.source_steps
        invalid_rows += int((~target.action_valid).sum())
    return {
        "segments": len(segments),
        "rows": rows,
        "invalid_rows": invalid_rows,
        "flat_action_exact": True,
        "action_valid_exact": True,
    }


def write_droid_action_target_shard(
    path: str | Path,
    *,
    contract_hash: str,
    segments: Sequence[DroidActionTargetSegment],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(path)
    rows = tuple(segments)
    if not rows:
        raise ValueError("DROID action-target shard must not be empty.")
    identifiers = [segment.episode_id for segment in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("DROID action-target shard repeats segment IDs.")
    if path.exists():
        raise FileExistsError(f"DROID action-target shard exists: {path}.")
    payload = {
        "schema": DROID_ACTION_TARGET_SHARD_SCHEMA,
        "contract_hash": str(contract_hash),
        "metadata": dict(metadata),
        "segments": [segment.to_payload() for segment in rows],
    }
    atomic_torch_save(payload, path)
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "segments": len(rows),
        "source_steps": sum(segment.source_steps for segment in rows),
    }


def validate_droid_action_target_shard(
    path: str | Path,
    *,
    contract_hash: str,
    expected_sha256: str | None = None,
) -> tuple[tuple[DroidActionTargetSegment, ...], dict[str, Any]]:
    path = Path(path)
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise RuntimeError(f"DROID action-target shard SHA-256 changed: {path}.")
    payload = load_torch_payload(path, map_location="cpu")
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != DROID_ACTION_TARGET_SHARD_SCHEMA
        or payload.get("contract_hash") != contract_hash
    ):
        raise RuntimeError(f"DROID action-target shard is invalid: {path}.")
    segments = tuple(
        DroidActionTargetSegment.from_payload(row)
        for row in payload.get("segments", ())
    )
    if not segments:
        raise RuntimeError(f"DROID action-target shard is empty: {path}.")
    identifiers = [segment.episode_id for segment in segments]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError(f"DROID action-target shard repeats IDs: {path}.")
    return segments, dict(payload.get("metadata", {}))


def write_droid_action_target_index(
    output_dir: str | Path,
    *,
    contract_hash: str,
    file_rows: Sequence[Mapping[str, Any]],
    segment_rows: Sequence[Mapping[str, Any]],
    mapping_statistics: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    ordered_segments = sorted(
        (dict(row) for row in segment_rows),
        key=lambda row: str(row["episode_id"]),
    )
    identifiers = [str(row["episode_id"]) for row in ordered_segments]
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("DROID action-target index segment IDs are invalid.")
    index_path = output_dir / "episodes.jsonl"
    _atomic_jsonl(index_path, ordered_segments)
    rows = sorted(
        (dict(row) for row in file_rows),
        key=lambda row: str(row["path"]),
    )
    summary = {
        "schema": DROID_ACTION_TARGET_SUMMARY_SCHEMA,
        "contract_hash": str(contract_hash),
        "episodes": len(ordered_segments),
        "source_steps": sum(int(row["source_steps"]) for row in rows),
        "shards": len(rows),
        "indices": {
            "episodes": {
                "path": index_path.name,
                "sha256": file_sha256(index_path),
            }
        },
        "flat_action_mapping": dict(mapping_statistics),
        "files": rows,
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


class FrozenDroidActionTargetCache:
    """Read-only raw action alternatives bound to one joint cache contract."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_joint_cache_contract_hash: str | None = None,
        verify_hashes: bool = True,
        max_cached_shards: int = 2,
    ):
        self.root = Path(root)
        self.contract = json.loads(
            (self.root / "contract.json").read_text(encoding="utf-8")
        )
        validate_droid_action_target_contract(self.contract)
        actual_joint = str(self.contract["joint_cache"]["contract_hash"])
        if (
            expected_joint_cache_contract_hash is not None
            and actual_joint != expected_joint_cache_contract_hash
        ):
            raise RuntimeError("Action targets belong to a different joint cache.")
        self.summary = json.loads(
            (self.root / "summary.json").read_text(encoding="utf-8")
        )
        if (
            self.summary.get("schema") != DROID_ACTION_TARGET_SUMMARY_SCHEMA
            or self.summary.get("contract_hash") != self.contract["contract_hash"]
        ):
            raise RuntimeError("DROID action-target summary is invalid.")
        index_row = self.summary.get("indices", {}).get("episodes", {})
        index_path = self.root / str(index_row.get("path", ""))
        if verify_hashes and file_sha256(index_path) != index_row.get("sha256"):
            raise RuntimeError("DROID action-target index hash changed.")
        self._locators: dict[str, dict[str, Any]] = {}
        with index_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                episode_id = str(row["episode_id"])
                if episode_id in self._locators:
                    raise RuntimeError("DROID action-target index repeats IDs.")
                self._locators[episode_id] = row
        if len(self._locators) != int(self.summary["episodes"]):
            raise RuntimeError("DROID action-target episode count changed.")
        self.max_cached_shards = int(max_cached_shards)
        if self.max_cached_shards <= 0:
            raise ValueError("Action-target shard cache size must be positive.")
        self.verify_hashes = bool(verify_hashes)
        self._shards: OrderedDict[
            str, tuple[DroidActionTargetSegment, ...]
        ] = OrderedDict()
        self._file_sha256 = {
            str(row["path"]): str(row["sha256"])
            for row in self.summary["files"]
        }
        self._verified: set[str] = set()

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return tuple(self._locators)

    def segment(self, episode_id: str) -> DroidActionTargetSegment:
        try:
            locator = self._locators[episode_id]
        except KeyError as exc:
            raise KeyError(f"Unknown DROID action-target segment `{episode_id}`.") from exc
        relative = str(locator["shard"])
        segments = self._shards.pop(relative, None)
        if segments is None:
            segments, _ = validate_droid_action_target_shard(
                self.root / relative,
                contract_hash=str(self.contract["contract_hash"]),
                expected_sha256=(
                    self._file_sha256[relative]
                    if self.verify_hashes and relative not in self._verified
                    else None
                ),
            )
            self._verified.add(relative)
        self._shards[relative] = segments
        while len(self._shards) > self.max_cached_shards:
            self._shards.popitem(last=False)
        offset = int(locator["offset"])
        if offset < 0 or offset >= len(segments):
            raise RuntimeError(f"Invalid action-target offset for `{episode_id}`.")
        segment = segments[offset]
        if segment.episode_id != episode_id:
            raise RuntimeError(f"Action-target locator differs for `{episode_id}`.")
        return segment
