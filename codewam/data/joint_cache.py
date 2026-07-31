from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch

from codewam.codebook_eval.manifest import VALID_SPLITS
from codewam.codebook_eval.shards import (
    atomic_torch_save,
    file_sha256,
    load_torch_payload,
)
from codewam.models.contracts import (
    ActionBatch,
    CodeMeasurements,
    CodeWAMBatch,
    FutureCodeTargets,
    PolicyCondition,
    StateInputs,
)

from .droid_endpoint import DROID_ENDPOINT_POLICY
from .frozen_assignment import FrozenArtifactChart
from .roles import TrajectoryRole, build_supervision_masks


JOINT_CACHE_SCHEMA = "codewam.joint-window-cache.v1"
JOINT_EPISODE_SHARD_SCHEMA = "codewam.joint-episode-shard.v1"
JOINT_SHARD_INDEX_SCHEMA = "codewam.joint-shard-index.v1"
JOINT_CACHE_SUMMARY_SCHEMA = "codewam.joint-cache-summary.v1"
JOINT_ACTION_INDEX_SCHEMA = "codewam.joint-action-index.v1"
JOINT_WINDOW_INDEX_SCHEMA = "codewam.joint-window-index.v1"

_JOINT_SPLITS = ("train", "val", "test")
_JOINT_WINDOW_ID_BYTES = 64
_JOINT_WINDOW_SCALAR_FIELDS = (
    "state_latent_start",
    "state_latent_stop",
    "current_latent_index",
    "future_latent_index",
    "proprio_start",
    "proprio_stop",
    "past_action_start",
    "past_action_stop",
    "action_start",
    "action_stop",
    "decision_source_index",
    "future_observation_source_index",
)


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        mode = (path.stat().st_mode & 0o777) if path.exists() else 0o644
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
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


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}.") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}.")
            yield payload


@dataclass(frozen=True)
class JointWindowConfig:
    action_horizon: int = 16
    state_latent_ticks: int = 8
    proprio_history_steps: int = 16
    past_action_steps: int = 16
    window_stride_ticks: int = 1
    require_all_code_families: bool = True
    require_full_history: bool = True

    def __post_init__(self) -> None:
        dimensions = (
            self.action_horizon,
            self.state_latent_ticks,
            self.proprio_history_steps,
            self.past_action_steps,
            self.window_stride_ticks,
        )
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("Joint window dimensions must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_horizon": self.action_horizon,
            "state_latent_ticks": self.state_latent_ticks,
            "proprio_history_steps": self.proprio_history_steps,
            "past_action_steps": self.past_action_steps,
            "window_stride_ticks": self.window_stride_ticks,
            "require_all_code_families": self.require_all_code_families,
            "require_full_history": self.require_full_history,
        }


def create_joint_cache_contract(
    *,
    dataset_revision: str,
    source_manifest_fingerprint: str,
    source_manifest_sha256: str,
    endpoint_audit_sha256: str,
    chart: FrozenArtifactChart,
    camera_ids: Sequence[str],
    wan_model_id: str,
    wan_revision: str,
    preprocess_revision: str,
    nominal_fps: float,
    action_dim: int,
    proprio_dim: int,
    latent_channels: int,
    window: JointWindowConfig,
    language_encoder_id: str | None = None,
    language_encoder_revision: str | None = None,
    language_dim: int | None = None,
    implementation_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    strings = (
        dataset_revision,
        source_manifest_fingerprint,
        source_manifest_sha256,
        endpoint_audit_sha256,
        wan_model_id,
        wan_revision,
        preprocess_revision,
    )
    if any(not value for value in strings):
        raise ValueError("Joint cache provenance strings must not be empty.")
    ordered_cameras = tuple(str(value) for value in camera_ids)
    if not ordered_cameras or len(set(ordered_cameras)) != len(ordered_cameras):
        raise ValueError("Joint cache camera IDs must be nonempty and unique.")
    if nominal_fps <= 0 or min(action_dim, proprio_dim, latent_channels) <= 0:
        raise ValueError("Joint cache dimensions and FPS must be positive.")
    language_values = (
        language_encoder_id,
        language_encoder_revision,
        language_dim,
    )
    if any(value is not None for value in language_values) and any(
        value is None for value in language_values
    ):
        raise ValueError(
            "Language encoder id, revision and dimension must be set together."
        )
    chart_identity = chart.compact_identity()
    training = chart_identity["training_provenance"]
    expected = {
        "wan_model_id": wan_model_id,
        "wan_revision": wan_revision,
        "preprocess_revision": preprocess_revision,
    }
    mismatches = [
        key for key, value in expected.items() if training.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"Joint cache Wan provenance differs from its chart in {mismatches}."
        )
    payload = {
        "schema": JOINT_CACHE_SCHEMA,
        "dataset_revision": dataset_revision,
        "source_manifest_fingerprint": source_manifest_fingerprint,
        "source_manifest_sha256": source_manifest_sha256,
        "endpoint": {
            "policy": DROID_ENDPOINT_POLICY,
            "audit_sha256": endpoint_audit_sha256,
            "last_step_action_valid": False,
        },
        "chart": chart_identity,
        "camera_ids": list(ordered_cameras),
        "wan_model_id": wan_model_id,
        "wan_revision": wan_revision,
        "preprocess_revision": preprocess_revision,
        "nominal_fps": float(nominal_fps),
        "action_dim": int(action_dim),
        "proprio_dim": int(proprio_dim),
        "latent_channels": int(latent_channels),
        "window": window.to_dict(),
        "language_encoder": (
            None
            if language_encoder_id is None
            else {
                "id": language_encoder_id,
                "revision": language_encoder_revision,
                "dim": int(language_dim),
            }
        ),
        "implementation_sha256": {
            name: file_sha256(path)
            for name, path in sorted((implementation_paths or {}).items())
        },
    }
    return {**payload, "contract_hash": _canonical_hash(payload)}


def validate_joint_cache_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema") != JOINT_CACHE_SCHEMA:
        raise ValueError("Unsupported joint cache contract schema.")
    payload = {
        key: value for key, value in contract.items() if key != "contract_hash"
    }
    if contract.get("contract_hash") != _canonical_hash(payload):
        raise RuntimeError("Joint cache contract hash is invalid.")
    endpoint = contract.get("endpoint")
    if (
        not isinstance(endpoint, dict)
        or endpoint.get("policy") != DROID_ENDPOINT_POLICY
        or endpoint.get("last_step_action_valid") is not False
    ):
        raise ValueError("Joint cache uses an unsupported endpoint contract.")


def write_joint_cache_contract(
    output_dir: str | Path,
    contract: Mapping[str, Any],
) -> Path:
    validate_joint_cache_contract(contract)
    path = Path(output_dir) / "contract.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != dict(contract):
            raise RuntimeError("Existing joint cache contract differs.")
        return path
    _write_json(path, dict(contract))
    return path


@dataclass(frozen=True)
class JointEpisode:
    episode_id: str
    parent_episode_id: str
    manifest_key: str
    range_index: int
    range_start: int
    range_stop: int
    split: str
    chart_name: str
    role: TrajectoryRole | str
    camera_ids: tuple[str, ...]
    latents: torch.Tensor
    latent_source_indices: torch.Tensor
    latent_valid: torch.Tensor
    source_actions: torch.Tensor
    source_proprio: torch.Tensor
    source_action_valid: torch.Tensor
    code_ids: torch.Tensor
    code_available: torch.Tensor
    descriptor_source_indices: torch.Tensor
    families: tuple[str, ...]
    language_instruction: str
    language_tokens: torch.Tensor | None = None
    language_valid: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id or not self.parent_episode_id or not self.manifest_key:
            raise ValueError("Joint episode identity fields must not be empty.")
        if self.range_index < 0 or self.range_start < 0:
            raise ValueError("Joint episode range indices must be non-negative.")
        if self.range_stop <= self.range_start:
            raise ValueError("Joint episode source range must be nonempty.")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"Unsupported joint episode split `{self.split}`.")
        role = TrajectoryRole(self.role)
        cameras = tuple(str(value) for value in self.camera_ids)
        families = tuple(str(value) for value in self.families)
        if not self.chart_name or not cameras or not families:
            raise ValueError("Joint episode chart, cameras and families are required.")
        if len(set(cameras)) != len(cameras) or len(set(families)) != len(families):
            raise ValueError("Joint episode camera/family identities must be unique.")
        if self.latents.ndim != 5:
            raise ValueError("Joint latents must be [T,V,C,H,W].")
        ticks, views = self.latents.shape[:2]
        if views != len(cameras):
            raise ValueError("Joint latent views do not match camera IDs.")
        if (
            self.latent_source_indices.dtype != torch.long
            or tuple(self.latent_source_indices.shape) != (ticks,)
        ):
            raise ValueError("Joint latent source indices must be long [T].")
        if ticks > 1 and not torch.all(
            self.latent_source_indices[1:]
            > self.latent_source_indices[:-1]
        ):
            raise ValueError("Joint latent source indices must increase.")
        if (
            int(self.latent_source_indices.min()) < self.range_start
            or int(self.latent_source_indices.max()) >= self.range_stop
        ):
            raise ValueError("Joint latent source indices leave the keep range.")
        if (
            self.latent_valid.dtype != torch.bool
            or tuple(self.latent_valid.shape) != (ticks, views)
        ):
            raise ValueError("Joint latent validity must be bool [T,V].")
        source_steps = self.range_stop - self.range_start
        for name, value in (
            ("source_actions", self.source_actions),
            ("source_proprio", self.source_proprio),
        ):
            if value.ndim != 2 or int(value.shape[0]) != source_steps:
                raise ValueError(f"Joint {name} must be [T_source,D].")
            if not torch.isfinite(value).all():
                raise ValueError(f"Joint {name} contains NaN or Inf.")
        if (
            self.source_action_valid.dtype != torch.bool
            or tuple(self.source_action_valid.shape) != (source_steps,)
        ):
            raise ValueError("Joint source action validity must be bool [T_source].")
        if (
            self.code_ids.dtype != torch.long
            or self.code_ids.ndim != 3
            or tuple(self.code_ids.shape[:2]) != (ticks, len(families))
        ):
            raise ValueError("Joint code IDs must be long [T,F,L].")
        if (
            self.code_available.dtype != torch.bool
            or tuple(self.code_available.shape) != (ticks, len(families))
        ):
            raise ValueError("Joint code availability must be bool [T,F].")
        if (
            self.descriptor_source_indices.dtype != torch.long
            or tuple(self.descriptor_source_indices.shape)
            != (ticks, len(families), 3)
        ):
            raise ValueError("Joint descriptor sources must be long [T,F,3].")
        unavailable = ~self.code_available
        if unavailable.any() and (
            not torch.all(self.code_ids[unavailable] == -1)
            or not torch.all(self.descriptor_source_indices[unavailable] == -1)
        ):
            raise ValueError("Unavailable joint codes must use -1 sentinels.")
        available_positions = self.code_available.nonzero(as_tuple=False)
        if available_positions.numel():
            time_index = available_positions[:, 0]
            family_index = available_positions[:, 1]
            endpoint = self.descriptor_source_indices[
                time_index,
                family_index,
                2,
            ]
            if not torch.equal(endpoint, self.latent_source_indices[time_index]):
                raise ValueError(
                    "A causal descriptor endpoint must equal its latent source."
                )
        if not torch.isfinite(self.latents).all():
            raise ValueError("Joint latents contain NaN or Inf.")
        if self.language_tokens is None:
            if self.language_valid is not None:
                raise ValueError("Language validity requires cached language tokens.")
        else:
            if self.language_tokens.ndim != 2:
                raise ValueError("Language tokens must be [N,D].")
            if not torch.isfinite(self.language_tokens).all():
                raise ValueError("Language tokens contain NaN or Inf.")
            valid = self.language_valid
            if valid is None:
                valid = torch.ones(
                    self.language_tokens.shape[0],
                    dtype=torch.bool,
                )
            if (
                valid.dtype != torch.bool
                or tuple(valid.shape) != (self.language_tokens.shape[0],)
            ):
                raise ValueError("Language validity must be bool [N].")
            object.__setattr__(self, "language_valid", valid)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "camera_ids", cameras)
        object.__setattr__(self, "families", families)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def latent_ticks(self) -> int:
        return int(self.latents.shape[0])

    @property
    def source_steps(self) -> int:
        return self.range_stop - self.range_start

    def to_payload(self) -> dict[str, Any]:
        def cpu(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.detach().cpu()

        return {
            "episode_id": self.episode_id,
            "parent_episode_id": self.parent_episode_id,
            "manifest_key": self.manifest_key,
            "range_index": self.range_index,
            "range_start": self.range_start,
            "range_stop": self.range_stop,
            "split": self.split,
            "chart_name": self.chart_name,
            "role": self.role.value,
            "camera_ids": list(self.camera_ids),
            "latents": cpu(self.latents),
            "latent_source_indices": cpu(self.latent_source_indices),
            "latent_valid": cpu(self.latent_valid),
            "source_actions": cpu(self.source_actions),
            "source_proprio": cpu(self.source_proprio),
            "source_action_valid": cpu(self.source_action_valid),
            "code_ids": cpu(self.code_ids),
            "code_available": cpu(self.code_available),
            "descriptor_source_indices": cpu(
                self.descriptor_source_indices
            ),
            "families": list(self.families),
            "language_instruction": self.language_instruction,
            "language_tokens": cpu(self.language_tokens),
            "language_valid": cpu(self.language_valid),
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> JointEpisode:
        return cls(
            episode_id=str(payload["episode_id"]),
            parent_episode_id=str(payload["parent_episode_id"]),
            manifest_key=str(payload["manifest_key"]),
            range_index=int(payload["range_index"]),
            range_start=int(payload["range_start"]),
            range_stop=int(payload["range_stop"]),
            split=str(payload["split"]),
            chart_name=str(payload["chart_name"]),
            role=str(payload["role"]),
            camera_ids=tuple(payload["camera_ids"]),
            latents=payload["latents"],
            latent_source_indices=payload["latent_source_indices"],
            latent_valid=payload["latent_valid"],
            source_actions=payload["source_actions"],
            source_proprio=payload["source_proprio"],
            source_action_valid=payload["source_action_valid"],
            code_ids=payload["code_ids"],
            code_available=payload["code_available"],
            descriptor_source_indices=payload[
                "descriptor_source_indices"
            ],
            families=tuple(payload["families"]),
            language_instruction=str(payload.get("language_instruction", "")),
            language_tokens=payload.get("language_tokens"),
            language_valid=payload.get("language_valid"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class JointWindowRecord:
    window_id: str
    episode_id: str
    parent_episode_id: str
    split: str
    chart_name: str
    role: TrajectoryRole | str
    families: tuple[str, ...]
    state_latent_start: int
    state_latent_stop: int
    current_latent_index: int
    future_latent_index: int
    proprio_start: int
    proprio_stop: int
    past_action_start: int
    past_action_stop: int
    action_start: int
    action_stop: int
    decision_source_index: int
    future_observation_source_index: int
    current_code_ids: tuple[tuple[int, ...], ...]
    future_code_ids: tuple[tuple[int, ...], ...]
    code_available: tuple[bool, ...]
    current_descriptor_sources: tuple[tuple[int, int, int], ...]
    future_descriptor_sources: tuple[tuple[int, int, int], ...]
    descriptor_overlap: tuple[int, ...]
    artifact_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.window_id
            or not self.episode_id
            or not self.parent_episode_id
            or not self.chart_name
        ):
            raise ValueError("Joint window identity fields must not be empty.")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"Unsupported joint window split `{self.split}`.")
        role = TrajectoryRole(self.role)
        families = tuple(str(value) for value in self.families)
        family_count = len(families)
        aligned = (
            self.current_code_ids,
            self.future_code_ids,
            self.code_available,
            self.current_descriptor_sources,
            self.future_descriptor_sources,
            self.descriptor_overlap,
            self.artifact_sha256,
        )
        if not families or any(len(value) != family_count for value in aligned):
            raise ValueError("Joint window family fields are not aligned.")
        intervals = (
            (self.state_latent_start, self.state_latent_stop),
            (self.proprio_start, self.proprio_stop),
            (self.past_action_start, self.past_action_stop),
            (self.action_start, self.action_stop),
        )
        if any(start < 0 or stop <= start for start, stop in intervals):
            raise ValueError("Joint window slices must be nonempty half-open ranges.")
        if self.current_latent_index != self.state_latent_stop - 1:
            raise ValueError("Current latent must be the final state latent.")
        if self.future_latent_index <= self.current_latent_index:
            raise ValueError("Future latent must follow the current latent.")
        if self.past_action_stop != self.action_start:
            raise ValueError("Past actions must stop where the target chunk starts.")
        if self.proprio_stop != self.action_start + 1:
            raise ValueError("Proprio history must end at the decision observation.")
        if (
            self.future_observation_source_index
            <= self.decision_source_index
        ):
            raise ValueError("Future observation must follow the decision.")
        if self.future_observation_source_index - self.decision_source_index != (
            self.action_stop - self.action_start
        ):
            raise ValueError("Action horizon and observation endpoint disagree.")
        if any(value < 0 or value > 3 for value in self.descriptor_overlap):
            raise ValueError("Descriptor overlap must be in [0,3].")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "families", families)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "episode_id": self.episode_id,
            "parent_episode_id": self.parent_episode_id,
            "split": self.split,
            "chart_name": self.chart_name,
            "role": self.role.value,
            "families": list(self.families),
            "state_latent_start": self.state_latent_start,
            "state_latent_stop": self.state_latent_stop,
            "current_latent_index": self.current_latent_index,
            "future_latent_index": self.future_latent_index,
            "proprio_start": self.proprio_start,
            "proprio_stop": self.proprio_stop,
            "past_action_start": self.past_action_start,
            "past_action_stop": self.past_action_stop,
            "action_start": self.action_start,
            "action_stop": self.action_stop,
            "decision_source_index": self.decision_source_index,
            "future_observation_source_index": (
                self.future_observation_source_index
            ),
            "current_code_ids": [list(value) for value in self.current_code_ids],
            "future_code_ids": [list(value) for value in self.future_code_ids],
            "code_available": list(self.code_available),
            "current_descriptor_sources": [
                list(value) for value in self.current_descriptor_sources
            ],
            "future_descriptor_sources": [
                list(value) for value in self.future_descriptor_sources
            ],
            "descriptor_overlap": list(self.descriptor_overlap),
            "artifact_sha256": list(self.artifact_sha256),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> JointWindowRecord:
        return cls(
            window_id=str(payload["window_id"]),
            episode_id=str(payload["episode_id"]),
            parent_episode_id=str(payload["parent_episode_id"]),
            split=str(payload["split"]),
            chart_name=str(payload["chart_name"]),
            role=str(payload["role"]),
            families=tuple(payload["families"]),
            state_latent_start=int(payload["state_latent_start"]),
            state_latent_stop=int(payload["state_latent_stop"]),
            current_latent_index=int(payload["current_latent_index"]),
            future_latent_index=int(payload["future_latent_index"]),
            proprio_start=int(payload["proprio_start"]),
            proprio_stop=int(payload["proprio_stop"]),
            past_action_start=int(payload["past_action_start"]),
            past_action_stop=int(payload["past_action_stop"]),
            action_start=int(payload["action_start"]),
            action_stop=int(payload["action_stop"]),
            decision_source_index=int(payload["decision_source_index"]),
            future_observation_source_index=int(
                payload["future_observation_source_index"]
            ),
            current_code_ids=tuple(
                tuple(int(code) for code in value)
                for value in payload["current_code_ids"]
            ),
            future_code_ids=tuple(
                tuple(int(code) for code in value)
                for value in payload["future_code_ids"]
            ),
            code_available=tuple(bool(value) for value in payload["code_available"]),
            current_descriptor_sources=tuple(
                tuple(int(index) for index in value)
                for value in payload["current_descriptor_sources"]
            ),
            future_descriptor_sources=tuple(
                tuple(int(index) for index in value)
                for value in payload["future_descriptor_sources"]
            ),
            descriptor_overlap=tuple(
                int(value) for value in payload["descriptor_overlap"]
            ),
            artifact_sha256=tuple(payload["artifact_sha256"]),
        )


def build_joint_windows(
    episode: JointEpisode,
    *,
    config: JointWindowConfig,
    artifact_sha256: Sequence[str],
) -> tuple[JointWindowRecord, ...]:
    hashes = tuple(str(value) for value in artifact_sha256)
    if len(hashes) != len(episode.families) or any(not value for value in hashes):
        raise ValueError("One artifact SHA-256 is required per code family.")
    source_to_latent = {
        int(source): index
        for index, source in enumerate(episode.latent_source_indices.tolist())
    }
    windows = []
    for current in range(0, episode.latent_ticks, config.window_stride_ticks):
        decision_source = int(episode.latent_source_indices[current])
        future_source = decision_source + config.action_horizon
        future = source_to_latent.get(future_source)
        if future is None:
            continue
        state_start = current + 1 - config.state_latent_ticks
        decision_local = decision_source - episode.range_start
        future_local = future_source - episode.range_start
        proprio_start = decision_local + 1 - config.proprio_history_steps
        past_action_start = decision_local - config.past_action_steps
        if config.require_full_history and min(
            state_start,
            proprio_start,
            past_action_start,
        ) < 0:
            continue
        state_start = max(state_start, 0)
        proprio_start = max(proprio_start, 0)
        past_action_start = max(past_action_start, 0)
        if (
            decision_local < 0
            or future_local >= episode.source_steps
            or future <= current
        ):
            continue
        action_start = decision_local
        action_stop = future_local
        if action_stop - action_start != config.action_horizon:
            continue
        if not episode.source_action_valid[action_start:action_stop].all():
            continue
        common = (
            episode.code_available[current]
            & episode.code_available[future]
        )
        if config.require_all_code_families and not common.all():
            continue
        if not common.any():
            continue
        current_ids = episode.code_ids[current].clone()
        future_ids = episode.code_ids[future].clone()
        current_sources = episode.descriptor_source_indices[current].clone()
        future_sources = episode.descriptor_source_indices[future].clone()
        current_ids[~common] = -1
        future_ids[~common] = -1
        current_sources[~common] = -1
        future_sources[~common] = -1
        overlaps = []
        for family_index, available in enumerate(common.tolist()):
            if not available:
                overlaps.append(0)
                continue
            current_set = set(current_sources[family_index].tolist())
            future_set = set(future_sources[family_index].tolist())
            overlaps.append(len(current_set & future_set))
        identity = {
            "episode_id": episode.episode_id,
            "current_latent_index": current,
            "future_latent_index": future,
            "action_start": action_start,
            "action_stop": action_stop,
            "chart_name": episode.chart_name,
            "artifacts": hashes,
        }
        windows.append(
            JointWindowRecord(
                window_id=_canonical_hash(identity)[:24],
                episode_id=episode.episode_id,
                parent_episode_id=episode.parent_episode_id,
                split=episode.split,
                chart_name=episode.chart_name,
                role=episode.role,
                families=episode.families,
                state_latent_start=state_start,
                state_latent_stop=current + 1,
                current_latent_index=current,
                future_latent_index=future,
                proprio_start=proprio_start,
                proprio_stop=decision_local + 1,
                past_action_start=past_action_start,
                past_action_stop=decision_local,
                action_start=action_start,
                action_stop=action_stop,
                decision_source_index=decision_source,
                future_observation_source_index=future_source,
                current_code_ids=tuple(
                    tuple(int(code) for code in row)
                    for row in current_ids.tolist()
                ),
                future_code_ids=tuple(
                    tuple(int(code) for code in row)
                    for row in future_ids.tolist()
                ),
                code_available=tuple(bool(value) for value in common.tolist()),
                current_descriptor_sources=tuple(
                    tuple(int(index) for index in row)
                    for row in current_sources.tolist()
                ),
                future_descriptor_sources=tuple(
                    tuple(int(index) for index in row)
                    for row in future_sources.tolist()
                ),
                descriptor_overlap=tuple(overlaps),
                artifact_sha256=hashes,
            )
        )
    return tuple(windows)


def _validate_window_against_episode(
    record: JointWindowRecord,
    episode: JointEpisode,
) -> None:
    if (
        record.episode_id != episode.episode_id
        or record.parent_episode_id != episode.parent_episode_id
        or record.split != episode.split
        or record.chart_name != episode.chart_name
        or record.role != episode.role
        or record.families != episode.families
    ):
        raise RuntimeError(
            f"Joint window `{record.window_id}` identity differs from its episode."
        )
    if not (
        0 <= record.state_latent_start
        < record.state_latent_stop
        <= episode.latent_ticks
        and 0 <= record.future_latent_index < episode.latent_ticks
        and 0 <= record.proprio_start < record.proprio_stop <= episode.source_steps
        and 0
        <= record.past_action_start
        < record.past_action_stop
        <= episode.source_steps
        and 0 <= record.action_start < record.action_stop <= episode.source_steps
    ):
        raise RuntimeError(
            f"Joint window `{record.window_id}` has an out-of-range slice."
        )
    current = record.current_latent_index
    future = record.future_latent_index
    if int(episode.latent_source_indices[current]) != record.decision_source_index:
        raise RuntimeError(f"Joint window `{record.window_id}` decision changed.")
    if (
        int(episode.latent_source_indices[future])
        != record.future_observation_source_index
    ):
        raise RuntimeError(f"Joint window `{record.window_id}` endpoint changed.")
    if (
        episode.range_start + record.action_start
        != record.decision_source_index
        or episode.range_start + record.action_stop
        != record.future_observation_source_index
    ):
        raise RuntimeError(
            f"Joint window `{record.window_id}` source/action indices disagree."
        )
    if not episode.source_action_valid[
        record.action_start : record.action_stop
    ].all():
        raise RuntimeError(f"Joint window `{record.window_id}` uses invalid actions.")
    available = torch.tensor(record.code_available, dtype=torch.bool)
    expected_available = (
        episode.code_available[current] & episode.code_available[future]
    )
    if not torch.equal(available, expected_available):
        raise RuntimeError(
            f"Joint window `{record.window_id}` availability changed."
        )
    current_codes = episode.code_ids[current].clone()
    future_codes = episode.code_ids[future].clone()
    current_sources = episode.descriptor_source_indices[current].clone()
    future_sources = episode.descriptor_source_indices[future].clone()
    current_codes[~available] = -1
    future_codes[~available] = -1
    current_sources[~available] = -1
    future_sources[~available] = -1
    if (
        tuple(tuple(int(value) for value in row) for row in current_codes)
        != record.current_code_ids
        or tuple(tuple(int(value) for value in row) for row in future_codes)
        != record.future_code_ids
        or tuple(
            tuple(int(value) for value in row) for row in current_sources
        )
        != record.current_descriptor_sources
        or tuple(
            tuple(int(value) for value in row) for row in future_sources
        )
        != record.future_descriptor_sources
    ):
        raise RuntimeError(f"Joint window `{record.window_id}` code label changed.")
    overlap = []
    for family, family_available in enumerate(available.tolist()):
        if not family_available:
            overlap.append(0)
        else:
            overlap.append(
                len(
                    set(current_sources[family].tolist())
                    & set(future_sources[family].tolist())
                )
            )
    if tuple(overlap) != record.descriptor_overlap:
        raise RuntimeError(
            f"Joint window `{record.window_id}` overlap metadata changed."
        )


def _sidecar_path(shard_path: Path) -> Path:
    return shard_path.with_suffix(".index.json")


def _load_joint_shard_sidecar(
    path: Path,
    *,
    contract_hash: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != JOINT_SHARD_INDEX_SCHEMA
        or payload.get("contract_hash") != contract_hash
    ):
        raise RuntimeError(f"Invalid joint shard sidecar `{path}`.")
    return payload


def _transition_coverage(
    windows: Sequence[JointWindowRecord],
) -> dict[str, Any]:
    states: dict[str, dict[str, Any]] = {}
    for window in windows:
        split = states.setdefault(
            window.split,
            {
                "any_available_windows": 0,
                "any_changed_windows": 0,
                "available_parents": set(),
                "changed_parents": set(),
                "families": {},
            },
        )
        any_available = False
        any_changed = False
        for family, current, future, available in zip(
            window.families,
            window.current_code_ids,
            window.future_code_ids,
            window.code_available,
        ):
            if not available:
                continue
            family_state = split["families"].setdefault(
                family,
                {
                    "available_windows": 0,
                    "changed_windows": 0,
                    "available_parents": set(),
                    "changed_parents": set(),
                    "level_changed_windows": [0] * len(current),
                    "prefix_changed_windows": [0] * len(current),
                },
            )
            changed = tuple(current) != tuple(future)
            any_available = True
            any_changed |= changed
            family_state["available_windows"] += 1
            family_state["available_parents"].add(window.parent_episode_id)
            if changed:
                family_state["changed_windows"] += 1
                family_state["changed_parents"].add(window.parent_episode_id)
            for level, (current_code, future_code) in enumerate(
                zip(current, future)
            ):
                family_state["level_changed_windows"][level] += int(
                    current_code != future_code
                )
                family_state["prefix_changed_windows"][level] += int(
                    tuple(current[: level + 1]) != tuple(future[: level + 1])
                )
        if any_available:
            split["any_available_windows"] += 1
            split["available_parents"].add(window.parent_episode_id)
        if any_changed:
            split["any_changed_windows"] += 1
            split["changed_parents"].add(window.parent_episode_id)

    report = {}
    for split_name, split in sorted(states.items()):
        available = int(split["any_available_windows"])
        family_report = {}
        for family, state in sorted(split["families"].items()):
            family_available = int(state["available_windows"])
            family_changed = int(state["changed_windows"])
            family_report[family] = {
                "available_windows": family_available,
                "changed_windows": family_changed,
                "changed_fraction": (
                    family_changed / family_available
                    if family_available
                    else float("nan")
                ),
                "available_parent_episodes": len(state["available_parents"]),
                "changed_parent_episodes": len(state["changed_parents"]),
                "level_changed_windows": list(
                    state["level_changed_windows"]
                ),
                "prefix_changed_windows": list(
                    state["prefix_changed_windows"]
                ),
            }
        changed = int(split["any_changed_windows"])
        report[split_name] = {
            "any_family": {
                "available_windows": available,
                "changed_windows": changed,
                "changed_fraction": (
                    changed / available if available else float("nan")
                ),
                "available_parent_episodes": len(split["available_parents"]),
                "changed_parent_episodes": len(split["changed_parents"]),
            },
            "families": family_report,
        }
    return report


def validate_joint_episode_shard(
    path: str | Path,
    *,
    contract_hash: str,
    expected_sha256: str | None = None,
) -> tuple[JointEpisode, ...]:
    path = Path(path)
    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise RuntimeError(f"Joint episode shard SHA-256 changed: {path}.")
    payload = load_torch_payload(path, map_location="cpu")
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != JOINT_EPISODE_SHARD_SCHEMA
    ):
        raise ValueError(f"Unsupported joint episode shard `{path}`.")
    if payload.get("contract_hash") != contract_hash:
        raise RuntimeError(f"Joint episode shard `{path}` uses another contract.")
    episodes = tuple(
        JointEpisode.from_payload(value)
        for value in payload.get("episodes", ())
    )
    if not episodes:
        raise ValueError(f"Joint episode shard `{path}` is empty.")
    identifiers = [episode.episode_id for episode in episodes]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Joint episode shard `{path}` has duplicate episodes.")
    return episodes


def write_joint_episode_shard(
    cache_dir: str | Path,
    shard_name: str,
    episodes: Sequence[JointEpisode],
    windows: Sequence[JointWindowRecord],
    *,
    contract_hash: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    if not shard_name or Path(shard_name).name != shard_name:
        raise ValueError("Joint shard name must be a plain file stem.")
    episode_rows = tuple(episodes)
    window_rows = tuple(windows)
    if not episode_rows:
        raise ValueError("A joint episode shard must not be empty.")
    identifiers = [episode.episode_id for episode in episode_rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("A joint episode shard cannot repeat episode IDs.")
    unknown_windows = sorted(
        {window.episode_id for window in window_rows} - set(identifiers)
    )
    if unknown_windows:
        raise ValueError(
            f"Joint shard windows reference unknown episodes: {unknown_windows}."
        )
    episodes_by_id = {
        episode.episode_id: episode for episode in episode_rows
    }
    for window in window_rows:
        _validate_window_against_episode(
            window,
            episodes_by_id[window.episode_id],
        )
    shard_path = cache_dir / "episode_shards" / f"{shard_name}.pt"
    sidecar_path = _sidecar_path(shard_path)
    if shard_path.exists() or sidecar_path.exists():
        raise FileExistsError(f"Joint shard already exists: {shard_path}.")
    payload = {
        "schema": JOINT_EPISODE_SHARD_SCHEMA,
        "contract_hash": contract_hash,
        "metadata": dict(metadata or {}),
        "episodes": [episode.to_payload() for episode in episode_rows],
    }
    atomic_torch_save(payload, shard_path)
    sha256 = file_sha256(shard_path)
    relative = str(shard_path.relative_to(cache_dir))
    sidecar = {
        "schema": JOINT_SHARD_INDEX_SCHEMA,
        "contract_hash": contract_hash,
        "episode_shard": relative,
        "episode_shard_sha256": sha256,
        "episode_shard_bytes": shard_path.stat().st_size,
        "metadata": dict(metadata or {}),
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "offset": offset,
                "split": episode.split,
                "role": episode.role.value,
                "latent_ticks": episode.latent_ticks,
                "source_steps": episode.source_steps,
            }
            for offset, episode in enumerate(episode_rows)
        ],
        "windows": [window.to_dict() for window in window_rows],
    }
    _write_json(sidecar_path, sidecar)
    return sidecar


def _joint_chart_fields(
    contract: Mapping[str, Any],
) -> tuple[str, tuple[str, ...], tuple[str, ...], int]:
    chart = contract["chart"]
    family_rows = tuple(chart["families"])
    families = tuple(str(row["family"]) for row in family_rows)
    artifact_sha256 = tuple(str(row["sha256"]) for row in family_rows)
    levels = {int(row["levels"]) for row in family_rows}
    if not families or len(levels) != 1:
        raise RuntimeError(
            "Compact joint indices require nonempty, level-aligned code families."
        )
    return str(chart["name"]), families, artifact_sha256, levels.pop()


def _write_compact_joint_window_index(
    cache_dir: Path,
    *,
    contract: Mapping[str, Any],
    episode_rows: Sequence[Mapping[str, Any]],
    windows: Iterable[JointWindowRecord],
    window_count: int,
    windows_sha256: str,
) -> dict[str, Any]:
    if window_count <= 0:
        raise ValueError("Compact joint window index cannot be empty.")
    chart_name, families, artifact_sha256, levels = _joint_chart_fields(
        contract
    )
    family_count = len(families)
    episode_ids = tuple(str(row["episode_id"]) for row in episode_rows)
    if len(episode_ids) != len(set(episode_ids)):
        raise RuntimeError("Joint episode rows contain duplicate IDs.")
    episode_positions = {
        episode_id: index for index, episode_id in enumerate(episode_ids)
    }
    parent_episode_ids = [
        str(row.get("parent_episode_id", "")) for row in episode_rows
    ]

    episode_index = np.empty(window_count, dtype=np.int32)
    split_ids = np.empty(window_count, dtype=np.uint8)
    window_id_bytes = np.zeros(
        (window_count, _JOINT_WINDOW_ID_BYTES),
        dtype=np.uint8,
    )
    window_id_lengths = np.empty(window_count, dtype=np.uint8)
    scalars = np.empty(
        (window_count, len(_JOINT_WINDOW_SCALAR_FIELDS)),
        dtype=np.int32,
    )
    current_code_ids = np.empty(
        (window_count, family_count, levels),
        dtype=np.int32,
    )
    future_code_ids = np.empty_like(current_code_ids)
    code_available = np.empty(
        (window_count, family_count),
        dtype=np.bool_,
    )
    current_descriptor_sources = np.empty(
        (window_count, family_count, 3),
        dtype=np.int32,
    )
    future_descriptor_sources = np.empty_like(
        current_descriptor_sources
    )
    descriptor_overlap = np.empty(
        (window_count, family_count),
        dtype=np.uint8,
    )
    split_positions = {
        split: index for index, split in enumerate(_JOINT_SPLITS)
    }
    previous_window_id: str | None = None
    written = 0
    for position, record in enumerate(windows):
        if position >= window_count:
            raise RuntimeError(
                "Joint window source exceeds its declared summary count."
            )
        if previous_window_id is not None and record.window_id <= previous_window_id:
            raise RuntimeError(
                "Joint windows must be strictly ordered by unique window ID."
            )
        previous_window_id = record.window_id
        try:
            episode_row = episode_positions[record.episode_id]
        except KeyError as exc:
            raise RuntimeError(
                f"Joint window references unknown episode `{record.episode_id}`."
            ) from exc
        locator = episode_rows[episode_row]
        if (
            str(locator["split"]) != record.split
            or str(locator["role"]) != record.role.value
        ):
            raise RuntimeError(
                f"Joint window `{record.window_id}` differs from its episode locator."
            )
        known_parent = parent_episode_ids[episode_row]
        if known_parent and known_parent != record.parent_episode_id:
            raise RuntimeError(
                f"Joint episode `{record.episode_id}` has inconsistent parents."
            )
        parent_episode_ids[episode_row] = record.parent_episode_id
        if (
            record.chart_name != chart_name
            or record.families != families
            or record.artifact_sha256 != artifact_sha256
        ):
            raise RuntimeError(
                f"Joint window `{record.window_id}` differs from its chart contract."
            )
        encoded_id = record.window_id.encode("utf-8")
        if len(encoded_id) > _JOINT_WINDOW_ID_BYTES:
            raise ValueError(
                f"Joint window ID exceeds {_JOINT_WINDOW_ID_BYTES} UTF-8 bytes."
            )
        episode_index[position] = episode_row
        split_ids[position] = split_positions[record.split]
        window_id_lengths[position] = len(encoded_id)
        window_id_bytes[position, : len(encoded_id)] = np.frombuffer(
            encoded_id,
            dtype=np.uint8,
        )
        scalars[position] = tuple(
            int(getattr(record, field))
            for field in _JOINT_WINDOW_SCALAR_FIELDS
        )
        current_code_ids[position] = record.current_code_ids
        future_code_ids[position] = record.future_code_ids
        code_available[position] = record.code_available
        current_descriptor_sources[position] = (
            record.current_descriptor_sources
        )
        future_descriptor_sources[position] = (
            record.future_descriptor_sources
        )
        descriptor_overlap[position] = record.descriptor_overlap
        written = position + 1
    if written != window_count:
        raise RuntimeError(
            f"Joint window source yielded {written}, expected {window_count}."
        )

    payload = {
        "schema": JOINT_WINDOW_INDEX_SCHEMA,
        "contract_hash": str(contract["contract_hash"]),
        "windows_sha256": str(windows_sha256),
        "count": int(window_count),
        "split_names": list(_JOINT_SPLITS),
        "scalar_fields": list(_JOINT_WINDOW_SCALAR_FIELDS),
        "chart_name": chart_name,
        "families": list(families),
        "artifact_sha256": list(artifact_sha256),
        "episode_ids": list(episode_ids),
        "parent_episode_ids": parent_episode_ids,
        "episode_index": torch.from_numpy(episode_index),
        "split_ids": torch.from_numpy(split_ids),
        "window_id_bytes": torch.from_numpy(window_id_bytes),
        "window_id_lengths": torch.from_numpy(window_id_lengths),
        "scalars": torch.from_numpy(scalars),
        "current_code_ids": torch.from_numpy(current_code_ids),
        "future_code_ids": torch.from_numpy(future_code_ids),
        "code_available": torch.from_numpy(code_available),
        "current_descriptor_sources": torch.from_numpy(
            current_descriptor_sources
        ),
        "future_descriptor_sources": torch.from_numpy(
            future_descriptor_sources
        ),
        "descriptor_overlap": torch.from_numpy(descriptor_overlap),
    }
    path = cache_dir / "window_records.pt"
    atomic_torch_save(payload, path)
    return {
        "path": path.name,
        "sha256": file_sha256(path),
        "source_windows_sha256": str(windows_sha256),
    }


def add_compact_joint_window_index(
    cache_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Add an mmap-friendly index to a finalized legacy joint cache."""

    cache_dir = Path(cache_dir)
    contract = json.loads(
        (cache_dir / "contract.json").read_text(encoding="utf-8")
    )
    validate_joint_cache_contract(contract)
    summary_path = cache_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("schema") != JOINT_CACHE_SUMMARY_SCHEMA
        or summary.get("contract_hash") != contract["contract_hash"]
    ):
        raise RuntimeError("Joint cache summary does not match its contract.")
    indices = summary.get("indices")
    if not isinstance(indices, dict):
        raise RuntimeError("Joint cache summary has no index registry.")
    existing = indices.get("window_records")
    if existing is not None and not force:
        path = cache_dir / str(existing["path"])
        if file_sha256(path) != existing["sha256"]:
            raise RuntimeError(f"Joint cache index hash changed: {path}.")
        return summary
    for name in ("episodes", "windows"):
        row = indices[name]
        path = cache_dir / str(row["path"])
        if file_sha256(path) != row["sha256"]:
            raise RuntimeError(f"Joint cache index hash changed: {path}.")
    episode_rows = list(
        _iter_jsonl(cache_dir / str(indices["episodes"]["path"]))
    )
    windows_path = cache_dir / str(indices["windows"]["path"])
    compact_row = _write_compact_joint_window_index(
        cache_dir,
        contract=contract,
        episode_rows=episode_rows,
        windows=(
            JointWindowRecord.from_dict(row)
            for row in _iter_jsonl(windows_path)
        ),
        window_count=int(summary["windows"]),
        windows_sha256=str(indices["windows"]["sha256"]),
    )
    updated = dict(summary)
    updated["indices"] = {**indices, "window_records": compact_row}
    _write_json(summary_path, updated)
    return updated


def finalize_joint_cache(
    cache_dir: str | Path,
    *,
    export_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    contract_path = cache_dir / "contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(f"Missing joint cache contract `{contract_path}`.")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_joint_cache_contract(contract)
    contract_hash = str(contract["contract_hash"])
    sidecar_paths = sorted(
        (cache_dir / "episode_shards").glob("*.index.json")
    )
    if not sidecar_paths:
        raise FileNotFoundError("Joint cache has no shard sidecars.")
    parsed_windows: list[JointWindowRecord] = []
    for sidecar_path in sidecar_paths:
        sidecar = _load_joint_shard_sidecar(
            sidecar_path,
            contract_hash=contract_hash,
        )
        for payload in sidecar.get("windows", ()):
            parsed_windows.append(JointWindowRecord.from_dict(payload))
    if not parsed_windows:
        raise RuntimeError("Joint cache contains no valid windows.")
    parsed_windows.sort(key=lambda row: row.window_id)
    if any(
        left.window_id == right.window_id
        for left, right in pairwise(parsed_windows)
    ):
        raise RuntimeError("Joint cache contains duplicate window IDs.")
    window_positions = {
        window.window_id: index
        for index, window in enumerate(parsed_windows)
    }
    action_horizon = int(contract["window"]["action_horizon"])
    action_dim = int(contract["action_dim"])
    action_chunks = torch.empty(
        (len(parsed_windows), action_horizon, action_dim),
        dtype=torch.float32,
    )
    action_valid = torch.empty(
        (len(parsed_windows), action_horizon),
        dtype=torch.bool,
    )
    filled_actions = torch.zeros(len(parsed_windows), dtype=torch.bool)

    episode_rows: list[dict[str, Any]] = []
    episode_ids: set[str] = set()
    shard_rows = []
    for sidecar_path in sidecar_paths:
        sidecar = _load_joint_shard_sidecar(
            sidecar_path,
            contract_hash=contract_hash,
        )
        shard_path = cache_dir / str(sidecar["episode_shard"])
        episodes = validate_joint_episode_shard(
            shard_path,
            contract_hash=contract_hash,
            expected_sha256=str(sidecar["episode_shard_sha256"]),
        )
        sidecar_episodes = sidecar.get("episodes", ())
        if len(sidecar_episodes) != len(episodes):
            raise RuntimeError(
                f"Joint sidecar episode count differs: {shard_path}."
            )
        episodes_by_id = {episode.episode_id: episode for episode in episodes}
        for offset, (locator, episode) in enumerate(
            zip(sidecar_episodes, episodes)
        ):
            if (
                locator.get("episode_id") != episode.episode_id
                or int(locator.get("offset", -1)) != offset
            ):
                raise RuntimeError(
                    f"Joint sidecar episode locator differs: {shard_path}."
                )
            if episode.episode_id in episode_ids:
                raise RuntimeError("Joint cache contains duplicate episode IDs.")
            episode_ids.add(episode.episode_id)
            episode_rows.append(
                {
                    **locator,
                    "parent_episode_id": episode.parent_episode_id,
                    "episode_shard": str(sidecar["episode_shard"]),
                    "episode_shard_sha256": str(
                        sidecar["episode_shard_sha256"]
                    ),
                }
            )
        for window_payload in sidecar.get("windows", ()):
            window = JointWindowRecord.from_dict(window_payload)
            try:
                episode = episodes_by_id[window.episode_id]
            except KeyError as exc:
                raise RuntimeError(
                    f"Joint shard window references unknown episode "
                    f"`{window.episode_id}`."
                ) from exc
            _validate_window_against_episode(window, episode)
            position = window_positions[window.window_id]
            actions = episode.source_actions[
                window.action_start : window.action_stop
            ]
            valid = episode.source_action_valid[
                window.action_start : window.action_stop
            ]
            if (
                tuple(actions.shape) != (action_horizon, action_dim)
                or tuple(valid.shape) != (action_horizon,)
            ):
                raise RuntimeError(
                    f"Joint window `{window.window_id}` action shape changed."
                )
            action_chunks[position].copy_(actions)
            action_valid[position].copy_(valid)
            filled_actions[position] = True
        shard_rows.append(
            {
                "path": str(sidecar["episode_shard"]),
                "sha256": str(sidecar["episode_shard_sha256"]),
                "bytes": int(sidecar["episode_shard_bytes"]),
                "episodes": len(episodes),
                "windows": len(sidecar.get("windows", ())),
            }
        )
    unknown = sorted(
        {window.episode_id for window in parsed_windows} - episode_ids
    )
    if unknown:
        raise RuntimeError(f"Joint cache windows reference unknown episodes: {unknown}.")
    if not filled_actions.all():
        raise RuntimeError("Joint cache action index is incomplete.")
    del filled_actions, window_positions
    episode_rows.sort(key=lambda row: row["episode_id"])
    episodes_path = cache_dir / "episodes.jsonl"
    windows_path = cache_dir / "windows.jsonl"
    _write_jsonl(episodes_path, episode_rows)
    _write_jsonl(
        windows_path,
        (window.to_dict() for window in parsed_windows),
    )
    windows_sha256 = file_sha256(windows_path)
    compact_window_row = _write_compact_joint_window_index(
        cache_dir,
        contract=contract,
        episode_rows=episode_rows,
        windows=parsed_windows,
        window_count=len(parsed_windows),
        windows_sha256=windows_sha256,
    )
    actions_path = cache_dir / "window_actions.pt"
    atomic_torch_save(
        {
            "schema": JOINT_ACTION_INDEX_SCHEMA,
            "contract_hash": contract_hash,
            "windows_sha256": windows_sha256,
            "actions": action_chunks,
            "valid": action_valid,
        },
        actions_path,
    )
    overlap_counts: Counter[str] = Counter()
    for window in parsed_windows:
        for family, overlap, available in zip(
            window.families,
            window.descriptor_overlap,
            window.code_available,
        ):
            if available:
                overlap_counts[f"{family}/overlap-{overlap}"] += 1
    summary = {
        "schema": JOINT_CACHE_SUMMARY_SCHEMA,
        "contract_hash": contract_hash,
        "export_audit": (
            None if export_audit is None else dict(export_audit)
        ),
        "episodes": len(episode_rows),
        "windows": len(parsed_windows),
        "episode_shards": len(shard_rows),
        "splits": dict(sorted(Counter(
            window.split for window in parsed_windows
        ).items())),
        "roles": dict(sorted(Counter(
            window.role.value for window in parsed_windows
        ).items())),
        "descriptor_overlap": dict(sorted(overlap_counts.items())),
        "transition_coverage": _transition_coverage(parsed_windows),
        "indices": {
            "episodes": {
                "path": episodes_path.name,
                "sha256": file_sha256(episodes_path),
            },
            "windows": {
                "path": windows_path.name,
                "sha256": windows_sha256,
            },
            "window_records": compact_window_row,
            "actions": {
                "path": actions_path.name,
                "sha256": file_sha256(actions_path),
            },
        },
        "shards": shard_rows,
    }
    _write_json(cache_dir / "summary.json", summary)
    return summary


@dataclass(frozen=True)
class JointWindowSample:
    record: JointWindowRecord
    latents: torch.Tensor
    latent_valid: torch.Tensor
    proprio_history: torch.Tensor
    past_actions: torch.Tensor
    actions: torch.Tensor
    action_valid: torch.Tensor
    current_codes: torch.Tensor
    future_codes: torch.Tensor
    code_available: torch.Tensor
    language_tokens: torch.Tensor | None
    language_valid: torch.Tensor | None


def _sequence_position(index: int, length: int) -> int:
    position = int(index)
    if position < 0:
        position += length
    if position < 0 or position >= length:
        raise IndexError("Joint window index is out of range.")
    return position


class _CompactJointWindowIndex:
    def __init__(
        self,
        payload: Mapping[str, Any],
        *,
        contract: Mapping[str, Any],
        summary: Mapping[str, Any],
        locators: Mapping[str, Mapping[str, Any]],
    ):
        windows_row = summary["indices"]["windows"]
        if (
            payload.get("schema") != JOINT_WINDOW_INDEX_SCHEMA
            or payload.get("contract_hash") != contract["contract_hash"]
            or payload.get("windows_sha256") != windows_row["sha256"]
            or int(payload.get("count", -1)) != int(summary["windows"])
            or tuple(payload.get("split_names", ())) != _JOINT_SPLITS
            or tuple(payload.get("scalar_fields", ()))
            != _JOINT_WINDOW_SCALAR_FIELDS
        ):
            raise RuntimeError(
                "Compact joint window index does not match its cache."
            )
        chart_name, families, artifact_sha256, levels = _joint_chart_fields(
            contract
        )
        if (
            payload.get("chart_name") != chart_name
            or tuple(payload.get("families", ())) != families
            or tuple(payload.get("artifact_sha256", ())) != artifact_sha256
        ):
            raise RuntimeError(
                "Compact joint window index differs from its chart contract."
            )
        self.count = int(payload["count"])
        self.chart_name = chart_name
        self.families = families
        self.artifact_sha256 = artifact_sha256
        self.episode_ids = tuple(
            str(value) for value in payload.get("episode_ids", ())
        )
        self.parent_episode_ids = tuple(
            str(value) for value in payload.get("parent_episode_ids", ())
        )
        if (
            len(self.episode_ids) != len(self.parent_episode_ids)
            or len(self.episode_ids) != len(locators)
            or len(set(self.episode_ids)) != len(self.episode_ids)
        ):
            raise RuntimeError(
                "Compact joint window episode identities are invalid."
            )
        try:
            self.episode_locators = tuple(
                locators[episode_id] for episode_id in self.episode_ids
            )
        except KeyError as exc:
            raise RuntimeError(
                f"Compact joint index references unknown episode `{exc.args[0]}`."
            ) from exc

        tensor_specs = {
            "episode_index": (torch.int32, (self.count,)),
            "split_ids": (torch.uint8, (self.count,)),
            "window_id_bytes": (
                torch.uint8,
                (self.count, _JOINT_WINDOW_ID_BYTES),
            ),
            "window_id_lengths": (torch.uint8, (self.count,)),
            "scalars": (
                torch.int32,
                (self.count, len(_JOINT_WINDOW_SCALAR_FIELDS)),
            ),
            "current_code_ids": (
                torch.int32,
                (self.count, len(families), levels),
            ),
            "future_code_ids": (
                torch.int32,
                (self.count, len(families), levels),
            ),
            "code_available": (
                torch.bool,
                (self.count, len(families)),
            ),
            "current_descriptor_sources": (
                torch.int32,
                (self.count, len(families), 3),
            ),
            "future_descriptor_sources": (
                torch.int32,
                (self.count, len(families), 3),
            ),
            "descriptor_overlap": (
                torch.uint8,
                (self.count, len(families)),
            ),
        }
        for name, (dtype, shape) in tensor_specs.items():
            value = payload.get(name)
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != dtype
                or tuple(value.shape) != shape
            ):
                raise RuntimeError(
                    f"Compact joint window tensor `{name}` is invalid."
                )
            setattr(self, name, value)
        if self.count:
            if (
                int(self.episode_index.min()) < 0
                or int(self.episode_index.max()) >= len(self.episode_ids)
                or int(self.split_ids.max()) >= len(_JOINT_SPLITS)
                or int(self.window_id_lengths.max())
                > _JOINT_WINDOW_ID_BYTES
            ):
                raise RuntimeError(
                    "Compact joint window index contains invalid row IDs."
                )

    def window_id(self, position: int) -> str:
        length = int(self.window_id_lengths[position])
        return bytes(
            self.window_id_bytes[position, :length].tolist()
        ).decode("utf-8")

    def episode_row(self, position: int) -> int:
        return int(self.episode_index[position])

    def split(self, position: int) -> str:
        return _JOINT_SPLITS[int(self.split_ids[position])]

    def parent_episode_id(self, position: int) -> str:
        return self.parent_episode_ids[self.episode_row(position)]

    def action_horizon(self, position: int) -> int:
        row = self.scalars[position]
        return int(row[9]) - int(row[8])

    def episode_shard(self, position: int) -> str:
        locator = self.episode_locators[self.episode_row(position)]
        return str(locator["episode_shard"])

    def record(self, position: int) -> JointWindowRecord:
        episode_row = self.episode_row(position)
        locator = self.episode_locators[episode_row]
        scalar_values = self.scalars[position].tolist()
        scalar_fields = dict(
            zip(_JOINT_WINDOW_SCALAR_FIELDS, scalar_values)
        )
        return JointWindowRecord(
            window_id=self.window_id(position),
            episode_id=self.episode_ids[episode_row],
            parent_episode_id=self.parent_episode_ids[episode_row],
            split=self.split(position),
            chart_name=self.chart_name,
            role=str(locator["role"]),
            families=self.families,
            **scalar_fields,
            current_code_ids=tuple(
                tuple(int(code) for code in row)
                for row in self.current_code_ids[position].tolist()
            ),
            future_code_ids=tuple(
                tuple(int(code) for code in row)
                for row in self.future_code_ids[position].tolist()
            ),
            code_available=tuple(
                bool(value)
                for value in self.code_available[position].tolist()
            ),
            current_descriptor_sources=tuple(
                tuple(int(value) for value in row)
                for row in self.current_descriptor_sources[position].tolist()
            ),
            future_descriptor_sources=tuple(
                tuple(int(value) for value in row)
                for row in self.future_descriptor_sources[position].tolist()
            ),
            descriptor_overlap=tuple(
                int(value)
                for value in self.descriptor_overlap[position].tolist()
            ),
            artifact_sha256=self.artifact_sha256,
        )


class _CompactJointWindowSequence(Sequence[JointWindowRecord]):
    def __init__(
        self,
        index: _CompactJointWindowIndex,
        positions: torch.Tensor | None,
    ):
        self.index = index
        self.positions = positions

    def __len__(self) -> int:
        if self.positions is None:
            return self.index.count
        return int(self.positions.numel())

    def global_position(self, index: int) -> int:
        position = _sequence_position(index, len(self))
        if self.positions is None:
            return position
        return int(self.positions[position])

    def __getitem__(
        self,
        index: int | slice,
    ) -> JointWindowRecord | tuple[JointWindowRecord, ...]:
        if isinstance(index, slice):
            return tuple(
                self[position]
                for position in range(*index.indices(len(self)))
            )
        return self.index.record(self.global_position(index))

    def split_indices(self, split: str) -> tuple[int, ...]:
        split_id = _JOINT_SPLITS.index(split)
        if self.positions is None:
            selected = self.index.split_ids == split_id
        else:
            selected = (
                self.index.split_ids.index_select(0, self.positions) == split_id
            )
        return tuple(
            int(value)
            for value in torch.nonzero(selected, as_tuple=False).flatten().tolist()
        )

    def permutation_rows(
        self,
    ) -> Iterator[tuple[int, str, int, str, str]]:
        for local_position in range(len(self)):
            global_position = self.global_position(local_position)
            yield (
                local_position,
                self.index.split(global_position),
                self.index.action_horizon(global_position),
                self.index.parent_episode_id(global_position),
                self.index.window_id(global_position),
            )

    def episode_shard(self, index: int) -> str:
        return self.index.episode_shard(self.global_position(index))


class _CompactWindowShardSequence(Sequence[str]):
    def __init__(self, windows: _CompactJointWindowSequence):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int | slice) -> str | tuple[str, ...]:
        if isinstance(index, slice):
            return tuple(
                self[position] for position in range(*index.indices(len(self)))
            )
        return self.windows.episode_shard(index)


class JointWindowCache:
    """Verified random access to deduplicated episode tensors and logical windows."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        split: str | None = None,
        max_cached_shards: int = 2,
        verify_index_hashes: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        if split is not None and split not in VALID_SPLITS:
            raise ValueError(f"Unsupported joint cache split `{split}`.")
        if max_cached_shards <= 0:
            raise ValueError("Joint cache must retain at least one loaded shard.")
        self.contract = json.loads(
            (self.cache_dir / "contract.json").read_text(encoding="utf-8")
        )
        validate_joint_cache_contract(self.contract)
        self.summary = json.loads(
            (self.cache_dir / "summary.json").read_text(encoding="utf-8")
        )
        if (
            self.summary.get("schema") != JOINT_CACHE_SUMMARY_SCHEMA
            or self.summary.get("contract_hash")
            != self.contract["contract_hash"]
        ):
            raise RuntimeError("Joint cache summary does not match its contract.")
        if verify_index_hashes:
            for row in self.summary["indices"].values():
                path = self.cache_dir / row["path"]
                if file_sha256(path) != row["sha256"]:
                    raise RuntimeError(
                        f"Joint cache index hash changed: {path}."
                    )
        self._locators: dict[str, dict[str, Any]] = {}
        episode_row = self.summary["indices"]["episodes"]
        for row in _iter_jsonl(self.cache_dir / episode_row["path"]):
            episode_id = str(row["episode_id"])
            if episode_id in self._locators:
                raise RuntimeError("Joint cache episode index contains duplicates.")
            self._locators[episode_id] = row
        compact_row = self.summary["indices"].get("window_records")
        self._compact_index: _CompactJointWindowIndex | None = None
        if compact_row is not None:
            compact_path = self.cache_dir / str(compact_row["path"])
            try:
                compact_payload = torch.load(
                    compact_path,
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
            except TypeError:
                compact_payload = load_torch_payload(
                    compact_path,
                    map_location="cpu",
                )
            if not isinstance(compact_payload, dict):
                raise RuntimeError("Compact joint window index is invalid.")
            self._compact_index = _CompactJointWindowIndex(
                compact_payload,
                contract=self.contract,
                summary=self.summary,
                locators=self._locators,
            )
            window_count = self._compact_index.count
            positions = (
                None
                if split is None
                else torch.nonzero(
                    self._compact_index.split_ids
                    == _JOINT_SPLITS.index(split),
                    as_tuple=False,
                ).flatten()
            )
            compact_windows = _CompactJointWindowSequence(
                self._compact_index,
                positions,
            )
            self.windows: Sequence[JointWindowRecord] = compact_windows
            self._window_positions: tuple[int, ...] | torch.Tensor | None = (
                positions
            )
            self.window_shards: Sequence[str] = (
                _CompactWindowShardSequence(compact_windows)
            )
        else:
            window_positions = []
            selected_windows = []
            window_count = 0
            windows_row = self.summary["indices"]["windows"]
            for index, row in enumerate(
                _iter_jsonl(self.cache_dir / windows_row["path"])
            ):
                window = JointWindowRecord.from_dict(row)
                window_count += 1
                if split is None or window.split == split:
                    window_positions.append(index)
                    selected_windows.append(window)
            self._window_positions = tuple(window_positions)
            self.windows = tuple(selected_windows)
            del selected_windows, window_positions
            try:
                self.window_shards = tuple(
                    str(self._locators[window.episode_id]["episode_shard"])
                    for window in self.windows
                )
            except KeyError as exc:
                raise RuntimeError(
                    f"Joint window references an unknown episode `{exc.args[0]}`."
                ) from exc
        if window_count != int(self.summary["windows"]):
            raise RuntimeError("Joint cache window count differs from its summary.")
        if len(self._locators) != int(self.summary["episodes"]):
            raise RuntimeError("Joint cache episode count differs from its summary.")
        self.max_cached_shards = int(max_cached_shards)
        self._shards: OrderedDict[str, tuple[JointEpisode, ...]] = OrderedDict()
        self._verified_shards: set[str] = set()
        action_row = self.summary["indices"]["actions"]
        action_payload = torch.load(
            self.cache_dir / action_row["path"],
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if (
            action_payload.get("schema") != JOINT_ACTION_INDEX_SCHEMA
            or action_payload.get("contract_hash")
            != self.contract["contract_hash"]
            or action_payload.get("windows_sha256")
            != self.summary["indices"]["windows"]["sha256"]
        ):
            raise RuntimeError("Joint cache action index does not match its windows.")
        self._action_chunks = action_payload.get("actions")
        self._action_valid = action_payload.get("valid")
        if (
            not isinstance(self._action_chunks, torch.Tensor)
            or not isinstance(self._action_valid, torch.Tensor)
            or self._action_chunks.ndim != 3
            or self._action_valid.dtype != torch.bool
            or tuple(self._action_valid.shape)
            != tuple(self._action_chunks.shape[:2])
            or int(self._action_chunks.shape[0]) != window_count
        ):
            raise RuntimeError("Joint cache action index has invalid tensors.")

    def __len__(self) -> int:
        return len(self.windows)

    @property
    def parent_episode_ids(self) -> tuple[str, ...]:
        if self._compact_index is not None:
            return tuple(
                dict.fromkeys(
                    parent
                    for parent in self._compact_index.parent_episode_ids
                    if parent
                )
            )
        return tuple(
            dict.fromkeys(window.parent_episode_id for window in self.windows)
        )

    @property
    def referenced_episode_ids(self) -> tuple[str, ...]:
        """Segment IDs that contribute at least one logical window."""
        if self._compact_index is not None:
            return tuple(
                episode
                for episode, parent in zip(
                    self._compact_index.episode_ids,
                    self._compact_index.parent_episode_ids,
                )
                if parent
            )
        return tuple(dict.fromkeys(window.episode_id for window in self.windows))

    def split_indices(self, split: str) -> tuple[int, ...]:
        if split not in VALID_SPLITS:
            raise ValueError(f"Unsupported joint cache split `{split}`.")
        if isinstance(self.windows, _CompactJointWindowSequence):
            return self.windows.split_indices(split)
        return tuple(
            index
            for index, window in enumerate(self.windows)
            if window.split == split
        )

    def permutation_rows(
        self,
    ) -> Iterator[tuple[int, str, int, str, str]]:
        if isinstance(self.windows, _CompactJointWindowSequence):
            yield from self.windows.permutation_rows()
            return
        for index, window in enumerate(self.windows):
            yield (
                index,
                window.split,
                window.action_stop - window.action_start,
                window.parent_episode_id,
                window.window_id,
            )

    def _load_episode(self, episode_id: str) -> JointEpisode:
        try:
            locator = self._locators[episode_id]
        except KeyError as exc:
            raise KeyError(f"Unknown joint episode `{episode_id}`.") from exc
        relative = str(locator["episode_shard"])
        episodes = self._shards.pop(relative, None)
        if episodes is None:
            episodes = validate_joint_episode_shard(
                self.cache_dir / relative,
                contract_hash=str(self.contract["contract_hash"]),
                expected_sha256=(
                    None
                    if relative in self._verified_shards
                    else str(locator["episode_shard_sha256"])
                ),
            )
            self._verified_shards.add(relative)
        self._shards[relative] = episodes
        while len(self._shards) > self.max_cached_shards:
            self._shards.popitem(last=False)
        offset = int(locator["offset"])
        if offset < 0 or offset >= len(episodes):
            raise RuntimeError(f"Joint episode offset is invalid for `{episode_id}`.")
        episode = episodes[offset]
        if episode.episode_id != episode_id:
            raise RuntimeError(f"Joint episode locator mismatch for `{episode_id}`.")
        return episode

    def __getitem__(self, index: int) -> JointWindowSample:
        record = self.windows[index]
        episode = self._load_episode(record.episode_id)
        _validate_window_against_episode(record, episode)
        current = record.current_latent_index
        future = record.future_latent_index
        available = torch.tensor(record.code_available, dtype=torch.bool)
        current_codes = episode.code_ids[current].clone()
        future_codes = episode.code_ids[future].clone()
        return JointWindowSample(
            record=record,
            latents=episode.latents[
                record.state_latent_start : record.state_latent_stop
            ],
            latent_valid=episode.latent_valid[
                record.state_latent_start : record.state_latent_stop
            ],
            proprio_history=episode.source_proprio[
                record.proprio_start : record.proprio_stop
            ],
            past_actions=episode.source_actions[
                record.past_action_start : record.past_action_stop
            ],
            actions=episode.source_actions[
                record.action_start : record.action_stop
            ],
            action_valid=episode.source_action_valid[
                record.action_start : record.action_stop
            ],
            current_codes=current_codes,
            future_codes=future_codes,
            code_available=available,
            language_tokens=episode.language_tokens,
            language_valid=episode.language_valid,
        )

    def action_chunk(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        local_position = _sequence_position(index, len(self))
        position = (
            local_position
            if self._window_positions is None
            else int(self._window_positions[local_position])
        )
        actions = self._action_chunks[position]
        valid = self._action_valid[position]
        record = self.windows[index]
        expected_horizon = record.action_stop - record.action_start
        if (
            tuple(actions.shape)
            != (expected_horizon, int(self.contract["action_dim"]))
            or tuple(valid.shape) != (expected_horizon,)
        ):
            raise RuntimeError(
                f"Joint window `{record.window_id}` compact action changed."
            )
        return actions, valid


@dataclass(frozen=True)
class JointModelBatch:
    model: CodeWAMBatch
    window_ids: tuple[str, ...]
    episode_ids: tuple[str, ...]
    parent_episode_ids: tuple[str, ...]
    splits: tuple[str, ...]
    descriptor_overlap: torch.Tensor


def _pad_sequence(
    values: Sequence[torch.Tensor],
    *,
    left: bool,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not values:
        raise ValueError("Cannot pad an empty sequence list.")
    trailing = tuple(values[0].shape[1:])
    if any(tuple(value.shape[1:]) != trailing for value in values):
        raise ValueError("Padded sequences must share trailing dimensions.")
    maximum = max(int(value.shape[0]) for value in values)
    output = values[0].new_full(
        (len(values), maximum, *trailing),
        pad_value,
    )
    valid = torch.zeros((len(values), maximum), dtype=torch.bool)
    for index, value in enumerate(values):
        length = int(value.shape[0])
        start = maximum - length if left else 0
        output[index, start : start + length] = value
        valid[index, start : start + length] = True
    return output, valid


def collate_joint_windows(
    samples: Sequence[JointWindowSample],
    *,
    language_dim: int,
    require_language_for_action: bool = True,
) -> JointModelBatch:
    if not samples:
        raise ValueError("Cannot collate an empty joint window batch.")
    if language_dim <= 0:
        raise ValueError("Joint batch language width must be positive.")
    latents, latent_time_valid = _pad_sequence(
        [sample.latents for sample in samples],
        left=True,
    )
    latent_view_valid, _ = _pad_sequence(
        [sample.latent_valid for sample in samples],
        left=True,
    )
    latent_valid = latent_view_valid & latent_time_valid[:, :, None]
    proprio, proprio_valid = _pad_sequence(
        [sample.proprio_history for sample in samples],
        left=True,
    )
    past_actions, past_action_valid = _pad_sequence(
        [sample.past_actions for sample in samples],
        left=True,
    )
    actions, action_padding_valid = _pad_sequence(
        [sample.actions for sample in samples],
        left=False,
    )
    explicit_action_valid, _ = _pad_sequence(
        [sample.action_valid for sample in samples],
        left=False,
    )
    action_valid = action_padding_valid & explicit_action_valid
    language_values = []
    language_availability = []
    for sample in samples:
        if sample.language_tokens is None:
            language_values.append(torch.zeros((1, language_dim)))
            language_availability.append(torch.zeros(1, dtype=torch.bool))
            continue
        if sample.language_tokens.shape[1] != language_dim:
            raise ValueError("Cached language token width changed within a batch.")
        language_values.append(sample.language_tokens)
        language_availability.append(
            sample.language_valid
            if sample.language_valid is not None
            else torch.ones(
                sample.language_tokens.shape[0],
                dtype=torch.bool,
            )
        )
    language, language_padding_valid = _pad_sequence(
        language_values,
        left=False,
    )
    language_explicit_valid, _ = _pad_sequence(
        language_availability,
        left=False,
    )
    language_valid = language_padding_valid & language_explicit_valid
    action_available = action_valid.any(dim=1)
    if require_language_for_action:
        imitation_available = action_available & language_valid.any(dim=1)
    else:
        imitation_available = action_available
    roles = tuple(sample.record.role for sample in samples)
    supervision = build_supervision_masks(
        roles,
        action_available=imitation_available,
    )
    supervision = type(supervision)(
        temporal=supervision.temporal,
        action=supervision.action,
        dynamics=build_supervision_masks(
            roles,
            action_available=action_available,
        ).dynamics,
    )
    current_codes = torch.stack(
        [sample.current_codes for sample in samples]
    ).long()
    future_codes = torch.stack(
        [sample.future_codes for sample in samples]
    ).long()
    available = torch.stack(
        [sample.code_available for sample in samples]
    ).bool()
    batch = CodeWAMBatch(
        state=StateInputs(
            latents=latents,
            proprio_history=proprio,
            past_actions=past_actions,
            latent_valid=latent_valid,
            proprio_valid=proprio_valid,
            past_action_valid=past_action_valid,
        ),
        policy=PolicyCondition(
            language=language,
            language_valid=language_valid,
        ),
        actions=ActionBatch(values=actions, valid=action_valid),
        supervision=supervision,
        codes=CodeMeasurements(
            code_ids=current_codes,
            available=available,
            chart_names=tuple(
                sample.record.chart_name for sample in samples
            ),
        ),
        future_codes=FutureCodeTargets(
            code_ids=future_codes,
            available=available,
        ),
    )
    return JointModelBatch(
        model=batch,
        window_ids=tuple(sample.record.window_id for sample in samples),
        episode_ids=tuple(sample.record.episode_id for sample in samples),
        parent_episode_ids=tuple(
            sample.record.parent_episode_id for sample in samples
        ),
        splits=tuple(sample.record.split for sample in samples),
        descriptor_overlap=torch.tensor(
            [sample.record.descriptor_overlap for sample in samples],
            dtype=torch.long,
        ),
    )
