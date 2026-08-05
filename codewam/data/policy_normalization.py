from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from codewam.codebook_eval.shards import atomic_torch_save, file_sha256
from codewam.models import ActionBatch, CodeWAMBatch, StateInputs


POLICY_NORMALIZATION_SCHEMA = "codewam.policy-normalization.v1"
POLICY_NORMALIZATION_STATS_SCHEMA = "codewam.policy-normalization-stats.v1"
POLICY_NORMALIZATION_SUMMARY_SCHEMA = "codewam.policy-normalization-summary.v1"
DROID_POLICY_REPRESENTATION = "droid-cartesian-position-euler-sincos-v1"
DROID_ACTION_SEMANTICS = (
    "commanded cartesian position xyz/euler plus commanded gripper position"
)


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def encode_droid_actions(values: torch.Tensor) -> torch.Tensor:
    """Encode xyz/euler/gripper actions without an Euler wrap discontinuity."""
    if values.shape[-1] != 7:
        raise ValueError(f"Raw DROID actions must end in 7, got {values.shape}.")
    euler = values[..., 3:6]
    return torch.cat(
        (
            values[..., :3],
            torch.sin(euler),
            torch.cos(euler),
            values[..., 6:7],
        ),
        dim=-1,
    )


def decode_droid_actions(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] != 10:
        raise ValueError(f"Encoded DROID actions must end in 10, got {values.shape}.")
    euler = torch.atan2(values[..., 3:6], values[..., 6:9])
    return torch.cat((values[..., :3], euler, values[..., 9:10]), dim=-1)


def encode_droid_proprio(values: torch.Tensor) -> torch.Tensor:
    """Encode Cartesian Euler state while preserving joints and gripper."""
    if values.shape[-1] != 14:
        raise ValueError(f"Raw DROID proprio must end in 14, got {values.shape}.")
    euler = values[..., 3:6]
    return torch.cat(
        (
            values[..., :3],
            torch.sin(euler),
            torch.cos(euler),
            values[..., 6:],
        ),
        dim=-1,
    )


def create_policy_normalization_contract(
    *,
    joint_cache_contract_hash: str,
    joint_cache_summary_sha256: str,
    implementation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    payload = {
        "schema": POLICY_NORMALIZATION_SCHEMA,
        "joint_cache": {
            "contract_hash": str(joint_cache_contract_hash),
            "summary_sha256": str(joint_cache_summary_sha256),
        },
        "fit": {
            "split": "train",
            "weighting": "referenced-source-step-once",
            "variance": "population",
            "minimum_std": 1e-6,
        },
        "representation": {
            "name": DROID_POLICY_REPRESENTATION,
            "action_semantics": DROID_ACTION_SEMANTICS,
            "raw_action_dim": 7,
            "encoded_action_dim": 10,
            "raw_proprio_dim": 14,
            "encoded_proprio_dim": 17,
            "euler_encoding": "sin-then-cos",
        },
        "implementation_sha256": dict(sorted(implementation_sha256.items())),
    }
    return {**payload, "contract_hash": _canonical_hash(payload)}


def validate_policy_normalization_contract(contract: Mapping[str, Any]) -> None:
    payload = dict(contract)
    contract_hash = str(payload.pop("contract_hash", ""))
    if (
        payload.get("schema") != POLICY_NORMALIZATION_SCHEMA
        or contract_hash != _canonical_hash(payload)
        or payload.get("representation", {}).get("name")
        != DROID_POLICY_REPRESENTATION
    ):
        raise RuntimeError("Policy-normalization contract is invalid.")


def _validate_moment(name: str, value: torch.Tensor, width: int) -> None:
    if (
        value.dtype != torch.float32
        or tuple(value.shape) != (width,)
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"Policy-normalization `{name}` is invalid.")


def write_policy_normalization(
    output_dir: str | Path,
    *,
    contract: Mapping[str, Any],
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    proprio_mean: torch.Tensor,
    proprio_std: torch.Tensor,
    action_rows: int,
    proprio_rows: int,
    source_segments: int,
) -> dict[str, Any]:
    validate_policy_normalization_contract(contract)
    if min(action_rows, proprio_rows, source_segments) <= 0:
        raise ValueError("Policy-normalization fit counts must be positive.")
    for name, value, width in (
        ("action_mean", action_mean, 10),
        ("action_std", action_std, 10),
        ("proprio_mean", proprio_mean, 17),
        ("proprio_std", proprio_std, 17),
    ):
        _validate_moment(name, value, width)
        if name.endswith("std") and (value <= 0).any():
            raise ValueError(
                "Policy-normalization standard deviations must be positive."
            )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != dict(contract):
            raise RuntimeError("Existing normalization uses another contract.")
    else:
        _atomic_json(contract_path, dict(contract))
    stats_path = output / "stats.pt"
    atomic_torch_save(
        {
            "schema": POLICY_NORMALIZATION_STATS_SCHEMA,
            "contract_hash": contract["contract_hash"],
            "action_mean": action_mean.detach().float().cpu(),
            "action_std": action_std.detach().float().cpu(),
            "proprio_mean": proprio_mean.detach().float().cpu(),
            "proprio_std": proprio_std.detach().float().cpu(),
        },
        stats_path,
    )
    summary = {
        "schema": POLICY_NORMALIZATION_SUMMARY_SCHEMA,
        "contract_hash": contract["contract_hash"],
        "counts": {
            "action_rows": int(action_rows),
            "proprio_rows": int(proprio_rows),
            "source_segments": int(source_segments),
        },
        "stats": {
            "path": stats_path.name,
            "sha256": file_sha256(stats_path),
            "bytes": stats_path.stat().st_size,
        },
    }
    _atomic_json(output / "summary.json", summary)
    return summary


class PolicyNormalizer:
    """Immutable train-only normalization and reversible action representation."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_joint_cache_contract_hash: str | None = None,
        verify_hashes: bool = True,
    ):
        self.root = Path(root)
        self.contract = json.loads(
            (self.root / "contract.json").read_text(encoding="utf-8")
        )
        validate_policy_normalization_contract(self.contract)
        if (
            expected_joint_cache_contract_hash is not None
            and self.contract["joint_cache"]["contract_hash"]
            != expected_joint_cache_contract_hash
        ):
            raise RuntimeError("Normalization belongs to a different joint cache.")
        self.summary = json.loads(
            (self.root / "summary.json").read_text(encoding="utf-8")
        )
        if (
            self.summary.get("schema") != POLICY_NORMALIZATION_SUMMARY_SCHEMA
            or self.summary.get("contract_hash") != self.contract["contract_hash"]
        ):
            raise RuntimeError("Policy-normalization summary is invalid.")
        stats_path = self.root / self.summary["stats"]["path"]
        if verify_hashes and file_sha256(stats_path) != self.summary["stats"]["sha256"]:
            raise RuntimeError("Policy-normalization stats hash changed.")
        payload = torch.load(
            stats_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if (
            payload.get("schema") != POLICY_NORMALIZATION_STATS_SCHEMA
            or payload.get("contract_hash") != self.contract["contract_hash"]
        ):
            raise RuntimeError("Policy-normalization stats are invalid.")
        for name, width in (
            ("action_mean", 10),
            ("action_std", 10),
            ("proprio_mean", 17),
            ("proprio_std", 17),
        ):
            value = payload.get(name)
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(f"Policy-normalization `{name}` is missing.")
            _validate_moment(name, value, width)
            setattr(self, name, value)

    @property
    def action_dim(self) -> int:
        return 10

    @property
    def proprio_dim(self) -> int:
        return 17

    @staticmethod
    def _on(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return value.to(device=reference.device, dtype=reference.dtype)

    def normalize_actions(self, raw: torch.Tensor) -> torch.Tensor:
        encoded = encode_droid_actions(raw)
        return (
            encoded - self._on(self.action_mean, encoded)
        ) / self._on(self.action_std, encoded)

    def denormalize_actions(self, normalized: torch.Tensor) -> torch.Tensor:
        encoded = (
            normalized * self._on(self.action_std, normalized)
            + self._on(self.action_mean, normalized)
        )
        return decode_droid_actions(encoded)

    def normalize_proprio(self, raw: torch.Tensor) -> torch.Tensor:
        encoded = encode_droid_proprio(raw)
        return (
            encoded - self._on(self.proprio_mean, encoded)
        ) / self._on(self.proprio_std, encoded)

    def transform_batch(self, batch: CodeWAMBatch) -> CodeWAMBatch:
        state = batch.state
        return replace(
            batch,
            state=StateInputs(
                latents=state.latents,
                proprio_history=self.normalize_proprio(state.proprio_history),
                past_actions=self.normalize_actions(state.past_actions),
                latent_valid=state.latent_valid,
                proprio_valid=state.proprio_valid,
                past_action_valid=state.past_action_valid,
                latent_time_offsets=state.latent_time_offsets,
                proprio_time_offsets=state.proprio_time_offsets,
                past_action_time_offsets=state.past_action_time_offsets,
            ),
            actions=ActionBatch(
                values=self.normalize_actions(batch.actions.values),
                valid=batch.actions.valid,
            ),
        )


def moments_from_sums(
    count: int,
    total: torch.Tensor,
    squared_total: torch.Tensor,
    *,
    minimum_std: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if count <= 0 or total.ndim != 1 or total.shape != squared_total.shape:
        raise ValueError("Moment aggregates are empty or misaligned.")
    mean = total.double() / count
    variance = squared_total.double() / count - mean.square()
    variance = variance.clamp_min(0.0)
    std = variance.sqrt().clamp_min(float(minimum_std))
    return mean.float(), std.float()


def combine_encoded_rows(
    rows: Sequence[torch.Tensor],
) -> tuple[int, torch.Tensor, torch.Tensor]:
    if not rows:
        raise ValueError("Cannot aggregate an empty policy feature sequence.")
    width = int(rows[0].shape[-1])
    total = torch.zeros(width, dtype=torch.float64)
    squared = torch.zeros_like(total)
    count = 0
    for value in rows:
        if value.ndim != 2 or value.shape[1] != width:
            raise ValueError("Policy feature rows have inconsistent widths.")
        value = value.double()
        if not torch.isfinite(value).all():
            raise ValueError("Policy feature rows contain non-finite values.")
        total += value.sum(dim=0)
        squared += value.square().sum(dim=0)
        count += int(value.shape[0])
    return count, total, squared
