from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import torch

from codewam.codebook_eval.manifest import EpisodeRecord
from codewam.models.contracts import SupervisionMasks


class TrajectoryRole(str, Enum):
    EXPERT = "expert"
    FAILURE = "failure"
    RECOVERY = "recovery"
    UNLABELED_INTERACTION = "unlabeled_interaction"
    ACTION_FREE_VIDEO = "action_free_video"


@dataclass(frozen=True)
class RoleSupervision:
    codebook_fit: bool
    temporal_pretraining: bool
    action_imitation: bool
    code_dynamics: bool


ROLE_SUPERVISION: dict[TrajectoryRole, RoleSupervision] = {
    TrajectoryRole.EXPERT: RoleSupervision(True, True, True, True),
    TrajectoryRole.FAILURE: RoleSupervision(True, True, False, True),
    TrajectoryRole.RECOVERY: RoleSupervision(True, True, True, True),
    TrajectoryRole.UNLABELED_INTERACTION: RoleSupervision(
        True,
        True,
        False,
        True,
    ),
    TrajectoryRole.ACTION_FREE_VIDEO: RoleSupervision(
        True,
        True,
        False,
        False,
    ),
}


def trajectory_role(record: EpisodeRecord) -> TrajectoryRole:
    """Read an explicit role, with a narrow DROID-compatible success fallback."""

    explicit = record.metadata.get("trajectory_role")
    if explicit is not None:
        try:
            return TrajectoryRole(str(explicit))
        except ValueError as exc:
            raise ValueError(
                f"Episode `{record.key}` has unsupported trajectory role `{explicit}`."
            ) from exc
    if bool(record.metadata.get("action_free", False)):
        return TrajectoryRole.ACTION_FREE_VIDEO
    if bool(record.metadata.get("recovery", False)):
        return TrajectoryRole.RECOVERY
    if "success" in record.metadata:
        return (
            TrajectoryRole.EXPERT
            if bool(record.metadata["success"])
            else TrajectoryRole.FAILURE
        )
    raise ValueError(
        f"Episode `{record.key}` needs explicit `trajectory_role`; "
        "only records with success/action_free/recovery metadata have a fallback."
    )


def role_supervision(role: TrajectoryRole | str) -> RoleSupervision:
    return ROLE_SUPERVISION[TrajectoryRole(role)]


def build_supervision_masks(
    roles: Iterable[TrajectoryRole | str],
    *,
    action_available: Iterable[bool] | torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> SupervisionMasks:
    resolved = tuple(TrajectoryRole(value) for value in roles)
    if not resolved:
        raise ValueError("At least one trajectory role is required.")
    target_device = None if device is None else torch.device(device)
    if (
        target_device is None
        and isinstance(action_available, torch.Tensor)
    ):
        target_device = action_available.device
    if action_available is None:
        has_action = torch.ones(
            len(resolved),
            dtype=torch.bool,
            device=target_device,
        )
    elif isinstance(action_available, torch.Tensor):
        if (
            action_available.dtype != torch.bool
            or tuple(action_available.shape) != (len(resolved),)
        ):
            raise ValueError("Action availability must be bool [B].")
        has_action = action_available.to(device=target_device)
    else:
        availability_values = tuple(action_available)
        if len(availability_values) != len(resolved) or any(
            not isinstance(value, bool) for value in availability_values
        ):
            raise ValueError("Action availability must contain one bool per role.")
        has_action = torch.tensor(
            availability_values,
            dtype=torch.bool,
            device=target_device,
        )
    action_by_role = torch.tensor(
        [ROLE_SUPERVISION[value].action_imitation for value in resolved],
        dtype=torch.bool,
        device=target_device,
    )
    dynamics_by_role = torch.tensor(
        [ROLE_SUPERVISION[value].code_dynamics for value in resolved],
        dtype=torch.bool,
        device=target_device,
    )
    return SupervisionMasks(
        temporal=torch.tensor(
            [ROLE_SUPERVISION[value].temporal_pretraining for value in resolved],
            dtype=torch.bool,
            device=target_device,
        ),
        action=action_by_role & has_action,
        dynamics=dynamics_by_role & has_action,
    )


def codebook_fit_records(
    records: Iterable[EpisodeRecord],
) -> tuple[EpisodeRecord, ...]:
    return tuple(
        record
        for record in records
        if ROLE_SUPERVISION[trajectory_role(record)].codebook_fit
    )
