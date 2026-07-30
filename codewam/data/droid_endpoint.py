from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F

from .droid_rlds import DroidRLDSEpisode


DROID_ENDPOINT_AUDIT_SCHEMA = "codewam.droid-endpoint-audit.v1"
DROID_ENDPOINT_POLICY = "observation[t]--action[t:t+h]-->observation[t+h]"


def _cosine_rows(
    actions: torch.Tensor,
    deltas: torch.Tensor,
) -> torch.Tensor:
    if actions.ndim != 2 or deltas.ndim != 2 or actions.shape != deltas.shape:
        raise ValueError("Endpoint cosine inputs must be same-shaped [T,D].")
    action_norm = actions.float().norm(dim=1)
    delta_norm = deltas.float().norm(dim=1)
    valid = (action_norm > 1e-8) & (delta_norm > 1e-8)
    if not valid.any():
        return torch.empty(0, dtype=torch.float32)
    return F.cosine_similarity(
        actions[valid].float(),
        deltas[valid].float(),
        dim=1,
    ).cpu()


def _alignment(
    action: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if action.shape[0] != state.shape[0] or action.shape[0] < 2:
        raise ValueError("Endpoint alignment needs at least two aligned steps.")
    delta = state[1:] - state[:-1]
    return (
        _cosine_rows(action[:-1], delta),
        _cosine_rows(action[1:], delta),
    )


def audit_droid_endpoints(
    episodes: Iterable[DroidRLDSEpisode],
    *,
    minimum_alignment_margin: float = 0.005,
) -> dict[str, Any]:
    """Audit the RLDS current-action to successor-observation convention."""

    if minimum_alignment_margin < 0:
        raise ValueError("Endpoint alignment margin must be non-negative.")
    rows: list[dict[str, Any]] = []
    joint_current: list[torch.Tensor] = []
    joint_shifted: list[torch.Tensor] = []
    cart_current: list[torch.Tensor] = []
    cart_shifted: list[torch.Tensor] = []
    for episode in episodes:
        if (
            episode.is_first is None
            or episode.is_last is None
            or episode.is_terminal is None
        ):
            raise ValueError(
                f"Episode `{episode.episode_id}` lacks required RLDS flags."
            )
        first = episode.is_first.nonzero(as_tuple=False).flatten().tolist()
        last = episode.is_last.nonzero(as_tuple=False).flatten().tolist()
        terminal = episode.is_terminal.nonzero(as_tuple=False).flatten().tolist()
        row: dict[str, Any] = {
            "episode_id": episode.episode_id,
            "steps": episode.steps,
            "is_first_indices": first,
            "is_last_indices": last,
            "is_terminal_indices": terminal,
            "boundary_flags_valid": (
                first == [0]
                and last == [episode.steps - 1]
                and all(value == episode.steps - 1 for value in terminal)
            ),
        }
        joint_action = episode.action_components.get("joint_velocity")
        if joint_action is not None and episode.proprio.shape[1] >= 13:
            current, shifted = _alignment(
                joint_action[:, :7],
                episode.proprio[:, 6:13],
            )
            joint_current.append(current)
            joint_shifted.append(shifted)
            row["joint_current_count"] = int(current.numel())
            row["joint_current_cosine"] = (
                float(current.mean()) if current.numel() else float("nan")
            )
            row["joint_shifted_cosine"] = (
                float(shifted.mean()) if shifted.numel() else float("nan")
            )
        cart_action = episode.action_components.get("cartesian_velocity")
        if cart_action is not None and episode.proprio.shape[1] >= 3:
            current, shifted = _alignment(
                cart_action[:, :3],
                episode.proprio[:, :3],
            )
            cart_current.append(current)
            cart_shifted.append(shifted)
            row["cartesian_current_count"] = int(current.numel())
            row["cartesian_current_cosine"] = (
                float(current.mean()) if current.numel() else float("nan")
            )
            row["cartesian_shifted_cosine"] = (
                float(shifted.mean()) if shifted.numel() else float("nan")
            )
        rows.append(row)
    if not rows:
        raise ValueError("Endpoint audit received no DROID episodes.")

    def summarize(
        current_values: list[torch.Tensor],
        shifted_values: list[torch.Tensor],
    ) -> dict[str, float | int] | None:
        current = torch.cat(
            [value for value in current_values if value.numel()],
            dim=0,
        ) if any(value.numel() for value in current_values) else None
        shifted = torch.cat(
            [value for value in shifted_values if value.numel()],
            dim=0,
        ) if any(value.numel() for value in shifted_values) else None
        if current is None or shifted is None:
            return None
        return {
            "count": int(current.numel()),
            "current_action_cosine": float(current.mean()),
            "shifted_action_cosine": float(shifted.mean()),
            "current_minus_shifted": float(current.mean() - shifted.mean()),
        }

    joint = summarize(joint_current, joint_shifted)
    cartesian = summarize(cart_current, cart_shifted)
    alignments = [
        value
        for value in (joint, cartesian)
        if value is not None
    ]
    flags_pass = all(bool(row["boundary_flags_valid"]) for row in rows)
    alignment_pass = bool(alignments) and all(
        float(value["current_minus_shifted"]) >= minimum_alignment_margin
        for value in alignments
    )
    return {
        "schema": DROID_ENDPOINT_AUDIT_SCHEMA,
        "endpoint_policy": DROID_ENDPOINT_POLICY,
        "official_rlds_semantics": (
            "A step stores the current observation and the action taken from it; "
            "is_last actions are invalid."
        ),
        "episodes": len(rows),
        "steps": sum(int(row["steps"]) for row in rows),
        "minimum_alignment_margin": minimum_alignment_margin,
        "boundary_flags_pass": flags_pass,
        "alignment_pass": alignment_pass,
        "verdict": "pass" if flags_pass and alignment_pass else "fail",
        "joint_velocity_alignment": joint,
        "cartesian_velocity_alignment": cartesian,
        "rows": rows,
    }
