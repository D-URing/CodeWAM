from __future__ import annotations

from dataclasses import dataclass

import torch


def _require_shape(name: str, value: torch.Tensor, ndim: int) -> None:
    if value.ndim != ndim:
        raise ValueError(f"`{name}` must be {ndim}D, got {tuple(value.shape)}.")


def _require_bool_mask(
    name: str,
    value: torch.Tensor | None,
    shape: tuple[int, ...],
) -> None:
    if value is None:
        return
    if value.dtype != torch.bool or tuple(value.shape) != shape:
        raise ValueError(
            f"`{name}` must be bool with shape {shape}, got "
            f"{value.dtype} {tuple(value.shape)}."
        )


def _require_time_offsets(
    name: str,
    value: torch.Tensor | None,
    shape: tuple[int, ...],
) -> None:
    if value is None:
        return
    if not torch.is_floating_point(value) or tuple(value.shape) != shape:
        raise ValueError(
            f"`{name}` must be floating point with shape {shape}, got "
            f"{value.dtype} {tuple(value.shape)}."
        )
    if value.device.type != "cpu" or value.numel() == 0:
        return
    if not torch.isfinite(value).all():
        raise ValueError(f"`{name}` must contain finite relative times.")
    if value.shape[1] > 1 and (value[:, 1:] < value[:, :-1]).any():
        raise ValueError(f"`{name}` must be nondecreasing within each sample.")
    if (value > 1e-6).any():
        raise ValueError(f"`{name}` cannot contain information after decision time.")


@dataclass(frozen=True)
class StateInputs:
    """Task-free observations available before the current action is generated."""

    latents: torch.Tensor
    proprio_history: torch.Tensor
    past_actions: torch.Tensor
    latent_valid: torch.Tensor | None = None
    proprio_valid: torch.Tensor | None = None
    past_action_valid: torch.Tensor | None = None
    latent_time_offsets: torch.Tensor | None = None
    proprio_time_offsets: torch.Tensor | None = None
    past_action_time_offsets: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _require_shape("latents", self.latents, 6)
        _require_shape("proprio_history", self.proprio_history, 3)
        _require_shape("past_actions", self.past_actions, 3)
        batch, time, views = self.latents.shape[:3]
        if self.proprio_history.shape[0] != batch:
            raise ValueError("Latents and proprio history must share the batch size.")
        if self.past_actions.shape[0] != batch:
            raise ValueError("Latents and past actions must share the batch size.")
        if time <= 0 or views <= 0:
            raise ValueError("Latent time and view dimensions must be positive.")
        if self.proprio_history.shape[1] <= 0:
            raise ValueError("Proprio history must contain the current state.")
        _require_bool_mask(
            "latent_valid",
            self.latent_valid,
            (batch, time, views),
        )
        _require_bool_mask(
            "proprio_valid",
            self.proprio_valid,
            tuple(self.proprio_history.shape[:2]),
        )
        _require_bool_mask(
            "past_action_valid",
            self.past_action_valid,
            tuple(self.past_actions.shape[:2]),
        )
        _require_time_offsets(
            "latent_time_offsets",
            self.latent_time_offsets,
            (batch, time),
        )
        _require_time_offsets(
            "proprio_time_offsets",
            self.proprio_time_offsets,
            tuple(self.proprio_history.shape[:2]),
        )
        _require_time_offsets(
            "past_action_time_offsets",
            self.past_action_time_offsets,
            tuple(self.past_actions.shape[:2]),
        )

    @property
    def batch_size(self) -> int:
        return int(self.latents.shape[0])

    @property
    def current_proprio(self) -> torch.Tensor:
        return self.proprio_history[:, -1]


@dataclass(frozen=True)
class CodeMeasurements:
    """Frozen RQ assignments for one chart per sample."""

    code_ids: torch.Tensor
    available: torch.Tensor
    chart_names: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_shape("code_ids", self.code_ids, 3)
        _require_shape("available", self.available, 2)
        if self.code_ids.dtype != torch.long:
            raise ValueError("`code_ids` must use torch.long.")
        if self.available.dtype != torch.bool:
            raise ValueError("`available` must use torch.bool.")
        batch, families, _ = self.code_ids.shape
        if tuple(self.available.shape) != (batch, families):
            raise ValueError("Code availability must be batch/family aligned.")
        if len(self.chart_names) != batch or any(not value for value in self.chart_names):
            raise ValueError("One nonempty chart name is required per sample.")

    @property
    def batch_size(self) -> int:
        return int(self.code_ids.shape[0])


@dataclass(frozen=True)
class PolicyCondition:
    """Task condition that is intentionally excluded from world belief."""

    language: torch.Tensor
    language_valid: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _require_shape("language", self.language, 3)
        _require_bool_mask(
            "language_valid",
            self.language_valid,
            tuple(self.language.shape[:2]),
        )

    @property
    def batch_size(self) -> int:
        return int(self.language.shape[0])


@dataclass(frozen=True)
class ActionBatch:
    values: torch.Tensor
    valid: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _require_shape("action values", self.values, 3)
        if self.values.shape[1] <= 0 or self.values.shape[2] <= 0:
            raise ValueError("Action horizon and dimension must be positive.")
        _require_bool_mask("action valid", self.valid, tuple(self.values.shape[:2]))

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])


@dataclass(frozen=True)
class TransitionSchedule:
    """Per-family action prefix and physical horizon for one transition target."""

    action_prefix_lengths: torch.Tensor
    delta_times: torch.Tensor

    def __post_init__(self) -> None:
        _require_shape(
            "transition action prefix lengths",
            self.action_prefix_lengths,
            2,
        )
        _require_shape("transition delta times", self.delta_times, 2)
        if self.action_prefix_lengths.dtype != torch.long:
            raise ValueError("Transition action prefix lengths must use torch.long.")
        if not torch.is_floating_point(self.delta_times):
            raise ValueError("Transition delta times must be floating point.")
        if self.action_prefix_lengths.shape != self.delta_times.shape:
            raise ValueError("Transition prefix lengths and delta times must align.")
        if self.action_prefix_lengths.shape[1] <= 0:
            raise ValueError("Transition schedules need at least one clock family.")
        if self.action_prefix_lengths.device.type == "cpu":
            if (self.action_prefix_lengths < 0).any():
                raise ValueError("Transition action prefix lengths cannot be negative.")
            if not torch.isfinite(self.delta_times).all() or (
                self.delta_times <= 0
            ).any():
                raise ValueError("Transition delta times must be finite and positive.")

    @property
    def batch_size(self) -> int:
        return int(self.action_prefix_lengths.shape[0])

    @property
    def families(self) -> int:
        return int(self.action_prefix_lengths.shape[1])


@dataclass(frozen=True)
class FutureCodeTargets:
    code_ids: torch.Tensor
    available: torch.Tensor
    schedule: TransitionSchedule | None = None

    def __post_init__(self) -> None:
        _require_shape("future code ids", self.code_ids, 3)
        _require_shape("future code availability", self.available, 2)
        if self.code_ids.dtype != torch.long:
            raise ValueError("Future code IDs must use torch.long.")
        if self.available.dtype != torch.bool:
            raise ValueError("Future code availability must use torch.bool.")
        if tuple(self.available.shape) != tuple(self.code_ids.shape[:2]):
            raise ValueError("Future code availability must be batch/family aligned.")
        if self.schedule is not None and (
            self.schedule.batch_size != int(self.code_ids.shape[0])
            or self.schedule.families != int(self.code_ids.shape[1])
        ):
            raise ValueError("Future code targets and transition schedule must align.")


@dataclass(frozen=True)
class SupervisionMasks:
    """Per-sample objective availability derived from trajectory roles."""

    temporal: torch.Tensor
    action: torch.Tensor
    dynamics: torch.Tensor

    def __post_init__(self) -> None:
        shape = tuple(self.temporal.shape)
        if len(shape) != 1:
            raise ValueError("Supervision masks must be one-dimensional.")
        for name, value in (
            ("temporal", self.temporal),
            ("action", self.action),
            ("dynamics", self.dynamics),
        ):
            if value.dtype != torch.bool or tuple(value.shape) != shape:
                raise ValueError(
                    f"`{name}` supervision must be bool with shape {shape}."
                )

    @property
    def batch_size(self) -> int:
        return int(self.temporal.shape[0])


@dataclass(frozen=True)
class CodeWAMBatch:
    state: StateInputs
    policy: PolicyCondition
    actions: ActionBatch
    supervision: SupervisionMasks
    codes: CodeMeasurements | None = None
    future_codes: FutureCodeTargets | None = None

    def __post_init__(self) -> None:
        batch = self.state.batch_size
        sizes = {
            "policy": self.policy.batch_size,
            "actions": self.actions.batch_size,
            "supervision": self.supervision.batch_size,
        }
        if self.codes is not None:
            sizes["codes"] = self.codes.batch_size
        if self.future_codes is not None:
            sizes["future_codes"] = int(self.future_codes.code_ids.shape[0])
        mismatches = {name: size for name, size in sizes.items() if size != batch}
        if mismatches:
            raise ValueError(
                f"CodeWAM batch components do not share batch size {batch}: {mismatches}."
            )


@dataclass(frozen=True)
class ContinuousState:
    tokens: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self) -> None:
        _require_shape("continuous state tokens", self.tokens, 3)
        _require_bool_mask(
            "continuous state valid",
            self.valid,
            tuple(self.tokens.shape[:2]),
        )


@dataclass(frozen=True)
class CodeTokens:
    tokens: torch.Tensor
    valid: torch.Tensor
    families: tuple[str, ...]
    levels: int

    def __post_init__(self) -> None:
        _require_shape("code tokens", self.tokens, 3)
        _require_bool_mask("code token valid", self.valid, tuple(self.tokens.shape[:2]))
        if self.tokens.shape[1] != len(self.families) * int(self.levels):
            raise ValueError("Code token count does not match family/level identities.")


@dataclass(frozen=True)
class MultiClockCodeState:
    """One code token per clock plus cumulative RQ-prefix evidence."""

    tokens: torch.Tensor
    prefix_tokens: torch.Tensor
    valid: torch.Tensor
    families: tuple[str, ...]
    levels: int

    def __post_init__(self) -> None:
        _require_shape("multi-clock code tokens", self.tokens, 3)
        _require_shape("multi-clock RQ prefix tokens", self.prefix_tokens, 4)
        _require_bool_mask(
            "multi-clock code validity",
            self.valid,
            tuple(self.tokens.shape[:2]),
        )
        batch, families, width = self.tokens.shape
        expected = (batch, families, int(self.levels), width)
        if tuple(self.prefix_tokens.shape) != expected:
            raise ValueError(
                f"RQ prefix tokens must be {expected}, got "
                f"{tuple(self.prefix_tokens.shape)}."
            )
        if families != len(self.families) or self.levels <= 0:
            raise ValueError("Multi-clock family/depth identities are misaligned.")


@dataclass(frozen=True)
class WorldBelief:
    tokens: torch.Tensor

    def __post_init__(self) -> None:
        _require_shape("world belief tokens", self.tokens, 3)


@dataclass(frozen=True)
class StructuredWorldState:
    """Control state retaining global belief and local continuous evidence."""

    belief: WorldBelief
    continuous: ContinuousState
    codes: MultiClockCodeState | None = None

    def __post_init__(self) -> None:
        batch, _, width = self.belief.tokens.shape
        if (
            self.continuous.tokens.shape[0] != batch
            or self.continuous.tokens.shape[2] != width
        ):
            raise ValueError("Global belief and continuous detail must align.")
        if self.codes is not None and (
            self.codes.tokens.shape[0] != batch
            or self.codes.tokens.shape[2] != width
        ):
            raise ValueError("Global belief and multi-clock codes must align.")
