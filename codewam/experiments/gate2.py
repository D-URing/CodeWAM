from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler

from codewam.codebook_eval.shards import atomic_torch_save, file_sha256
from codewam.data.frozen_assignment import (
    FrozenArtifactChart,
    load_frozen_artifact_chart,
)
from codewam.data.joint_cache import (
    JointModelBatch,
    JointWindowCache,
    JointWindowRecord,
    JointWindowSample,
    collate_joint_windows,
)
from codewam.models import (
    ActionBatch,
    CodeMeasurements,
    CodeWAMBatch,
    CodeWAMConfig,
    FutureCodePrediction,
    FutureCodeTargets,
    PolicyCondition,
    StateInputs,
    SupervisionMasks,
    build_codewam_v1,
    encode_prefix_ids,
    transition_family_masks,
)
from codewam.models.codewam_v1 import CodeWAMV1


GATE2_SCHEMA = "codewam.gate2.v1"
GATE2_PROTOCOL_SCHEMA = "codewam.gate2-protocol.v1"
GATE2_CHECKPOINT_SCHEMA = "codewam.gate2-checkpoint.v1"
LEARNED_CONDITIONS = ("NOACT", "TRUE", "SHUFFLE")
ActionMode = Literal["none", "true", "shuffle"]


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


@dataclass(frozen=True)
class Gate2RunConfig:
    cache_dir: str
    output_dir: str
    artifact_paths: dict[str, str]
    chart_name: str = "droid"
    seed: int = 20260730
    batch_size: int = 16
    eval_batch_size: int = 32
    epochs: int = 10
    max_steps: int | None = None
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    device: str = "cuda"
    amp_dtype: str = "bfloat16"
    calibration_bins: int = 15
    bootstrap_samples: int = 2000
    minimum_gate_episodes: int = 30
    model: CodeWAMConfig = field(default_factory=CodeWAMConfig)

    def __post_init__(self) -> None:
        if not self.cache_dir or not self.output_dir or not self.chart_name:
            raise ValueError("Gate2 paths and chart name must not be empty.")
        if set(self.artifact_paths) != {"Q2", "Q3", "Q5"}:
            raise ValueError("Gate2 requires exactly Q2/Q3/Q5 artifact paths.")
        positive = (
            self.batch_size,
            self.eval_batch_size,
            self.epochs,
            self.calibration_bins,
            self.bootstrap_samples,
            self.minimum_gate_episodes,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Gate2 batch, epoch and metric counts must be positive.")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("Gate2 maximum steps must be positive.")
        if self.num_workers < 0:
            raise ValueError("Gate2 worker count must be non-negative.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Gate2 optimizer values are invalid.")
        if not math.isfinite(self.grad_clip_norm) or self.grad_clip_norm <= 0:
            raise ValueError("Gate2 gradient clipping must be finite and positive.")
        if self.amp_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"Unsupported Gate2 AMP dtype `{self.amp_dtype}`.")
        if self.model.variant != "C2":
            raise ValueError("Gate2 requires the C2 model variant.")
        if self.model.dropout != 0.0:
            raise ValueError("Gate2 fixes dropout at zero for paired controls.")


@dataclass(frozen=True)
class FixedActionPermutation:
    donor_indices: tuple[int, ...]
    seed: int
    permutation_hash: str
    groups: int
    singleton_groups: int
    cross_episode_fraction: float

    def __post_init__(self) -> None:
        if not self.donor_indices:
            raise ValueError("Action permutation must not be empty.")
        if sorted(self.donor_indices) != list(range(len(self.donor_indices))):
            raise ValueError("Action donors must form a global permutation.")
        payload = {
            "donor_indices": list(self.donor_indices),
            "seed": self.seed,
        }
        if self.permutation_hash != _canonical_hash(payload):
            raise RuntimeError("Action permutation hash is invalid.")


def build_fixed_action_permutation(
    windows: Sequence[JointWindowRecord] | JointWindowCache,
    *,
    seed: int,
) -> FixedActionPermutation:
    if not windows:
        raise ValueError("Cannot permute an empty joint cache.")
    grouped: dict[tuple[str, int], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    window_ids = [""] * len(windows)
    parent_ids = [""] * len(windows)
    splits = [""] * len(windows)
    horizons = [0] * len(windows)
    rows = (
        windows.permutation_rows()
        if isinstance(windows, JointWindowCache)
        else (
            (
                index,
                window.split,
                window.action_stop - window.action_start,
                window.parent_episode_id,
                window.window_id,
            )
            for index, window in enumerate(windows)
        )
    )
    for index, split, horizon, parent_id, window_id in rows:
        grouped[(split, horizon)][parent_id].append(index)
        window_ids[index] = window_id
        parent_ids[index] = parent_id
        splits[index] = split
        horizons[index] = horizon
    donors = [-1] * len(windows)
    singleton_groups = 0
    cross_episode = 0
    paired = 0
    for key, parent_groups in sorted(grouped.items()):
        size = sum(len(indices) for indices in parent_groups.values())
        if size == 1:
            only = next(iter(next(iter(parent_groups.values()))))
            donors[only] = only
            singleton_groups += 1
            continue
        ordered_parents = sorted(
            parent_groups,
            key=lambda parent: (
                -len(parent_groups[parent]),
                hashlib.sha256(
                    f"{seed}|parent|{parent}".encode("utf-8")
                ).digest(),
            ),
        )
        ordered = []
        for parent in ordered_parents:
            ordered.extend(
                sorted(
                    parent_groups[parent],
                    key=lambda index: hashlib.sha256(
                        f"{seed}|window|{window_ids[index]}".encode("utf-8")
                    ).digest(),
                )
            )
        largest_parent = len(parent_groups[ordered_parents[0]])
        # Contiguous parent groups rotated by the largest group attain the
        # lower bound max(0, 2 * largest_parent - size) on same-parent pairs.
        offset = largest_parent if largest_parent < size else 1
        for position, source in enumerate(ordered):
            donor = ordered[(position + offset) % size]
            donors[source] = donor
            paired += 1
            cross_episode += int(
                parent_ids[source] != parent_ids[donor]
            )
            if source == donor:
                raise RuntimeError("Non-singleton action permutation retained itself.")
            if (
                splits[donor] != key[0]
                or horizons[donor] != key[1]
            ):
                raise RuntimeError("Action permutation crossed a protocol group.")
    payload = {"donor_indices": donors, "seed": int(seed)}
    return FixedActionPermutation(
        donor_indices=tuple(donors),
        seed=int(seed),
        permutation_hash=_canonical_hash(payload),
        groups=len(grouped),
        singleton_groups=singleton_groups,
        cross_episode_fraction=(
            float(cross_episode / paired) if paired else float("nan")
        ),
    )


@dataclass(frozen=True)
class _Gate2Sample:
    index: int
    primary: JointWindowSample
    shuffled_actions: torch.Tensor
    shuffled_action_valid: torch.Tensor


class _Gate2Dataset(Dataset[_Gate2Sample]):
    def __init__(
        self,
        cache: JointWindowCache,
        permutation: FixedActionPermutation,
    ):
        if len(cache) != len(permutation.donor_indices):
            raise ValueError("Gate2 cache and action permutation differ in size.")
        self.cache = cache
        self.permutation = permutation

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, index: int) -> _Gate2Sample:
        primary = self.cache[index]
        shuffled_actions, shuffled_action_valid = self.cache.action_chunk(
            self.permutation.donor_indices[index]
        )
        if tuple(primary.actions.shape) != tuple(shuffled_actions.shape):
            raise RuntimeError("Permuted actions changed target shape.")
        return _Gate2Sample(
            index=index,
            primary=primary,
            shuffled_actions=shuffled_actions,
            shuffled_action_valid=shuffled_action_valid,
        )


@dataclass(frozen=True)
class Gate2Batch:
    indices: torch.Tensor
    joint: JointModelBatch
    shuffled_actions: ActionBatch


def _collate_gate2(
    rows: Sequence[_Gate2Sample],
    *,
    language_dim: int,
) -> Gate2Batch:
    joint = collate_joint_windows(
        [row.primary for row in rows],
        language_dim=language_dim,
    )
    shuffled_actions = torch.stack(
        [row.shuffled_actions for row in rows],
        dim=0,
    )
    shuffled_valid = torch.stack(
        [row.shuffled_action_valid for row in rows],
        dim=0,
    )
    return Gate2Batch(
        indices=torch.tensor([row.index for row in rows], dtype=torch.long),
        joint=joint,
        shuffled_actions=ActionBatch(
            values=shuffled_actions,
            valid=shuffled_valid,
        ),
    )


class _IndexSampler(Sampler[int]):
    def __init__(self, indices: Sequence[int]):
        self.indices = tuple(int(value) for value in indices)

    def set_indices(self, indices: Sequence[int]) -> None:
        self.indices = tuple(int(value) for value in indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _split_indices(
    windows: Sequence[JointWindowRecord],
    split: str,
) -> tuple[int, ...]:
    return tuple(
        index for index, window in enumerate(windows) if window.split == split
    )


def _rank_indices(
    indices: Sequence[int],
    *,
    rank: int,
    world_size: int,
    seed: int,
    epoch: int,
    training: bool,
    group_keys: Sequence[str] | None = None,
) -> tuple[int, ...]:
    if rank < 0 or rank >= world_size or world_size <= 0:
        raise ValueError("Gate2 rank/world size is invalid.")
    if group_keys is not None:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            grouped[str(group_keys[index])].append(int(index))
        names = sorted(grouped)
        if training:
            generator = random.Random(seed + 1_000_003 * epoch)
            generator.shuffle(names)
            for name in names:
                generator.shuffle(grouped[name])
        values = [
            index
            for name in names
            for index in grouped[name]
        ]
    else:
        values = list(indices)
    if training:
        if group_keys is None:
            generator = random.Random(seed + 1_000_003 * epoch)
            generator.shuffle(values)
        usable = len(values) - len(values) % world_size
        values = values[:usable]
        per_rank = usable // world_size
        return tuple(values[rank * per_rank : (rank + 1) * per_rank])
    start = rank * len(values) // world_size
    stop = (rank + 1) * len(values) // world_size
    return tuple(values[start:stop])


def _make_loader(
    dataset: _Gate2Dataset,
    sampler: Sampler[int],
    *,
    config: Gate2RunConfig,
    batch_size: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
        persistent_workers=config.num_workers > 0,
        collate_fn=partial(
            _collate_gate2,
            language_dim=config.model.language_dim,
        ),
    )


def _move_optional(
    value: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    return None if value is None else value.to(device, non_blocking=True)


def _move_gate2_batch(
    batch: Gate2Batch,
    device: torch.device,
) -> Gate2Batch:
    source = batch.joint.model
    state = source.state
    moved = CodeWAMBatch(
        state=StateInputs(
            latents=state.latents.to(device, non_blocking=True),
            proprio_history=state.proprio_history.to(device, non_blocking=True),
            past_actions=state.past_actions.to(device, non_blocking=True),
            latent_valid=_move_optional(state.latent_valid, device),
            proprio_valid=_move_optional(state.proprio_valid, device),
            past_action_valid=_move_optional(state.past_action_valid, device),
        ),
        policy=PolicyCondition(
            language=source.policy.language.to(device, non_blocking=True),
            language_valid=_move_optional(source.policy.language_valid, device),
        ),
        actions=ActionBatch(
            values=source.actions.values.to(device, non_blocking=True),
            valid=_move_optional(source.actions.valid, device),
        ),
        supervision=SupervisionMasks(
            temporal=source.supervision.temporal.to(device, non_blocking=True),
            action=source.supervision.action.to(device, non_blocking=True),
            dynamics=source.supervision.dynamics.to(device, non_blocking=True),
        ),
        codes=(
            None
            if source.codes is None
            else CodeMeasurements(
                code_ids=source.codes.code_ids.to(device, non_blocking=True),
                available=source.codes.available.to(device, non_blocking=True),
                chart_names=source.codes.chart_names,
            )
        ),
        future_codes=(
            None
            if source.future_codes is None
            else FutureCodeTargets(
                code_ids=source.future_codes.code_ids.to(
                    device,
                    non_blocking=True,
                ),
                available=source.future_codes.available.to(
                    device,
                    non_blocking=True,
                ),
            )
        ),
    )
    return Gate2Batch(
        indices=batch.indices.to(device, non_blocking=True),
        joint=replace(
            batch.joint,
            model=moved,
            descriptor_overlap=batch.joint.descriptor_overlap.to(
                device,
                non_blocking=True,
            ),
        ),
        shuffled_actions=ActionBatch(
            values=batch.shuffled_actions.values.to(device, non_blocking=True),
            valid=_move_optional(batch.shuffled_actions.valid, device),
        ),
    )


def _condition_actions(
    batch: Gate2Batch,
    mode: ActionMode,
) -> ActionBatch | None:
    if mode == "none":
        return None
    if mode == "true":
        return batch.joint.model.actions
    if mode == "shuffle":
        return batch.shuffled_actions
    raise ValueError(f"Unknown Gate2 action mode `{mode}`.")


def _condition_mode(condition: str) -> ActionMode:
    return {
        "NOACT": "none",
        "TRUE": "true",
        "SHUFFLE": "shuffle",
    }[condition]


@dataclass
class _MetricState:
    family_count: float = 0.0
    normalized_nll_sum: float = 0.0
    family_prefix_nll_sum: float = 0.0
    family_correct_sum: float = 0.0
    classification_count: float = 0.0
    classification_nll_sum: float = 0.0
    classification_correct_sum: float = 0.0
    brier_sum: float = 0.0
    entropy_sum: float = 0.0
    center_mse_sum: float = 0.0
    calibration_count: list[float] = field(default_factory=list)
    calibration_confidence: list[float] = field(default_factory=list)
    calibration_correct: list[float] = field(default_factory=list)

    @classmethod
    def with_bins(cls, bins: int) -> _MetricState:
        return cls(
            calibration_count=[0.0] * bins,
            calibration_confidence=[0.0] * bins,
            calibration_correct=[0.0] * bins,
        )

    def merge(self, other: _MetricState) -> None:
        for name in (
            "family_count",
            "normalized_nll_sum",
            "family_prefix_nll_sum",
            "family_correct_sum",
            "classification_count",
            "classification_nll_sum",
            "classification_correct_sum",
            "brier_sum",
            "entropy_sum",
            "center_mse_sum",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for target, source in (
            (self.calibration_count, other.calibration_count),
            (self.calibration_confidence, other.calibration_confidence),
            (self.calibration_correct, other.calibration_correct),
        ):
            for index, value in enumerate(source):
                target[index] += value

    def report(self, *, classification_unit: str) -> dict[str, Any]:
        family_count = self.family_count
        classification_count = self.classification_count
        if family_count == 0:
            return {
                "family_count": 0,
                "classification_count": 0,
                "normalized_nll": float("nan"),
                "family_prefix_nll": float("nan"),
                "family_prefix_accuracy": float("nan"),
                "classification_nll": float("nan"),
                "classification_accuracy": float("nan"),
                "brier": float("nan"),
                "entropy": float("nan"),
                "ece": float("nan"),
                "normalized_center_mse": float("nan"),
                "classification_unit": classification_unit,
            }
        ece = 0.0
        for count, confidence, correct in zip(
            self.calibration_count,
            self.calibration_confidence,
            self.calibration_correct,
        ):
            if count:
                ece += count / classification_count * abs(
                    correct / count - confidence / count
                )
        return {
            "family_count": int(family_count),
            "classification_count": int(classification_count),
            "normalized_nll": self.normalized_nll_sum / family_count,
            "family_prefix_nll": self.family_prefix_nll_sum / family_count,
            "family_prefix_accuracy": self.family_correct_sum / family_count,
            "classification_nll": (
                self.classification_nll_sum / classification_count
            ),
            "classification_accuracy": (
                self.classification_correct_sum / classification_count
            ),
            "brier": self.brier_sum / classification_count,
            "entropy": self.entropy_sum / classification_count,
            "ece": ece,
            "normalized_center_mse": self.center_mse_sum / family_count,
            "classification_unit": classification_unit,
        }


def _stratum_masks(
    current: CodeMeasurements,
    targets: FutureCodeTargets,
    overlap: torch.Tensor,
    families: Sequence[str],
) -> dict[str, torch.Tensor]:
    transitions = transition_family_masks(current, targets)
    common = transitions["common"]
    masks = {
        "all": common,
        "changed": transitions["changed"],
        "stable": transitions["stable"],
    }
    for family_index, family in enumerate(families):
        family_mask = torch.zeros_like(common)
        family_mask[:, family_index] = common[:, family_index]
        masks[f"family/{family}"] = family_mask
        masks[f"family/{family}/changed"] = (
            family_mask & transitions["changed"]
        )
    for value in range(4):
        masks[f"overlap/{value}"] = common & overlap.eq(value)
        for family_index, family in enumerate(families):
            family_overlap = torch.zeros_like(common)
            family_overlap[:, family_index] = (
                common[:, family_index]
                & overlap[:, family_index].eq(value)
            )
            masks[f"family/{family}/overlap/{value}"] = family_overlap
    return masks


def _prediction_values(
    prediction: FutureCodePrediction,
    targets: FutureCodeTargets,
) -> tuple[
    torch.Tensor,
    list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
]:
    batch, families, levels = targets.code_ids.shape
    family_nll = torch.zeros(
        (batch, families),
        dtype=prediction.logits[0].dtype,
        device=prediction.logits[0].device,
    )
    classification: list[
        list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ] = [[] for _ in range(families)]
    if prediction.mode == "independent":
        head = 0
        for family in range(families):
            for level in range(levels):
                target = targets.code_ids[:, family, level].clamp_min(0)
                logits = prediction.logits[head]
                nll = F.cross_entropy(logits, target, reduction="none")
                family_nll[:, family] += nll
                classification[family].append((logits, target, nll))
                head += 1
    else:
        for family, sizes in enumerate(prediction.codebook_sizes):
            target = encode_prefix_ids(
                targets.code_ids[:, family].clamp_min(0),
                sizes,
            )
            logits = prediction.logits[family]
            nll = F.cross_entropy(logits, target, reduction="none")
            family_nll[:, family] = nll
            classification[family].append((logits, target, nll))
    return family_nll, classification


class Gate2MetricAccumulator:
    def __init__(
        self,
        *,
        families: Sequence[str],
        levels: int,
        calibration_bins: int,
    ):
        self.families = tuple(families)
        self.levels = int(levels)
        self.calibration_bins = int(calibration_bins)
        self.states: dict[str, _MetricState] = {}
        self.episode_changed_nll: dict[str, list[float]] = defaultdict(
            lambda: [0.0, 0.0]
        )
        self.classification_unit: str | None = None

    @torch.no_grad()
    def update(
        self,
        prediction: FutureCodePrediction,
        batch: JointModelBatch,
        adapter: nn.Module,
    ) -> None:
        model_batch = batch.model
        if model_batch.codes is None or model_batch.future_codes is None:
            raise ValueError("Gate2 metrics require current and future codes.")
        targets = model_batch.future_codes
        current = model_batch.codes
        sample_valid = model_batch.supervision.dynamics
        masks = _stratum_masks(
            current,
            targets,
            batch.descriptor_overlap,
            self.families,
        )
        family_nll, classification = _prediction_values(
            prediction,
            targets,
        )
        predicted_ids = prediction.predicted_ids()
        prefix_correct = predicted_ids.eq(targets.code_ids).all(dim=-1)
        self.classification_unit = (
            "rq_level"
            if prediction.mode == "independent"
            else "family_prefix"
        )
        changed_valid = masks["changed"] & sample_valid[:, None]
        normalized_nll = family_nll / float(self.levels)
        for sample, episode_id in enumerate(batch.parent_episode_ids):
            selected = changed_valid[sample]
            if selected.any():
                values = normalized_nll[sample, selected]
                self.episode_changed_nll[episode_id][0] += float(
                    values.sum().cpu()
                )
                self.episode_changed_nll[episode_id][1] += float(
                    values.numel()
                )
        for name, family_mask in masks.items():
            valid = family_mask & sample_valid[:, None]
            count = int(valid.sum().item())
            if count == 0:
                self.states.setdefault(
                    name,
                    _MetricState.with_bins(self.calibration_bins),
                )
                continue
            state = self.states.setdefault(
                name,
                _MetricState.with_bins(self.calibration_bins),
            )
            state.family_count += count
            selected_family_nll = family_nll[valid]
            state.family_prefix_nll_sum += float(
                selected_family_nll.sum().cpu()
            )
            state.normalized_nll_sum += float(
                (selected_family_nll / float(self.levels)).sum().cpu()
            )
            state.family_correct_sum += float(
                prefix_correct[valid].float().sum().cpu()
            )
            center_mse = adapter.normalized_center_mse(
                predicted_ids,
                targets.code_ids,
                available=valid,
                chart_names=current.chart_names,
            )
            state.center_mse_sum += float(center_mse.cpu()) * count
            for family in range(len(self.families)):
                family_valid = valid[:, family]
                if not family_valid.any():
                    continue
                for logits, target, nll in classification[family]:
                    selected_logits = logits[family_valid]
                    selected_target = target[family_valid]
                    selected_nll = nll[family_valid]
                    probabilities = selected_logits.softmax(dim=-1)
                    confidence, predicted = probabilities.max(dim=-1)
                    correct = predicted.eq(selected_target)
                    one_hot = F.one_hot(
                        selected_target,
                        num_classes=selected_logits.shape[-1],
                    ).to(probabilities.dtype)
                    brier = (probabilities - one_hot).square().sum(dim=-1)
                    entropy = -(
                        probabilities
                        * probabilities.clamp_min(1e-12).log()
                    ).sum(dim=-1)
                    amount = int(confidence.numel())
                    state.classification_count += amount
                    state.classification_nll_sum += float(
                        selected_nll.sum().cpu()
                    )
                    state.classification_correct_sum += float(
                        correct.float().sum().cpu()
                    )
                    state.brier_sum += float(brier.sum().cpu())
                    state.entropy_sum += float(entropy.sum().cpu())
                    bin_index = (
                        confidence.mul(self.calibration_bins)
                        .floor()
                        .long()
                        .clamp_max(self.calibration_bins - 1)
                    )
                    for bin_value in range(self.calibration_bins):
                        member = bin_index.eq(bin_value)
                        if member.any():
                            state.calibration_count[bin_value] += int(
                                member.sum().item()
                            )
                            state.calibration_confidence[bin_value] += float(
                                confidence[member].sum().cpu()
                            )
                            state.calibration_correct[bin_value] += float(
                                correct[member].float().sum().cpu()
                            )

    def merge(self, other: Gate2MetricAccumulator) -> None:
        if (
            self.families != other.families
            or self.levels != other.levels
            or self.calibration_bins != other.calibration_bins
        ):
            raise ValueError("Cannot merge different Gate2 metric layouts.")
        for name, other_state in other.states.items():
            self.states.setdefault(
                name,
                _MetricState.with_bins(self.calibration_bins),
            ).merge(other_state)
        for episode, values in other.episode_changed_nll.items():
            self.episode_changed_nll[episode][0] += values[0]
            self.episode_changed_nll[episode][1] += values[1]
        if self.classification_unit is None:
            self.classification_unit = other.classification_unit
        elif (
            other.classification_unit is not None
            and self.classification_unit != other.classification_unit
        ):
            raise ValueError("Cannot merge different classification units.")

    def report(self) -> dict[str, Any]:
        unit = self.classification_unit or "unknown"
        return {
            "strata": {
                name: state.report(classification_unit=unit)
                for name, state in sorted(self.states.items())
            },
            "episode_changed_nll": {
                episode: {
                    "sum": values[0],
                    "count": int(values[1]),
                }
                for episode, values in sorted(
                    self.episode_changed_nll.items()
                )
            },
        }

    def payload(self) -> dict[str, Any]:
        return {
            "families": self.families,
            "levels": self.levels,
            "calibration_bins": self.calibration_bins,
            "classification_unit": self.classification_unit,
            "states": {
                name: asdict(state) for name, state in self.states.items()
            },
            "episode_changed_nll": {
                episode: list(values)
                for episode, values in self.episode_changed_nll.items()
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Gate2MetricAccumulator:
        result = cls(
            families=tuple(payload["families"]),
            levels=int(payload["levels"]),
            calibration_bins=int(payload["calibration_bins"]),
        )
        result.classification_unit = payload.get("classification_unit")
        result.states = {
            str(name): _MetricState(**dict(value))
            for name, value in payload["states"].items()
        }
        for episode, values in payload["episode_changed_nll"].items():
            result.episode_changed_nll[str(episode)] = [
                float(values[0]),
                float(values[1]),
            ]
        return result


@dataclass
class _PersistenceState:
    family_count: float = 0.0
    level_count: float = 0.0
    family_correct_sum: float = 0.0
    level_correct_sum: float = 0.0
    center_mse_sum: float = 0.0

    def merge(self, other: _PersistenceState) -> None:
        self.family_count += other.family_count
        self.level_count += other.level_count
        self.family_correct_sum += other.family_correct_sum
        self.level_correct_sum += other.level_correct_sum
        self.center_mse_sum += other.center_mse_sum

    def report(self) -> dict[str, Any]:
        if self.family_count == 0:
            return {
                "family_count": 0,
                "level_count": 0,
                "family_prefix_accuracy": float("nan"),
                "level_accuracy": float("nan"),
                "normalized_center_mse": float("nan"),
            }
        return {
            "family_count": int(self.family_count),
            "level_count": int(self.level_count),
            "family_prefix_accuracy": (
                self.family_correct_sum / self.family_count
            ),
            "level_accuracy": self.level_correct_sum / self.level_count,
            "normalized_center_mse": (
                self.center_mse_sum / self.family_count
            ),
        }


class PersistenceAccumulator:
    def __init__(self, *, families: Sequence[str], levels: int):
        self.families = tuple(families)
        self.levels = int(levels)
        self.states: dict[str, _PersistenceState] = {}

    @torch.no_grad()
    def update(self, batch: JointModelBatch, adapter: nn.Module) -> None:
        model_batch = batch.model
        if model_batch.codes is None or model_batch.future_codes is None:
            raise ValueError("Persistence requires current and future codes.")
        current = model_batch.codes
        targets = model_batch.future_codes
        masks = _stratum_masks(
            current,
            targets,
            batch.descriptor_overlap,
            self.families,
        )
        level_equal = current.code_ids.eq(targets.code_ids)
        family_equal = level_equal.all(dim=-1)
        sample_valid = model_batch.supervision.dynamics
        for name, family_mask in masks.items():
            valid = family_mask & sample_valid[:, None]
            count = int(valid.sum().item())
            state = self.states.setdefault(name, _PersistenceState())
            if count == 0:
                continue
            level_valid = valid[:, :, None].expand_as(level_equal)
            state.family_count += count
            state.level_count += int(level_valid.sum().item())
            state.family_correct_sum += float(
                family_equal[valid].float().sum().cpu()
            )
            state.level_correct_sum += float(
                level_equal[level_valid].float().sum().cpu()
            )
            center_mse = adapter.normalized_center_mse(
                current.code_ids,
                targets.code_ids,
                available=valid,
                chart_names=current.chart_names,
            )
            state.center_mse_sum += float(center_mse.cpu()) * count

    def merge(self, other: PersistenceAccumulator) -> None:
        if self.families != other.families or self.levels != other.levels:
            raise ValueError("Cannot merge different persistence layouts.")
        for name, other_state in other.states.items():
            self.states.setdefault(name, _PersistenceState()).merge(
                other_state
            )

    def report(self) -> dict[str, Any]:
        return {
            "strata": {
                name: state.report()
                for name, state in sorted(self.states.items())
            }
        }

    def payload(self) -> dict[str, Any]:
        return {
            "families": self.families,
            "levels": self.levels,
            "states": {
                name: asdict(state) for name, state in self.states.items()
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PersistenceAccumulator:
        result = cls(
            families=tuple(payload["families"]),
            levels=int(payload["levels"]),
        )
        result.states = {
            str(name): _PersistenceState(**dict(value))
            for name, value in payload["states"].items()
        }
        return result


@dataclass(frozen=True)
class _DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    initialized_here: bool

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def _distributed_context(device_name: str) -> _DistributedContext:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", str(rank)))
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed Gate2 currently requires CUDA/NCCL.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device("cuda", local_rank),
        )
        initialized_here = True
    if world_size > 1:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_name)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Gate2 requested unavailable CUDA.")
            torch.cuda.set_device(device)
    return _DistributedContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        initialized_here=initialized_here,
    )


def _barrier(context: _DistributedContext) -> None:
    if context.world_size > 1:
        dist.barrier()


def _broadcast_action_permutation(
    permutation: FixedActionPermutation | None,
    context: _DistributedContext,
) -> FixedActionPermutation:
    if context.world_size == 1:
        if permutation is None:
            raise RuntimeError("Primary Gate2 rank lost its action permutation.")
        return permutation
    payload = [permutation if context.is_primary else None]
    dist.broadcast_object_list(payload, src=0)
    received = payload[0]
    if not isinstance(received, FixedActionPermutation):
        raise RuntimeError("Gate2 action permutation broadcast is invalid.")
    return received


def _gather_payloads(
    payload: dict[str, Any],
    context: _DistributedContext,
) -> list[dict[str, Any]]:
    if context.world_size == 1:
        return [payload]
    outputs: list[dict[str, Any] | None] = [None] * context.world_size
    dist.all_gather_object(outputs, payload)
    return [value for value in outputs if value is not None]


class _Gate2DynamicsModel(nn.Module):
    def __init__(self, model: CodeWAMV1):
        super().__init__()
        self.codewam = model

    def forward(
        self,
        state: StateInputs,
        codes: CodeMeasurements,
        actions: ActionBatch | None,
    ) -> FutureCodePrediction:
        belief = self.codewam.build_belief(state, codes)
        return self.codewam.code_dynamics(belief, actions)


def _unwrap_model(
    model: _Gate2DynamicsModel | DistributedDataParallel,
) -> _Gate2DynamicsModel:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def _autocast_context(
    config: Gate2RunConfig,
    device: torch.device,
):
    enabled = device.type == "cuda" and config.amp_dtype != "float32"
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[config.amp_dtype]
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=enabled,
    )


def _validate_cache_and_chart(
    cache: JointWindowCache,
    chart: FrozenArtifactChart,
    config: Gate2RunConfig,
) -> None:
    contract = cache.contract
    chart_contract = contract["chart"]
    if chart_contract["name"] != chart.name:
        raise ValueError("Gate2 cache and requested chart names differ.")
    expected_hashes = {
        row["family"]: row["sha256"]
        for row in chart_contract["families"]
    }
    actual_hashes = dict(zip(chart.families, chart.artifact_sha256))
    if expected_hashes != actual_hashes:
        raise RuntimeError("Gate2 artifacts differ from the joint cache contract.")
    dimensions = {
        "latent_channels": int(contract["latent_channels"]),
        "proprio_dim": int(contract["proprio_dim"]),
        "action_dim": int(contract["action_dim"]),
    }
    changed = {
        name: (getattr(config.model, name), expected)
        for name, expected in dimensions.items()
        if getattr(config.model, name) != expected
    }
    if changed:
        raise ValueError(f"Gate2 model/cache dimensions differ: {changed}.")
    window = contract["window"]
    if config.model.max_time < int(window["state_latent_ticks"]):
        raise ValueError("Gate2 model max_time is shorter than cached state.")
    if config.model.max_action_horizon < int(window["action_horizon"]):
        raise ValueError("Gate2 model action horizon is shorter than the cache.")
    if config.model.max_cameras < len(contract["camera_ids"]):
        raise ValueError("Gate2 model camera capacity is smaller than the cache.")
    split_counts = {
        split: len(cache.split_indices(split))
        for split in ("train", "val", "test")
    }
    missing = [split for split, count in split_counts.items() if count == 0]
    if missing:
        raise ValueError(f"Gate2 cache has no windows for splits {missing}.")


def _build_gate2_model(
    config: Gate2RunConfig,
    chart: FrozenArtifactChart,
    *,
    device: torch.device,
) -> _Gate2DynamicsModel:
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    codewam = build_codewam_v1(
        config.model,
        {chart.name: chart.artifacts},
    )
    codewam.action_flow.requires_grad_(False)
    return _Gate2DynamicsModel(codewam).to(device)


def _protocol(
    config: Gate2RunConfig,
    cache: JointWindowCache,
    chart: FrozenArtifactChart,
    permutation: FixedActionPermutation,
    *,
    world_size: int,
) -> dict[str, Any]:
    if world_size <= 0:
        raise ValueError("Gate2 protocol world size must be positive.")
    run_values = asdict(config)
    for key in ("cache_dir", "output_dir", "artifact_paths"):
        run_values.pop(key)
    payload = {
        "schema": GATE2_PROTOCOL_SCHEMA,
        "cache_contract_hash": cache.contract["contract_hash"],
        "cache_summary_sha256": file_sha256(
            Path(config.cache_dir) / "summary.json"
        ),
        "chart": chart.compact_identity(),
        "artifact_sha256": dict(
            zip(chart.families, chart.artifact_sha256)
        ),
        "permutation": asdict(permutation),
        "conditions": {
            "PERSIST": "current code is the future code; no fitted parameters",
            "NOACT": "state-conditioned predictor with no action tokens",
            "TRUE": "state-conditioned predictor with aligned action[t:t+h]",
            "SHUFFLE": "same predictor with fixed split-local wrong actions",
            "TRUE@NOACT": "TRUE-trained model evaluated without actions",
            "TRUE@SHUFFLE": "TRUE-trained model evaluated with wrong actions",
        },
        "primary_stratum": "changed",
        "primary_metric": "normalized_nll",
        "loss": "future_code_cross_entropy_only",
        "distributed": {
            "world_size": world_size,
            "per_rank_batch_size": config.batch_size,
            "effective_batch_size": world_size * config.batch_size,
            "per_rank_eval_batch_size": config.eval_batch_size,
            "effective_eval_batch_size": world_size * config.eval_batch_size,
        },
        "run_config": run_values,
        "implementation_sha256": {
            "gate2": file_sha256(Path(__file__)),
            "code_dynamics": file_sha256(
                Path(__file__).parents[1] / "models" / "code_dynamics.py"
            ),
            "joint_cache": file_sha256(
                Path(__file__).parents[1] / "data" / "joint_cache.py"
            ),
        },
    }
    return {**payload, "protocol_hash": _canonical_hash(payload)}


def _prepare_initialization(
    config: Gate2RunConfig,
    chart: FrozenArtifactChart,
    protocol: Mapping[str, Any],
    context: _DistributedContext,
) -> tuple[Path, str]:
    path = Path(config.output_dir) / "initialization.pt"
    if context.is_primary and not path.exists():
        model = _build_gate2_model(config, chart, device=torch.device("cpu"))
        atomic_torch_save(
            {
                "schema": "codewam.gate2-initialization.v1",
                "protocol_hash": protocol["protocol_hash"],
                "model": model.state_dict(),
            },
            path,
        )
    _barrier(context)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != "codewam.gate2-initialization.v1"
        or payload.get("protocol_hash") != protocol["protocol_hash"]
    ):
        raise RuntimeError("Gate2 initialization belongs to another protocol.")
    return path, file_sha256(path)


def _load_initial_state(
    path: Path,
) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["model"]


def _train_condition(
    condition: str,
    *,
    config: Gate2RunConfig,
    chart: FrozenArtifactChart,
    dataset: _Gate2Dataset,
    train_indices: Sequence[int],
    initialization_path: Path,
    protocol_hash: str,
    context: _DistributedContext,
) -> dict[str, Any]:
    if condition not in LEARNED_CONDITIONS:
        raise ValueError(f"Unsupported learned Gate2 condition `{condition}`.")
    condition_dir = Path(config.output_dir) / condition.lower()
    latest_path = condition_dir / "latest.pt"
    final_path = condition_dir / "final.pt"
    model = _build_gate2_model(config, chart, device=context.device)
    model.load_state_dict(_load_initial_state(initialization_path), strict=True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, Any]] = []
    start_epoch = 0
    global_step = 0
    scaler_state: Mapping[str, Any] | None = None
    if final_path.is_file():
        checkpoint = torch.load(
            final_path,
            map_location="cpu",
            weights_only=False,
        )
        if (
            checkpoint.get("schema") != GATE2_CHECKPOINT_SCHEMA
            or checkpoint.get("protocol_hash") != protocol_hash
            or checkpoint.get("condition") != condition
        ):
            raise RuntimeError(f"Gate2 final checkpoint mismatch: {final_path}.")
        return {
            "condition": condition,
            "status": "reused",
            "epochs_completed": int(checkpoint["epochs_completed"]),
            "optimizer_steps": int(checkpoint["global_step"]),
            "history": checkpoint["history"],
            "checkpoint": str(final_path),
            "checkpoint_sha256": file_sha256(final_path),
        }
    if latest_path.is_file():
        checkpoint = torch.load(
            latest_path,
            map_location="cpu",
            weights_only=False,
        )
        if (
            checkpoint.get("schema") != GATE2_CHECKPOINT_SCHEMA
            or checkpoint.get("protocol_hash") != protocol_hash
            or checkpoint.get("condition") != condition
        ):
            raise RuntimeError(f"Gate2 checkpoint mismatch: {latest_path}.")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler_state = checkpoint.get("scaler")
        start_epoch = int(checkpoint["epochs_completed"])
        global_step = int(checkpoint["global_step"])
        history = list(checkpoint["history"])
    wrapped: _Gate2DynamicsModel | DistributedDataParallel = model
    if context.world_size > 1:
        wrapped = DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            find_unused_parameters=condition == "NOACT",
        )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            context.device.type == "cuda"
            and config.amp_dtype == "float16"
        )
    )
    if scaler_state is not None:
        scaler.load_state_dict(dict(scaler_state))
    mode = _condition_mode(condition)
    stopped = False
    sampler = _IndexSampler(())
    loader = _make_loader(
        dataset,
        sampler,
        config=config,
        batch_size=config.batch_size,
    )
    for epoch in range(start_epoch, config.epochs):
        if config.max_steps is not None and global_step >= config.max_steps:
            break
        sampler.set_indices(
            _rank_indices(
                train_indices,
                rank=context.rank,
                world_size=context.world_size,
                seed=config.seed,
                epoch=epoch,
                training=True,
                group_keys=dataset.cache.window_shards,
            )
        )
        wrapped.train()
        loss_sum = 0.0
        updates = 0
        started = time.monotonic()
        loader_iterator = iter(loader)
        while True:
            if config.max_steps is not None and global_step >= config.max_steps:
                stopped = True
                break
            try:
                cpu_batch = next(loader_iterator)
            except StopIteration:
                break
            batch = _move_gate2_batch(cpu_batch, context.device)
            model_batch = batch.joint.model
            if model_batch.codes is None or model_batch.future_codes is None:
                raise RuntimeError("Gate2 training batch lost code supervision.")
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(config, context.device):
                prediction = wrapped(
                    model_batch.state,
                    model_batch.codes,
                    _condition_actions(batch, mode),
                )
                loss = _unwrap_model(wrapped).codewam.code_dynamics.loss(
                    prediction,
                    model_batch.future_codes,
                    sample_valid=model_batch.supervision.dynamics,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in wrapped.parameters()
                    if parameter.requires_grad
                ],
                config.grad_clip_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().cpu())
            updates += 1
            global_step += 1
        totals = torch.tensor(
            [loss_sum, updates],
            dtype=torch.float64,
            device=context.device,
        )
        if context.world_size > 1:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        history.append(
            {
                "epoch": epoch + 1,
                "optimizer_steps": global_step,
                "train_loss": (
                    float(totals[0].item() / totals[1].item())
                    if totals[1].item()
                    else float("nan")
                ),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        checkpoint = {
            "schema": GATE2_CHECKPOINT_SCHEMA,
            "protocol_hash": protocol_hash,
            "condition": condition,
            "epochs_completed": epoch + 1,
            "global_step": global_step,
            "history": history,
            "model": _unwrap_model(wrapped).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        }
        if context.is_primary:
            atomic_torch_save(checkpoint, latest_path)
        _barrier(context)
        if stopped:
            break
    final_checkpoint = {
        "schema": GATE2_CHECKPOINT_SCHEMA,
        "protocol_hash": protocol_hash,
        "condition": condition,
        "epochs_completed": len(history),
        "global_step": global_step,
        "history": history,
        "model": _unwrap_model(wrapped).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }
    if context.is_primary:
        atomic_torch_save(final_checkpoint, final_path)
    _barrier(context)
    return {
        "condition": condition,
        "status": "trained",
        "epochs_completed": len(history),
        "optimizer_steps": global_step,
        "history": history,
        "checkpoint": str(final_path),
        "checkpoint_sha256": file_sha256(final_path),
    }


def _load_trained_model(
    condition: str,
    *,
    config: Gate2RunConfig,
    chart: FrozenArtifactChart,
    protocol_hash: str,
    device: torch.device,
) -> _Gate2DynamicsModel:
    path = Path(config.output_dir) / condition.lower() / "final.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema") != GATE2_CHECKPOINT_SCHEMA
        or checkpoint.get("protocol_hash") != protocol_hash
        or checkpoint.get("condition") != condition
    ):
        raise RuntimeError(f"Gate2 evaluation checkpoint mismatch: {path}.")
    model = _build_gate2_model(config, chart, device=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()


@torch.no_grad()
def _evaluate_split_conditions(
    models: Mapping[str, _Gate2DynamicsModel],
    *,
    split: str,
    config: Gate2RunConfig,
    loader: DataLoader,
    split_windows: int,
    context: _DistributedContext,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if set(models) != set(LEARNED_CONDITIONS):
        raise ValueError("Combined Gate2 evaluation needs every trained condition.")
    true_model = models["TRUE"]
    adapter = true_model.codewam.frozen_codebook
    persistence = PersistenceAccumulator(
        families=adapter.families,
        levels=adapter.levels,
    )
    evaluations: tuple[
        tuple[str, _Gate2DynamicsModel, ActionMode], ...
    ] = (
        ("TRUE", true_model, "true"),
        ("TRUE@NOACT", true_model, "none"),
        ("TRUE@SHUFFLE", true_model, "shuffle"),
        ("NOACT", models["NOACT"], "none"),
        ("SHUFFLE", models["SHUFFLE"], "shuffle"),
    )
    metrics = {
        name: Gate2MetricAccumulator(
            families=model.codewam.frozen_codebook.families,
            levels=model.codewam.frozen_codebook.levels,
            calibration_bins=config.calibration_bins,
        )
        for name, model, _ in evaluations
    }
    for model in models.values():
        model.eval()
    for cpu_batch in loader:
        batch = _move_gate2_batch(cpu_batch, context.device)
        model_batch = batch.joint.model
        if model_batch.codes is None:
            raise RuntimeError("Gate2 evaluation batch lost current codes.")
        persistence.update(batch.joint, adapter)
        for name, model, mode in evaluations:
            with _autocast_context(config, context.device):
                prediction = model(
                    model_batch.state,
                    model_batch.codes,
                    _condition_actions(batch, mode),
                )
            metrics[name].update(
                prediction,
                batch.joint,
                model.codewam.frozen_codebook,
            )

    persistence_payloads = _gather_payloads(
        persistence.payload(),
        context,
    )
    merged_persistence = PersistenceAccumulator.from_payload(
        persistence_payloads[0]
    )
    for payload in persistence_payloads[1:]:
        merged_persistence.merge(PersistenceAccumulator.from_payload(payload))
    persistence_report = merged_persistence.report()
    persistence_report["split"] = split
    persistence_report["windows"] = split_windows

    reports = {}
    for name, _, _ in evaluations:
        payloads = _gather_payloads(metrics[name].payload(), context)
        merged = Gate2MetricAccumulator.from_payload(payloads[0])
        for payload in payloads[1:]:
            merged.merge(Gate2MetricAccumulator.from_payload(payload))
        report = merged.report()
        report["split"] = split
        report["windows"] = split_windows
        reports[name] = report
    return persistence_report, reports


def _paired_episode_bootstrap(
    true_rows: Mapping[str, Mapping[str, float | int]],
    baseline_rows: Mapping[str, Mapping[str, float | int]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    episodes = sorted(set(true_rows) & set(baseline_rows))
    deltas = []
    count_mismatches = []
    for episode in episodes:
        true_count = int(true_rows[episode]["count"])
        baseline_count = int(baseline_rows[episode]["count"])
        if true_count <= 0 or baseline_count <= 0:
            continue
        if true_count != baseline_count:
            count_mismatches.append(episode)
        true_mean = float(true_rows[episode]["sum"]) / true_count
        baseline_mean = float(baseline_rows[episode]["sum"]) / baseline_count
        deltas.append(true_mean - baseline_mean)
    if count_mismatches:
        raise RuntimeError(
            "Paired Gate2 conditions used different changed-family counts."
        )
    if not deltas:
        return {
            "episodes": 0,
            "mean_delta_true_minus_baseline": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "episode_win_fraction": float("nan"),
            "bootstrap_samples": samples,
        }
    values = torch.tensor(deltas, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    bootstrap = torch.empty(samples, dtype=torch.float64)
    chunk = 128
    for start in range(0, samples, chunk):
        amount = min(chunk, samples - start)
        indices = torch.randint(
            0,
            values.numel(),
            (amount, values.numel()),
            generator=generator,
        )
        bootstrap[start : start + amount] = values[indices].mean(dim=1)
    quantiles = torch.quantile(
        bootstrap,
        torch.tensor([0.025, 0.975], dtype=torch.float64),
    )
    return {
        "episodes": int(values.numel()),
        "mean_delta_true_minus_baseline": float(values.mean()),
        "ci95": [float(quantiles[0]), float(quantiles[1])],
        "episode_win_fraction": float(values.lt(0).float().mean()),
        "bootstrap_samples": samples,
        "unit": "episode_mean_changed_family_normalized_nll",
        "interpretation": "negative favors TRUE",
    }


def _gate_verdict(
    comparisons: Mapping[str, Mapping[str, Any]],
    permutation: FixedActionPermutation,
    *,
    minimum_episodes: int,
) -> dict[str, Any]:
    required = (
        "TRUE-vs-NOACT",
        "TRUE-vs-SHUFFLE",
        "TRUE-vs-TRUE@SHUFFLE",
    )
    if permutation.singleton_groups:
        return {
            "verdict": "invalid",
            "reason": "At least one split/horizon group had no wrong-action donor.",
            "required_comparisons": list(required),
        }
    missing = [
        name
        for name in required
        if int(comparisons[name]["episodes"]) < minimum_episodes
    ]
    if missing:
        return {
            "verdict": "invalid",
            "reason": (
                f"Fewer than {minimum_episodes} changed-code episodes for "
                f"{missing}."
            ),
            "required_comparisons": list(required),
            "minimum_gate_episodes": minimum_episodes,
        }
    upper = [float(comparisons[name]["ci95"][1]) for name in required]
    lower = [float(comparisons[name]["ci95"][0]) for name in required]
    if all(value < 0.0 for value in upper):
        verdict = "pass"
        reason = (
            "TRUE lowers changed-family normalized NLL against both trained "
            "controls, and wrong actions damage the TRUE-trained model."
        )
    elif any(value >= 0.0 for value in lower):
        verdict = "fail"
        reason = (
            "At least one preregistered control provides evidence that aligned "
            "actions do not improve changed-family prediction."
        )
    else:
        verdict = "inconclusive"
        reason = (
            "Point estimates do not satisfy all preregistered confidence "
            "interval tests."
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "required_comparisons": list(required),
        "minimum_gate_episodes": minimum_episodes,
        "decision_rule": "all paired 95% CI upper bounds must be below zero",
    }


def _strip_episode_rows(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key != "episode_changed_nll"
    }


def run_gate2(config: Gate2RunConfig) -> dict[str, Any]:
    context = _distributed_context(config.device)
    try:
        cache = JointWindowCache(
            config.cache_dir,
            verify_index_hashes=context.is_primary,
        )
        _barrier(context)
        chart = load_frozen_artifact_chart(
            config.chart_name,
            config.artifact_paths,
        )
        _validate_cache_and_chart(cache, chart, config)
        permutation = _broadcast_action_permutation(
            (
                build_fixed_action_permutation(cache, seed=config.seed)
                if context.is_primary
                else None
            ),
            context,
        )
        split_indices = {
            split: cache.split_indices(split)
            for split in ("train", "val", "test")
        }
        minimum_train = context.world_size * config.batch_size
        if len(split_indices["train"]) < minimum_train:
            raise ValueError(
                f"Gate2 needs at least {minimum_train} train windows for "
                f"{context.world_size} ranks."
            )
        protocol = _protocol(
            config,
            cache,
            chart,
            permutation,
            world_size=context.world_size,
        )
        protocol_path = Path(config.output_dir) / "protocol.json"
        if context.is_primary:
            if protocol_path.is_file():
                existing = json.loads(
                    protocol_path.read_text(encoding="utf-8")
                )
                if existing != protocol:
                    raise RuntimeError(
                        "Existing Gate2 output uses another protocol."
                    )
            else:
                _atomic_json(protocol_path, protocol)
        _barrier(context)
        dataset = _Gate2Dataset(cache, permutation)
        initialization_path, initialization_sha256 = _prepare_initialization(
            config,
            chart,
            protocol,
            context,
        )
        training = {}
        for condition in LEARNED_CONDITIONS:
            training[condition] = _train_condition(
                condition,
                config=config,
                chart=chart,
                dataset=dataset,
                train_indices=split_indices["train"],
                initialization_path=initialization_path,
                protocol_hash=protocol["protocol_hash"],
                context=context,
            )
        step_counts = {
            int(value["optimizer_steps"]) for value in training.values()
        }
        if len(step_counts) != 1:
            raise RuntimeError("Gate2 learned controls received unequal budgets.")

        conditions: dict[str, dict[str, Any]] = {
            name: {} for name in ("PERSIST", *LEARNED_CONDITIONS)
        }
        diagnostics: dict[str, dict[str, Any]] = {
            "TRUE@NOACT": {},
            "TRUE@SHUFFLE": {},
        }
        for split in ("val", "test"):
            rank_indices = _rank_indices(
                split_indices[split],
                rank=context.rank,
                world_size=context.world_size,
                seed=config.seed,
                epoch=0,
                training=False,
                group_keys=dataset.cache.window_shards,
            )
            loader = _make_loader(
                dataset,
                _IndexSampler(rank_indices),
                config=config,
                batch_size=config.eval_batch_size,
            )
            split_windows = len(split_indices[split])
            evaluation_models = {
                condition: _load_trained_model(
                    condition,
                    config=config,
                    chart=chart,
                    protocol_hash=protocol["protocol_hash"],
                    device=context.device,
                )
                for condition in LEARNED_CONDITIONS
            }
            persistence_report, evaluated = _evaluate_split_conditions(
                evaluation_models,
                split=split,
                config=config,
                loader=loader,
                split_windows=split_windows,
                context=context,
            )
            conditions["PERSIST"][split] = persistence_report
            for condition in LEARNED_CONDITIONS:
                conditions[condition][split] = evaluated[condition]
            for name in diagnostics:
                diagnostics[name][split] = evaluated[name]
            del loader, evaluation_models
            if context.device.type == "cuda":
                torch.cuda.empty_cache()

        true_rows = conditions["TRUE"]["test"]["episode_changed_nll"]
        comparison_sources = {
            "TRUE-vs-NOACT": conditions["NOACT"]["test"],
            "TRUE-vs-SHUFFLE": conditions["SHUFFLE"]["test"],
            "TRUE-vs-TRUE@NOACT": diagnostics["TRUE@NOACT"]["test"],
            "TRUE-vs-TRUE@SHUFFLE": diagnostics["TRUE@SHUFFLE"]["test"],
        }
        comparisons = {
            name: _paired_episode_bootstrap(
                true_rows,
                baseline["episode_changed_nll"],
                samples=config.bootstrap_samples,
                seed=config.seed + index,
            )
            for index, (name, baseline) in enumerate(
                comparison_sources.items()
            )
        }
        gate = _gate_verdict(
            comparisons,
            permutation,
            minimum_episodes=config.minimum_gate_episodes,
        )
        clean_conditions = {
            condition: {
                split: _strip_episode_rows(split_report)
                for split, split_report in reports.items()
            }
            for condition, reports in conditions.items()
        }
        clean_diagnostics = {
            condition: {
                split: _strip_episode_rows(split_report)
                for split, split_report in reports.items()
            }
            for condition, reports in diagnostics.items()
        }
        report = {
            "schema": GATE2_SCHEMA,
            "protocol_hash": protocol["protocol_hash"],
            "initialization_sha256": initialization_sha256,
            "action_index": {
                **cache.summary["indices"]["actions"],
                "rows": len(cache),
            },
            "cache_contract_hash": cache.contract["contract_hash"],
            "split_windows": {
                key: len(value) for key, value in split_indices.items()
            },
            "permutation": asdict(permutation),
            "training": training,
            "conditions": clean_conditions,
            "diagnostics": clean_diagnostics,
            "paired_episode_comparisons": comparisons,
            "gate": gate,
            "runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "world_size": context.world_size,
                "device": str(context.device),
            },
        }
        if context.is_primary:
            _atomic_json(Path(config.output_dir) / "report.json", report)
        _barrier(context)
        return report
    finally:
        if context.initialized_here and dist.is_initialized():
            dist.destroy_process_group()
