from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import QueryCrossAttentionBlock
from .contracts import (
    ActionBatch,
    CodeMeasurements,
    FutureCodeTargets,
    WorldBelief,
)


DynamicsMode = Literal["independent", "prefix"]


def encode_prefix_ids(code_ids: torch.Tensor, sizes: tuple[int, ...]) -> torch.Tensor:
    if code_ids.dtype != torch.long:
        raise ValueError("Prefix code IDs must use torch.long.")
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("Prefix vocabulary sizes must be positive.")
    if code_ids.shape[-1] != len(sizes):
        raise ValueError("Code IDs do not match the prefix depth.")
    encoded = torch.zeros(code_ids.shape[:-1], dtype=torch.long, device=code_ids.device)
    for level, size in enumerate(sizes):
        values = code_ids[..., level]
        if (
            values.device.type == "cpu"
            and ((values < 0) | (values >= size)).any()
        ):
            raise ValueError(f"Prefix level {level + 1} has IDs outside [0,{size}).")
        encoded = encoded * int(size) + values
    return encoded


def decode_prefix_ids(encoded: torch.Tensor, sizes: tuple[int, ...]) -> torch.Tensor:
    if encoded.dtype != torch.long:
        raise ValueError("Encoded prefix IDs must use torch.long.")
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("Prefix vocabulary sizes must be positive.")
    if (
        encoded.device.type == "cpu"
        and ((encoded < 0) | (encoded >= prod(sizes))).any()
    ):
        raise ValueError("Encoded prefix ID is outside its vocabulary.")
    remaining = encoded.long()
    levels: list[torch.Tensor] = []
    for size in reversed(sizes):
        levels.append(remaining.remainder(int(size)))
        remaining = torch.div(remaining, int(size), rounding_mode="floor")
    return torch.stack(list(reversed(levels)), dim=-1)


@dataclass(frozen=True)
class FutureCodePrediction:
    mode: DynamicsMode
    logits: tuple[torch.Tensor, ...]
    families: tuple[str, ...]
    codebook_sizes: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.mode not in {"independent", "prefix"}:
            raise ValueError(f"Unsupported future-code mode `{self.mode}`.")
        if (
            not self.families
            or len(set(self.families)) != len(self.families)
            or len(self.families) != len(self.codebook_sizes)
        ):
            raise ValueError("Future-code families and vocabularies are misaligned.")
        depths = {len(sizes) for sizes in self.codebook_sizes}
        if (
            len(depths) != 1
            or next(iter(depths)) <= 0
            or any(size <= 0 for sizes in self.codebook_sizes for size in sizes)
        ):
            raise ValueError("Future-code vocabularies need one common positive depth.")
        output_sizes = (
            tuple(size for sizes in self.codebook_sizes for size in sizes)
            if self.mode == "independent"
            else tuple(prod(sizes) for sizes in self.codebook_sizes)
        )
        if len(self.logits) != len(output_sizes):
            raise ValueError(
                f"Expected {len(output_sizes)} future-code heads, "
                f"got {len(self.logits)}."
            )
        batches = set()
        for index, (logits, output_size) in enumerate(
            zip(self.logits, output_sizes)
        ):
            if logits.ndim != 2 or logits.shape[1] != output_size:
                raise ValueError(
                    f"Future-code head {index} must be [B,{output_size}], "
                    f"got {tuple(logits.shape)}."
                )
            batches.add(int(logits.shape[0]))
        if len(batches) != 1:
            raise ValueError("Future-code heads must share one batch size.")

    def predicted_ids(self) -> torch.Tensor:
        if self.mode == "independent":
            outputs: list[torch.Tensor] = []
            index = 0
            for sizes in self.codebook_sizes:
                levels = []
                for _ in sizes:
                    levels.append(self.logits[index].argmax(dim=-1))
                    index += 1
                outputs.append(torch.stack(levels, dim=-1))
            return torch.stack(outputs, dim=1)
        outputs = [
            decode_prefix_ids(logits.argmax(dim=-1), sizes)
            for logits, sizes in zip(self.logits, self.codebook_sizes)
        ]
        return torch.stack(outputs, dim=1)


class CodeDynamicsDecoder(nn.Module):
    """Action-conditioned next-code model with explicit output factorization."""

    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        action_dim: int,
        max_horizon: int,
        families: tuple[str, ...],
        codebook_sizes: tuple[tuple[int, ...], ...],
        mode: DynamicsMode = "independent",
        layers: int = 3,
        action_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if min(dim, heads, action_dim, max_horizon, layers, action_layers) <= 0:
            raise ValueError(
                "Code-dynamics dimensions and layer counts must be positive."
            )
        if mode not in {"independent", "prefix"}:
            raise ValueError(f"Unsupported code-dynamics mode `{mode}`.")
        if len(families) != len(codebook_sizes) or not families:
            raise ValueError("Families and codebook sizes must be nonempty and aligned.")
        depths = {len(sizes) for sizes in codebook_sizes}
        if len(depths) != 1 or next(iter(depths)) <= 0:
            raise ValueError("All dynamics families must use one common positive depth.")
        if any(size <= 1 for sizes in codebook_sizes for size in sizes):
            raise ValueError("Every future-code vocabulary must contain at least two IDs.")
        if dim % heads:
            raise ValueError("Code-dynamics dim must be divisible by heads.")
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.max_horizon = int(max_horizon)
        self.families = tuple(families)
        self.codebook_sizes = tuple(tuple(int(v) for v in row) for row in codebook_sizes)
        self.levels = next(iter(depths))
        self.mode: DynamicsMode = mode
        self.action_projection = nn.Linear(action_dim, dim)
        self.action_position = nn.Parameter(
            torch.randn(max_horizon, dim) * (dim**-0.5)
        )
        action_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_encoder = nn.TransformerEncoder(
            action_layer,
            num_layers=action_layers,
            norm=nn.LayerNorm(dim),
            enable_nested_tensor=False,
        )
        query_count = (
            len(families) * self.levels
            if mode == "independent"
            else len(families)
        )
        self.future_queries = nn.Parameter(
            torch.randn(query_count, dim) * (dim**-0.5)
        )
        self.blocks = nn.ModuleList(
            [
                QueryCrossAttentionBlock(
                    dim=dim,
                    heads=heads,
                    dropout=dropout,
                )
                for _ in range(layers)
            ]
        )
        self.output_norm = nn.LayerNorm(dim)
        if mode == "independent":
            output_sizes = [
                size
                for family_sizes in self.codebook_sizes
                for size in family_sizes
            ]
        else:
            output_sizes = [prod(sizes) for sizes in self.codebook_sizes]
        self.heads = nn.ModuleList([nn.Linear(dim, size) for size in output_sizes])

    def forward(
        self,
        belief: WorldBelief,
        actions: ActionBatch,
    ) -> FutureCodePrediction:
        batch, horizon, action_dim = actions.values.shape
        if action_dim != self.action_dim:
            raise ValueError(
                f"Expected dynamics action dim {self.action_dim}, got {action_dim}."
            )
        if horizon > self.max_horizon:
            raise ValueError(
                f"Dynamics action horizon {horizon} exceeds {self.max_horizon}."
            )
        if belief.tokens.shape[0] != batch or belief.tokens.shape[2] != self.dim:
            raise ValueError("World belief does not match dynamics batch/width.")
        action_valid = (
            actions.valid
            if actions.valid is not None
            else torch.ones(
                (batch, horizon),
                dtype=torch.bool,
                device=actions.values.device,
            )
        )
        action_tokens = self.action_projection(actions.values)
        action_tokens = action_tokens + self.action_position[:horizon][None]
        action_padding = ~action_valid
        all_missing = action_padding.all(dim=1)
        safe_action_padding = action_padding.clone()
        safe_action_padding[all_missing, 0] = False
        action_tokens = self.action_encoder(
            action_tokens,
            src_key_padding_mask=safe_action_padding,
        )
        action_tokens = action_tokens * action_valid[:, :, None].to(
            action_tokens.dtype
        )
        context = torch.cat((belief.tokens, action_tokens), dim=1)
        context_valid = torch.cat(
            (
                torch.ones(
                    belief.tokens.shape[:2],
                    dtype=torch.bool,
                    device=belief.tokens.device,
                ),
                action_valid,
            ),
            dim=1,
        )
        queries = self.future_queries[None].expand(batch, -1, -1)
        for block in self.blocks:
            queries = block(
                queries,
                context,
                context_valid=context_valid,
            )
        queries = self.output_norm(queries)
        logits = tuple(head(queries[:, index]) for index, head in enumerate(self.heads))
        return FutureCodePrediction(
            mode=self.mode,
            logits=logits,
            families=self.families,
            codebook_sizes=self.codebook_sizes,
        )

    def loss(
        self,
        prediction: FutureCodePrediction,
        targets: FutureCodeTargets,
        *,
        sample_valid: torch.Tensor,
    ) -> torch.Tensor:
        batch, families, levels = targets.code_ids.shape
        if (
            prediction.mode != self.mode
            or prediction.families != self.families
            or prediction.codebook_sizes != self.codebook_sizes
        ):
            raise ValueError("Future-code prediction does not belong to this decoder.")
        if (families, levels) != (len(self.families), self.levels):
            raise ValueError("Future-code targets do not match dynamics layout.")
        if sample_valid.dtype != torch.bool or sample_valid.shape != (batch,):
            raise ValueError("Dynamics supervision must be bool [B].")
        losses: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        if prediction.mode == "independent":
            index = 0
            for family in range(families):
                valid = targets.available[:, family] & sample_valid
                for level in range(levels):
                    safe_targets = torch.where(
                        valid,
                        targets.code_ids[:, family, level],
                        torch.zeros_like(targets.code_ids[:, family, level]),
                    )
                    losses.append(
                        F.cross_entropy(
                            prediction.logits[index],
                            safe_targets,
                            reduction="none",
                        )
                    )
                    masks.append(valid)
                    index += 1
        else:
            for family, sizes in enumerate(self.codebook_sizes):
                valid = targets.available[:, family] & sample_valid
                safe_targets = torch.where(
                    valid[:, None],
                    targets.code_ids[:, family],
                    torch.zeros_like(targets.code_ids[:, family]),
                )
                encoded = encode_prefix_ids(safe_targets, sizes)
                losses.append(
                    F.cross_entropy(
                        prediction.logits[family],
                        encoded,
                        reduction="none",
                    )
                    / float(self.levels)
                )
                masks.append(valid)
        stacked_loss = torch.stack(losses, dim=1)
        stacked_mask = torch.stack(masks, dim=1)
        weights = stacked_mask.to(stacked_loss.dtype)
        return (stacked_loss * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def future_code_metrics(
    prediction: FutureCodePrediction,
    targets: FutureCodeTargets,
    *,
    sample_valid: torch.Tensor,
    family_valid: torch.Tensor | None = None,
    calibration_bins: int = 10,
) -> dict[str, float | int | str]:
    """Held-out ID metrics; center-space error is computed by the adapter."""

    if calibration_bins <= 1:
        raise ValueError("Calibration requires at least two bins.")
    batch, families, levels = targets.code_ids.shape
    if (
        families != len(prediction.families)
        or any(len(sizes) != levels for sizes in prediction.codebook_sizes)
    ):
        raise ValueError("Metric targets do not match future-code prediction layout.")
    if sample_valid.dtype != torch.bool or sample_valid.shape != (batch,):
        raise ValueError("Metric supervision must be bool [B].")
    if family_valid is not None and (
        family_valid.dtype != torch.bool
        or tuple(family_valid.shape) != (batch, families)
    ):
        raise ValueError("Metric family validity must be bool [B,F].")
    available = targets.available
    if family_valid is not None:
        available = available & family_valid
    confidences: list[torch.Tensor] = []
    correctness: list[torch.Tensor] = []
    nll_values: list[torch.Tensor] = []
    brier_values: list[torch.Tensor] = []
    entropy_values: list[torch.Tensor] = []
    family_prefix_nll_values: list[torch.Tensor] = []

    def collect(
        logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor | None:
        if not valid.any():
            return None
        selected_logits = logits[valid]
        selected_target = target[valid]
        probabilities = selected_logits.softmax(dim=-1)
        confidence, predicted = probabilities.max(dim=-1)
        confidences.append(confidence)
        correctness.append(predicted.eq(selected_target))
        nll_values.append(
            F.cross_entropy(selected_logits, selected_target, reduction="none")
        )
        one_hot = F.one_hot(
            selected_target,
            num_classes=selected_logits.shape[-1],
        ).to(probabilities.dtype)
        brier_values.append((probabilities - one_hot).square().sum(dim=-1))
        entropy_values.append(
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        )
        return nll_values[-1]

    if prediction.mode == "independent":
        index = 0
        for family in range(families):
            valid = available[:, family] & sample_valid
            level_nll: list[torch.Tensor] = []
            for level in range(levels):
                safe_target = torch.where(
                    valid,
                    targets.code_ids[:, family, level],
                    torch.zeros_like(targets.code_ids[:, family, level]),
                )
                selected_nll = collect(
                    prediction.logits[index],
                    safe_target,
                    valid,
                )
                if selected_nll is not None:
                    level_nll.append(selected_nll)
                index += 1
            if level_nll:
                family_prefix_nll_values.append(
                    torch.stack(level_nll, dim=1).sum(dim=1)
                )
    else:
        for family, sizes in enumerate(prediction.codebook_sizes):
            valid = available[:, family] & sample_valid
            safe_target = torch.where(
                valid[:, None],
                targets.code_ids[:, family],
                torch.zeros_like(targets.code_ids[:, family]),
            )
            selected_nll = collect(
                prediction.logits[family],
                encode_prefix_ids(safe_target, sizes),
                valid,
            )
            if selected_nll is not None:
                family_prefix_nll_values.append(selected_nll)

    if not confidences:
        return {
            "count": 0,
            "classification_count": 0,
            "family_count": 0,
            "nll": float("nan"),
            "classification_nll": float("nan"),
            "family_prefix_nll": float("nan"),
            "classification_accuracy": float("nan"),
            "family_prefix_accuracy": float("nan"),
            "brier": float("nan"),
            "entropy": float("nan"),
            "ece": float("nan"),
            "classification_unit": (
                "rq_level"
                if prediction.mode == "independent"
                else "family_prefix"
            ),
        }
    confidence = torch.cat(confidences)
    correct = torch.cat(correctness)
    nll = torch.cat(nll_values)
    family_prefix_nll = torch.cat(family_prefix_nll_values)
    brier = torch.cat(brier_values)
    entropy = torch.cat(entropy_values)
    ece = confidence.new_zeros(())
    boundaries = torch.linspace(
        0.0,
        1.0,
        calibration_bins + 1,
        device=confidence.device,
    )
    for index in range(calibration_bins):
        if index == 0:
            member = (confidence >= boundaries[index]) & (
                confidence <= boundaries[index + 1]
            )
        else:
            member = (confidence > boundaries[index]) & (
                confidence <= boundaries[index + 1]
            )
        if member.any():
            weight = member.float().mean()
            ece = ece + weight * (
                correct[member].float().mean() - confidence[member].mean()
            ).abs()

    predicted_ids = prediction.predicted_ids()
    prefix_valid = available & sample_valid[:, None]
    prefix_correct = predicted_ids.eq(targets.code_ids).all(dim=-1)
    prefix_weights = prefix_valid.float()
    prefix_accuracy = (
        (prefix_correct.float() * prefix_weights).sum()
        / prefix_weights.sum().clamp_min(1.0)
    )
    return {
        "count": int(confidence.numel()),
        "classification_count": int(confidence.numel()),
        "family_count": int(prefix_valid.sum().item()),
        "nll": float((family_prefix_nll.mean() / float(levels)).cpu()),
        "classification_nll": float(nll.mean().cpu()),
        "family_prefix_nll": float(family_prefix_nll.mean().cpu()),
        "classification_accuracy": float(correct.float().mean().cpu()),
        "family_prefix_accuracy": float(prefix_accuracy.cpu()),
        "brier": float(brier.mean().cpu()),
        "entropy": float(entropy.mean().cpu()),
        "ece": float(ece.cpu()),
        "classification_unit": (
            "rq_level"
            if prediction.mode == "independent"
            else "family_prefix"
        ),
    }


@torch.no_grad()
def transition_family_masks(
    current: CodeMeasurements,
    targets: FutureCodeTargets,
) -> dict[str, torch.Tensor]:
    """Partition jointly available families by exact RQ-prefix change."""

    if tuple(current.code_ids.shape) != tuple(targets.code_ids.shape):
        raise ValueError("Current and future code layouts must match.")
    if current.code_ids.device != targets.code_ids.device:
        raise ValueError("Current and future code IDs must share one device.")
    if current.available.device != targets.available.device:
        raise ValueError("Current and future availability must share one device.")
    common = current.available & targets.available
    changed = common & current.code_ids.ne(targets.code_ids).any(dim=-1)
    return {
        "common": common,
        "changed": changed,
        "stable": common & ~changed,
    }


@torch.no_grad()
def persistence_code_metrics(
    current: CodeMeasurements,
    targets: FutureCodeTargets,
    *,
    sample_valid: torch.Tensor,
) -> dict[str, float | int]:
    """Deterministic current-code-as-future baseline without invented NLL."""

    masks = transition_family_masks(current, targets)
    batch, _, levels = targets.code_ids.shape
    if sample_valid.dtype != torch.bool or sample_valid.shape != (batch,):
        raise ValueError("Persistence supervision must be bool [B].")
    valid = masks["common"] & sample_valid[:, None]
    family_count = int(valid.sum().item())
    if family_count == 0:
        return {
            "family_count": 0,
            "level_count": 0,
            "family_prefix_accuracy": float("nan"),
            "level_accuracy": float("nan"),
            "changed_family_fraction": float("nan"),
        }
    level_equal = current.code_ids.eq(targets.code_ids)
    level_valid = valid[:, :, None].expand_as(level_equal)
    family_equal = level_equal.all(dim=-1)
    return {
        "family_count": family_count,
        "level_count": family_count * levels,
        "family_prefix_accuracy": float(
            family_equal[valid].float().mean().cpu()
        ),
        "level_accuracy": float(level_equal[level_valid].float().mean().cpu()),
        "changed_family_fraction": float(
            masks["changed"][valid].float().mean().cpu()
        ),
    }
