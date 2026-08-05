from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Literal

import torch
from torch import nn

from codewam.codebook_eval.streaming import FrozenRQArtifact

from .contracts import CodeMeasurements, CodeTokens, MultiClockCodeState


CHART_IDENTITY_KEYS = (
    "manifest_fingerprint",
    "dataset_revision",
    "wan_model_id",
    "wan_revision",
    "preprocess_revision",
    "source_checksums",
)
ARTIFACT_IDENTITY_KEYS = (*CHART_IDENTITY_KEYS, "config_hash")
ADAPTER_EXTRA_STATE_SCHEMA = "codewam.frozen-codebook-adapter.v1"
HIERARCHICAL_ADAPTER_EXTRA_STATE_SCHEMA = (
    "codewam.hierarchical-frozen-codebook-adapter.v1"
)
CodeLayout = Literal["flat", "hierarchical"]


class FrozenCodebookAdapter(nn.Module):
    """Projects preassigned chart-local RQ centers into a shared belief width."""

    def __init__(
        self,
        artifact_sets: Mapping[str, Sequence[FrozenRQArtifact]],
        *,
        dim: int,
        families: Sequence[str] = ("Q2", "Q3", "Q5"),
        layout: CodeLayout = "flat",
    ):
        super().__init__()
        if not artifact_sets:
            raise ValueError("At least one frozen codebook chart is required.")
        if dim <= 0:
            raise ValueError("Codebook adapter width must be positive.")
        if layout not in {"flat", "hierarchical"}:
            raise ValueError(f"Unsupported codebook layout `{layout}`.")
        if any(not isinstance(name, str) or not name for name in artifact_sets):
            raise ValueError("Codebook chart names must be nonempty strings.")
        self.chart_names = tuple(sorted(artifact_sets))
        self.families = tuple(str(value) for value in families)
        if not self.families or len(set(self.families)) != len(self.families):
            raise ValueError("Codebook families must be nonempty and unique.")
        self.dim = int(dim)
        self.layout: CodeLayout = layout
        self._chart_index = {name: index for index, name in enumerate(self.chart_names)}
        self._buffer_names: dict[tuple[int, int, int], str] = {}
        self.chart_metadata: dict[str, dict[str, object]] = {}
        self.artifact_metadata: dict[str, dict[str, dict[str, object]]] = {}
        self.projections = nn.ModuleDict()

        reference_sizes: tuple[tuple[int, ...], ...] | None = None
        for chart_index, chart_name in enumerate(self.chart_names):
            chart_artifacts = tuple(artifact_sets[chart_name])
            by_family = {
                artifact.family: artifact
                for artifact in chart_artifacts
            }
            if len(by_family) != len(chart_artifacts):
                raise ValueError(
                    f"Chart `{chart_name}` contains duplicate family artifacts."
                )
            if set(by_family) != set(self.families):
                raise ValueError(
                    f"Chart `{chart_name}` families {sorted(by_family)} do not match "
                    f"{list(self.families)}."
                )
            first_artifact = by_family[self.families[0]]
            chart_identity = {
                key: deepcopy(first_artifact.metadata[key])
                for key in CHART_IDENTITY_KEYS
            }
            for family in self.families[1:]:
                artifact_identity = {
                    key: by_family[family].metadata[key]
                    for key in CHART_IDENTITY_KEYS
                }
                if artifact_identity != chart_identity:
                    changed = [
                        key
                        for key in CHART_IDENTITY_KEYS
                        if artifact_identity[key] != chart_identity[key]
                    ]
                    raise ValueError(
                        f"Chart `{chart_name}` mixes provenance across families; "
                        f"{family} differs in {changed}."
                    )
            self.chart_metadata[chart_name] = chart_identity
            self.artifact_metadata[chart_name] = {
                family: {
                    key: deepcopy(by_family[family].metadata[key])
                    for key in ARTIFACT_IDENTITY_KEYS
                }
                for family in self.families
            }
            chart_sizes: list[tuple[int, ...]] = []
            for family_index, family in enumerate(self.families):
                artifact = by_family[family]
                sizes: list[int] = []
                for level_index, centers in enumerate(artifact.centers):
                    if not torch.isfinite(centers).all():
                        raise ValueError(
                            f"Chart `{chart_name}`/{family} contains non-finite centers."
                        )
                    name = (
                        f"centers_chart{chart_index}_family{family_index}_"
                        f"level{level_index}"
                    )
                    self.register_buffer(name, centers.detach().float().clone())
                    self._buffer_names[
                        (chart_index, family_index, level_index)
                    ] = name
                    self.projections[
                        self._projection_key(
                            chart_index,
                            family_index,
                            level_index,
                        )
                    ] = nn.Linear(int(centers.shape[1]), dim)
                    sizes.append(int(centers.shape[0]))
                chart_sizes.append(tuple(sizes))
            current = tuple(chart_sizes)
            if reference_sizes is None:
                reference_sizes = current
            elif current != reference_sizes:
                raise ValueError(
                    "All charts must use the same family/level codebook sizes; "
                    f"expected {reference_sizes}, got {current} for `{chart_name}`."
                )

        assert reference_sizes is not None
        level_counts = {len(value) for value in reference_sizes}
        if len(level_counts) != 1:
            raise ValueError("All codebook families must have the same RQ depth.")
        self.codebook_sizes = reference_sizes
        self.levels = next(iter(level_counts))
        self.family_embedding = nn.Parameter(
            torch.randn(len(self.families), dim) * (dim**-0.5)
        )
        self.level_embedding = nn.Parameter(
            torch.randn(self.levels, dim) * (dim**-0.5)
        )
        self.chart_embedding = nn.Parameter(
            torch.randn(len(self.chart_names), dim) * (dim**-0.5)
        )
        self.missing = nn.Parameter(
            torch.randn(len(self.families), self.levels, dim) * (dim**-0.5)
        )
        self.output_norm = nn.LayerNorm(dim)
        if self.layout == "hierarchical":
            self.family_fusion = nn.Sequential(
                nn.LayerNorm(self.levels * dim),
                nn.Linear(self.levels * dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )

    @staticmethod
    def _projection_key(chart: int, family: int, level: int) -> str:
        return f"chart{chart}_family{family}_level{level}"

    def _centers(self, chart: int, family: int, level: int) -> torch.Tensor:
        return getattr(self, self._buffer_names[(chart, family, level)])

    def get_extra_state(self) -> dict[str, object]:
        state = {
            "schema": (
                ADAPTER_EXTRA_STATE_SCHEMA
                if self.layout == "flat"
                else HIERARCHICAL_ADAPTER_EXTRA_STATE_SCHEMA
            ),
            "chart_names": self.chart_names,
            "families": self.families,
            "codebook_sizes": self.codebook_sizes,
            "artifact_metadata": deepcopy(self.artifact_metadata),
        }
        if self.layout == "hierarchical":
            state["layout"] = self.layout
        return state

    def set_extra_state(self, state: object) -> None:
        if state != self.get_extra_state():
            raise RuntimeError(
                "Frozen codebook checkpoint provenance does not match the "
                "constructed chart artifacts."
            )

    def forward(
        self,
        measurements: CodeMeasurements,
    ) -> CodeTokens | MultiClockCodeState:
        if self.layout == "hierarchical":
            return self._forward_hierarchical(measurements)
        batch, families, levels = measurements.code_ids.shape
        if (families, levels) != (len(self.families), self.levels):
            raise ValueError(
                f"Expected code layout {(len(self.families), self.levels)}, "
                f"got {(families, levels)}."
            )
        unknown = sorted(set(measurements.chart_names) - set(self.chart_names))
        if unknown:
            raise ValueError(f"Unknown codebook charts: {unknown}.")

        device = self.family_embedding.device
        code_ids = measurements.code_ids.to(device=device)
        available = measurements.available.to(device=device)
        chart_ids_list = [
            self._chart_index[name] for name in measurements.chart_names
        ]
        chart_ids = torch.tensor(
            chart_ids_list,
            dtype=torch.long,
            device=device,
        )
        identity = (
            self.chart_embedding[chart_ids, None, None]
            + self.family_embedding[None, :, None]
            + self.level_embedding[None, None, :]
        )
        tokens = self.missing[None] + identity

        chart_groups: dict[int, list[int]] = {}
        for sample_index, chart_index in enumerate(chart_ids_list):
            chart_groups.setdefault(chart_index, []).append(sample_index)
        for chart_index, sample_indices in chart_groups.items():
            chart_samples = torch.tensor(
                sample_indices,
                dtype=torch.long,
                device=device,
            )
            for family_index in range(families):
                selected = chart_samples[available[chart_samples, family_index]]
                if selected.numel() == 0:
                    continue
                for level_index in range(levels):
                    codes = code_ids[selected, family_index, level_index]
                    centers = self._centers(
                        chart_index,
                        family_index,
                        level_index,
                    )
                    invalid = (codes < 0) | (codes >= centers.shape[0])
                    if codes.device.type == "cpu" and invalid.any():
                        bad_code = int(codes[invalid][0].item())
                        chart_name = self.chart_names[chart_index]
                        raise ValueError(
                            f"Code {bad_code} is outside [0,{centers.shape[0]}) for "
                            f"{chart_name}/{self.families[family_index]}/"
                            f"L{level_index + 1}."
                        )
                    projected = self.projections[
                        self._projection_key(
                            chart_index,
                            family_index,
                            level_index,
                        )
                    ](centers.index_select(0, codes))
                    tokens[selected, family_index, level_index] = (
                        projected + identity[selected, family_index, level_index]
                    )
        tokens = self.output_norm(tokens.flatten(1, 2))
        valid = torch.ones(
            (batch, families * levels),
            dtype=torch.bool,
            device=tokens.device,
        )
        return CodeTokens(
            tokens=tokens,
            valid=valid,
            families=self.families,
            levels=self.levels,
        )

    def _forward_hierarchical(
        self,
        measurements: CodeMeasurements,
    ) -> MultiClockCodeState:
        batch, families, levels = measurements.code_ids.shape
        if (families, levels) != (len(self.families), self.levels):
            raise ValueError(
                f"Expected code layout {(len(self.families), self.levels)}, "
                f"got {(families, levels)}."
            )
        unknown = sorted(set(measurements.chart_names) - set(self.chart_names))
        if unknown:
            raise ValueError(f"Unknown codebook charts: {unknown}.")

        device = self.family_embedding.device
        code_ids = measurements.code_ids.to(device=device)
        available = measurements.available.to(device=device)
        chart_ids_list = [
            self._chart_index[name] for name in measurements.chart_names
        ]
        chart_ids = torch.tensor(chart_ids_list, dtype=torch.long, device=device)
        identity = (
            self.chart_embedding[chart_ids, None, None]
            + self.family_embedding[None, :, None]
            + self.level_embedding[None, None, :]
        )
        prefix_tokens = self.missing[None] + identity

        chart_groups: dict[int, list[int]] = {}
        for sample_index, chart_index in enumerate(chart_ids_list):
            chart_groups.setdefault(chart_index, []).append(sample_index)
        for chart_index, sample_indices in chart_groups.items():
            chart_samples = torch.tensor(
                sample_indices,
                dtype=torch.long,
                device=device,
            )
            for family_index in range(families):
                selected = chart_samples[available[chart_samples, family_index]]
                if selected.numel() == 0:
                    continue
                descriptor_dim = int(
                    self._centers(chart_index, family_index, 0).shape[1]
                )
                cumulative = torch.zeros(
                    (selected.numel(), descriptor_dim),
                    dtype=self._centers(chart_index, family_index, 0).dtype,
                    device=device,
                )
                for level_index in range(levels):
                    centers = self._centers(
                        chart_index,
                        family_index,
                        level_index,
                    )
                    codes = code_ids[selected, family_index, level_index]
                    invalid = (codes < 0) | (codes >= centers.shape[0])
                    if codes.device.type == "cpu" and invalid.any():
                        bad_code = int(codes[invalid][0].item())
                        chart_name = self.chart_names[chart_index]
                        raise ValueError(
                            f"Code {bad_code} is outside [0,{centers.shape[0]}) for "
                            f"{chart_name}/{self.families[family_index]}/"
                            f"L{level_index + 1}."
                        )
                    cumulative = cumulative + centers.index_select(0, codes)
                    projected = self.projections[
                        self._projection_key(
                            chart_index,
                            family_index,
                            level_index,
                        )
                    ](cumulative)
                    prefix_tokens[selected, family_index, level_index] = (
                        projected + identity[selected, family_index, level_index]
                    )

        tokens = self.family_fusion(prefix_tokens.flatten(2, 3))
        tokens = self.output_norm(tokens)
        return MultiClockCodeState(
            tokens=tokens,
            prefix_tokens=prefix_tokens,
            valid=available,
            families=self.families,
            levels=self.levels,
        )

    @torch.no_grad()
    def normalized_center_mse(
        self,
        predicted_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        available: torch.Tensor,
        chart_names: tuple[str, ...],
    ) -> torch.Tensor:
        expected = (len(chart_names), len(self.families), self.levels)
        if tuple(predicted_ids.shape) != expected or tuple(target_ids.shape) != expected:
            raise ValueError(
                f"Center-distance IDs must both be {expected}, got "
                f"{tuple(predicted_ids.shape)} and {tuple(target_ids.shape)}."
            )
        if available.dtype != torch.bool or tuple(available.shape) != expected[:2]:
            raise ValueError("Center-distance availability must be bool [B,F].")
        unknown = sorted(set(chart_names) - set(self.chart_names))
        if unknown:
            raise ValueError(f"Unknown codebook charts: {unknown}.")

        device = self.family_embedding.device
        predicted_ids = predicted_ids.to(device=device)
        target_ids = target_ids.to(device=device)
        available = available.to(device=device)
        chart_groups: dict[int, list[int]] = {}
        for sample, chart_name in enumerate(chart_names):
            chart_groups.setdefault(self._chart_index[chart_name], []).append(sample)

        errors: list[torch.Tensor] = []
        for chart, sample_indices in chart_groups.items():
            chart_samples = torch.tensor(
                sample_indices,
                dtype=torch.long,
                device=device,
            )
            for family in range(len(self.families)):
                selected = chart_samples[available[chart_samples, family]]
                if selected.numel() == 0:
                    continue
                dimension = int(self._centers(chart, family, 0).shape[1])
                predicted = torch.zeros(
                    (selected.numel(), dimension),
                    dtype=self._centers(chart, family, 0).dtype,
                    device=device,
                )
                target = torch.zeros_like(predicted)
                for level in range(self.levels):
                    centers = self._centers(chart, family, level)
                    predicted_codes = predicted_ids[selected, family, level]
                    target_codes = target_ids[selected, family, level]
                    invalid = (
                        (predicted_codes < 0)
                        | (target_codes < 0)
                        | (predicted_codes >= centers.shape[0])
                        | (target_codes >= centers.shape[0])
                    )
                    if invalid.any():
                        raise ValueError("Center-distance code is outside its vocabulary.")
                    predicted += centers.index_select(0, predicted_codes)
                    target += centers.index_select(0, target_codes)
                errors.append((predicted - target).square().mean(dim=1))
        if not errors:
            raise ValueError("Center-distance metric has no available family.")
        return torch.cat(errors).mean()
