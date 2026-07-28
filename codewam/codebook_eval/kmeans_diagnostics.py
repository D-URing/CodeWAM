from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import torch

from .streaming import assign_nearest


def _resolve_device(spec: str | torch.device) -> torch.device:
    if isinstance(spec, torch.device):
        return spec
    if str(spec).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device `{spec}` was requested but CUDA is unavailable.")
    return device


@dataclass(frozen=True)
class DiagnosticKMeansConfig:
    k: int
    max_iters: int = 50
    min_iters: int = 3
    tol: float = 1e-4
    patience: int = 3
    seed: int = 0
    chunk_size: int = 8192
    center_block_size: int = 1024
    device: str = "auto"

    def __post_init__(self) -> None:
        if int(self.k) <= 1:
            raise ValueError(f"`k` must be greater than one, got {self.k}.")
        if int(self.max_iters) <= 0:
            raise ValueError(f"`max_iters` must be positive, got {self.max_iters}.")
        if int(self.min_iters) <= 0 or int(self.min_iters) > int(self.max_iters):
            raise ValueError("`min_iters` must be in [1, max_iters].")
        if float(self.tol) < 0:
            raise ValueError(f"`tol` must be non-negative, got {self.tol}.")
        if int(self.patience) <= 0:
            raise ValueError(f"`patience` must be positive, got {self.patience}.")
        if int(self.chunk_size) <= 0 or int(self.center_block_size) <= 0:
            raise ValueError("K-Means chunk sizes must be positive.")


@dataclass(frozen=True)
class KMeansIteration:
    iteration: int
    train_inertia: float
    validation_inertia: float | None
    relative_improvement: float | None
    relative_center_shift: float
    assignment_change: float | None
    empty_clusters: int
    plateau_steps: int

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "iteration": self.iteration,
            "train_inertia": self.train_inertia,
            "validation_inertia": self.validation_inertia,
            "relative_improvement": self.relative_improvement,
            "relative_center_shift": self.relative_center_shift,
            "assignment_change": self.assignment_change,
            "empty_clusters": self.empty_clusters,
            "plateau_steps": self.plateau_steps,
        }


@dataclass(frozen=True)
class DiagnosticKMeansResult:
    centers: torch.Tensor
    train_codes: torch.Tensor
    validation_codes: torch.Tensor | None
    train_distances: torch.Tensor
    validation_distances: torch.Tensor | None
    history: tuple[KMeansIteration, ...]
    converged: bool
    stop_reason: str

    @property
    def iterations(self) -> int:
        return len(self.history)

    @property
    def train_inertia(self) -> float:
        return float(self.train_distances.float().mean().item())

    @property
    def validation_inertia(self) -> float | None:
        if self.validation_distances is None:
            return None
        return float(self.validation_distances.float().mean().item())


@dataclass(frozen=True)
class DiagnosticRQResult:
    levels: tuple[DiagnosticKMeansResult, ...]
    train_residual_mse: tuple[float, ...]
    validation_residual_mse: tuple[float, ...] | None
    test_residual_mse: tuple[float, ...] | None
    train_codes: torch.Tensor
    validation_codes: torch.Tensor | None
    test_codes: torch.Tensor | None


def _validate_matrix(name: str, values: torch.Tensor, dimension: int | None = None) -> None:
    if values.ndim != 2:
        raise ValueError(f"`{name}` must be [N,D], got {tuple(values.shape)}.")
    if values.shape[0] == 0:
        raise ValueError(f"`{name}` must not be empty.")
    if dimension is not None and values.shape[1] != dimension:
        raise ValueError(
            f"`{name}` dimension differs from training data: {values.shape[1]} vs {dimension}."
        )
    if not torch.isfinite(values).all():
        raise ValueError(f"`{name}` contains NaN or Inf.")


def _generator_for(device: torch.device, seed: int) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(int(seed))


def kmeans_plus_plus_gpu(
    vectors: torch.Tensor,
    k: int,
    seed: int = 0,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """K-Means++ that keeps both the sample and distance vector on one device."""

    values = vectors.detach().float().contiguous()
    _validate_matrix("vectors", values)
    if values.shape[0] < int(k):
        raise ValueError(f"Need at least K samples, got N={values.shape[0]}, K={k}.")

    generator = _generator_for(values.device, seed)
    first = int(
        torch.randint(
            values.shape[0],
            (1,),
            generator=generator,
            device=values.device,
        ).item()
    )
    selected = [first]
    closest = torch.full(
        (values.shape[0],),
        float("inf"),
        device=values.device,
        dtype=torch.float32,
    )

    def update_closest(center: torch.Tensor) -> None:
        for start in range(0, values.shape[0], int(chunk_size)):
            chunk = values[start : start + int(chunk_size)]
            distances = (chunk - center).square().sum(dim=1)
            closest[start : start + chunk.shape[0]] = torch.minimum(
                closest[start : start + chunk.shape[0]],
                distances,
            )

    update_closest(values[first])
    while len(selected) < int(k):
        total = closest.sum()
        if not torch.isfinite(total) or float(total.item()) <= 0.0:
            selected_set = set(selected)
            next_index = next(
                index for index in range(values.shape[0]) if index not in selected_set
            )
        else:
            next_index = int(
                torch.multinomial(
                    closest / total,
                    1,
                    generator=generator,
                ).item()
            )
        selected.append(next_index)
        update_closest(values[next_index])
    indices = torch.tensor(selected, dtype=torch.long, device=values.device)
    return values[indices].clone().contiguous()


def _assign_in_chunks(
    values: torch.Tensor,
    centers: torch.Tensor,
    chunk_size: int,
    center_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    codes: list[torch.Tensor] = []
    distances: list[torch.Tensor] = []
    for start in range(0, values.shape[0], int(chunk_size)):
        code, distance = assign_nearest(
            values[start : start + int(chunk_size)],
            centers,
            center_block_size=center_block_size,
        )
        codes.append(code)
        distances.append(distance)
    return torch.cat(codes), torch.cat(distances)


def _refill_empty_centers(
    values: torch.Tensor,
    distances: torch.Tensor,
    centers: torch.Tensor,
    empty: torch.Tensor,
) -> None:
    count = int(empty.sum().item())
    if count == 0:
        return
    hardest = torch.topk(distances, k=count, largest=True, sorted=True).indices
    centers[empty] = values[hardest]


def fit_diagnostic_kmeans(
    train: torch.Tensor,
    validation: torch.Tensor | None,
    config: DiagnosticKMeansConfig,
    initial_centers: torch.Tensor | None = None,
) -> DiagnosticKMeansResult:
    """Fit full-batch Lloyd K-Means while exposing convergence diagnostics."""

    device = _resolve_device(config.device)
    train_values = train.detach().to(device=device, dtype=torch.float32).contiguous()
    _validate_matrix("train", train_values)
    if train_values.shape[0] < config.k:
        raise ValueError(
            f"K-Means needs at least K training vectors, got {train_values.shape[0]} and {config.k}."
        )
    validation_values = None
    if validation is not None:
        validation_values = validation.detach().to(
            device=device,
            dtype=torch.float32,
        ).contiguous()
        _validate_matrix("validation", validation_values, train_values.shape[1])

    if initial_centers is None:
        centers = kmeans_plus_plus_gpu(
            train_values,
            k=config.k,
            seed=config.seed,
            chunk_size=config.chunk_size,
        )
    else:
        centers = initial_centers.detach().to(device=device, dtype=torch.float32).clone()
        if tuple(centers.shape) != (config.k, train_values.shape[1]):
            raise ValueError(
                "Initial centers must have shape "
                f"{(config.k, train_values.shape[1])}, got {tuple(centers.shape)}."
            )

    history: list[KMeansIteration] = []
    previous_inertia: float | None = None
    previous_codes: torch.Tensor | None = None
    plateau_steps = 0
    converged = False
    stop_reason = "max_iters"

    for iteration in range(1, config.max_iters + 1):
        old_centers = centers.clone()
        codes, distances = _assign_in_chunks(
            train_values,
            old_centers,
            config.chunk_size,
            config.center_block_size,
        )
        raw_counts = torch.bincount(codes, minlength=config.k)
        sums = torch.zeros_like(old_centers)
        sums.index_add_(0, codes, train_values)
        centers = old_centers.clone()
        nonempty = raw_counts > 0
        centers[nonempty] = sums[nonempty] / raw_counts[nonempty].unsqueeze(1)
        empty = ~nonempty
        _refill_empty_centers(train_values, distances, centers, empty)

        train_codes, train_distances = _assign_in_chunks(
            train_values,
            centers,
            config.chunk_size,
            config.center_block_size,
        )
        train_inertia = float(train_distances.mean().item())
        validation_inertia = None
        if validation_values is not None:
            _, validation_distances = _assign_in_chunks(
                validation_values,
                centers,
                config.chunk_size,
                config.center_block_size,
            )
            validation_inertia = float(validation_distances.mean().item())

        relative_improvement = None
        if previous_inertia is not None:
            relative_improvement = (previous_inertia - train_inertia) / max(
                abs(previous_inertia),
                1e-12,
            )
        center_scale = old_centers.square().mean().sqrt().clamp_min(1e-12)
        center_shift = float(
            ((centers - old_centers).square().mean().sqrt() / center_scale).item()
        )
        assignment_change = None
        if previous_codes is not None:
            assignment_change = float((train_codes != previous_codes).float().mean().item())

        if relative_improvement is not None and relative_improvement <= config.tol:
            plateau_steps += 1
        else:
            plateau_steps = 0
        history.append(
            KMeansIteration(
                iteration=iteration,
                train_inertia=train_inertia,
                validation_inertia=validation_inertia,
                relative_improvement=relative_improvement,
                relative_center_shift=center_shift,
                assignment_change=assignment_change,
                empty_clusters=int(empty.sum().item()),
                plateau_steps=plateau_steps,
            )
        )
        previous_inertia = train_inertia
        previous_codes = train_codes

        if (
            iteration >= config.min_iters
            and plateau_steps >= config.patience
            and not empty.any()
        ):
            converged = True
            stop_reason = "inertia_plateau"
            break

    final_train_codes, final_train_distances = _assign_in_chunks(
        train_values,
        centers,
        config.chunk_size,
        config.center_block_size,
    )
    final_validation_codes = None
    final_validation_distances = None
    if validation_values is not None:
        final_validation_codes, final_validation_distances = _assign_in_chunks(
            validation_values,
            centers,
            config.chunk_size,
            config.center_block_size,
        )
    return DiagnosticKMeansResult(
        centers=centers.detach().cpu(),
        train_codes=final_train_codes.detach().cpu(),
        validation_codes=(
            None
            if final_validation_codes is None
            else final_validation_codes.detach().cpu()
        ),
        train_distances=final_train_distances.detach().cpu(),
        validation_distances=(
            None
            if final_validation_distances is None
            else final_validation_distances.detach().cpu()
        ),
        history=tuple(history),
        converged=converged,
        stop_reason=stop_reason,
    )


def encode_with_centers(
    values: torch.Tensor,
    centers: Sequence[torch.Tensor],
    device: str | torch.device = "auto",
    chunk_size: int = 8192,
    center_block_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = _resolve_device(device)
    residual = values.detach().to(device=target, dtype=torch.float32).contiguous()
    _validate_matrix("values", residual)
    all_codes: list[torch.Tensor] = []
    for level_centers in centers:
        device_centers = level_centers.to(device=target, dtype=torch.float32)
        codes, _ = _assign_in_chunks(
            residual,
            device_centers,
            chunk_size,
            center_block_size,
        )
        residual = residual - device_centers[codes]
        all_codes.append(codes.detach().cpu())
    return torch.stack(all_codes, dim=1), residual.detach().cpu()


def fit_diagnostic_rq(
    train: torch.Tensor,
    validation: torch.Tensor | None,
    test: torch.Tensor | None,
    config: DiagnosticKMeansConfig,
    levels: int = 3,
) -> DiagnosticRQResult:
    if int(levels) <= 0:
        raise ValueError(f"`levels` must be positive, got {levels}.")
    device = _resolve_device(config.device)
    train_residual = train.detach().to(device=device, dtype=torch.float32).contiguous()
    _validate_matrix("train", train_residual)
    validation_residual = None
    if validation is not None:
        validation_residual = validation.detach().to(
            device=device,
            dtype=torch.float32,
        ).contiguous()
        _validate_matrix("validation", validation_residual, train_residual.shape[1])
    test_residual = None
    if test is not None:
        test_residual = test.detach().to(device=device, dtype=torch.float32).contiguous()
        _validate_matrix("test", test_residual, train_residual.shape[1])

    train_mse = [float(train_residual.square().mean().item())]
    validation_mse = (
        None
        if validation_residual is None
        else [float(validation_residual.square().mean().item())]
    )
    test_mse = (
        None if test_residual is None else [float(test_residual.square().mean().item())]
    )
    level_results: list[DiagnosticKMeansResult] = []
    train_codes: list[torch.Tensor] = []
    validation_codes: list[torch.Tensor] = []
    test_codes: list[torch.Tensor] = []

    for level in range(int(levels)):
        level_config = replace(config, seed=int(config.seed) + level * 9973)
        result = fit_diagnostic_kmeans(
            train_residual,
            validation_residual,
            level_config,
        )
        centers = result.centers.to(device=device, dtype=torch.float32)
        train_code, _ = _assign_in_chunks(
            train_residual,
            centers,
            config.chunk_size,
            config.center_block_size,
        )
        train_residual = train_residual - centers[train_code]
        train_codes.append(train_code.detach().cpu())
        train_mse.append(float(train_residual.square().mean().item()))

        if validation_residual is not None:
            validation_code, _ = _assign_in_chunks(
                validation_residual,
                centers,
                config.chunk_size,
                config.center_block_size,
            )
            validation_residual = validation_residual - centers[validation_code]
            validation_codes.append(validation_code.detach().cpu())
            assert validation_mse is not None
            validation_mse.append(float(validation_residual.square().mean().item()))

        if test_residual is not None:
            test_code, _ = _assign_in_chunks(
                test_residual,
                centers,
                config.chunk_size,
                config.center_block_size,
            )
            test_residual = test_residual - centers[test_code]
            test_codes.append(test_code.detach().cpu())
            assert test_mse is not None
            test_mse.append(float(test_residual.square().mean().item()))
        level_results.append(result)

    return DiagnosticRQResult(
        levels=tuple(level_results),
        train_residual_mse=tuple(train_mse),
        validation_residual_mse=(
            None if validation_mse is None else tuple(validation_mse)
        ),
        test_residual_mse=None if test_mse is None else tuple(test_mse),
        train_codes=torch.stack(train_codes, dim=1),
        validation_codes=(
            None
            if not validation_codes
            else torch.stack(validation_codes, dim=1)
        ),
        test_codes=None if not test_codes else torch.stack(test_codes, dim=1),
    )


def usage_summary(codes: torch.Tensor, k: int) -> dict[str, float | int]:
    labels = codes.detach().reshape(-1).cpu().long()
    counts = torch.bincount(labels, minlength=int(k)).float()
    probabilities = counts / counts.sum().clamp_min(1.0)
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * nonzero.log()).sum().item())
    perplexity = math.exp(entropy)
    return {
        "used": int((counts > 0).sum().item()),
        "dead": int((counts == 0).sum().item()),
        "dead_fraction": float((counts == 0).float().mean().item()),
        "perplexity": float(perplexity),
        "perplexity_fraction": float(perplexity / float(k)),
        "maximum_cluster_fraction": float(
            counts.max().item() / counts.sum().clamp_min(1.0).item()
        ),
    }


def adjusted_rand_index(labels_a: torch.Tensor, labels_b: torch.Tensor) -> float:
    """Permutation-invariant agreement between two cluster assignments."""

    a = labels_a.detach().reshape(-1).cpu().long()
    b = labels_b.detach().reshape(-1).cpu().long()
    if a.numel() != b.numel() or a.numel() < 2:
        raise ValueError("ARI needs equal label lengths with at least two samples.")
    _, inverse_a = torch.unique(a, return_inverse=True)
    _, inverse_b = torch.unique(b, return_inverse=True)
    width = int(inverse_b.max().item()) + 1
    joint = inverse_a * width + inverse_b
    joint_counts = torch.bincount(joint).double()
    counts_a = torch.bincount(inverse_a).double()
    counts_b = torch.bincount(inverse_b).double()

    def choose_two(values: torch.Tensor) -> torch.Tensor:
        return (values * (values - 1.0) / 2.0).sum()

    same_joint = choose_two(joint_counts)
    same_a = choose_two(counts_a)
    same_b = choose_two(counts_b)
    total_pairs = float(a.numel() * (a.numel() - 1) / 2)
    expected = same_a * same_b / total_pairs
    maximum = 0.5 * (same_a + same_b)
    denominator = maximum - expected
    if abs(float(denominator.item())) < 1e-12:
        return 1.0
    return float(((same_joint - expected) / denominator).item())

