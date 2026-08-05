from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler

from codewam.codebook_eval.shards import atomic_torch_save, file_sha256
from codewam.data import (
    FrozenLanguageCache,
    JointModelBatch,
    JointWindowCache,
    LanguageConditionedJointWindowCache,
    PolicyNormalizer,
    collate_joint_windows,
    load_frozen_artifact_chart,
)
from codewam.data.frozen_assignment import FrozenArtifactChart
from codewam.models import (
    ActionBatch,
    CodeMeasurements,
    CodeWAMBatch,
    CodeWAMConfig,
    FutureCodeTargets,
    PolicyCondition,
    StateInputs,
    SupervisionMasks,
    TransitionSchedule,
    build_codewam_v1,
    build_codewam_v2,
)
from codewam.models.codewam_v1 import CodeWAMV1
from codewam.models.codewam_v2 import CodeWAMV2


POLICY_ABLATION_SCHEMA = "codewam.policy-ablation.v1"
POLICY_ABLATION_PROTOCOL_SCHEMA = "codewam.policy-ablation-protocol.v1"
POLICY_ABLATION_CHECKPOINT_SCHEMA = "codewam.policy-ablation-checkpoint.v1"
POLICY_VARIANTS = ("C0", "C1", "C2")
PolicyModel = CodeWAMV1 | CodeWAMV2


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
class PolicyAblationRunConfig:
    cache_dir: str
    language_cache_dir: str
    normalization_dir: str
    output_dir: str
    artifact_paths: dict[str, str]
    chart_name: str = "droid"
    architecture: str = "v1"
    seed: int = 20260731
    batch_size: int = 4
    eval_batch_size: int = 8
    epochs: int = 10
    max_steps: int | None = 200
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    device: str = "cuda"
    amp_dtype: str = "bfloat16"
    checkpoint_every: int = 50
    log_every: int = 10
    eval_windows: int = 2048
    eval_flow_draws: int = 1
    sample_steps: int = 10
    bootstrap_samples: int = 2000
    model: CodeWAMConfig = field(default_factory=CodeWAMConfig)

    def __post_init__(self) -> None:
        paths = (
            self.cache_dir,
            self.language_cache_dir,
            self.normalization_dir,
            self.output_dir,
            self.chart_name,
        )
        if any(not value for value in paths):
            raise ValueError("Policy-ablation paths and chart name must be nonempty.")
        if set(self.artifact_paths) != {"Q2", "Q3", "Q5"}:
            raise ValueError("Policy ablation requires exactly Q2/Q3/Q5 artifacts.")
        if self.architecture not in {"v1", "v2"}:
            raise ValueError("Policy architecture must be `v1` or `v2`.")
        positive = (
            self.batch_size,
            self.eval_batch_size,
            self.epochs,
            self.checkpoint_every,
            self.log_every,
            self.eval_windows,
            self.eval_flow_draws,
            self.sample_steps,
            self.bootstrap_samples,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("Policy-ablation counts must be positive.")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("Policy-ablation maximum steps must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Policy-ablation optimizer values are invalid.")
        if self.num_workers < 0:
            raise ValueError("Policy-ablation worker count must be non-negative.")
        if not math.isfinite(self.grad_clip_norm) or self.grad_clip_norm <= 0:
            raise ValueError("Policy-ablation gradient clipping is invalid.")
        if self.amp_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"Unsupported AMP dtype `{self.amp_dtype}`.")
        if self.model.variant != "C2":
            raise ValueError("The policy-ablation base model must use C2.")
        if self.model.dropout != 0.0:
            raise ValueError("Policy ablation fixes dropout at zero.")
        if self.model.lambda_code <= 0:
            raise ValueError("C2 policy ablation needs a positive code weight.")


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
            raise RuntimeError("Distributed policy ablation requires CUDA/NCCL.")
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
                raise RuntimeError("Policy ablation requested unavailable CUDA.")
            device = torch.device("cuda", 0 if device.index is None else device.index)
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


def _gather_payloads(
    payload: dict[str, Any],
    context: _DistributedContext,
) -> list[dict[str, Any]]:
    if context.world_size == 1:
        return [payload]
    outputs: list[dict[str, Any] | None] = [None] * context.world_size
    dist.all_gather_object(outputs, payload)
    return [value for value in outputs if value is not None]


class _IndexSampler(Sampler[int]):
    def __init__(self, indices: Sequence[int]):
        self.indices = tuple(int(value) for value in indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class _PolicyDataset(Dataset):
    def __init__(self, cache: LanguageConditionedJointWindowCache):
        self.cache = cache

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, index: int):
        return self.cache[index]


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
        raise ValueError("Policy-ablation rank/world size is invalid.")
    if group_keys is None:
        values = list(indices)
        if training:
            random.Random(seed + 1_000_003 * epoch).shuffle(values)
    else:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            grouped[str(group_keys[index])].append(int(index))
        names = sorted(grouped)
        if training:
            generator = random.Random(seed + 1_000_003 * epoch)
            generator.shuffle(names)
            for name in names:
                generator.shuffle(grouped[name])
        values = [index for name in names for index in grouped[name]]
    if training:
        usable = len(values) - len(values) % world_size
        values = values[:usable]
        per_rank = usable // world_size
        return tuple(values[rank * per_rank : (rank + 1) * per_rank])
    start = rank * len(values) // world_size
    stop = (rank + 1) * len(values) // world_size
    return tuple(values[start:stop])


def fixed_eval_subset(
    cache: JointWindowCache,
    indices: Sequence[int],
    *,
    split: str,
    seed: int,
    maximum: int,
) -> tuple[int, ...]:
    if maximum <= 0:
        raise ValueError("Evaluation subset size must be positive.")
    ranked = sorted(
        (int(index) for index in indices),
        key=lambda index: hashlib.sha256(
            f"{seed}|{split}|{cache.windows[index].window_id}".encode("utf-8")
        ).digest(),
    )
    return tuple(ranked[:maximum])


def _make_loader(
    dataset: _PolicyDataset,
    indices: Sequence[int],
    *,
    config: PolicyAblationRunConfig,
    batch_size: int,
    training: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=_IndexSampler(indices),
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
        persistent_workers=config.num_workers > 0,
        drop_last=training,
        collate_fn=partial(
            collate_joint_windows,
            language_dim=config.model.language_dim,
        ),
    )


def _move_optional(
    value: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    return None if value is None else value.to(device, non_blocking=True)


def _move_batch(batch: CodeWAMBatch, device: torch.device) -> CodeWAMBatch:
    state = batch.state
    return CodeWAMBatch(
        state=StateInputs(
            latents=state.latents.to(device, non_blocking=True),
            proprio_history=state.proprio_history.to(device, non_blocking=True),
            past_actions=state.past_actions.to(device, non_blocking=True),
            latent_valid=_move_optional(state.latent_valid, device),
            proprio_valid=_move_optional(state.proprio_valid, device),
            past_action_valid=_move_optional(state.past_action_valid, device),
            latent_time_offsets=_move_optional(state.latent_time_offsets, device),
            proprio_time_offsets=_move_optional(
                state.proprio_time_offsets,
                device,
            ),
            past_action_time_offsets=_move_optional(
                state.past_action_time_offsets,
                device,
            ),
        ),
        policy=PolicyCondition(
            language=batch.policy.language.to(device, non_blocking=True),
            language_valid=_move_optional(batch.policy.language_valid, device),
        ),
        actions=ActionBatch(
            values=batch.actions.values.to(device, non_blocking=True),
            valid=_move_optional(batch.actions.valid, device),
        ),
        supervision=SupervisionMasks(
            temporal=batch.supervision.temporal.to(device, non_blocking=True),
            action=batch.supervision.action.to(device, non_blocking=True),
            dynamics=batch.supervision.dynamics.to(device, non_blocking=True),
        ),
        codes=(
            None
            if batch.codes is None
            else CodeMeasurements(
                code_ids=batch.codes.code_ids.to(device, non_blocking=True),
                available=batch.codes.available.to(device, non_blocking=True),
                chart_names=batch.codes.chart_names,
            )
        ),
        future_codes=(
            None
            if batch.future_codes is None
            else FutureCodeTargets(
                code_ids=batch.future_codes.code_ids.to(
                    device,
                    non_blocking=True,
                ),
                available=batch.future_codes.available.to(
                    device,
                    non_blocking=True,
                ),
                schedule=(
                    None
                    if batch.future_codes.schedule is None
                    else TransitionSchedule(
                        action_prefix_lengths=(
                            batch.future_codes.schedule.action_prefix_lengths.to(
                                device,
                                non_blocking=True,
                            )
                        ),
                        delta_times=batch.future_codes.schedule.delta_times.to(
                            device,
                            non_blocking=True,
                        ),
                    )
                ),
            )
        ),
    )


def _autocast_context(
    config: PolicyAblationRunConfig,
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


def _flow_inputs(
    actions: torch.Tensor,
    *,
    seed: int,
    phase: str,
    step: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    digest = hashlib.sha256(
        f"{seed}|{phase}|{step}|{rank}".encode("utf-8")
    ).digest()
    flow_seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
    generator = torch.Generator(device=actions.device)
    generator.manual_seed(flow_seed)
    noise = torch.randn(
        actions.shape,
        dtype=actions.dtype,
        device=actions.device,
        generator=generator,
    )
    flow_time = torch.rand(
        (actions.shape[0],),
        dtype=actions.dtype,
        device=actions.device,
        generator=generator,
    )
    return noise, flow_time


def _validate_inputs(
    cache: JointWindowCache,
    language: FrozenLanguageCache,
    normalizer: PolicyNormalizer,
    chart: FrozenArtifactChart,
    config: PolicyAblationRunConfig,
) -> None:
    contract = cache.contract
    expected_hashes = {
        row["family"]: row["sha256"]
        for row in contract["chart"]["families"]
    }
    actual_hashes = dict(zip(chart.families, chart.artifact_sha256))
    if contract["chart"]["name"] != chart.name or expected_hashes != actual_hashes:
        raise RuntimeError("Policy-ablation chart differs from the joint cache.")
    if language.contract["joint_cache"]["contract_hash"] != contract["contract_hash"]:
        raise RuntimeError("Policy-ablation language cache belongs to another cache.")
    if (
        normalizer.contract["joint_cache"]["contract_hash"]
        != contract["contract_hash"]
    ):
        raise RuntimeError("Policy normalization belongs to another joint cache.")
    dimensions = {
        "latent_channels": int(contract["latent_channels"]),
        "proprio_dim": normalizer.proprio_dim,
        "action_dim": normalizer.action_dim,
        "language_dim": language.hidden_size,
    }
    changed = {
        name: (getattr(config.model, name), expected)
        for name, expected in dimensions.items()
        if getattr(config.model, name) != expected
    }
    if changed:
        raise ValueError(f"Policy model/input dimensions differ: {changed}.")
    window = contract["window"]
    if config.model.max_time < int(window["state_latent_ticks"]):
        raise ValueError("Policy model max_time is shorter than the cache.")
    if config.model.max_action_horizon < int(window["action_horizon"]):
        raise ValueError("Policy model action horizon is shorter than the cache.")
    if config.model.max_cameras < len(contract["camera_ids"]):
        raise ValueError("Policy model camera capacity is smaller than the cache.")
    missing = [
        split
        for split in ("train", "val", "test")
        if not cache.split_indices(split)
    ]
    if missing:
        raise ValueError(f"Policy cache has no windows for splits {missing}.")


def _variant_config(
    config: PolicyAblationRunConfig,
    variant: str,
) -> CodeWAMConfig:
    if variant not in POLICY_VARIANTS:
        raise ValueError(f"Unknown policy variant `{variant}`.")
    return replace(config.model, variant=variant)


def _build_model(
    config: PolicyAblationRunConfig,
    chart: FrozenArtifactChart,
    *,
    variant: str,
    device: torch.device,
) -> PolicyModel:
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    builder = build_codewam_v1 if config.architecture == "v1" else build_codewam_v2
    model = builder(_variant_config(config, variant), {chart.name: chart.artifacts})
    if variant == "C0":
        model.frozen_codebook.requires_grad_(False)
    if variant in {"C0", "C1"}:
        dynamics = (
            model.code_dynamics
            if isinstance(model, CodeWAMV1)
            else model.transition
        )
        dynamics.requires_grad_(False)
    return model.to(device)


def _unwrap_model(
    model: PolicyModel | DistributedDataParallel,
) -> PolicyModel:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def _protocol(
    config: PolicyAblationRunConfig,
    cache: JointWindowCache,
    language: FrozenLanguageCache,
    normalizer: PolicyNormalizer,
    chart: FrozenArtifactChart,
    eval_subsets: Mapping[str, Sequence[int]],
    *,
    world_size: int,
) -> dict[str, Any]:
    run_values = asdict(config)
    for key in (
        "cache_dir",
        "language_cache_dir",
        "normalization_dir",
        "output_dir",
        "artifact_paths",
    ):
        run_values.pop(key)
    subset_rows = {}
    for split, indices in eval_subsets.items():
        window_ids = [cache.windows[index].window_id for index in indices]
        subset_rows[split] = {
            "windows": len(window_ids),
            "window_ids_hash": _canonical_hash(window_ids),
        }
    payload = {
        "schema": POLICY_ABLATION_PROTOCOL_SCHEMA,
        "cache": {
            "contract_hash": cache.contract["contract_hash"],
            "summary_sha256": file_sha256(Path(config.cache_dir) / "summary.json"),
        },
        "language_cache": {
            "contract_hash": language.contract["contract_hash"],
            "summary_sha256": file_sha256(
                Path(config.language_cache_dir) / "summary.json"
            ),
        },
        "normalization": {
            "contract_hash": normalizer.contract["contract_hash"],
            "summary_sha256": file_sha256(
                Path(config.normalization_dir) / "summary.json"
            ),
            "action_representation": normalizer.contract["representation"],
        },
        "chart": chart.compact_identity(),
        "artifact_sha256": dict(zip(chart.families, chart.artifact_sha256)),
        "variants": {
            "C0": "continuous state plus proprio/history and language policy",
            "C1": "C0 plus frozen RQ measurements in world belief",
            "C2": "C1 plus action-conditioned future-code auxiliary loss",
        },
        "primary_metric": "fixed-noise normalized action flow MSE",
        "loss": "L_action for C0/C1; L_action + lambda_code*L_code for C2",
        "fairness": {
            "shared_initialization": True,
            "shared_window_order": True,
            "shared_flow_noise_and_time": True,
            "equal_optimizer_steps": True,
            "dropout": 0.0,
        },
        "evaluation_subsets": subset_rows,
        "distributed": {
            "world_size": world_size,
            "per_rank_batch_size": config.batch_size,
            "effective_batch_size": world_size * config.batch_size,
            "per_rank_eval_batch_size": config.eval_batch_size,
        },
        "run_config": run_values,
        "implementation_sha256": {
            "policy_ablation": file_sha256(Path(__file__)),
            "policy_normalization": file_sha256(
                Path(__file__).parents[1] / "data" / "policy_normalization.py"
            ),
            "codewam_v1": file_sha256(
                Path(__file__).parents[1] / "models" / "codewam_v1.py"
            ),
            "codewam_v2": file_sha256(
                Path(__file__).parents[1] / "models" / "codewam_v2.py"
            ),
            "world_state": file_sha256(
                Path(__file__).parents[1] / "models" / "world_state.py"
            ),
            "multiclock_dynamics": file_sha256(
                Path(__file__).parents[1]
                / "models"
                / "multiclock_dynamics.py"
            ),
            "action_flow": file_sha256(
                Path(__file__).parents[1] / "models" / "action_flow.py"
            ),
        },
    }
    return {**payload, "protocol_hash": _canonical_hash(payload)}


def _prepare_initialization(
    config: PolicyAblationRunConfig,
    chart: FrozenArtifactChart,
    protocol: Mapping[str, Any],
    context: _DistributedContext,
) -> tuple[Path, str]:
    path = Path(config.output_dir) / "initialization.pt"
    if context.is_primary and not path.exists():
        model = _build_model(
            config,
            chart,
            variant="C2",
            device=torch.device("cpu"),
        )
        atomic_torch_save(
            {
                "schema": "codewam.policy-ablation-initialization.v1",
                "protocol_hash": protocol["protocol_hash"],
                "model": model.state_dict(),
            },
            path,
        )
    _barrier(context)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("schema")
        != "codewam.policy-ablation-initialization.v1"
        or payload.get("protocol_hash") != protocol["protocol_hash"]
    ):
        raise RuntimeError("Policy initialization belongs to another protocol.")
    return path, file_sha256(path)


def _initial_state(path: Path) -> Mapping[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)["model"]


def _checkpoint_payload(
    *,
    variant: str,
    protocol_hash: str,
    epoch: int,
    sample_offset: int,
    global_step: int,
    history: Sequence[Mapping[str, Any]],
    model: PolicyModel | DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict[str, Any]:
    return {
        "schema": POLICY_ABLATION_CHECKPOINT_SCHEMA,
        "protocol_hash": protocol_hash,
        "variant": variant,
        "epoch": int(epoch),
        "sample_offset": int(sample_offset),
        "global_step": int(global_step),
        "history": list(history),
        "model": _unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    variant: str,
    protocol_hash: str,
    path: Path,
) -> None:
    if (
        checkpoint.get("schema") != POLICY_ABLATION_CHECKPOINT_SCHEMA
        or checkpoint.get("protocol_hash") != protocol_hash
        or checkpoint.get("variant") != variant
    ):
        raise RuntimeError(f"Policy checkpoint mismatch: {path}.")


def _train_variant(
    variant: str,
    *,
    config: PolicyAblationRunConfig,
    chart: FrozenArtifactChart,
    dataset: _PolicyDataset,
    normalizer: PolicyNormalizer,
    train_indices: Sequence[int],
    initialization_path: Path,
    protocol_hash: str,
    context: _DistributedContext,
) -> dict[str, Any]:
    variant_dir = Path(config.output_dir) / variant.lower()
    latest_path = variant_dir / "latest.pt"
    final_path = variant_dir / "final.pt"
    if final_path.is_file():
        checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
        _validate_checkpoint(
            checkpoint,
            variant=variant,
            protocol_hash=protocol_hash,
            path=final_path,
        )
        return {
            "variant": variant,
            "status": "reused",
            "optimizer_steps": int(checkpoint["global_step"]),
            "history": checkpoint["history"],
            "checkpoint": str(final_path),
            "checkpoint_sha256": file_sha256(final_path),
        }
    model = _build_model(
        config,
        chart,
        variant=variant,
        device=context.device,
    )
    model.load_state_dict(_initial_state(initialization_path), strict=True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(context.device.type == "cuda" and config.amp_dtype == "float16"),
    )
    epoch = 0
    sample_offset = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    if latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        _validate_checkpoint(
            checkpoint,
            variant=variant,
            protocol_hash=protocol_hash,
            path=latest_path,
        )
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        epoch = int(checkpoint["epoch"])
        sample_offset = int(checkpoint["sample_offset"])
        global_step = int(checkpoint["global_step"])
        history = list(checkpoint["history"])
    wrapped: PolicyModel | DistributedDataParallel = model
    if context.world_size > 1:
        wrapped = DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
        )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    interval = torch.zeros(4, dtype=torch.float64, device=context.device)
    interval_started = time.monotonic()
    while epoch < config.epochs:
        if config.max_steps is not None and global_step >= config.max_steps:
            break
        ranked = _rank_indices(
            train_indices,
            rank=context.rank,
            world_size=context.world_size,
            seed=config.seed,
            epoch=epoch,
            training=True,
            group_keys=dataset.cache.window_shards,
        )
        usable = len(ranked) - len(ranked) % config.batch_size
        ranked = ranked[:usable]
        if sample_offset < 0 or sample_offset > len(ranked):
            raise RuntimeError("Policy checkpoint sample offset is invalid.")
        remaining = ranked[sample_offset:]
        loader = _make_loader(
            dataset,
            remaining,
            config=config,
            batch_size=config.batch_size,
            training=True,
        )
        wrapped.train()
        for cpu_joint in loader:
            if config.max_steps is not None and global_step >= config.max_steps:
                break
            normalized = normalizer.transform_batch(cpu_joint.model)
            batch = _move_batch(normalized, context.device)
            noise, flow_time = _flow_inputs(
                batch.actions.values,
                seed=config.seed,
                phase="train",
                step=global_step,
                rank=context.rank,
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(config, context.device):
                output = wrapped(batch, noise=noise, flow_time=flow_time)
            scaler.scale(output.total).backward()
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
            global_step += 1
            sample_offset += config.batch_size
            interval += torch.tensor(
                [
                    float(output.total.detach()),
                    float(output.action.detach()),
                    float(output.code.detach()),
                    1.0,
                ],
                dtype=torch.float64,
                device=context.device,
            )
            if global_step % config.log_every == 0:
                totals = interval.clone()
                if context.world_size > 1:
                    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
                history.append(
                    {
                        "optimizer_step": global_step,
                        "total_loss": float(totals[0] / totals[3]),
                        "action_loss": float(totals[1] / totals[3]),
                        "code_loss": float(totals[2] / totals[3]),
                        "elapsed_seconds": time.monotonic() - interval_started,
                    }
                )
                interval.zero_()
                interval_started = time.monotonic()
            if global_step % config.checkpoint_every == 0:
                checkpoint = _checkpoint_payload(
                    variant=variant,
                    protocol_hash=protocol_hash,
                    epoch=epoch,
                    sample_offset=sample_offset,
                    global_step=global_step,
                    history=history,
                    model=wrapped,
                    optimizer=optimizer,
                    scaler=scaler,
                )
                if context.is_primary:
                    atomic_torch_save(checkpoint, latest_path)
                _barrier(context)
        if sample_offset >= len(ranked):
            epoch += 1
            sample_offset = 0
        del loader
    if interval[3].item() > 0:
        totals = interval.clone()
        if context.world_size > 1:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        history.append(
            {
                "optimizer_step": global_step,
                "total_loss": float(totals[0] / totals[3]),
                "action_loss": float(totals[1] / totals[3]),
                "code_loss": float(totals[2] / totals[3]),
                "elapsed_seconds": time.monotonic() - interval_started,
            }
        )
    final_checkpoint = _checkpoint_payload(
        variant=variant,
        protocol_hash=protocol_hash,
        epoch=epoch,
        sample_offset=sample_offset,
        global_step=global_step,
        history=history,
        model=wrapped,
        optimizer=optimizer,
        scaler=scaler,
    )
    if context.is_primary:
        atomic_torch_save(final_checkpoint, final_path)
        latest_path.unlink(missing_ok=True)
    _barrier(context)
    return {
        "variant": variant,
        "status": "trained",
        "optimizer_steps": global_step,
        "trainable_parameters": trainable_parameters,
        "history": history,
        "checkpoint": str(final_path),
        "checkpoint_sha256": file_sha256(final_path),
    }


def _load_model(
    variant: str,
    *,
    config: PolicyAblationRunConfig,
    chart: FrozenArtifactChart,
    protocol_hash: str,
    device: torch.device,
) -> PolicyModel:
    path = Path(config.output_dir) / variant.lower() / "final.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint(
        checkpoint,
        variant=variant,
        protocol_hash=protocol_hash,
        path=path,
    )
    model = _build_model(
        config,
        chart,
        variant=variant,
        device=device,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()


def _merge_episode_rows(
    payloads: Sequence[Mapping[str, Mapping[str, float | int]]],
) -> dict[str, dict[str, float | int]]:
    merged: dict[str, dict[str, float | int]] = {}
    for payload in payloads:
        for episode, row in payload.items():
            target = merged.setdefault(episode, {"sum": 0.0, "count": 0})
            target["sum"] = float(target["sum"]) + float(row["sum"])
            target["count"] = int(target["count"]) + int(row["count"])
    return merged


@torch.no_grad()
def _evaluate_variant(
    variant: str,
    *,
    split: str,
    indices: Sequence[int],
    config: PolicyAblationRunConfig,
    chart: FrozenArtifactChart,
    dataset: _PolicyDataset,
    normalizer: PolicyNormalizer,
    protocol_hash: str,
    context: _DistributedContext,
) -> dict[str, Any]:
    ranked = _rank_indices(
        indices,
        rank=context.rank,
        world_size=context.world_size,
        seed=config.seed,
        epoch=0,
        training=False,
        group_keys=dataset.cache.window_shards,
    )
    loader = _make_loader(
        dataset,
        ranked,
        config=config,
        batch_size=config.eval_batch_size,
        training=False,
    )
    model = _load_model(
        variant,
        config=config,
        chart=chart,
        protocol_hash=protocol_hash,
        device=context.device,
    )
    totals = torch.zeros(9, dtype=torch.float64, device=context.device)
    episode_rows: dict[str, dict[str, float | int]] = {}
    for batch_index, cpu_joint in enumerate(loader):
        normalized = normalizer.transform_batch(cpu_joint.model)
        batch = _move_batch(normalized, context.device)
        sample_error = torch.zeros(
            batch.actions.values.shape[0],
            dtype=torch.float64,
            device=context.device,
        )
        for draw in range(config.eval_flow_draws):
            noise, flow_time = _flow_inputs(
                batch.actions.values,
                seed=config.seed,
                phase=f"eval-{split}-{draw}",
                step=batch_index,
                rank=context.rank,
            )
            with _autocast_context(config, context.device):
                output = model(batch, noise=noise, flow_time=flow_time)
            squared = (output.flow.velocity - output.flow.target_velocity).square()
            valid = batch.actions.valid & batch.supervision.action[:, None]
            denominator = (
                valid.sum(dim=1).clamp_min(1) * batch.actions.values.shape[2]
            )
            per_sample = (
                squared * valid[:, :, None].to(squared.dtype)
            ).sum(dim=(1, 2)) / denominator
            sample_error += per_sample.double()
        sample_error /= config.eval_flow_draws
        totals[0] += sample_error.sum()
        totals[1] += sample_error.numel()
        for parent, value in zip(cpu_joint.parent_episode_ids, sample_error.tolist()):
            row = episode_rows.setdefault(parent, {"sum": 0.0, "count": 0})
            row["sum"] = float(row["sum"]) + float(value)
            row["count"] = int(row["count"]) + 1

        initial_noise, _ = _flow_inputs(
            batch.actions.values,
            seed=config.seed,
            phase=f"sample-{split}",
            step=batch_index,
            rank=context.rank,
        )
        with _autocast_context(config, context.device):
            predicted = model.infer_actions(
                state=batch.state,
                policy=batch.policy,
                codes=batch.codes,
                horizon=batch.actions.values.shape[1],
                steps=config.sample_steps,
                initial_noise=initial_noise,
            )
        valid = batch.actions.valid & batch.supervision.action[:, None]
        normalized_absolute = (predicted - batch.actions.values).abs()
        raw_predicted = normalizer.denormalize_actions(predicted.float())
        raw_target = normalizer.denormalize_actions(batch.actions.values.float())
        raw_absolute = (raw_predicted - raw_target).abs()
        angular = torch.atan2(
            torch.sin(raw_predicted[..., 3:6] - raw_target[..., 3:6]),
            torch.cos(raw_predicted[..., 3:6] - raw_target[..., 3:6]),
        ).abs()
        valid_values = valid.sum().double()
        totals[2] += (
            normalized_absolute * valid[:, :, None]
        ).sum().double()
        totals[3] += valid_values * normalized_absolute.shape[2]
        totals[4] += (raw_absolute[..., :3] * valid[:, :, None]).sum().double()
        totals[5] += valid_values * 3
        totals[6] += (angular * valid[:, :, None]).sum().double()
        totals[7] += valid_values * 3
        totals[8] += (raw_absolute[..., 6] * valid).sum().double()
    if context.world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    gathered = _gather_payloads(episode_rows, context)
    merged_rows = _merge_episode_rows(gathered)
    report = {
        "split": split,
        "windows": len(indices),
        "flow_mse": float(totals[0] / totals[1].clamp_min(1)),
        "sample_normalized_mae": float(
            totals[2] / totals[3].clamp_min(1)
        ),
        "sample_xyz_mae": float(totals[4] / totals[5].clamp_min(1)),
        "sample_angle_coordinate_mae_degrees": float(
            (totals[6] / totals[7].clamp_min(1)) * (180.0 / math.pi)
        ),
        "sample_gripper_mae": float(
            totals[8] / (totals[5] / 3).clamp_min(1)
        ),
        "episode_flow_mse": merged_rows,
    }
    del loader, model
    if context.device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def paired_episode_bootstrap(
    candidate: Mapping[str, Mapping[str, float | int]],
    baseline: Mapping[str, Mapping[str, float | int]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    episodes = sorted(set(candidate) & set(baseline))
    deltas = []
    for episode in episodes:
        candidate_count = int(candidate[episode]["count"])
        baseline_count = int(baseline[episode]["count"])
        if candidate_count <= 0 or baseline_count <= 0:
            continue
        if candidate_count != baseline_count:
            raise RuntimeError(
                "Paired policy variants evaluated different window counts."
            )
        deltas.append(
            float(candidate[episode]["sum"]) / candidate_count
            - float(baseline[episode]["sum"]) / baseline_count
        )
    if not deltas:
        return {
            "episodes": 0,
            "mean_delta_candidate_minus_baseline": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "episode_win_fraction": float("nan"),
        }
    generator = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(
            sum(generator.choice(deltas) for _ in deltas) / len(deltas)
        )
    means.sort()
    lower = means[max(0, int(0.025 * samples) - 1)]
    upper = means[min(samples - 1, int(0.975 * samples))]
    return {
        "episodes": len(deltas),
        "mean_delta_candidate_minus_baseline": sum(deltas) / len(deltas),
        "ci95": [lower, upper],
        "episode_win_fraction": sum(value < 0 for value in deltas) / len(deltas),
    }


def _strip_episode_rows(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key != "episode_flow_mse"
    }


def run_policy_ablation(config: PolicyAblationRunConfig) -> dict[str, Any]:
    context = _distributed_context(config.device)
    try:
        cache = JointWindowCache(
            config.cache_dir,
            verify_index_hashes=context.is_primary,
        )
        _barrier(context)
        language = FrozenLanguageCache(
            config.language_cache_dir,
            expected_joint_cache_contract_hash=cache.contract["contract_hash"],
            verify_hashes=context.is_primary,
        )
        normalizer = PolicyNormalizer(
            config.normalization_dir,
            expected_joint_cache_contract_hash=cache.contract["contract_hash"],
            verify_hashes=context.is_primary,
        )
        _barrier(context)
        chart = load_frozen_artifact_chart(
            config.chart_name,
            config.artifact_paths,
        )
        _validate_inputs(cache, language, normalizer, chart, config)
        conditioned = LanguageConditionedJointWindowCache(cache, language)
        dataset = _PolicyDataset(conditioned)
        split_indices = {
            split: cache.split_indices(split)
            for split in ("train", "val", "test")
        }
        minimum_train = context.world_size * config.batch_size
        if len(split_indices["train"]) < minimum_train:
            raise ValueError(
                f"Policy ablation needs at least {minimum_train} train windows."
            )
        eval_subsets = {
            split: fixed_eval_subset(
                cache,
                split_indices[split],
                split=split,
                seed=config.seed,
                maximum=config.eval_windows,
            )
            for split in ("val", "test")
        }
        protocol = _protocol(
            config,
            cache,
            language,
            normalizer,
            chart,
            eval_subsets,
            world_size=context.world_size,
        )
        protocol_path = Path(config.output_dir) / "protocol.json"
        if context.is_primary:
            if protocol_path.is_file():
                existing = json.loads(protocol_path.read_text(encoding="utf-8"))
                if existing != protocol:
                    raise RuntimeError(
                        "Existing policy output uses another protocol."
                    )
            else:
                _atomic_json(protocol_path, protocol)
        _barrier(context)
        initialization_path, initialization_sha256 = _prepare_initialization(
            config,
            chart,
            protocol,
            context,
        )
        training = {}
        for variant in POLICY_VARIANTS:
            training[variant] = _train_variant(
                variant,
                config=config,
                chart=chart,
                dataset=dataset,
                normalizer=normalizer,
                train_indices=split_indices["train"],
                initialization_path=initialization_path,
                protocol_hash=protocol["protocol_hash"],
                context=context,
            )
        step_counts = {
            int(value["optimizer_steps"]) for value in training.values()
        }
        if len(step_counts) != 1:
            raise RuntimeError("Policy variants received unequal update budgets.")

        evaluations: dict[str, dict[str, dict[str, Any]]] = {
            variant: {} for variant in POLICY_VARIANTS
        }
        for split, indices in eval_subsets.items():
            for variant in POLICY_VARIANTS:
                evaluations[variant][split] = _evaluate_variant(
                    variant,
                    split=split,
                    indices=indices,
                    config=config,
                    chart=chart,
                    dataset=dataset,
                    normalizer=normalizer,
                    protocol_hash=protocol["protocol_hash"],
                    context=context,
                )
        comparison_pairs = (
            ("C1-vs-C0", "C1", "C0"),
            ("C2-vs-C1", "C2", "C1"),
            ("C2-vs-C0", "C2", "C0"),
        )
        comparisons = {
            split: {
                name: paired_episode_bootstrap(
                    evaluations[candidate][split]["episode_flow_mse"],
                    evaluations[baseline][split]["episode_flow_mse"],
                    samples=config.bootstrap_samples,
                    seed=config.seed + pair_index + split_index * 100,
                )
                for pair_index, (name, candidate, baseline) in enumerate(
                    comparison_pairs
                )
            }
            for split_index, split in enumerate(("val", "test"))
        }
        report = {
            "schema": POLICY_ABLATION_SCHEMA,
            "protocol_hash": protocol["protocol_hash"],
            "initialization_sha256": initialization_sha256,
            "training": training,
            "evaluation": {
                variant: {
                    split: _strip_episode_rows(split_report)
                    for split, split_report in reports.items()
                }
                for variant, reports in evaluations.items()
            },
            "paired_episode_comparisons": comparisons,
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
