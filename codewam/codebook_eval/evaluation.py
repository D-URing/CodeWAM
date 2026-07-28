from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import torch
from omegaconf import DictConfig, OmegaConf

from codewam.data.droid_manifest import write_json_report

from .manifest import EpisodeManifest
from .shards import (
    PooledFeatureEpisode,
    expand_shard_paths,
    file_sha256,
    iter_pooled_feature_episodes,
)
from .streaming import (
    CausalDescriptorSource,
    FrozenRQArtifact,
    assign_nearest,
)


HELDOUT_EVALUATION_SCHEMA = "codewam.heldout-rq-evaluation.v1"
HELDOUT_CONTRACT_SCHEMA = "codewam.heldout-rq-contract.v1"


def _plain_config(config: DictConfig) -> dict[str, Any]:
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("The held-out evaluation config must be a mapping.")
    return payload


def _resolve_device(value: str) -> torch.device:
    if str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device `{value}` is unavailable.")
    return device


def _episode_factory(
    shard_paths: tuple[Path, ...],
    split: str,
    expected_episode_ids: set[str],
):
    def episodes() -> Iterator[PooledFeatureEpisode]:
        seen: set[str] = set()
        for episode in iter_pooled_feature_episodes(shard_paths, split=split):
            if episode.episode_id not in expected_episode_ids:
                raise ValueError(
                    f"Held-out shard episode `{episode.episode_id}` is absent "
                    "from the evaluation manifest."
                )
            if episode.episode_id in seen:
                raise ValueError(
                    f"Duplicate held-out episode `{episode.episode_id}`."
                )
            seen.add(episode.episode_id)
            yield episode
        missing = sorted(expected_episode_ids - seen)
        if missing:
            raise ValueError(
                f"Held-out manifest episodes are missing from shards: {missing[:8]}."
            )

    return episodes


def _usage_metrics(counts: torch.Tensor) -> dict[str, float | int]:
    counts = counts.detach().double().cpu()
    total = float(counts.sum().item())
    if total <= 0:
        raise ValueError("Cannot summarize empty code assignments.")
    active = counts > 0
    probabilities = counts[active] / total
    perplexity = float(
        torch.exp(-(probabilities * probabilities.log()).sum()).item()
    )
    return {
        "active_codes": int(active.sum().item()),
        "dead_fraction": float((~active).double().mean().item()),
        "perplexity": perplexity,
        "perplexity_fraction": perplexity / float(counts.numel()),
        "maximum_cluster_fraction": float(counts.max().item() / total),
    }


def _joint_usage_metrics(
    counts: Counter[int],
    *,
    capacity: int,
) -> dict[str, float | int]:
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("Cannot summarize empty joint code assignments.")
    probabilities = torch.tensor(
        list(counts.values()),
        dtype=torch.float64,
    ) / float(total)
    perplexity = float(
        torch.exp(-(probabilities * probabilities.log()).sum()).item()
    )
    return {
        "active_tuples": len(counts),
        "active_capacity_fraction": len(counts) / float(capacity),
        "perplexity": perplexity,
        "perplexity_fraction": perplexity / float(capacity),
        "maximum_tuple_fraction": max(counts.values()) / float(total),
    }


def _transition_metrics(counts: torch.Tensor) -> dict[str, float | int]:
    counts = counts.detach().double().cpu()
    total = float(counts.sum().item())
    if total <= 0:
        return {
            "adjacent_pairs": 0,
            "same_next_fraction": float("nan"),
            "change_next_fraction": float("nan"),
            "active_transitions": 0,
            "transition_perplexity": float("nan"),
            "maximum_transition_fraction": float("nan"),
        }
    active = counts > 0
    probabilities = counts[active] / total
    perplexity = float(
        torch.exp(-(probabilities * probabilities.log()).sum()).item()
    )
    same = float(counts.diagonal().sum().item() / total)
    return {
        "adjacent_pairs": int(total),
        "same_next_fraction": same,
        "change_next_fraction": 1.0 - same,
        "active_transitions": int(active.sum().item()),
        "transition_perplexity": perplexity,
        "maximum_transition_fraction": float(counts.max().item() / total),
    }


def _update_representatives(
    representatives: list[list[dict[str, Any]]],
    *,
    codes: torch.Tensor,
    distances: torch.Tensor,
    batch: Any,
    dimension: int,
    limit: int,
) -> None:
    if limit <= 0:
        return
    codes = codes.detach().cpu()
    distances = distances.detach().float().cpu()
    for code in torch.unique(codes).tolist():
        indices = torch.nonzero(codes == int(code), as_tuple=False).flatten()
        take = min(int(limit), int(indices.numel()))
        nearest = indices[
            torch.topk(
                distances[indices],
                k=take,
                largest=False,
                sorted=True,
            ).indices
        ]
        candidates = [
            {
                "episode_id": batch.episode_ids[int(index)],
                "time_index": int(batch.time_indices[int(index)].item()),
                "timestamp": float(batch.timestamps[int(index)].item()),
                "distance_mse": float(
                    distances[int(index)].item() / float(dimension)
                ),
            }
            for index in nearest.tolist()
        ]
        combined = [*representatives[int(code)], *candidates]
        representatives[int(code)] = sorted(
            combined,
            key=lambda value: (
                value["distance_mse"],
                value["episode_id"],
                value["time_index"],
            ),
        )[:limit]


def _evaluate_artifact(
    artifact: FrozenRQArtifact,
    *,
    episode_factory: Any,
    split: str,
    batch_size: int,
    center_block_size: int,
    device: torch.device,
    representatives_per_code: int,
) -> dict[str, Any]:
    source = CausalDescriptorSource(
        episode_factory=episode_factory,
        spec=artifact.descriptor,
        batch_size=batch_size,
        split=split,
    )
    centers = tuple(
        value.to(device=device, dtype=torch.float32)
        for value in artifact.centers
    )
    k = int(centers[0].shape[0])
    levels = len(centers)
    level_counts = [
        torch.zeros(k, dtype=torch.long) for _ in range(levels)
    ]
    joint_counts: Counter[int] = Counter()
    transition_counts = [
        torch.zeros((k, k), dtype=torch.long) for _ in range(levels)
    ]
    representatives: list[list[list[dict[str, Any]]]] = [
        [[] for _ in range(k)] for _ in range(levels)
    ]
    residual_sse = [0.0 for _ in range(levels + 1)]
    vector_count = 0
    dimension: int | None = None
    episode_ids: set[str] = set()
    previous_episode_id: str | None = None
    previous_time_index: int | None = None
    previous_codes: torch.Tensor | None = None

    for batch in source:
        values = artifact.normalization.normalize(batch.vectors)
        values = values.to(device=device, dtype=torch.float32)
        if dimension is None:
            dimension = int(values.shape[1])
        elif dimension != int(values.shape[1]):
            raise ValueError("Held-out descriptor dimension changed mid-stream.")
        residual = values
        residual_sse[0] += float(residual.square().sum().item())
        joint = torch.zeros(
            values.shape[0],
            dtype=torch.long,
            device=device,
        )
        batch_codes = []
        for level, level_centers in enumerate(centers):
            codes, distances = assign_nearest(
                residual,
                level_centers,
                center_block_size=center_block_size,
            )
            codes_cpu = codes.detach().cpu()
            level_counts[level] += torch.bincount(
                codes_cpu,
                minlength=k,
            )
            _update_representatives(
                representatives[level],
                codes=codes_cpu,
                distances=distances,
                batch=batch,
                dimension=int(values.shape[1]),
                limit=representatives_per_code,
            )
            residual = residual - level_centers[codes]
            residual_sse[level + 1] += float(residual.square().sum().item())
            joint = joint * k + codes
            batch_codes.append(codes_cpu)
        code_matrix = torch.stack(batch_codes, dim=1)

        if code_matrix.shape[0] > 1:
            adjacent = torch.tensor(
                [
                    batch.episode_ids[index] == batch.episode_ids[index - 1]
                    for index in range(1, code_matrix.shape[0])
                ],
                dtype=torch.bool,
            )
            adjacent &= batch.time_indices[1:] == batch.time_indices[:-1] + 1
            if adjacent.any():
                for level in range(levels):
                    pairs = (
                        code_matrix[:-1, level][adjacent] * k
                        + code_matrix[1:, level][adjacent]
                    )
                    transition_counts[level] += torch.bincount(
                        pairs,
                        minlength=k * k,
                    ).reshape(k, k)
        if (
            previous_episode_id == batch.episode_ids[0]
            and previous_time_index is not None
            and previous_time_index + 1 == int(batch.time_indices[0].item())
            and previous_codes is not None
        ):
            for level in range(levels):
                transition_counts[level][
                    int(previous_codes[level].item()),
                    int(code_matrix[0, level].item()),
                ] += 1
        previous_episode_id = batch.episode_ids[-1]
        previous_time_index = int(batch.time_indices[-1].item())
        previous_codes = code_matrix[-1].clone()

        joint_counts.update(int(value) for value in joint.detach().cpu().tolist())
        vector_count += int(values.shape[0])
        episode_ids.update(batch.episode_ids)

    if vector_count <= 0 or dimension is None:
        raise ValueError(
            f"No held-out descriptors were produced for {artifact.family}/{split}."
        )
    denominator = float(vector_count * dimension)
    residual_mse = [value / denominator for value in residual_sse]
    level_reductions = [
        1.0 - after / max(before, 1e-12)
        for before, after in zip(residual_mse, residual_mse[1:])
    ]
    return {
        "family": artifact.family,
        "split": split,
        "episodes": len(episode_ids),
        "vectors": vector_count,
        "dimension": dimension,
        "k": k,
        "levels": levels,
        "residual_mse": residual_mse,
        "residual_reduction_by_level": level_reductions,
        "residual_total_reduction": (
            1.0 - residual_mse[-1] / max(residual_mse[0], 1e-12)
        ),
        "code_usage": [
            {"level": level + 1, **_usage_metrics(counts)}
            for level, counts in enumerate(level_counts)
        ],
        "temporal": [
            {"level": level + 1, **_transition_metrics(counts)}
            for level, counts in enumerate(transition_counts)
        ],
        "representatives": [
            {
                "level": level + 1,
                "codes": [
                    {"code": code, "samples": samples}
                    for code, samples in enumerate(level_representatives)
                ],
            }
            for level, level_representatives in enumerate(representatives)
        ],
        "joint_usage": _joint_usage_metrics(
            joint_counts,
            capacity=int(math.pow(k, levels)),
        ),
    }


def _write_contract(path: Path, contract: dict[str, Any], resume: bool) -> None:
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != contract:
            raise RuntimeError(
                f"Existing held-out contract differs from `{path}`."
            )
        if not resume:
            raise FileExistsError(
                f"Held-out contract already exists at `{path}`."
            )
        return
    write_json_report(path, contract)


def evaluate_frozen_codebooks(
    config_path: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = OmegaConf.load(config_path)
    plain_config = _plain_config(config)
    input_config = config.get("input", {})
    evaluation = config.get("evaluation", {})
    metadata = config.get("metadata", {})
    artifact_config = config.get("artifacts", {})

    cpu_threads = int(evaluation.get("cpu_threads", 4))
    batch_size = int(evaluation.get("batch_size", 8192))
    center_block_size = int(evaluation.get("center_block_size", 1024))
    representatives_per_code = int(
        evaluation.get("representatives_per_code", 3)
    )
    if cpu_threads <= 0 or batch_size <= 0 or center_block_size <= 0:
        raise ValueError("Held-out thread and batch sizes must be positive.")
    if representatives_per_code < 0:
        raise ValueError("Held-out representatives_per_code must be non-negative.")
    torch.set_num_threads(cpu_threads)
    device = _resolve_device(str(evaluation.get("device", "auto")))
    resume = bool(evaluation.get("resume", True))

    dataset = str(metadata.get("dataset", ""))
    if not dataset:
        raise ValueError("Held-out `metadata.dataset` must not be empty.")
    manifest_path = Path(str(input_config.get("manifest", "")))
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing held-out manifest `{manifest_path}`.")
    manifest = EpisodeManifest.read_jsonl(manifest_path)
    manifest.assert_group_isolation(str(input_config.get("group_by", "scene")))
    manifest_fingerprint = manifest.fingerprint()

    patterns = input_config.get("pooled_shards", ())
    if not patterns:
        raise ValueError("Held-out `input.pooled_shards` must not be empty.")
    shard_paths = tuple(expand_shard_paths(patterns))
    shard_checksums = [file_sha256(path) for path in shard_paths]
    splits = tuple(str(value) for value in evaluation.get("splits", ("val", "test")))
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("Held-out splits must be nonempty and unique.")
    if any(split not in {"val", "test"} for split in splits):
        raise ValueError("Held-out evaluation accepts only `val` and `test`.")

    artifact_paths = {
        str(family): Path(str(path))
        for family, path in artifact_config.items()
    }
    if not artifact_paths:
        raise ValueError("Held-out `artifacts` must not be empty.")
    artifacts = {
        family: FrozenRQArtifact.load(path)
        for family, path in sorted(artifact_paths.items())
    }
    for family, artifact in artifacts.items():
        if artifact.family != family:
            raise ValueError(
                f"Artifact mapping `{family}` contains `{artifact.family}`."
            )
        expected_metadata = {
            "dataset": dataset,
            "manifest_fingerprint": manifest_fingerprint,
            "source_checksums": shard_checksums,
        }
        mismatches = [
            key
            for key, value in expected_metadata.items()
            if artifact.metadata.get(key) != value
        ]
        if mismatches:
            raise RuntimeError(
                f"Artifact `{family}` differs from held-out input in {mismatches}."
            )

    expected_by_split = {
        split: {
            record.episode_id
            for record in manifest
            if record.dataset == dataset and record.split == split
        }
        for split in splits
    }
    empty_splits = [
        split for split, identifiers in expected_by_split.items() if not identifiers
    ]
    if empty_splits:
        raise ValueError(
            f"Held-out manifest has no `{dataset}` episodes in {empty_splits}."
        )

    implementation_sha256 = {
        "evaluation": file_sha256(Path(__file__)),
        "streaming": file_sha256(Path(__file__).with_name("streaming.py")),
    }
    contract_payload = {
        "schema": HELDOUT_CONTRACT_SCHEMA,
        "config": plain_config,
        "config_sha256": hashlib.sha256(
            json.dumps(
                plain_config,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": file_sha256(manifest_path),
            "fingerprint": manifest_fingerprint,
        },
        "pooled_shards": [
            {"path": str(path), "sha256": checksum}
            for path, checksum in zip(shard_paths, shard_checksums)
        ],
        "artifacts": {
            family: {
                "path": str(artifact_paths[family].resolve()),
                "sha256": file_sha256(artifact_paths[family]),
            }
            for family in sorted(artifact_paths)
        },
        "implementation_sha256": implementation_sha256,
    }
    contract_hash = hashlib.sha256(
        json.dumps(
            contract_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    contract = {**contract_payload, "contract_hash": contract_hash}
    output_dir = Path(str(config.get("output_dir", "runs/codebook_eval/heldout")))
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    report_path = output_dir / "evaluation_report.json"
    _write_contract(contract_path, contract, resume=resume)
    if resume and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("contract_hash") != contract_hash:
            raise RuntimeError("Held-out report contract hash is invalid.")
        return report
    if report_path.exists():
        raise FileExistsError(f"Held-out report already exists at `{report_path}`.")

    rows = []
    for family, artifact in sorted(artifacts.items()):
        for split in splits:
            rows.append(
                _evaluate_artifact(
                    artifact,
                    episode_factory=_episode_factory(
                        shard_paths,
                        split,
                        expected_by_split[split],
                    ),
                    split=split,
                    batch_size=batch_size,
                    center_block_size=center_block_size,
                    device=device,
                    representatives_per_code=representatives_per_code,
                )
            )
    report = {
        "schema": HELDOUT_EVALUATION_SCHEMA,
        "contract_hash": contract_hash,
        "dataset": dataset,
        "manifest_fingerprint": manifest_fingerprint,
        "splits": list(splits),
        "rows": rows,
    }
    write_json_report(report_path, report)
    return report
