from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch

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
    encode_residual_quantizer,
)


TEMPORAL_SENSITIVITY_CONTRACT_SCHEMA = (
    "codewam.temporal-sensitivity-contract.v1"
)
TEMPORAL_SENSITIVITY_REPORT_SCHEMA = (
    "codewam.temporal-sensitivity-report.v1"
)
PERTURBATION_NAMES = (
    "history_swap",
    "reverse_time",
    "static_current",
)


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_device(value: str | torch.device) -> torch.device:
    if isinstance(value, torch.device):
        return value
    if str(value).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device `{device}` is unavailable.")
    return device


def _episode_factory(
    shard_paths: tuple[Path, ...],
    split: str,
    expected_episode_ids: set[str],
):
    def episodes() -> Iterator[PooledFeatureEpisode]:
        seen: set[str] = set()
        for episode in iter_pooled_feature_episodes(
            shard_paths,
            split=split,
        ):
            if episode.episode_id not in expected_episode_ids:
                raise ValueError(
                    f"Temporal probe shard episode `{episode.episode_id}` is "
                    "absent from the manifest."
                )
            if episode.episode_id in seen:
                raise ValueError(
                    f"Duplicate temporal probe episode "
                    f"`{episode.episode_id}`."
                )
            seen.add(episode.episode_id)
            yield episode
        missing = sorted(expected_episode_ids - seen)
        if missing:
            raise ValueError(
                f"Temporal probe manifest episodes are missing from shards: "
                f"{missing[:8]}."
            )

    return episodes


def _descriptor_perturbations(
    vectors: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if vectors.ndim != 2 or vectors.shape[1] % 3 != 0:
        raise ValueError(
            "Temporal sensitivity expects [N,3D] causal descriptors."
        )
    frames = vectors.reshape(vectors.shape[0], 3, vectors.shape[1] // 3)
    current = frames[:, 2:3]
    return {
        "history_swap": frames[:, [1, 0, 2]].reshape_as(vectors),
        "reverse_time": frames.flip(dims=(1,)).reshape_as(vectors),
        "static_current": current.expand(-1, 3, -1).reshape_as(vectors),
    }


def _prefix_reconstructions(
    codes: torch.Tensor,
    centers: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    quantized = torch.zeros(
        (codes.shape[0], centers[0].shape[1]),
        device=codes.device,
        dtype=torch.float32,
    )
    prefixes = []
    for level, level_centers in enumerate(centers):
        quantized = quantized + level_centers[codes[:, level]]
        prefixes.append(quantized.clone())
    return prefixes


def _probe_artifact(
    artifact: FrozenRQArtifact,
    *,
    episode_factory: Any,
    split: str,
    batch_size: int,
    center_block_size: int,
    device: torch.device,
) -> dict[str, Any]:
    source = CausalDescriptorSource(
        episode_factory=episode_factory,
        spec=artifact.descriptor,
        batch_size=batch_size,
        split=split,
    )
    centers = tuple(
        center.to(device=device, dtype=torch.float32)
        for center in artifact.centers
    )
    k = int(centers[0].shape[0])
    levels = len(centers)
    vector_count = 0
    dimension: int | None = None
    baseline_sse = [0.0 for _ in range(levels)]
    true_code_counts = [
        torch.zeros(k, dtype=torch.long) for _ in range(levels)
    ]
    accumulators = {
        name: {
            "displacement_sse": 0.0,
            "level_changed": [0 for _ in range(levels)],
            "prefix_changed": [0 for _ in range(levels)],
            "changed_levels": 0,
            "cross_reconstruction_sse": [
                0.0 for _ in range(levels)
            ],
            "transition_counts": [
                torch.zeros((k, k), dtype=torch.long)
                for _ in range(levels)
            ],
            "l1_changed_by_code": torch.zeros(k, dtype=torch.long),
        }
        for name in PERTURBATION_NAMES
    }

    with torch.inference_mode():
        for batch in source:
            raw = batch.vectors.to(device=device, dtype=torch.float32)
            true_values = artifact.normalization.normalize(raw)
            if dimension is None:
                dimension = int(true_values.shape[1])
            elif dimension != int(true_values.shape[1]):
                raise ValueError(
                    "Temporal probe descriptor dimension changed mid-stream."
                )
            true_codes, _, _ = encode_residual_quantizer(
                true_values,
                centers,
                center_block_size=center_block_size,
            )
            true_prefixes = _prefix_reconstructions(true_codes, centers)
            for level in range(levels):
                baseline_sse[level] += float(
                    (
                        true_values - true_prefixes[level]
                    ).square().sum().item()
                )
                true_code_counts[level] += torch.bincount(
                    true_codes[:, level].detach().cpu(),
                    minlength=k,
                )

            perturbations = _descriptor_perturbations(raw)
            for name, perturbed_raw in perturbations.items():
                perturbed_values = artifact.normalization.normalize(
                    perturbed_raw
                )
                perturbed_codes, _, _ = encode_residual_quantizer(
                    perturbed_values,
                    centers,
                    center_block_size=center_block_size,
                )
                comparison = perturbed_codes != true_codes
                accumulator = accumulators[name]
                accumulator["displacement_sse"] += float(
                    (
                        perturbed_values - true_values
                    ).square().sum().item()
                )
                accumulator["changed_levels"] += int(
                    comparison.sum().item()
                )
                perturbed_prefixes = _prefix_reconstructions(
                    perturbed_codes,
                    centers,
                )
                for level in range(levels):
                    level_changed = comparison[:, level]
                    prefix_changed = comparison[:, : level + 1].any(dim=1)
                    accumulator["level_changed"][level] += int(
                        level_changed.sum().item()
                    )
                    accumulator["prefix_changed"][level] += int(
                        prefix_changed.sum().item()
                    )
                    accumulator["cross_reconstruction_sse"][level] += float(
                        (
                            true_values - perturbed_prefixes[level]
                        ).square().sum().item()
                    )
                    pairs = (
                        true_codes[:, level] * k
                        + perturbed_codes[:, level]
                    )
                    accumulator["transition_counts"][level] += (
                        torch.bincount(
                            pairs.detach().cpu(),
                            minlength=k * k,
                        ).reshape(k, k)
                    )
                accumulator["l1_changed_by_code"] += torch.bincount(
                    true_codes[:, 0][comparison[:, 0]].detach().cpu(),
                    minlength=k,
                )
            vector_count += int(true_values.shape[0])

    if vector_count <= 0 or dimension is None:
        raise ValueError(
            f"No descriptors were produced for {artifact.family}/{split}."
        )
    denominator = float(vector_count * dimension)
    baseline_mse = [value / denominator for value in baseline_sse]
    perturbation_rows = []
    for name in PERTURBATION_NAMES:
        accumulator = accumulators[name]
        cross_mse = [
            value / denominator
            for value in accumulator["cross_reconstruction_sse"]
        ]
        by_l1_code = []
        for code in range(k):
            total = int(true_code_counts[0][code].item())
            changed = int(
                accumulator["l1_changed_by_code"][code].item()
            )
            by_l1_code.append(
                {
                    "code": code,
                    "vectors": total,
                    "change_fraction": (
                        None if total == 0 else changed / float(total)
                    ),
                }
            )
        perturbation_rows.append(
            {
                "name": name,
                "normalized_descriptor_displacement_mse": (
                    accumulator["displacement_sse"] / denominator
                ),
                "mean_changed_level_fraction": (
                    accumulator["changed_levels"]
                    / float(vector_count * levels)
                ),
                "level_code_change": [
                    {
                        "level": level + 1,
                        "change_fraction": (
                            accumulator["level_changed"][level]
                            / float(vector_count)
                        ),
                    }
                    for level in range(levels)
                ],
                "prefix_code_change": [
                    {
                        "depth": level + 1,
                        "change_fraction": (
                            accumulator["prefix_changed"][level]
                            / float(vector_count)
                        ),
                    }
                    for level in range(levels)
                ],
                "true_prefix_reconstruction_mse": baseline_mse,
                "perturbed_code_reconstruction_mse_to_true": cross_mse,
                "cross_reconstruction_penalty": [
                    cross - baseline
                    for cross, baseline in zip(cross_mse, baseline_mse)
                ],
                "l1_change_by_true_code": by_l1_code,
                "level_transition_counts": [
                    {
                        "level": level + 1,
                        "counts": counts.tolist(),
                    }
                    for level, counts in enumerate(
                        accumulator["transition_counts"]
                    )
                ],
            }
        )
    return {
        "family": artifact.family,
        "stride": artifact.descriptor.stride,
        "pool": artifact.descriptor.pool,
        "camera_ids": (
            None
            if artifact.descriptor.camera_ids is None
            else list(artifact.descriptor.camera_ids)
        ),
        "split": split,
        "vectors": vector_count,
        "dimension": dimension,
        "k": k,
        "levels": levels,
        "true_code_usage": [
            {
                "level": level + 1,
                "counts": counts.tolist(),
            }
            for level, counts in enumerate(true_code_counts)
        ],
        "perturbations": perturbation_rows,
    }


def probe_codebook_temporal_sensitivity(
    *,
    manifest_path: str | Path,
    pooled_shards: Iterable[str | Path],
    artifacts: Mapping[str, str | Path],
    output_dir: str | Path,
    dataset: str = "droid-1.0.1",
    splits: Iterable[str] = ("test",),
    device: str | torch.device = "auto",
    cpu_threads: int = 4,
    batch_size: int = 8192,
    center_block_size: int = 1024,
    resume: bool = True,
) -> dict[str, Any]:
    """Measure frozen RQ sensitivity to temporal counterfactuals."""

    if cpu_threads <= 0 or batch_size <= 0 or center_block_size <= 0:
        raise ValueError(
            "Temporal probe thread and batch settings must be positive."
        )
    if not dataset:
        raise ValueError("Temporal probe dataset must not be empty.")
    splits = tuple(str(value) for value in splits)
    if (
        not splits
        or len(set(splits)) != len(splits)
        or any(value not in {"val", "test"} for value in splits)
    ):
        raise ValueError("Temporal probe splits must be unique val/test values.")
    if not artifacts:
        raise ValueError("Temporal probe requires at least one artifact.")
    torch.set_num_threads(cpu_threads)
    target_device = _resolve_device(device)

    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest `{manifest_path}`.")
    manifest = EpisodeManifest.read_jsonl(manifest_path)
    manifest.assert_group_isolation("scene")
    manifest_fingerprint = manifest.fingerprint()
    shard_paths = tuple(expand_shard_paths(pooled_shards))
    shard_checksums = [file_sha256(path) for path in shard_paths]
    artifact_paths = {
        str(family): Path(path)
        for family, path in artifacts.items()
    }
    frozen = {
        family: FrozenRQArtifact.load(path)
        for family, path in sorted(artifact_paths.items())
    }
    for family, artifact in frozen.items():
        if family != artifact.family:
            raise ValueError(
                f"Artifact mapping `{family}` contains `{artifact.family}`."
            )
        expected = {
            "dataset": dataset,
            "manifest_fingerprint": manifest_fingerprint,
            "source_checksums": shard_checksums,
        }
        mismatches = [
            key
            for key, value in expected.items()
            if artifact.metadata.get(key) != value
        ]
        if mismatches:
            raise RuntimeError(
                f"Artifact `{family}` differs from temporal probe input in "
                f"{mismatches}."
            )

    expected_by_split = {
        split: {
            record.episode_id
            for record in manifest
            if record.dataset == dataset and record.split == split
        }
        for split in splits
    }
    empty = [
        split for split, episode_ids in expected_by_split.items()
        if not episode_ids
    ]
    if empty:
        raise ValueError(
            f"Temporal probe manifest has no `{dataset}` episodes in {empty}."
        )

    contract = {
        "schema": TEMPORAL_SENSITIVITY_CONTRACT_SCHEMA,
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
        "config": {
            "dataset": dataset,
            "splits": list(splits),
            "device": str(target_device),
            "cpu_threads": cpu_threads,
            "batch_size": batch_size,
            "center_block_size": center_block_size,
            "perturbations": list(PERTURBATION_NAMES),
        },
        "implementation_sha256": {
            "temporal_sensitivity": file_sha256(Path(__file__)),
            "streaming": file_sha256(
                Path(__file__).with_name("streaming.py")
            ),
            "shards": file_sha256(Path(__file__).with_name("shards.py")),
        },
    }
    contract["contract_hash"] = _canonical_hash(contract)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    report_path = output_dir / "temporal_sensitivity_report.json"
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != contract:
            raise RuntimeError(
                f"Existing temporal probe contract differs from "
                f"`{contract_path}`."
            )
        if not resume:
            raise FileExistsError(
                f"Temporal probe contract already exists at `{contract_path}`."
            )
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("contract_hash") != contract["contract_hash"]:
                raise RuntimeError(
                    "Temporal sensitivity report contract hash is invalid."
                )
            return report
    else:
        write_json_report(contract_path, contract)

    rows = []
    for family, artifact in sorted(frozen.items()):
        for split in splits:
            rows.append(
                _probe_artifact(
                    artifact,
                    episode_factory=_episode_factory(
                        shard_paths,
                        split,
                        expected_by_split[split],
                    ),
                    split=split,
                    batch_size=batch_size,
                    center_block_size=center_block_size,
                    device=target_device,
                )
            )
    summary_rows = []
    for row in rows:
        for perturbation in row["perturbations"]:
            summary_rows.append(
                {
                    "family": row["family"],
                    "split": row["split"],
                    "perturbation": perturbation["name"],
                    "descriptor_displacement_mse": perturbation[
                        "normalized_descriptor_displacement_mse"
                    ],
                    "l1_code_change_fraction": perturbation[
                        "level_code_change"
                    ][0]["change_fraction"],
                    "full_prefix_change_fraction": perturbation[
                        "prefix_code_change"
                    ][-1]["change_fraction"],
                    "mean_changed_level_fraction": perturbation[
                        "mean_changed_level_fraction"
                    ],
                    "full_cross_reconstruction_penalty": perturbation[
                        "cross_reconstruction_penalty"
                    ][-1],
                }
            )
    report = {
        "schema": TEMPORAL_SENSITIVITY_REPORT_SCHEMA,
        "contract_hash": contract["contract_hash"],
        "dataset": dataset,
        "manifest_fingerprint": manifest_fingerprint,
        "splits": list(splits),
        "interpretation": {
            "history_swap": (
                "Swaps only the two historical states while keeping z[t] "
                "fixed."
            ),
            "reverse_time": (
                "Reverses the same three states to test temporal direction."
            ),
            "static_current": (
                "Repeats z[t] three times to remove observed motion while "
                "preserving the current visual endpoint."
            ),
            "scope": (
                "This is a frozen held-out sensitivity diagnostic, not a "
                "causal intervention on the physical environment."
            ),
        },
        "rows": rows,
        "summary_rows": summary_rows,
    }
    write_json_report(report_path, report)
    return report
