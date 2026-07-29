from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from codewam.data.droid_manifest import write_json_report

from .association import _episode_factory, _resolve_device
from .family_association import (
    _iter_aligned_probe_batches,
    _validate_family_artifacts,
)
from .manifest import EpisodeManifest
from .shards import expand_shard_paths, file_sha256
from .streaming import FrozenRQArtifact, encode_residual_quantizer


SEED_STABILITY_CONTRACT_SCHEMA = "codewam.rq-seed-stability-contract.v1"
SEED_STABILITY_REPORT_SCHEMA = "codewam.rq-seed-stability-report.v1"


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _best_overlap_mapping(contingency: torch.Tensor) -> tuple[int, ...]:
    if (
        contingency.ndim != 2
        or contingency.shape[0] != contingency.shape[1]
        or contingency.shape[0] <= 0
    ):
        raise ValueError("Code contingency must be a nonempty square matrix.")
    capacity = int(contingency.shape[0])
    scores = contingency.long().cpu()
    states: dict[int, tuple[int, tuple[int, ...]]] = {0: (0, ())}
    for candidate_code in range(capacity):
        next_states: dict[int, tuple[int, tuple[int, ...]]] = {}
        for mask, (score, mapping) in states.items():
            for reference_code in range(capacity):
                bit = 1 << reference_code
                if mask & bit:
                    continue
                next_mask = mask | bit
                candidate = (
                    score
                    + int(scores[reference_code, candidate_code].item()),
                    (*mapping, reference_code),
                )
                previous = next_states.get(next_mask)
                if previous is None or candidate > previous:
                    next_states[next_mask] = candidate
        states = next_states
    return states[(1 << capacity) - 1][1]


def _nmi_ari(contingency: torch.Tensor) -> tuple[float, float]:
    values = contingency.double()
    total = float(values.sum().item())
    if total <= 0:
        raise ValueError("Cannot compare empty code assignments.")
    left = values.sum(dim=1)
    right = values.sum(dim=0)
    left_probabilities = left[left > 0] / total
    right_probabilities = right[right > 0] / total
    left_entropy = float(
        -(left_probabilities * left_probabilities.log()).sum().item()
    )
    right_entropy = float(
        -(right_probabilities * right_probabilities.log()).sum().item()
    )
    mutual_information = 0.0
    for row, column in values.nonzero().tolist():
        count = float(values[row, column].item())
        mutual_information += (count / total) * math.log(
            count * total / float(left[row] * right[column])
        )
    nmi = (
        mutual_information / math.sqrt(left_entropy * right_entropy)
        if left_entropy > 1e-12 and right_entropy > 1e-12
        else 0.0
    )

    def choose_two(value: torch.Tensor) -> float:
        return float((value * (value - 1.0) / 2.0).sum().item())

    sum_joint = choose_two(values)
    sum_left = choose_two(left)
    sum_right = choose_two(right)
    total_pairs = total * (total - 1.0) / 2.0
    expected = (
        sum_left * sum_right / total_pairs if total_pairs > 0 else 0.0
    )
    maximum = 0.5 * (sum_left + sum_right)
    ari = (
        (sum_joint - expected) / (maximum - expected)
        if maximum > expected + 1e-12
        else 1.0
    )
    return nmi, ari


def _mapped_codes(
    codes: torch.Tensor,
    mappings: list[tuple[int, ...]],
) -> torch.Tensor:
    result = codes.clone().long()
    for level, mapping in enumerate(mappings):
        lookup = torch.tensor(mapping, dtype=torch.long)
        result[:, level] = lookup[result[:, level]]
    return result


def _prefix_ids(codes: torch.Tensor, k: int) -> list[torch.Tensor]:
    codes = codes.detach().long().cpu()
    if codes.ndim != 2 or codes.shape[1] <= 0:
        raise ValueError("Seed-stability codes must be [N,L].")
    if k <= 0 or (
        codes.numel()
        and (int(codes.min()) < 0 or int(codes.max()) >= int(k))
    ):
        raise ValueError("Seed-stability code is outside the RQ capacity.")
    identifiers = []
    current = torch.zeros(codes.shape[0], dtype=torch.long)
    for level in range(codes.shape[1]):
        current = current * int(k) + codes[:, level]
        identifiers.append(current.clone())
    return identifiers


def _normalization_equal(
    left: FrozenRQArtifact,
    right: FrozenRQArtifact,
) -> bool:
    return (
        left.normalization.count == right.normalization.count
        and torch.equal(left.normalization.mean, right.normalization.mean)
        and torch.equal(left.normalization.std, right.normalization.std)
    )


def _training_contract_identity(
    artifact_path: Path,
    artifact: FrozenRQArtifact,
) -> dict[str, Any]:
    contract_path = artifact_path.parent / "contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"Missing training contract for `{artifact_path}`."
        )
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "codewam.family-run-contract.v1":
        raise ValueError(
            f"Unsupported training contract schema in `{contract_path}`."
        )
    expected = {
        "family": artifact.family,
        "stride": artifact.descriptor.stride,
        "pool": artifact.descriptor.pool,
        "max_gap_factor": artifact.descriptor.max_gap_factor,
        "camera_ids": (
            None
            if artifact.descriptor.camera_ids is None
            else list(artifact.descriptor.camera_ids)
        ),
        "k": int(artifact.centers[0].shape[0]),
        "levels": len(artifact.centers),
        "manifest_fingerprint": artifact.metadata["manifest_fingerprint"],
        "source_checksums": artifact.metadata["source_checksums"],
        "implementation_sha256": artifact.metadata[
            "implementation_sha256"
        ],
    }
    mismatches = [
        key for key, value in expected.items() if payload.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            f"Training contract `{contract_path}` differs in {mismatches}."
        )
    try:
        seed = int(payload["seed"])
        tolerance = float(payload["tol"])
        patience = int(payload["patience"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Training contract `{contract_path}` lacks optimizer identity."
        ) from exc
    if tolerance <= 0 or patience <= 0:
        raise ValueError(
            f"Training contract `{contract_path}` has invalid convergence."
        )
    return {
        "path": str(contract_path.resolve()),
        "sha256": file_sha256(contract_path),
        "seed": seed,
        "tol": tolerance,
        "patience": patience,
        "initialization_policy": payload.get("initialization_policy"),
    }


def _validate_runs(
    runs: dict[str, dict[str, FrozenRQArtifact]],
) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    if len(runs) < 3:
        raise ValueError("Seed stability requires at least three runs.")
    run_labels = tuple(sorted(runs))
    family_sets = {tuple(sorted(artifacts)) for artifacts in runs.values()}
    if len(family_sets) != 1:
        raise ValueError("Seed-stability runs must contain the same families.")
    family_labels = next(iter(family_sets))
    reference = runs[run_labels[0]]
    ordered_families, levels = _validate_family_artifacts(reference)
    if set(ordered_families) != set(family_labels):
        raise ValueError("Seed-stability family labels are inconsistent.")
    for run_label in run_labels[1:]:
        ordered, observed_levels = _validate_family_artifacts(runs[run_label])
        if ordered != ordered_families or observed_levels != levels:
            raise ValueError(
                f"Seed-stability run `{run_label}` changes the RQ contract."
            )
        for family in ordered_families:
            left = reference[family]
            right = runs[run_label][family]
            if left.descriptor != right.descriptor:
                raise ValueError(
                    f"Seed run `{run_label}` changes {family} descriptor."
                )
            if not _normalization_equal(left, right):
                raise ValueError(
                    f"Seed run `{run_label}` changes {family} normalization."
                )
            if len(left.centers) != len(right.centers):
                raise ValueError(
                    f"Seed run `{run_label}` changes {family} depth."
                )
    return run_labels, ordered_families, levels


def probe_rq_seed_stability(
    *,
    manifest_path: str | Path,
    pooled_shards: Iterable[str | Path],
    runs: dict[str, dict[str, str | Path]],
    output_dir: str | Path,
    reference_run: str,
    splits: tuple[str, ...] = ("val", "test"),
    device: str = "auto",
    cpu_threads: int = 4,
    batch_size: int = 8192,
    center_block_size: int = 1024,
    resume: bool = True,
) -> dict[str, Any]:
    if reference_run not in runs:
        raise ValueError(
            f"Unknown seed-stability reference run `{reference_run}`."
        )
    if (
        cpu_threads <= 0
        or batch_size <= 0
        or center_block_size <= 0
    ):
        raise ValueError(
            "Seed-stability thread, batch and block sizes must be positive."
        )
    if not splits or any(split not in {"val", "test"} for split in splits):
        raise ValueError("Seed-stability splits must be val/test.")
    torch.set_num_threads(int(cpu_threads))

    manifest_path = Path(manifest_path)
    manifest = EpisodeManifest.read_jsonl(manifest_path)
    manifest.assert_group_isolation("scene")
    manifest_fingerprint = manifest.fingerprint()
    shard_paths = tuple(expand_shard_paths(pooled_shards))
    shard_checksums = [file_sha256(path) for path in shard_paths]
    run_paths = {
        run_label: {
            family: Path(path)
            for family, path in sorted(artifacts.items())
        }
        for run_label, artifacts in sorted(runs.items())
    }
    loaded_runs = {
        run_label: {
            family: FrozenRQArtifact.load(path)
            for family, path in artifacts.items()
        }
        for run_label, artifacts in run_paths.items()
    }
    training_contracts = {
        run_label: {
            family: _training_contract_identity(
                run_paths[run_label][family],
                artifact,
            )
            for family, artifact in artifacts.items()
        }
        for run_label, artifacts in loaded_runs.items()
    }
    run_labels, family_labels, levels = _validate_runs(loaded_runs)
    run_seeds = {}
    for run_label in run_labels:
        seeds = {
            int(contract["seed"])
            for contract in training_contracts[run_label].values()
        }
        if len(seeds) != 1:
            raise ValueError(
                f"Seed run `{run_label}` mixes training seeds {sorted(seeds)}."
            )
        run_seeds[run_label] = next(iter(seeds))
    if len(set(run_seeds.values())) != len(run_seeds):
        raise ValueError("Seed-stability runs must use distinct training seeds.")
    for run_label, artifacts in loaded_runs.items():
        for family, artifact in artifacts.items():
            expected = {
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
                    f"Seed artifact `{run_label}/{family}` differs in "
                    f"{mismatches}."
                )

    expected_by_split = {
        split: {
            record.episode_id
            for record in manifest
            if record.split == split
        }
        for split in splits
    }
    empty = [
        split
        for split, identifiers in expected_by_split.items()
        if not identifiers
    ]
    if empty:
        raise ValueError(f"Seed-stability manifest has empty splits {empty}.")

    contract_payload = {
        "schema": SEED_STABILITY_CONTRACT_SCHEMA,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": file_sha256(manifest_path),
            "fingerprint": manifest_fingerprint,
        },
        "pooled_shards": [
            {"path": str(path), "sha256": checksum}
            for path, checksum in zip(shard_paths, shard_checksums)
        ],
        "runs": {
            run_label: {
                family: {
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                    "training_contract": training_contracts[run_label][
                        family
                    ],
                }
                for family, path in artifacts.items()
            }
            for run_label, artifacts in run_paths.items()
        },
        "reference_run": reference_run,
        "splits": list(splits),
        "device": device,
        "cpu_threads": cpu_threads,
        "batch_size": batch_size,
        "center_block_size": center_block_size,
        "implementation_sha256": {
            "seed_stability": file_sha256(Path(__file__)),
            "family_association": file_sha256(
                Path(__file__).with_name("family_association.py")
            ),
            "streaming": file_sha256(
                Path(__file__).with_name("streaming.py")
            ),
        },
    }
    contract_hash = _canonical_hash(contract_payload)
    contract = {**contract_payload, "contract_hash": contract_hash}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    report_path = output_dir / "seed_stability_report.json"
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != contract:
            raise RuntimeError("Existing seed-stability contract differs.")
        if not resume:
            raise FileExistsError(
                f"Seed-stability contract exists at `{contract_path}`."
            )
    else:
        write_json_report(contract_path, contract)
    if resume and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("contract_hash") != contract_hash:
            raise RuntimeError("Seed-stability report contract hash is invalid.")
        return report
    if report_path.exists():
        raise FileExistsError(
            f"Seed-stability report exists at `{report_path}`."
        )

    target_device = _resolve_device(device)
    centers = {
        run_label: {
            family: tuple(
                center.to(device=target_device, dtype=torch.float32)
                for center in artifact.centers
            )
            for family, artifact in artifacts.items()
        }
        for run_label, artifacts in loaded_runs.items()
    }
    pairs = tuple(itertools.combinations(run_labels, 2))
    accumulators: dict[
        tuple[str, str, str, str],
        dict[str, Any],
    ] = {}

    with torch.inference_mode():
        for split in splits:
            episode_factory = _episode_factory(
                shard_paths,
                split,
                expected_by_split[split],
            )
            for vectors, _ in _iter_aligned_probe_batches(
                episode_factory,
                loaded_runs[reference_run],
                future_offset=1,
                batch_size=batch_size,
            ):
                codes_by_run: dict[
                    str, dict[str, torch.Tensor]
                ] = {run_label: {} for run_label in run_labels}
                normalized_by_family: dict[str, torch.Tensor] = {}
                for family in family_labels:
                    raw = vectors[family]
                    normalized = loaded_runs[
                        reference_run
                    ][family].normalization.normalize(raw).to(
                        device=target_device,
                        dtype=torch.float32,
                    )
                    normalized_by_family[family] = normalized
                    for run_label in run_labels:
                        codes, _, residual = encode_residual_quantizer(
                            normalized,
                            centers[run_label][family],
                            center_block_size=center_block_size,
                        )
                        codes_by_run[run_label][family] = codes.cpu()
                        key = (split, family, run_label, run_label)
                        accumulator = accumulators.setdefault(
                            key,
                            {
                                "vectors": 0,
                                "elements": 0,
                                "residual_sse": 0.0,
                            },
                        )
                        accumulator["vectors"] += int(codes.shape[0])
                        accumulator["elements"] += int(
                            residual.numel()
                        )
                        accumulator["residual_sse"] += float(
                            residual.square().sum().item()
                        )

                for left_run, right_run in pairs:
                    for family in family_labels:
                        left_codes = codes_by_run[left_run][family]
                        right_codes = codes_by_run[right_run][family]
                        k = int(
                            centers[left_run][family][0].shape[0]
                        )
                        key = (split, family, left_run, right_run)
                        accumulator = accumulators.setdefault(
                            key,
                            {
                                "vectors": 0,
                                "contingencies": [
                                    torch.zeros(
                                        (k, k),
                                        dtype=torch.long,
                                    )
                                    for _ in range(levels)
                                ],
                                "left_codes": [],
                                "right_codes": [],
                            },
                        )
                        accumulator["vectors"] += int(left_codes.shape[0])
                        accumulator["left_codes"].append(left_codes)
                        accumulator["right_codes"].append(right_codes)
                        for level in range(levels):
                            flat = (
                                left_codes[:, level] * k
                                + right_codes[:, level]
                            )
                            accumulator["contingencies"][level] += (
                                torch.bincount(
                                    flat,
                                    minlength=k * k,
                                ).reshape(k, k)
                            )

    distortion_rows = []
    for split in splits:
        for family in family_labels:
            values = []
            for run_label in run_labels:
                accumulator = accumulators[
                    (split, family, run_label, run_label)
                ]
                mse = accumulator["residual_sse"] / float(
                    accumulator["elements"]
                )
                values.append(mse)
                distortion_rows.append(
                    {
                        "split": split,
                        "family": family,
                        "run": run_label,
                        "vectors": accumulator["vectors"],
                        "full_rq_normalized_residual_mse": mse,
                    }
                )
            mean = sum(values) / len(values)
            variance = sum(
                (value - mean) ** 2 for value in values
            ) / len(values)
            distortion_rows.append(
                {
                    "split": split,
                    "family": family,
                    "run": "__across_runs__",
                    "vectors": accumulators[
                        (split, family, run_labels[0], run_labels[0])
                    ]["vectors"],
                    "mean_full_rq_normalized_residual_mse": mean,
                    "coefficient_of_variation": (
                        math.sqrt(variance) / mean if mean > 0 else 0.0
                    ),
                    "maximum_relative_range": (
                        (max(values) - min(values)) / mean
                        if mean > 0
                        else 0.0
                    ),
                }
            )

    pair_rows = []
    for split in splits:
        for family in family_labels:
            for left_run, right_run in pairs:
                k = int(
                    loaded_runs[left_run][family].centers[0].shape[0]
                )
                accumulator = accumulators[
                    (split, family, left_run, right_run)
                ]
                mappings = [
                    _best_overlap_mapping(contingency)
                    for contingency in accumulator["contingencies"]
                ]
                left_codes = torch.cat(
                    accumulator["left_codes"],
                    dim=0,
                )
                right_codes = torch.cat(
                    accumulator["right_codes"],
                    dim=0,
                )
                mapped_right = _mapped_codes(right_codes, mappings)
                comparison = left_codes == mapped_right
                left_prefixes = _prefix_ids(left_codes, k)
                right_prefixes = _prefix_ids(right_codes, k)
                level_rows = []
                for level, contingency in enumerate(
                    accumulator["contingencies"]
                ):
                    nmi, ari = _nmi_ari(contingency)
                    level_rows.append(
                        {
                            "level": level + 1,
                            "mapping_right_to_left": list(
                                mappings[level]
                            ),
                            "mapped_agreement": float(
                                comparison[:, level]
                                .float()
                                .mean()
                                .item()
                            ),
                            "normalized_mutual_information": nmi,
                            "adjusted_rand_index": ari,
                            "contingency": contingency.tolist(),
                        }
                    )
                prefix_partition_rows = []
                for depth, (left_prefix, right_prefix) in enumerate(
                    zip(left_prefixes, right_prefixes),
                    start=1,
                ):
                    capacity = k**depth
                    contingency = torch.bincount(
                        left_prefix * capacity + right_prefix,
                        minlength=capacity * capacity,
                    ).reshape(capacity, capacity)
                    nmi, ari = _nmi_ari(contingency)
                    prefix_partition_rows.append(
                        {
                            "depth": depth,
                            "active_left_prefixes": int(
                                (contingency.sum(dim=1) > 0).sum().item()
                            ),
                            "active_right_prefixes": int(
                                (contingency.sum(dim=0) > 0).sum().item()
                            ),
                            "normalized_mutual_information": nmi,
                            "adjusted_rand_index": ari,
                        }
                    )
                pair_rows.append(
                    {
                        "split": split,
                        "family": family,
                        "left_run": left_run,
                        "right_run": right_run,
                        "vectors": accumulator["vectors"],
                        "levels": level_rows,
                        "prefix_partitions": prefix_partition_rows,
                        "mapped_prefix_agreement": [
                            float(
                                comparison[:, : depth]
                                .all(dim=1)
                                .float()
                                .mean()
                                .item()
                            )
                            for depth in range(1, levels + 1)
                        ],
                    }
                )

    report = {
        "schema": SEED_STABILITY_REPORT_SCHEMA,
        "contract_hash": contract_hash,
        "manifest_fingerprint": manifest_fingerprint,
        "reference_run": reference_run,
        "run_seeds": run_seeds,
        "distortion_rows": distortion_rows,
        "pair_rows": pair_rows,
        "interpretation": [
            "NMI and ARI are invariant to arbitrary code labels.",
            "Mapped agreement uses a maximum-overlap one-to-one code mapping "
            "at each RQ level.",
            "Low exact prefix agreement with stable distortion can indicate "
            "multiple equivalent partitions; both values must be reported.",
        ],
    }
    write_json_report(report_path, report)
    return report
