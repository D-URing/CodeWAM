from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import torch

from codewam.data.droid_manifest import write_json_report

from .manifest import EpisodeManifest, EpisodeRecord
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


CONCENTRATION_CONTRACT_SCHEMA = "codewam.rq-concentration-contract.v1"
CONCENTRATION_REPORT_SCHEMA = "codewam.rq-concentration-report.v1"
GROUPING_DEFINITIONS = {
    "scene": "Strict institution/building/scene identity from the manifest.",
    "institution": "Data-collection institution identity from the manifest.",
    "task": "Exact sorted task-id tuple from the manifest; no semantic merging.",
}
_MISSING_GROUP = "__missing__"


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
) -> Callable[[], Iterator[PooledFeatureEpisode]]:
    def episodes() -> Iterator[PooledFeatureEpisode]:
        seen: set[str] = set()
        for episode in iter_pooled_feature_episodes(
            shard_paths,
            split=split,
        ):
            if episode.episode_id not in expected_episode_ids:
                raise ValueError(
                    f"Concentration episode `{episode.episode_id}` is absent "
                    f"from the `{split}` manifest."
                )
            if episode.episode_id in seen:
                raise ValueError(
                    f"Duplicate concentration episode `{episode.episode_id}`."
                )
            seen.add(episode.episode_id)
            yield episode
        missing = sorted(expected_episode_ids - seen)
        if missing:
            raise ValueError(
                f"Concentration `{split}` episodes are missing: {missing[:8]}."
            )

    return episodes


def _group_label(record: EpisodeRecord, grouping: str) -> tuple[str, bool]:
    if grouping == "scene":
        if record.scene_id is None:
            return _MISSING_GROUP, True
        return record.group_key("scene"), False
    if grouping == "institution":
        if record.institution_id is None:
            return _MISSING_GROUP, True
        return record.group_key("institution"), False
    if grouping == "task":
        if not record.task_ids:
            return _MISSING_GROUP, True
        return json.dumps(
            sorted(record.task_ids),
            ensure_ascii=True,
            separators=(",", ":"),
        ), False
    raise ValueError(f"Unsupported concentration grouping `{grouping}`.")


def _entropy(counts: Iterable[int], total: int) -> float:
    if total <= 0:
        raise ValueError("Categorical entropy requires a positive total.")
    result = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = float(count) / float(total)
        result -= probability * math.log(probability)
    return result


def _categorical_association_metrics(
    joint_counts: Counter[tuple[int, str]],
    *,
    capacity: int,
    missing_group: str = _MISSING_GROUP,
) -> dict[str, Any]:
    if capacity <= 0:
        raise ValueError("Code capacity must be positive.")
    total = int(sum(joint_counts.values()))
    if total <= 0:
        raise ValueError("Cannot summarize empty concentration counts.")
    code_counts: Counter[int] = Counter()
    group_counts: Counter[str] = Counter()
    for (code, group), count in joint_counts.items():
        if code < 0 or code >= capacity or count <= 0:
            raise ValueError("Concentration counts contain an invalid entry.")
        code_counts[int(code)] += int(count)
        group_counts[str(group)] += int(count)

    code_entropy = _entropy(code_counts.values(), total)
    group_entropy = _entropy(group_counts.values(), total)
    mutual_information = 0.0
    for (code, group), count in joint_counts.items():
        probability = float(count) / float(total)
        mutual_information += probability * math.log(
            float(count * total)
            / float(code_counts[code] * group_counts[group])
        )
    mutual_information = max(mutual_information, 0.0)
    information_gain = (
        mutual_information / group_entropy
        if group_entropy > 1e-12
        else 0.0
    )
    normalized_mutual_information = (
        mutual_information / math.sqrt(code_entropy * group_entropy)
        if code_entropy > 1e-12 and group_entropy > 1e-12
        else 0.0
    )

    maximum_by_code: Counter[int] = Counter()
    for (code, _), count in joint_counts.items():
        maximum_by_code[code] = max(maximum_by_code[code], int(count))
    purity = sum(maximum_by_code.values()) / float(total)
    global_majority = max(group_counts.values()) / float(total)
    normalized_purity_gain = (
        (purity - global_majority) / (1.0 - global_majority)
        if global_majority < 1.0
        else 0.0
    )
    return {
        "vectors": total,
        "capacity": int(capacity),
        "active_codes": len(code_counts),
        "groups": len(group_counts),
        "missing_group_fraction": (
            group_counts.get(missing_group, 0) / float(total)
        ),
        "code_entropy_nats": code_entropy,
        "code_perplexity": math.exp(code_entropy),
        "group_entropy_nats": group_entropy,
        "mutual_information_nats": mutual_information,
        "group_information_gain": information_gain,
        "normalized_mutual_information": normalized_mutual_information,
        "group_purity": purity,
        "global_majority_fraction": global_majority,
        "normalized_purity_gain": normalized_purity_gain,
    }


def _cross_parent_prediction_metrics(
    fit_counts: Counter[tuple[int, str]],
    evaluation_counts: Counter[tuple[int, str]],
) -> dict[str, Any]:
    fit_total = int(sum(fit_counts.values()))
    evaluation_total = int(sum(evaluation_counts.values()))
    if fit_total <= 0 or evaluation_total <= 0:
        return {
            "cross_parent_available": False,
            "cross_parent_fit_vectors": fit_total,
            "cross_parent_evaluation_vectors": evaluation_total,
            "cross_parent_fit_groups": 0,
            "cross_parent_fit_active_codes": 0,
            "cross_parent_exact_code_coverage": None,
            "cross_parent_accuracy": None,
            "cross_parent_global_baseline_accuracy": None,
            "cross_parent_normalized_accuracy_gain": None,
        }
    fit_group_counts: Counter[str] = Counter()
    fit_code_counts: Counter[int] = Counter()
    labels_by_code: dict[int, Counter[str]] = defaultdict(Counter)
    for (code, group), count in fit_counts.items():
        fit_group_counts[group] += int(count)
        fit_code_counts[int(code)] += int(count)
        labels_by_code[int(code)][group] += int(count)
    global_label = min(
        (
            (-count, group)
            for group, count in fit_group_counts.items()
        )
    )[1]
    predicted_by_code = {
        code: min(
            (-count, group)
            for group, count in group_counts.items()
        )[1]
        for code, group_counts in labels_by_code.items()
    }

    correct = 0
    baseline_correct = 0
    covered = 0
    for (code, group), count in evaluation_counts.items():
        prediction = predicted_by_code.get(int(code), global_label)
        correct += int(count) if prediction == group else 0
        baseline_correct += int(count) if global_label == group else 0
        covered += int(count) if int(code) in fit_code_counts else 0
    accuracy = correct / float(evaluation_total)
    baseline_accuracy = baseline_correct / float(evaluation_total)
    normalized_accuracy_gain = (
        (accuracy - baseline_accuracy) / (1.0 - baseline_accuracy)
        if baseline_accuracy < 1.0
        else 0.0
    )
    return {
        "cross_parent_available": True,
        "cross_parent_fit_vectors": fit_total,
        "cross_parent_evaluation_vectors": evaluation_total,
        "cross_parent_fit_groups": len(fit_group_counts),
        "cross_parent_fit_active_codes": len(fit_code_counts),
        "cross_parent_exact_code_coverage": (
            covered / float(evaluation_total)
        ),
        "cross_parent_accuracy": accuracy,
        "cross_parent_global_baseline_accuracy": baseline_accuracy,
        "cross_parent_normalized_accuracy_gain": (
            normalized_accuracy_gain
        ),
    }


def _parent_episode_id(record: EpisodeRecord) -> str:
    value = str(record.metadata.get("parent_episode_id", ""))
    return value or record.episode_id


def _cross_parent_folds(
    records_by_episode: dict[str, EpisodeRecord],
    group_labels: dict[str, dict[str, str]],
    groupings: tuple[str, ...],
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, dict[str, int]]],
]:
    folds = {
        episode_id: {} for episode_id in records_by_episode
    }
    summaries: dict[str, dict[str, dict[str, int]]] = {}
    for grouping in groupings:
        parents_by_group: dict[tuple[str, str], set[str]] = defaultdict(set)
        for episode_id, record in records_by_episode.items():
            if record.split is None:
                raise ValueError(
                    f"Concentration episode `{episode_id}` has no split."
                )
            parents_by_group[
                (record.split, group_labels[episode_id][grouping])
            ].add(
                _parent_episode_id(record)
            )

        fold_by_group_parent: dict[tuple[str, str, str], str] = {}
        summaries[grouping] = {}
        for split in ("train", "val", "test"):
            split_groups = {
                group: parents
                for (group_split, group), parents in parents_by_group.items()
                if group_split == split
            }
            summaries[grouping][split] = {
                "groups": len(split_groups),
                "eligible_groups": 0,
                "fit_parent_episodes": 0,
                "evaluation_parent_episodes": 0,
            }
        for (split, group), parents in parents_by_group.items():
            ordered = sorted(
                parents,
                key=lambda parent: hashlib.sha256(
                    f"{split}|{grouping}|{group}|{parent}".encode("utf-8")
                ).hexdigest(),
            )
            if len(ordered) < 2:
                for parent in ordered:
                    fold_by_group_parent[
                        (split, group, parent)
                    ] = "ineligible"
                continue
            summaries[grouping][split]["eligible_groups"] += 1
            for index, parent in enumerate(ordered):
                fold = "fit" if index % 2 == 0 else "evaluation"
                fold_by_group_parent[(split, group, parent)] = fold
                summaries[grouping][split][
                    f"{fold}_parent_episodes"
                ] += 1

        for episode_id, record in records_by_episode.items():
            if record.split is None:
                raise RuntimeError("Concentration split validation drifted.")
            group = group_labels[episode_id][grouping]
            parent = _parent_episode_id(record)
            folds[episode_id][grouping] = fold_by_group_parent[
                (record.split, group, parent)
            ]
    return folds, summaries


def _write_contract(
    path: Path,
    contract: dict[str, Any],
    *,
    resume: bool,
) -> None:
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != contract:
            raise RuntimeError(
                f"Existing concentration contract differs from `{path}`."
            )
        if not resume:
            raise FileExistsError(
                f"Concentration contract exists at `{path}`."
            )
        return
    write_json_report(path, contract)


def probe_frozen_codebook_concentration(
    *,
    manifest_path: str | Path,
    pooled_shards: Iterable[str | Path],
    artifacts: dict[str, str | Path],
    output_dir: str | Path,
    splits: tuple[str, ...] = ("val", "test"),
    groupings: tuple[str, ...] = ("scene", "institution", "task"),
    device: str = "auto",
    cpu_threads: int = 4,
    batch_size: int = 8192,
    center_block_size: int = 1024,
    resume: bool = True,
) -> dict[str, Any]:
    if not artifacts or any(not label for label in artifacts):
        raise ValueError(
            "Concentration artifact labels must be nonempty and unique."
        )
    if cpu_threads <= 0 or batch_size <= 0 or center_block_size <= 0:
        raise ValueError(
            "Concentration thread, batch and block values must be positive."
        )
    if (
        not splits
        or len(splits) != len(set(splits))
        or any(split not in {"val", "test"} for split in splits)
    ):
        raise ValueError("Concentration splits must be unique val/test values.")
    if (
        not groupings
        or len(groupings) != len(set(groupings))
        or any(value not in GROUPING_DEFINITIONS for value in groupings)
    ):
        raise ValueError(
            "Concentration groupings must be unique supported values."
        )
    torch.set_num_threads(int(cpu_threads))

    manifest_path = Path(manifest_path)
    manifest = EpisodeManifest.read_jsonl(manifest_path)
    manifest.assert_group_isolation("scene")
    manifest_fingerprint = manifest.fingerprint()
    records_by_episode = {
        record.episode_id: record for record in manifest
    }
    if len(records_by_episode) != len(manifest):
        raise ValueError(
            "Concentration requires episode ids to be globally unique."
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
        raise ValueError(f"Concentration manifest has empty splits {empty}.")

    group_labels = {
        episode_id: {
            grouping: _group_label(record, grouping)[0]
            for grouping in groupings
        }
        for episode_id, record in records_by_episode.items()
    }
    cross_parent_folds, cross_parent_summaries = _cross_parent_folds(
        records_by_episode,
        group_labels,
        groupings,
    )
    shard_paths = tuple(expand_shard_paths(pooled_shards))
    shard_checksums = [file_sha256(path) for path in shard_paths]
    artifact_paths = {
        str(label): Path(path) for label, path in artifacts.items()
    }
    loaded_artifacts = {
        label: FrozenRQArtifact.load(path)
        for label, path in sorted(artifact_paths.items())
    }
    for label, artifact in loaded_artifacts.items():
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
                f"Concentration artifact `{label}` differs in {mismatches}."
            )

    implementation_sha256 = {
        "concentration": file_sha256(Path(__file__)),
        "shards": file_sha256(Path(__file__).with_name("shards.py")),
        "streaming": file_sha256(Path(__file__).with_name("streaming.py")),
    }
    contract_payload = {
        "schema": CONCENTRATION_CONTRACT_SCHEMA,
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
            label: {
                "path": str(artifact_paths[label].resolve()),
                "sha256": file_sha256(artifact_paths[label]),
            }
            for label in sorted(artifact_paths)
        },
        "splits": list(splits),
        "groupings": list(groupings),
        "grouping_definitions": GROUPING_DEFINITIONS,
        "weighting": "descriptor-tick",
        "cross_parent_policy": (
            "Within each held-out split and grouping, parent episodes are "
            "hash-ordered inside groups with at least two parents, then "
            "alternated between fit and evaluation. Single-parent groups are "
            "excluded from cross-parent prediction."
        ),
        "device": device,
        "cpu_threads": int(cpu_threads),
        "batch_size": int(batch_size),
        "center_block_size": int(center_block_size),
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    report_path = output_dir / "concentration_report.json"
    _write_contract(contract_path, contract, resume=resume)
    if resume and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("contract_hash") != contract_hash:
            raise RuntimeError("Concentration report contract hash is invalid.")
        return report
    if report_path.exists():
        raise FileExistsError(
            f"Concentration report exists at `{report_path}`."
        )

    target_device = _resolve_device(device)
    rows = []
    for label, artifact in loaded_artifacts.items():
        centers = tuple(
            center.to(device=target_device, dtype=torch.float32)
            for center in artifact.centers
        )
        k = int(centers[0].shape[0])
        levels = len(centers)
        for split in splits:
            joint_counts = {
                (grouping, depth): Counter()
                for grouping in groupings
                for depth in range(1, levels + 1)
            }
            cross_parent_fit_counts = {
                (grouping, depth): Counter()
                for grouping in groupings
                for depth in range(1, levels + 1)
            }
            cross_parent_evaluation_counts = {
                (grouping, depth): Counter()
                for grouping in groupings
                for depth in range(1, levels + 1)
            }
            descriptor_vectors = 0
            eligible_vectors = Counter()
            source = CausalDescriptorSource(
                episode_factory=_episode_factory(
                    shard_paths,
                    split,
                    expected_by_split[split],
                ),
                spec=artifact.descriptor,
                batch_size=batch_size,
                split=split,
            )
            for batch in source:
                normalized = artifact.normalization.normalize(
                    batch.vectors
                ).to(
                    device=target_device,
                    dtype=torch.float32,
                )
                codes, _, _ = encode_residual_quantizer(
                    normalized,
                    centers,
                    center_block_size=center_block_size,
                )
                codes = codes.detach().long().cpu()
                batch_group_labels = {
                    grouping: [
                        group_labels[episode_id][grouping]
                        for episode_id in batch.episode_ids
                    ]
                    for grouping in groupings
                }
                batch_folds = {
                    grouping: [
                        cross_parent_folds[episode_id][grouping]
                        for episode_id in batch.episode_ids
                    ]
                    for grouping in groupings
                }
                descriptor_vectors += len(batch.episode_ids)
                prefix = torch.zeros(codes.shape[0], dtype=torch.long)
                for depth in range(1, levels + 1):
                    prefix = prefix * k + codes[:, depth - 1]
                    keys = prefix.tolist()
                    for grouping in groupings:
                        joint_counts[(grouping, depth)].update(
                            zip(keys, batch_group_labels[grouping])
                        )
                        fit = cross_parent_fit_counts[(grouping, depth)]
                        evaluation = cross_parent_evaluation_counts[
                            (grouping, depth)
                        ]
                        for key, group, fold in zip(
                            keys,
                            batch_group_labels[grouping],
                            batch_folds[grouping],
                        ):
                            if fold == "fit":
                                fit[(key, group)] += 1
                            elif fold == "evaluation":
                                evaluation[(key, group)] += 1
                        if depth == 1:
                            eligible_vectors[grouping] += sum(
                                fold != "ineligible"
                                for fold in batch_folds[grouping]
                            )

            for (grouping, depth), counts in sorted(
                joint_counts.items()
            ):
                fit_counts = cross_parent_fit_counts[(grouping, depth)]
                evaluation_counts = cross_parent_evaluation_counts[
                    (grouping, depth)
                ]
                rows.append(
                    {
                        "label": label,
                        "family": artifact.family,
                        "stride": artifact.descriptor.stride,
                        "pool": artifact.descriptor.pool,
                        "camera_ids": (
                            None
                            if artifact.descriptor.camera_ids is None
                            else list(artifact.descriptor.camera_ids)
                        ),
                        "k": k,
                        "levels": levels,
                        "split": split,
                        "grouping": grouping,
                        "prefix_depth": depth,
                        "cross_parent_eligible_vector_fraction": (
                            eligible_vectors[grouping]
                            / float(descriptor_vectors)
                        ),
                        "cross_parent_split": cross_parent_summaries[
                            grouping
                        ][split],
                        **_categorical_association_metrics(
                            counts,
                            capacity=k**depth,
                        ),
                        **_cross_parent_prediction_metrics(
                            fit_counts,
                            evaluation_counts,
                        ),
                    }
                )

    report = {
        "schema": CONCENTRATION_REPORT_SCHEMA,
        "contract_hash": contract_hash,
        "manifest_fingerprint": manifest_fingerprint,
        "grouping_definitions": GROUPING_DEFINITIONS,
        "weighting": "descriptor-tick",
        "cross_parent_splits": cross_parent_summaries,
        "rows": rows,
    }
    write_json_report(report_path, report)
    return report
