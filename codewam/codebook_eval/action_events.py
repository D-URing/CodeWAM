from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import torch

from codewam.data.droid_manifest import write_json_report

from .association import _episode_factory, _resolve_device, _selected_features
from .family_association import _validate_family_artifacts
from .manifest import EpisodeManifest
from .shards import (
    PooledFeatureEpisode,
    expand_shard_paths,
    file_sha256,
)
from .streaming import FrozenRQArtifact, encode_residual_quantizer


ACTION_EVENT_CONTRACT_SCHEMA = "codewam.rq-action-event-contract.v1"
ACTION_EVENT_REPORT_SCHEMA = "codewam.rq-action-event-report.v1"
EVENT_NAMES = (
    "translation_magnitude_quartile",
    "translation_direction",
    "rotation_magnitude_quartile",
    "rotation_direction",
    "gripper_change",
)
EVENT_DEFINITIONS = {
    "translation_magnitude_quartile": (
        "Quartile of Cartesian translation between t-stride and t. "
        "Thresholds are fit on train only."
    ),
    "translation_direction": (
        "Low-motion or signed dominant Cartesian translation axis between "
        "t-stride and t. The low-motion threshold is the train Q25 norm."
    ),
    "rotation_magnitude_quartile": (
        "Quartile of Cartesian rotation-coordinate change between t-stride "
        "and t. Thresholds are fit on train only."
    ),
    "rotation_direction": (
        "Low-motion or signed dominant Cartesian rotation-coordinate axis "
        "between t-stride and t."
    ),
    "gripper_change": (
        "Negative, stationary, or positive gripper-position change between "
        "t-stride and t. The deadband is fit on train nonzero magnitudes."
    ),
}
EVENT_CLASS_NAMES = {
    "translation_magnitude_quartile": ("q1", "q2", "q3", "q4"),
    "translation_direction": (
        "low_motion",
        "x_negative",
        "x_positive",
        "y_negative",
        "y_positive",
        "z_negative",
        "z_positive",
    ),
    "rotation_magnitude_quartile": ("q1", "q2", "q3", "q4"),
    "rotation_direction": (
        "low_motion",
        "x_negative",
        "x_positive",
        "y_negative",
        "y_positive",
        "z_negative",
        "z_positive",
    ),
    "gripper_change": ("negative", "stationary", "positive"),
}


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prefix_keys(codes: torch.Tensor, k: int) -> list[torch.Tensor]:
    codes = codes.detach().long().cpu()
    if codes.ndim != 2 or codes.shape[1] <= 0:
        raise ValueError("Action-event codes must be [N,L].")
    keys = []
    key = torch.zeros(codes.shape[0], dtype=torch.long)
    for level in range(codes.shape[1]):
        key = key * int(k) + codes[:, level]
        keys.append(key.clone())
    return keys


def _quantiles(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 1 or values.numel() <= 0:
        raise ValueError("Event threshold values must be a nonempty vector.")
    return torch.quantile(
        values.float(),
        torch.tensor((0.25, 0.5, 0.75), dtype=torch.float32),
    )


def _fit_event_thresholds(values: torch.Tensor) -> dict[str, Any]:
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError("Event values must be [N,7].")
    translation_norm = values[:, :3].float().norm(dim=1)
    rotation_norm = values[:, 3:6].float().norm(dim=1)
    gripper_absolute = values[:, 6].float().abs()
    nonzero = gripper_absolute[gripper_absolute > 1e-6]
    gripper_deadband = (
        1e-4
        if nonzero.numel() == 0
        else max(1e-4, 0.1 * float(torch.quantile(nonzero, 0.25).item()))
    )
    return {
        "translation_norm_quartiles": _quantiles(
            translation_norm
        ).tolist(),
        "rotation_norm_quartiles": _quantiles(rotation_norm).tolist(),
        "gripper_deadband": gripper_deadband,
    }


def _direction_labels(
    vectors: torch.Tensor,
    low_motion_threshold: float,
) -> torch.Tensor:
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("Direction vectors must be [N,3].")
    values = vectors.float()
    norms = values.norm(dim=1)
    axes = values.abs().argmax(dim=1)
    dominant = values.gather(1, axes.unsqueeze(1)).squeeze(1)
    labels = 1 + axes * 2 + (dominant >= 0).long()
    labels[norms <= float(low_motion_threshold)] = 0
    return labels


def _event_labels(
    values: torch.Tensor,
    thresholds: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError("Event values must be [N,7].")
    translation_thresholds = torch.tensor(
        thresholds["translation_norm_quartiles"],
        dtype=torch.float32,
    )
    rotation_thresholds = torch.tensor(
        thresholds["rotation_norm_quartiles"],
        dtype=torch.float32,
    )
    if translation_thresholds.shape != (3,) or rotation_thresholds.shape != (3,):
        raise ValueError("Magnitude events require three quartile thresholds.")
    translation_norm = values[:, :3].float().norm(dim=1)
    rotation_norm = values[:, 3:6].float().norm(dim=1)
    gripper = values[:, 6].float()
    deadband = float(thresholds["gripper_deadband"])
    gripper_labels = torch.ones(gripper.shape[0], dtype=torch.long)
    gripper_labels[gripper < -deadband] = 0
    gripper_labels[gripper > deadband] = 2
    return {
        "translation_magnitude_quartile": torch.bucketize(
            translation_norm,
            translation_thresholds,
        ),
        "translation_direction": _direction_labels(
            values[:, :3],
            float(translation_thresholds[0]),
        ),
        "rotation_magnitude_quartile": torch.bucketize(
            rotation_norm,
            rotation_thresholds,
        ),
        "rotation_direction": _direction_labels(
            values[:, 3:6],
            float(rotation_thresholds[0]),
        ),
        "gripper_change": gripper_labels,
    }


def _episode_aligned_event_values(
    episode: PooledFeatureEpisode,
    artifacts: dict[str, FrozenRQArtifact],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    labels, _ = _validate_family_artifacts(artifacts)
    reference = artifacts[labels[0]].descriptor
    maximum_stride = max(
        artifacts[label].descriptor.stride for label in labels
    )
    if episode.proprio is None or episode.proprio.shape[1] < 7:
        raise ValueError(
            f"Episode `{episode.episode_id}` lacks Cartesian/gripper proprio."
        )
    if episode.ticks <= 2 * maximum_stride:
        return {}, {}

    pooled, valid_mask = _selected_features(episode, reference)
    features = pooled.reshape(episode.ticks, -1)
    current = torch.arange(
        2 * maximum_stride,
        episode.ticks,
        dtype=torch.long,
        device=features.device,
    )
    timestamps = episode.timestamps.to(device=features.device)
    valid_mask = valid_mask.to(device=features.device)
    valid = torch.ones(current.shape[0], dtype=torch.bool, device=features.device)
    for label in labels:
        spec = artifacts[label].descriptor
        stride = spec.stride
        valid &= (
            valid_mask[current - 2 * stride].all(dim=1)
            & valid_mask[current - stride].all(dim=1)
            & valid_mask[current].all(dim=1)
        )
        if spec.max_gap_factor is not None and episode.ticks > 1:
            cadence = torch.median(timestamps[1:] - timestamps[:-1])
            maximum_gap = cadence * stride * float(spec.max_gap_factor)
            valid &= (
                timestamps[current - stride]
                - timestamps[current - 2 * stride]
                <= maximum_gap
            )
            valid &= (
                timestamps[current] - timestamps[current - stride]
                <= maximum_gap
            )
    current = current[valid]
    if current.numel() == 0:
        return {}, {}

    vectors: dict[str, torch.Tensor] = {}
    event_values: dict[str, torch.Tensor] = {}
    proprio = episode.proprio.to(device=features.device).float()
    for label in labels:
        stride = artifacts[label].descriptor.stride
        vectors[label] = torch.cat(
            (
                features[current - 2 * stride],
                features[current - stride],
                features[current],
            ),
            dim=1,
        ).contiguous()
        cartesian_change = (
            proprio[current, :6] - proprio[current - stride, :6]
        )
        gripper_change = (
            proprio[current, -1:] - proprio[current - stride, -1:]
        )
        event_values[label] = torch.cat(
            (cartesian_change, gripper_change),
            dim=1,
        ).contiguous()
    return vectors, event_values


def _iter_aligned_event_batches(
    episode_factory: Callable[[], Iterable[PooledFeatureEpisode]],
    artifacts: dict[str, FrozenRQArtifact],
    batch_size: int,
) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]]:
    labels, _ = _validate_family_artifacts(artifacts)
    vector_parts: dict[str, list[torch.Tensor]] = {
        label: [] for label in labels
    }
    event_parts: dict[str, list[torch.Tensor]] = {
        label: [] for label in labels
    }
    pending = 0

    def emit() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        nonlocal vector_parts, event_parts, pending
        result = (
            {
                label: torch.cat(parts, dim=0).contiguous()
                for label, parts in vector_parts.items()
            },
            {
                label: torch.cat(parts, dim=0).contiguous()
                for label, parts in event_parts.items()
            },
        )
        vector_parts = {label: [] for label in labels}
        event_parts = {label: [] for label in labels}
        pending = 0
        return result

    for episode in episode_factory():
        vectors, events = _episode_aligned_event_values(episode, artifacts)
        if not vectors:
            continue
        rows = next(iter(vectors.values())).shape[0]
        offset = 0
        while offset < rows:
            take = min(batch_size - pending, rows - offset)
            for label in labels:
                vector_parts[label].append(
                    vectors[label][offset : offset + take]
                )
                event_parts[label].append(
                    events[label][offset : offset + take]
                )
            pending += take
            offset += take
            if pending == batch_size:
                yield emit()
    if pending:
        yield emit()


class _CategoricalPrefixTable:
    def __init__(self, *, k: int, levels: int, classes: int) -> None:
        self.k = int(k)
        self.levels = int(levels)
        self.classes = int(classes)
        self.counts = [
            torch.zeros(
                (self.k ** (level + 1), self.classes),
                dtype=torch.long,
            )
            for level in range(self.levels)
        ]
        self.global_counts = torch.zeros(self.classes, dtype=torch.long)

    def update(self, codes: torch.Tensor, labels: torch.Tensor) -> None:
        labels = labels.detach().long().cpu()
        if labels.ndim != 1 or labels.numel() != codes.shape[0]:
            raise ValueError("Categorical labels do not match code rows.")
        if labels.numel() and (
            int(labels.min()) < 0 or int(labels.max()) >= self.classes
        ):
            raise ValueError("Categorical label is outside the class range.")
        self.global_counts += torch.bincount(
            labels,
            minlength=self.classes,
        )
        for level, keys in enumerate(_prefix_keys(codes, self.k)):
            flat = keys * self.classes + labels
            self.counts[level] += torch.bincount(
                flat,
                minlength=self.counts[level].numel(),
            ).reshape_as(self.counts[level])

    def predict(
        self,
        codes: torch.Tensor,
        *,
        depth: int,
        min_train_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if depth <= 0 or depth > self.levels:
            raise ValueError(f"Invalid categorical prefix depth `{depth}`.")
        keys = _prefix_keys(codes, self.k)
        global_label = int(self.global_counts.argmax().item())
        predictions = torch.full(
            (codes.shape[0],),
            global_label,
            dtype=torch.long,
        )
        chosen_depth = torch.zeros(codes.shape[0], dtype=torch.long)
        for level in range(depth):
            selected = self.counts[level][keys[level]]
            usable = selected.sum(dim=1) >= int(min_train_count)
            if usable.any():
                predictions[usable] = selected[usable].argmax(dim=1)
                chosen_depth[usable] = level + 1
        return predictions, chosen_depth


def _entropy(counts: torch.Tensor) -> float:
    counts = counts.double()
    total = float(counts.sum().item())
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(
        -(probabilities * probabilities.log()).sum().item()
    )


@dataclass
class _ClassificationAccumulator:
    classes: int
    depth: int
    train_global_label: int

    def __post_init__(self) -> None:
        self.confusion = torch.zeros(
            (self.classes, self.classes),
            dtype=torch.long,
        )
        self.prefix_event = torch.zeros(
            (0, self.classes),
            dtype=torch.long,
        )
        self.exact_matches = 0
        self.any_matches = 0
        self.backoff_depth_counts = [0 for _ in range(self.depth + 1)]

    def update(
        self,
        *,
        prefix_keys: torch.Tensor,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        chosen_depth: torch.Tensor,
        prefix_capacity: int,
    ) -> None:
        labels = labels.detach().long().cpu()
        predictions = predictions.detach().long().cpu()
        prefix_keys = prefix_keys.detach().long().cpu()
        pairs = labels * self.classes + predictions
        self.confusion += torch.bincount(
            pairs,
            minlength=self.classes * self.classes,
        ).reshape(self.classes, self.classes)
        if self.prefix_event.shape[0] == 0:
            self.prefix_event = torch.zeros(
                (prefix_capacity, self.classes),
                dtype=torch.long,
            )
        flat = prefix_keys * self.classes + labels
        self.prefix_event += torch.bincount(
            flat,
            minlength=prefix_capacity * self.classes,
        ).reshape(prefix_capacity, self.classes)
        self.exact_matches += int((chosen_depth == self.depth).sum().item())
        self.any_matches += int((chosen_depth > 0).sum().item())
        backoff = torch.bincount(
            chosen_depth,
            minlength=self.depth + 1,
        )
        for index, value in enumerate(backoff.tolist()):
            self.backoff_depth_counts[index] += int(value)

    def row(self) -> dict[str, Any]:
        total = int(self.confusion.sum().item())
        if total <= 0:
            raise ValueError("Cannot summarize an empty classification.")
        correct = int(self.confusion.diag().sum().item())
        accuracy = correct / float(total)
        true_counts = self.confusion.sum(dim=1)
        predicted_counts = self.confusion.sum(dim=0)
        recalls = self.confusion.diag().float() / true_counts.clamp_min(1)
        present = true_counts > 0
        balanced_accuracy = float(recalls[present].mean().item())
        baseline_accuracy = float(
            true_counts[self.train_global_label].item()
        ) / float(total)
        normalized_gain = (
            (accuracy - baseline_accuracy) / (1.0 - baseline_accuracy)
            if baseline_accuracy < 1.0
            else 0.0
        )

        precision = (
            self.confusion.diag().float()
            / predicted_counts.clamp_min(1)
        )
        f1 = (
            2.0
            * precision
            * recalls
            / (precision + recalls).clamp_min(1e-12)
        )
        macro_f1 = float(f1[present].mean().item())

        joint = self.prefix_event
        prefix_counts = joint.sum(dim=1)
        event_counts = joint.sum(dim=0)
        prefix_entropy = _entropy(prefix_counts)
        event_entropy = _entropy(event_counts)
        mutual_information = 0.0
        joint_total = float(joint.sum().item())
        if joint_total > 0:
            nonzero = joint.nonzero()
            for prefix, event in nonzero.tolist():
                count = float(joint[prefix, event].item())
                mutual_information += (count / joint_total) * math.log(
                    count
                    * joint_total
                    / float(prefix_counts[prefix] * event_counts[event])
                )
        normalized_mutual_information = (
            mutual_information
            / math.sqrt(prefix_entropy * event_entropy)
            if prefix_entropy > 1e-12 and event_entropy > 1e-12
            else 0.0
        )
        return {
            "vectors": total,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "macro_f1": macro_f1,
            "train_global_baseline_accuracy": baseline_accuracy,
            "normalized_accuracy_gain": normalized_gain,
            "normalized_mutual_information": normalized_mutual_information,
            "event_class_counts": true_counts.tolist(),
            "confusion_matrix": self.confusion.tolist(),
            "exact_prefix_coverage": self.exact_matches / float(total),
            "any_code_coverage": self.any_matches / float(total),
            "backoff_depth_counts": self.backoff_depth_counts,
        }


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
                f"Existing action-event contract differs from `{path}`."
            )
        if not resume:
            raise FileExistsError(
                f"Action-event contract exists at `{path}`."
            )
        return
    write_json_report(path, contract)


def probe_codebook_action_events(
    *,
    manifest_path: str | Path,
    pooled_shards: Iterable[str | Path],
    artifacts: dict[str, str | Path],
    output_dir: str | Path,
    splits: tuple[str, ...] = ("val", "test"),
    device: str = "auto",
    cpu_threads: int = 4,
    batch_size: int = 8192,
    center_block_size: int = 1024,
    min_train_count: int = 8,
    resume: bool = True,
) -> dict[str, Any]:
    if len(artifacts) < 2 or any(not label for label in artifacts):
        raise ValueError(
            "Action-event probe requires at least two labeled artifacts."
        )
    if (
        cpu_threads <= 0
        or batch_size <= 0
        or center_block_size <= 0
        or min_train_count <= 0
    ):
        raise ValueError(
            "Action-event thread, batch, block and count values must be positive."
        )
    if not splits or any(split not in {"val", "test"} for split in splits):
        raise ValueError("Action-event splits must be val/test.")
    if len(splits) != len(set(splits)):
        raise ValueError("Action-event splits must be unique.")
    torch.set_num_threads(int(cpu_threads))

    manifest_path = Path(manifest_path)
    manifest = EpisodeManifest.read_jsonl(manifest_path)
    manifest.assert_group_isolation("scene")
    manifest_fingerprint = manifest.fingerprint()
    shard_paths = tuple(expand_shard_paths(pooled_shards))
    shard_checksums = [file_sha256(path) for path in shard_paths]
    artifact_paths = {
        str(label): Path(path) for label, path in sorted(artifacts.items())
    }
    loaded_artifacts = {
        label: FrozenRQArtifact.load(path)
        for label, path in artifact_paths.items()
    }
    labels, levels = _validate_family_artifacts(loaded_artifacts)
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
                f"Action-event artifact `{label}` differs in {mismatches}."
            )

    expected_by_split = {
        split: {
            record.episode_id
            for record in manifest
            if record.split == split
        }
        for split in ("train", *splits)
    }
    empty = [
        split
        for split, identifiers in expected_by_split.items()
        if not identifiers
    ]
    if empty:
        raise ValueError(f"Action-event manifest has empty splits {empty}.")

    implementation_sha256 = {
        "action_events": file_sha256(Path(__file__)),
        "association": file_sha256(Path(__file__).with_name("association.py")),
        "family_association": file_sha256(
            Path(__file__).with_name("family_association.py")
        ),
        "shards": file_sha256(Path(__file__).with_name("shards.py")),
        "streaming": file_sha256(Path(__file__).with_name("streaming.py")),
    }
    contract_payload = {
        "schema": ACTION_EVENT_CONTRACT_SCHEMA,
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
            for label in labels
        },
        "splits": list(splits),
        "device": device,
        "cpu_threads": cpu_threads,
        "batch_size": batch_size,
        "center_block_size": center_block_size,
        "min_train_count": min_train_count,
        "event_definitions": EVENT_DEFINITIONS,
        "implementation_sha256": implementation_sha256,
    }
    contract_hash = _canonical_hash(contract_payload)
    contract = {**contract_payload, "contract_hash": contract_hash}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    report_path = output_dir / "action_event_report.json"
    _write_contract(contract_path, contract, resume=resume)
    if resume and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("contract_hash") != contract_hash:
            raise RuntimeError("Action-event report contract hash is invalid.")
        return report
    if report_path.exists():
        raise FileExistsError(
            f"Action-event report exists at `{report_path}`."
        )

    target_device = _resolve_device(device)
    centers = {
        label: tuple(
            center.to(device=target_device, dtype=torch.float32)
            for center in loaded_artifacts[label].centers
        )
        for label in labels
    }
    k_by_label = {
        label: int(centers[label][0].shape[0]) for label in labels
    }

    train_codes: dict[str, list[torch.Tensor]] = {
        label: [] for label in labels
    }
    train_values: dict[str, list[torch.Tensor]] = {
        label: [] for label in labels
    }
    train_factory = _episode_factory(
        shard_paths,
        "train",
        expected_by_split["train"],
    )
    with torch.inference_mode():
        for vectors, event_values in _iter_aligned_event_batches(
            train_factory,
            loaded_artifacts,
            batch_size,
        ):
            for label in labels:
                normalized = loaded_artifacts[label].normalization.normalize(
                    vectors[label]
                ).to(device=target_device, dtype=torch.float32)
                codes, _, _ = encode_residual_quantizer(
                    normalized,
                    centers[label],
                    center_block_size=center_block_size,
                )
                train_codes[label].append(codes.cpu())
                train_values[label].append(event_values[label].float().cpu())

    thresholds: dict[str, dict[str, Any]] = {}
    tables: dict[tuple[str, str], _CategoricalPrefixTable] = {}
    train_vector_counts: dict[str, int] = {}
    for label in labels:
        codes = torch.cat(train_codes[label], dim=0)
        values = torch.cat(train_values[label], dim=0)
        train_vector_counts[label] = int(codes.shape[0])
        thresholds[label] = _fit_event_thresholds(values)
        event_labels = _event_labels(values, thresholds[label])
        for event_name in EVENT_NAMES:
            table = _CategoricalPrefixTable(
                k=k_by_label[label],
                levels=levels,
                classes=len(EVENT_CLASS_NAMES[event_name]),
            )
            table.update(codes, event_labels[event_name])
            tables[(label, event_name)] = table
    del train_codes, train_values

    accumulators: dict[
        tuple[str, str, str, int],
        _ClassificationAccumulator,
    ] = {}
    with torch.inference_mode():
        for split in splits:
            split_factory = _episode_factory(
                shard_paths,
                split,
                expected_by_split[split],
            )
            for vectors, event_values in _iter_aligned_event_batches(
                split_factory,
                loaded_artifacts,
                batch_size,
            ):
                for label in labels:
                    normalized = (
                        loaded_artifacts[label].normalization.normalize(
                            vectors[label]
                        ).to(device=target_device, dtype=torch.float32)
                    )
                    codes, _, _ = encode_residual_quantizer(
                        normalized,
                        centers[label],
                        center_block_size=center_block_size,
                    )
                    codes = codes.cpu()
                    prefix_keys = _prefix_keys(
                        codes,
                        k_by_label[label],
                    )
                    event_labels = _event_labels(
                        event_values[label].cpu(),
                        thresholds[label],
                    )
                    for event_name in EVENT_NAMES:
                        table = tables[(label, event_name)]
                        values = event_labels[event_name]
                        for depth in range(1, levels + 1):
                            predictions, chosen_depth = table.predict(
                                codes,
                                depth=depth,
                                min_train_count=min_train_count,
                            )
                            key = (label, split, event_name, depth)
                            accumulator = accumulators.setdefault(
                                key,
                                _ClassificationAccumulator(
                                    classes=len(
                                        EVENT_CLASS_NAMES[event_name]
                                    ),
                                    depth=depth,
                                    train_global_label=int(
                                        table.global_counts.argmax().item()
                                    ),
                                ),
                            )
                            accumulator.update(
                                prefix_keys=prefix_keys[depth - 1],
                                labels=values,
                                predictions=predictions,
                                chosen_depth=chosen_depth,
                                prefix_capacity=(
                                    k_by_label[label] ** depth
                                ),
                            )

    rows = []
    for (label, split, event_name, depth), accumulator in sorted(
        accumulators.items()
    ):
        artifact = loaded_artifacts[label]
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
                "k": k_by_label[label],
                "levels": levels,
                "split": split,
                "event": event_name,
                "event_classes": list(EVENT_CLASS_NAMES[event_name]),
                "prefix_depth": depth,
                "min_train_count": min_train_count,
                "train_vectors": train_vector_counts[label],
                **accumulator.row(),
            }
        )

    report = {
        "schema": ACTION_EVENT_REPORT_SCHEMA,
        "contract_hash": contract_hash,
        "manifest_fingerprint": manifest_fingerprint,
        "event_definitions": EVENT_DEFINITIONS,
        "train_thresholds": thresholds,
        "rows": rows,
    }
    write_json_report(report_path, report)
    return report
