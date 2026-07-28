from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F

from codewam.codebook_eval.manifest import EpisodeManifest, EpisodeRecord
from codewam.data.droid_rlds import DroidRLDSEpisode


DROID_MOTION_AUDIT_SCHEMA = "codewam.droid-motion-audit.v1"


@dataclass(frozen=True)
class DroidMotionAuditConfig:
    cameras: tuple[str, ...]
    episodes_per_institution: int = 2
    split: str = "train"
    thumbnail_size: int = 32
    minimum_idle_steps: int = 16
    minimum_active_segment_steps: int = 41
    salt: str = "codewam-droid-motion-audit-v1"

    def __post_init__(self) -> None:
        if not self.cameras:
            raise ValueError("DROID motion audit requires at least one camera.")
        if len(set(self.cameras)) != len(self.cameras):
            raise ValueError("DROID motion audit cameras must be unique.")
        if self.episodes_per_institution <= 0:
            raise ValueError("Episodes per institution must be positive.")
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported audit split `{self.split}`.")
        if self.thumbnail_size <= 0:
            raise ValueError("Audit thumbnail size must be positive.")
        if self.minimum_idle_steps < 0:
            raise ValueError("Minimum idle steps must be non-negative.")
        if self.minimum_active_segment_steps <= 0:
            raise ValueError("Minimum active segment steps must be positive.")
        if not self.salt:
            raise ValueError("DROID motion audit salt must not be empty.")


def _score(value: str) -> bytes:
    return hashlib.sha256(value.encode("utf-8")).digest()


def _record_ranges(record: EpisodeRecord) -> tuple[tuple[int, int], ...]:
    value = record.metadata.get("keep_ranges")
    if not isinstance(value, list):
        raise ValueError(f"Episode `{record.key}` has no keep-range list.")
    ranges: list[tuple[int, int]] = []
    previous_stop = 0
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"Episode `{record.key}` has malformed keep ranges.")
        start, stop = int(item[0]), int(item[1])
        if start < previous_stop or stop <= start or stop > record.num_steps:
            raise ValueError(f"Episode `{record.key}` has invalid keep ranges.")
        ranges.append((start, stop))
        previous_stop = stop
    return tuple(ranges)


def select_droid_motion_audit_manifest(
    manifest: EpisodeManifest,
    config: DroidMotionAuditConfig,
) -> EpisodeManifest:
    by_institution: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for record in manifest:
        if record.split != config.split:
            continue
        ranges = _record_ranges(record)
        eligible_steps = sum(stop - start for start, stop in ranges)
        if record.num_steps - eligible_steps < config.minimum_idle_steps:
            continue
        if max((stop - start for start, stop in ranges), default=0) < (
            config.minimum_active_segment_steps
        ):
            continue
        by_institution[str(record.institution_id or "_")].append(record)

    selected: list[EpisodeRecord] = []
    for institution, candidates in sorted(by_institution.items()):
        candidates.sort(
            key=lambda record: _score(
                f"{config.salt}|{institution}|{record.group_key('scene')}|{record.key}"
            )
        )
        used_scenes: set[str] = set()
        for record in candidates:
            scene = record.group_key("scene")
            if scene in used_scenes:
                continue
            selected.append(record)
            used_scenes.add(scene)
            if len(used_scenes) == config.episodes_per_institution:
                break
        if len(used_scenes) < config.episodes_per_institution:
            raise ValueError(
                f"Institution `{institution}` has only {len(used_scenes)} "
                "eligible audit scenes."
            )
    if not selected:
        raise ValueError("No episodes satisfy the DROID motion-audit contract.")
    return EpisodeManifest.from_records(sorted(selected, key=lambda record: record.key))


def _thumbnails(frames: torch.Tensor, size: int, chunk_size: int = 64) -> torch.Tensor:
    chunks = []
    for start in range(0, int(frames.shape[0]), chunk_size):
        values = frames[start : start + chunk_size].permute(0, 3, 1, 2).float()
        chunks.append(F.adaptive_avg_pool2d(values, output_size=(size, size)))
    return torch.cat(chunks).div_(255.0)


def _keep_transition_masks(
    steps: int,
    ranges: Sequence[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keep = torch.zeros(steps, dtype=torch.bool)
    for start, stop in ranges:
        keep[start:stop] = True
    inside = keep[:-1] & keep[1:]
    outside = ~keep[:-1] & ~keep[1:]
    boundary = ~(inside | outside)
    return inside, outside, boundary


def _summary(values: torch.Tensor) -> dict[str, float | int | None]:
    values = values.detach().double().flatten()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "p10": None,
            "p50": None,
            "p90": None,
        }
    quantiles = torch.quantile(
        values,
        torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64),
    )
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "p10": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p90": float(quantiles[2].item()),
    }


def _median_ratio(
    inside: dict[str, float | int | None],
    outside: dict[str, float | int | None],
) -> float | None:
    numerator = inside["p50"]
    denominator = outside["p50"]
    if not isinstance(numerator, float) or not isinstance(denominator, float):
        return None
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    return numerator / max(denominator, 1e-12)


def episode_motion_measurements(
    episode: DroidRLDSEpisode,
    *,
    thumbnail_size: int,
) -> dict[str, Any]:
    inside, outside, boundary = _keep_transition_masks(
        episode.steps,
        episode.keep_ranges,
    )
    proprio_motion = (
        episode.proprio[1:].float()
        .sub(episode.proprio[:-1].float())
        .square()
        .mean(dim=1)
        .sqrt()
    )
    action_magnitude = episode.action[1:].float().square().mean(dim=1).sqrt()
    channels: dict[str, torch.Tensor] = {
        "proprio_motion": proprio_motion,
        "action_magnitude": action_magnitude,
    }
    for camera, frames in episode.frames.items():
        thumbnails = _thumbnails(frames, thumbnail_size)
        channels[f"image_motion/{camera}"] = (
            thumbnails[1:].sub(thumbnails[:-1]).abs().mean(dim=(1, 2, 3))
        )

    metrics = {}
    for name, values in channels.items():
        inside_summary = _summary(values[inside])
        outside_summary = _summary(values[outside])
        metrics[name] = {
            "inside": inside_summary,
            "outside": outside_summary,
            "inside_outside_median_ratio": _median_ratio(
                inside_summary,
                outside_summary,
            ),
        }
    return {
        "episode_id": episode.episode_id,
        "manifest_key": episode.manifest_key,
        "split": episode.split,
        "source_shard": episode.source_shard,
        "record_index": episode.record_index,
        "steps": episode.steps,
        "keep_ranges": [list(value) for value in episode.keep_ranges],
        "transition_counts": {
            "inside": int(inside.sum().item()),
            "outside": int(outside.sum().item()),
            "boundary": int(boundary.sum().item()),
        },
        "metrics": metrics,
        "_values": channels,
        "_inside": inside,
        "_outside": outside,
    }


def aggregate_motion_audit(
    manifest: EpisodeManifest,
    episodes: Iterable[DroidRLDSEpisode],
    config: DroidMotionAuditConfig,
) -> dict[str, Any]:
    records = {record.key: record for record in manifest}
    rows = []
    values: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: {"inside": [], "outside": []}
    )
    institution_counts: Counter[str] = Counter()
    for episode in episodes:
        if episode.manifest_key not in records:
            raise ValueError(
                f"Decoded episode `{episode.episode_id}` is not in the audit manifest."
            )
        record = records[episode.manifest_key]
        institution_counts[str(record.institution_id or "_")] += 1
        row = episode_motion_measurements(
            episode,
            thumbnail_size=config.thumbnail_size,
        )
        for name, channel_values in row.pop("_values").items():
            values[name]["inside"].append(channel_values[row["_inside"]])
            values[name]["outside"].append(channel_values[row["_outside"]])
        row.pop("_inside")
        row.pop("_outside")
        row.update(
            {
                "institution_id": record.institution_id,
                "building_id": record.building_id,
                "scene_id": record.scene_id,
                "collector_id": record.metadata.get("collector_id"),
            }
        )
        rows.append(row)
    if len(rows) != len(manifest):
        raise RuntimeError(
            f"Decoded {len(rows)} audit episodes, expected {len(manifest)}."
        )

    aggregate = {}
    for name, categories in values.items():
        inside = _summary(torch.cat(categories["inside"]))
        outside = _summary(torch.cat(categories["outside"]))
        episode_ratios = [
            row["metrics"][name]["inside_outside_median_ratio"]
            for row in rows
            if row["metrics"][name]["inside_outside_median_ratio"] is not None
        ]
        aggregate[name] = {
            "inside": inside,
            "outside": outside,
            "inside_outside_median_ratio": _median_ratio(inside, outside),
            "episode_median_ratio": (
                float(torch.tensor(episode_ratios).median().item())
                if episode_ratios
                else None
            ),
        }

    return {
        "schema": DROID_MOTION_AUDIT_SCHEMA,
        "source_manifest_fingerprint": manifest.fingerprint(),
        "config": {
            "cameras": list(config.cameras),
            "episodes_per_institution": config.episodes_per_institution,
            "split": config.split,
            "thumbnail_size": config.thumbnail_size,
            "minimum_idle_steps": config.minimum_idle_steps,
            "minimum_active_segment_steps": config.minimum_active_segment_steps,
            "salt": config.salt,
        },
        "selection": {
            "episodes": len(manifest),
            "institutions": dict(sorted(institution_counts.items())),
            "scenes": len({record.group_key("scene") for record in manifest}),
        },
        "aggregate": aggregate,
        "episodes": rows,
    }
