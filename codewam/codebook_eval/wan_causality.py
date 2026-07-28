from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from codewam.data.droid_manifest import write_json_report
from codewam.data.droid_rlds import iter_manifest_droid_rlds_episodes

from .manifest import EpisodeManifest, EpisodeRecord
from .shards import file_sha256
from .wan_probe_export import (
    WanProbeExportConfig,
    _load_wan_vae,
    _preprocess_video,
    _torch_dtype,
)


WAN_CAUSALITY_AUDIT_SCHEMA = "codewam.wan-causality-audit.v1"


@dataclass(frozen=True)
class WanCausalityAuditConfig:
    source_manifest: str
    data_dir: str
    output_path: str
    vae_path: str
    fastwam_src: str
    cameras: tuple[str, ...] = (
        "exterior_image_1_left",
        "wrist_image_left",
    )
    split: str = "train"
    latent_ticks: int = 6
    image_height: int = 224
    image_width: int = 224
    device: str = "cuda"
    dtype: str = "bfloat16"
    atol: float = 1.0e-2
    rtol: float = 1.0e-2

    def __post_init__(self) -> None:
        if not self.cameras or len(set(self.cameras)) != len(self.cameras):
            raise ValueError("Causality-audit cameras must be nonempty and unique.")
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported audit split `{self.split}`.")
        if int(self.latent_ticks) < 2:
            raise ValueError("Causality audit requires at least two latent ticks.")
        if int(self.image_height) % 16 or int(self.image_width) % 16:
            raise ValueError("Wan input height and width must be divisible by 16.")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"Unsupported Wan dtype `{self.dtype}`.")
        if float(self.atol) < 0 or float(self.rtol) < 0:
            raise ValueError("Causality-audit tolerances must be non-negative.")

    @property
    def required_frames(self) -> int:
        return 1 + 4 * (int(self.latent_ticks) - 1)


def compare_causal_prefixes(
    videos: Sequence[torch.Tensor],
    vae: Any,
    *,
    device: str | torch.device,
    latent_ticks: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if not videos:
        raise ValueError("Causality audit requires at least one video.")
    if int(latent_ticks) < 2:
        raise ValueError("Causality audit requires at least two latent ticks.")
    required_frames = 1 + 4 * (int(latent_ticks) - 1)
    shapes = [tuple(video.shape) for video in videos]
    if any(video.ndim != 4 for video in videos):
        raise ValueError(f"Wan audit videos must be [C,T,H,W], got {shapes}.")
    if any(int(video.shape[1]) < required_frames for video in videos):
        raise ValueError(
            f"Wan audit needs {required_frames} frames per view, got {shapes}."
        )
    if any(tuple(video.shape) != tuple(videos[0].shape) for video in videos[1:]):
        raise ValueError(f"Wan audit video shapes differ: {shapes}.")

    trimmed = [
        video[:, :required_frames].contiguous()
        for video in videos
    ]
    with torch.inference_mode():
        full = vae.encode(trimmed, device=torch.device(device), tiled=False)
    if full.ndim != 5:
        raise RuntimeError(
            f"Wan full-sequence latent must be [V,C,T,H,W], got {tuple(full.shape)}."
        )
    expected_shape = (
        len(trimmed),
        int(full.shape[1]),
        int(latent_ticks),
        int(full.shape[3]),
        int(full.shape[4]),
    )
    if tuple(full.shape) != expected_shape:
        raise RuntimeError(
            "Wan full-sequence latent shape is inconsistent with causal cadence: "
            f"expected {expected_shape}, got {tuple(full.shape)}."
        )

    rows = []
    for prefix_ticks in range(1, int(latent_ticks) + 1):
        prefix_frames = 1 + 4 * (prefix_ticks - 1)
        prefix_videos = [
            video[:, :prefix_frames].contiguous()
            for video in trimmed
        ]
        with torch.inference_mode():
            observed = vae.encode(
                prefix_videos,
                device=torch.device(device),
                tiled=False,
            )
        expected = full[:, :, :prefix_ticks]
        if tuple(observed.shape) != tuple(expected.shape):
            raise RuntimeError(
                f"Wan prefix shape mismatch at {prefix_frames} frames: "
                f"{tuple(observed.shape)} vs {tuple(expected.shape)}."
            )
        difference = (observed.float() - expected.float()).abs()
        tolerance = float(atol) + float(rtol) * expected.float().abs()
        within = difference <= tolerance
        rows.append(
            {
                "prefix_frames": prefix_frames,
                "prefix_latent_ticks": prefix_ticks,
                "elements": int(difference.numel()),
                "max_abs_error": float(difference.max().item()),
                "mean_abs_error": float(difference.mean().item()),
                "rmse": float(difference.square().mean().sqrt().item()),
                "mismatch_fraction": float((~within).float().mean().item()),
                "passed": bool(within.all().item()),
            }
        )
        del observed, expected, difference, tolerance, within

    return {
        "frames": required_frames,
        "latent_ticks": int(latent_ticks),
        "views": len(trimmed),
        "latent_shape": list(full.shape),
        "atol": float(atol),
        "rtol": float(rtol),
        "rows": rows,
        "passed": all(bool(row["passed"]) for row in rows),
    }


def _select_record(
    manifest: EpisodeManifest,
    *,
    split: str,
    required_frames: int,
) -> EpisodeRecord:
    for record in manifest:
        if record.split != split:
            continue
        ranges = record.metadata.get("keep_ranges")
        if not isinstance(ranges, list):
            continue
        if any(int(stop) - int(start) >= required_frames for start, stop in ranges):
            return record
    raise ValueError(
        f"No `{split}` DROID keep range contains {required_frames} frames."
    )


def run_wan_causality_audit(
    config: WanCausalityAuditConfig,
) -> dict[str, Any]:
    source_manifest_path = Path(config.source_manifest)
    source_manifest = EpisodeManifest.read_jsonl(source_manifest_path)
    record = _select_record(
        source_manifest,
        split=config.split,
        required_frames=config.required_frames,
    )
    singleton_manifest = EpisodeManifest.from_records((record,))
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Wan causality audit requested unavailable CUDA.")
        torch.cuda.set_device(
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        torch.cuda.init()

    episode = next(
        iter_manifest_droid_rlds_episodes(
            config.data_dir,
            singleton_manifest,
            cameras=config.cameras,
        )
    )
    segment = next(
        value
        for value in episode.iter_eligible_segments()
        if value.steps >= config.required_frames
    )
    dtype = _torch_dtype(config.dtype)
    videos = [
        _preprocess_video(
            segment.frames[camera][: config.required_frames],
            height=config.image_height,
            width=config.image_width,
            dtype=dtype,
        )
        for camera in config.cameras
    ]
    loader_config = WanProbeExportConfig(
        data_dir=config.data_dir,
        output_dir=str(Path(config.output_path).parent),
        vae_path=config.vae_path,
        fastwam_src=config.fastwam_src,
        max_episodes=1,
        cameras=config.cameras,
        image_height=config.image_height,
        image_width=config.image_width,
        device=config.device,
        dtype=config.dtype,
        hash_source_shards=False,
    )
    vae = _load_wan_vae(loader_config)
    comparison = compare_causal_prefixes(
        videos,
        vae,
        device=config.device,
        latent_ticks=config.latent_ticks,
        atol=config.atol,
        rtol=config.rtol,
    )

    fastwam_implementation = (
        Path(config.fastwam_src)
        / "fastwam/models/wan22/wan_video_vae.py"
    )
    report = {
        "schema": WAN_CAUSALITY_AUDIT_SCHEMA,
        "passed": comparison["passed"],
        "audit_implementation_sha256": file_sha256(Path(__file__)),
        "source": {
            "manifest": str(source_manifest_path.resolve()),
            "manifest_sha256": file_sha256(source_manifest_path),
            "manifest_fingerprint": source_manifest.fingerprint(),
            "record_key": record.key,
            "segment_id": segment.segment_id,
            "source_shard": segment.source_shard,
            "source_range": [segment.start, segment.stop],
        },
        "model": {
            "model_id": "Wan-AI/Wan2.2-TI2V-5B",
            "vae_path": str(Path(config.vae_path).resolve()),
            "vae_sha256": file_sha256(config.vae_path),
            "fastwam_implementation_sha256": file_sha256(
                fastwam_implementation
            ),
        },
        "preprocess": {
            "cameras": list(config.cameras),
            "image_height": config.image_height,
            "image_width": config.image_width,
            "dtype": config.dtype,
            "device": config.device,
            "temporal_policy": "first-frame-then-four-frame-causal-prefixes",
        },
        "comparison": comparison,
    }
    write_json_report(config.output_path, report)

    del vae, videos, episode, segment
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report
