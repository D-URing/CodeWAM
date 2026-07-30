from __future__ import annotations

import gc
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
import torch.nn.functional as F

from codewam.data.droid_rlds import (
    DROID_CAMERA_KEYS,
    DroidRLDSEpisode,
    iter_droid_rlds_episodes,
    probe_split,
)

from .io import ensure_dir, save_json
from .shards import (
    POOLED_SHARD_SCHEMA,
    PooledFeatureEpisode,
    file_sha256,
    load_torch_payload,
    write_pooled_feature_shard,
)


WAN_PROBE_EXPORT_SCHEMA = "codewam.wan-latent-probe-export.v1"


@dataclass(frozen=True)
class WanProbeExportConfig:
    data_dir: str
    output_dir: str
    vae_path: str
    fastwam_src: str
    max_episodes: int = 12
    cameras: tuple[str, ...] = DROID_CAMERA_KEYS
    nominal_fps: float = 15.0
    image_height: int = 224
    image_width: int = 224
    thumbnail_size: int = 64
    device: str = "cuda"
    dtype: str = "bfloat16"
    resume: bool = True
    hash_source_shards: bool = True

    def __post_init__(self) -> None:
        if int(self.max_episodes) <= 0:
            raise ValueError("`max_episodes` must be positive.")
        if float(self.nominal_fps) <= 0:
            raise ValueError("`nominal_fps` must be positive.")
        if int(self.image_height) % 16 or int(self.image_width) % 16:
            raise ValueError("Wan input height and width must be divisible by 16.")
        if int(self.thumbnail_size) <= 0:
            raise ValueError("`thumbnail_size` must be positive.")
        if str(self.dtype) not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"Unsupported Wan dtype `{self.dtype}`.")


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[str(name)]


def _construct_wan_vae(model_type: type[torch.nn.Module]) -> torch.nn.Module:
    # FastWAM keeps the VAE normalization statistics as ordinary tensor
    # attributes, so meta-device construction cannot materialize them later.
    model = model_type()
    for name in ("mean", "std"):
        value = getattr(model, name, None)
        if not isinstance(value, torch.Tensor) or value.is_meta:
            raise RuntimeError(
                f"Wan VAE normalization tensor `{name}` was not materialized."
            )
    return model


def _source_checksums(data_dir: Path, include_tfrecords: bool) -> list[str]:
    patterns = ["dataset_info.json", "features.json"]
    if include_tfrecords:
        patterns.append("*.tfrecord*")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(data_dir.glob(pattern)))
    if not paths:
        raise FileNotFoundError(f"No DROID source files found under {data_dir}.")
    return [f"{path.name}:{file_sha256(path)}" for path in paths]


def _load_wan_vae(config: WanProbeExportConfig):
    fastwam_src = str(Path(config.fastwam_src).resolve())
    if fastwam_src not in sys.path:
        sys.path.insert(0, fastwam_src)
    try:
        from fastwam.models.wan22.helpers.io import load_state_dict
        from fastwam.models.wan22.helpers.state_dict_converters import (
            wan_video_vae_state_dict_converter,
        )
        from fastwam.models.wan22.wan_video_vae import WanVideoVAE38
    except ImportError as exc:
        raise RuntimeError(
            f"Could not import the FastWAM Wan-VAE implementation from {fastwam_src}."
        ) from exc

    vae_path = Path(config.vae_path)
    if not vae_path.is_file():
        raise FileNotFoundError(f"Missing Wan VAE checkpoint: {vae_path}.")
    dtype = _torch_dtype(config.dtype)
    model = _construct_wan_vae(WanVideoVAE38)
    state_dict = load_state_dict(str(vae_path), torch_dtype=dtype, device="cpu")
    converted = wan_video_vae_state_dict_converter(state_dict)
    incompatibility = model.load_state_dict(converted, strict=True, assign=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f"Wan VAE state-dict mismatch: {incompatibility}.")
    del state_dict, converted
    model = model.to(device=torch.device(config.device), dtype=dtype)
    model.eval().requires_grad_(False)
    return model


def latent_frame_indices(num_frames: int, latent_ticks: int) -> torch.Tensor:
    if int(num_frames) <= 0 or int(latent_ticks) <= 0:
        raise ValueError("Frame and latent tick counts must be positive.")
    expected = 1 + (int(num_frames) - 1) // 4
    if expected != int(latent_ticks):
        raise ValueError(
            f"Wan temporal shape mismatch: frames={num_frames}, "
            f"expected ticks={expected}, observed ticks={latent_ticks}."
        )
    return (torch.arange(latent_ticks, dtype=torch.long) * 4).clamp_max(
        int(num_frames) - 1
    )


def _preprocess_video(
    frames: torch.Tensor,
    height: int,
    width: int,
    dtype: torch.dtype,
    frame_batch_size: int = 64,
) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 [T,H,W,3] frames, got {tuple(frames.shape)}.")
    if int(frame_batch_size) <= 0:
        raise ValueError("Wan preprocessing frame batch size must be positive.")
    output = torch.empty(
        (3, int(frames.shape[0]), int(height), int(width)),
        dtype=dtype,
        device=frames.device,
    )
    for start in range(0, int(frames.shape[0]), int(frame_batch_size)):
        stop = min(start + int(frame_batch_size), int(frames.shape[0]))
        values = frames[start:stop].permute(0, 3, 1, 2).float()
        values = F.interpolate(
            values,
            size=(int(height), int(width)),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        values = values / 127.5 - 1.0
        output[:, start:stop].copy_(
            values.to(dtype=dtype).permute(1, 0, 2, 3)
        )
    return output


@torch.inference_mode()
def _encode_wan_views_streaming(
    camera_frames: Iterable[torch.Tensor],
    *,
    vae: Any,
    device: torch.device,
    height: int,
    width: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    hidden_states = []
    for frames in camera_frames:
        video = _preprocess_video(
            frames,
            height=height,
            width=width,
            dtype=dtype,
        ).to(device=device)
        encoded = vae.encode([video], device=device, tiled=False)
        if encoded.ndim != 5 or int(encoded.shape[0]) != 1:
            raise RuntimeError(
                "Wan streaming view encode must return [1,C,T,H,W], "
                f"got {tuple(encoded.shape)}."
            )
        hidden_states.append(encoded[0])
        del video, encoded
    if not hidden_states:
        raise ValueError("Wan streaming encode needs at least one camera view.")
    return torch.stack(hidden_states)


def _thumbnails(
    frames: torch.Tensor,
    indices: torch.Tensor,
    size: int,
) -> torch.Tensor:
    selected = frames[indices].permute(0, 3, 1, 2).float()
    resized = F.interpolate(
        selected,
        size=(int(size), int(size)),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return resized.round().clamp(0, 255).to(torch.uint8)


def _encode_episode(
    episode: DroidRLDSEpisode,
    vae: Any,
    config: WanProbeExportConfig,
) -> PooledFeatureEpisode:
    dtype = _torch_dtype(config.dtype)
    latent = _encode_wan_views_streaming(
        (episode.frames[camera] for camera in config.cameras),
        vae=vae,
        device=torch.device(config.device),
        height=config.image_height,
        width=config.image_width,
        dtype=dtype,
    )
    if latent.ndim != 5 or latent.shape[0] != len(config.cameras):
        raise RuntimeError(f"Unexpected Wan latent shape: {tuple(latent.shape)}.")
    views, channels, ticks, latent_height, latent_width = latent.shape
    if channels != 48:
        raise RuntimeError(f"Expected 48 Wan latent channels, got {channels}.")
    pooled = F.adaptive_avg_pool2d(
        latent.permute(0, 2, 1, 3, 4).reshape(
            views * ticks,
            channels,
            latent_height,
            latent_width,
        ),
        output_size=(4, 4),
    )
    pooled = (
        pooled.reshape(views, ticks, channels, 4, 4)
        .permute(1, 0, 2, 3, 4)
        .half()
        .cpu()
        .contiguous()
    )
    frame_indices = latent_frame_indices(episode.steps, ticks)
    thumbnails = torch.stack(
        [
            _thumbnails(
                episode.frames[camera],
                frame_indices,
                size=config.thumbnail_size,
            )
            for camera in config.cameras
        ],
        dim=1,
    )
    timestamps = frame_indices.double() / float(config.nominal_fps)
    return PooledFeatureEpisode(
        episode_id=episode.episode_id,
        split=probe_split(episode.index),
        timestamps=timestamps,
        pooled_g4=pooled,
        camera_ids=tuple(config.cameras),
        action=episode.action[frame_indices].half(),
        proprio=episode.proprio[frame_indices].half(),
        metadata={
            "schema": WAN_PROBE_EXPORT_SCHEMA,
            "source_index": int(episode.index),
            "source_file": episode.source_file,
            "recording_folder": episode.recording_folder,
            "language_instruction": episode.language_instruction,
            "source_frame_count": int(episode.steps),
            "latent_frame_indices": frame_indices,
            "probe_thumbnails": thumbnails,
            "timestamp_source": "step_index/nominal_fps",
            "nominal_fps": float(config.nominal_fps),
            "source_image_shape": list(episode.frames[config.cameras[0]].shape[1:]),
            "model_input_shape": [config.image_height, config.image_width],
            "latent_shape": [channels, ticks, latent_height, latent_width],
        },
    )


def _probe_shard_metadata(
    config: WanProbeExportConfig,
    vae_sha256: str,
    source_checksums: list[str],
) -> dict[str, Any]:
    return {
        "dataset_revision": "droid-r2d2-faceblur-1.0.0",
        "wan_model_id": "Wan-AI/Wan2.2-TI2V-5B",
        "wan_revision": vae_sha256,
        "preprocess_revision": (
            f"rgb-direct-bilinear-{config.image_height}x{config.image_width}"
            "-minus1-plus1-v1"
        ),
        "source_checksums": source_checksums,
    }


def _reused_episode_row(
    path: Path,
    episode: DroidRLDSEpisode,
    expected_metadata: dict[str, Any],
    expected_cameras: tuple[str, ...],
) -> dict[str, Any]:
    payload = load_torch_payload(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema") != POOLED_SHARD_SCHEMA:
        raise RuntimeError(f"Cannot resume from unsupported pooled shard `{path}`.")
    shard_metadata = payload.get("metadata")
    if not isinstance(shard_metadata, dict):
        raise RuntimeError(f"Resume shard `{path}` has no metadata mapping.")
    mismatched_metadata = [
        key
        for key, value in expected_metadata.items()
        if shard_metadata.get(key) != value
    ]
    if mismatched_metadata:
        raise RuntimeError(
            f"Resume shard `{path}` does not match the current export contract: "
            f"{mismatched_metadata}."
        )
    episode_payloads = payload.get("episodes")
    if not isinstance(episode_payloads, list) or len(episode_payloads) != 1:
        raise RuntimeError(f"Resume shard `{path}` must contain exactly one episode.")
    pooled_episode = PooledFeatureEpisode.from_payload(episode_payloads[0])
    expected_values = {
        "episode_id": episode.episode_id,
        "split": probe_split(episode.index),
        "camera_ids": expected_cameras,
        "source_index": episode.index,
        "source_frame_count": episode.steps,
        "source_file": episode.source_file,
        "recording_folder": episode.recording_folder,
    }
    observed_values = {
        "episode_id": pooled_episode.episode_id,
        "split": pooled_episode.split,
        "camera_ids": pooled_episode.camera_ids,
        "source_index": pooled_episode.metadata.get("source_index"),
        "source_frame_count": pooled_episode.metadata.get("source_frame_count"),
        "source_file": pooled_episode.metadata.get("source_file"),
        "recording_folder": pooled_episode.metadata.get("recording_folder"),
    }
    mismatched_episode = [
        key
        for key, value in expected_values.items()
        if observed_values.get(key) != value
    ]
    if mismatched_episode:
        raise RuntimeError(
            f"Resume shard `{path}` does not match source episode "
            f"`{episode.episode_id}`: {mismatched_episode}."
        )
    latent_shape = pooled_episode.metadata.get("latent_shape")
    if not isinstance(latent_shape, list):
        raise RuntimeError(f"Resume shard `{path}` has no latent shape metadata.")
    return {
        "episode_id": episode.episode_id,
        "source_index": episode.index,
        "split": pooled_episode.split,
        "source_frames": episode.steps,
        "latent_ticks": pooled_episode.ticks,
        "latent_shape": latent_shape,
        "path": str(path),
        "sha256": file_sha256(path),
        "elapsed_seconds": 0.0,
        "peak_cuda_memory_gib": None,
        "status": "reused",
    }


def export_droid_wan_probe(config: WanProbeExportConfig) -> dict[str, Any]:
    output_dir = ensure_dir(config.output_dir)
    data_dir = Path(config.data_dir)
    vae_path = Path(config.vae_path)
    started = time.time()
    source_checksums = _source_checksums(
        data_dir,
        include_tfrecords=bool(config.hash_source_shards),
    )
    vae_sha256 = file_sha256(vae_path)
    shard_metadata = _probe_shard_metadata(config, vae_sha256, source_checksums)

    # Constructing the iterator disables TensorFlow CUDA before the Wan model is loaded.
    episode_iterator: Iterator[DroidRLDSEpisode] = iter_droid_rlds_episodes(
        data_dir,
        cameras=config.cameras,
        max_episodes=config.max_episodes,
    )
    vae = None
    rows: list[dict[str, Any]] = []
    for episode in episode_iterator:
        path = output_dir / f"episode-{episode.index:05d}.pt"
        if config.resume and path.is_file():
            rows.append(
                _reused_episode_row(
                    path,
                    episode,
                    expected_metadata=shard_metadata,
                    expected_cameras=config.cameras,
                )
            )
            print(
                f"Reused {episode.episode_id}: frames={episode.steps}, "
                f"ticks={rows[-1]['latent_ticks']}, sha256={rows[-1]['sha256'][:12]}",
                flush=True,
            )
            continue
        if path.exists() and not config.resume:
            raise FileExistsError(f"Probe shard already exists: {path}.")
        device = torch.device(config.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Wan probe requested CUDA, but PyTorch cannot see CUDA.")
            torch.cuda.reset_peak_memory_stats(device)
        if vae is None:
            vae = _load_wan_vae(config)
        episode_started = time.time()
        pooled_episode = _encode_episode(episode, vae, config)
        info = write_pooled_feature_shard(
            path,
            [pooled_episode],
            metadata=shard_metadata,
        )
        rows.append(
            {
                "episode_id": episode.episode_id,
                "source_index": episode.index,
                "split": pooled_episode.split,
                "source_frames": episode.steps,
                "latent_ticks": pooled_episode.ticks,
                "latent_shape": pooled_episode.metadata["latent_shape"],
                "path": str(path),
                "sha256": info.sha256,
                "elapsed_seconds": time.time() - episode_started,
                "peak_cuda_memory_gib": (
                    float(torch.cuda.max_memory_allocated(device) / 1024**3)
                    if device.type == "cuda"
                    else None
                ),
                "status": "exported",
            }
        )
        print(
            f"Exported {episode.episode_id}: frames={episode.steps}, "
            f"ticks={pooled_episode.ticks}, elapsed={rows[-1]['elapsed_seconds']:.1f}s, "
            f"peak_cuda_gib={rows[-1]['peak_cuda_memory_gib']}",
            flush=True,
        )
        del pooled_episode
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "schema": WAN_PROBE_EXPORT_SCHEMA,
        "config": asdict(config),
        "vae_sha256": vae_sha256,
        "source_checksums": source_checksums,
        "episodes": rows,
        "elapsed_seconds": time.time() - started,
    }
    save_json(output_dir / "export_manifest.json", payload)
    return payload


def export_config_from_mapping(mapping: dict[str, Any]) -> WanProbeExportConfig:
    values = dict(mapping)
    values["cameras"] = tuple(values.get("cameras", DROID_CAMERA_KEYS))
    return WanProbeExportConfig(**values)
