from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader

from codewam.data import PackageScanV6Dataset


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }
    if value.numel() and value.is_floating_point():
        result["min"] = float(value.min().item())
        result["max"] = float(value.max().item())
    return result


def _save_video_preview(video: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = video.detach().cpu().permute(1, 2, 3, 0)
    if float(frames.min()) < 0.0:
        frames = (frames + 1.0) * 0.5
    frames = (frames.clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()

    num_frames, height, width, _ = frames.shape
    canvas = Image.new("RGB", (width * num_frames, height))
    for i, frame in enumerate(frames):
        canvas.paste(Image.fromarray(frame, mode="RGB"), (i * width, 0))
    canvas.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local Package Scan v6 data smoke demo.")
    parser.add_argument("--root", default="package_scan_v6")
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--global-sample-stride", type=int, default=1)
    parser.add_argument("--action-video-freq-ratio", type=int, default=4)
    parser.add_argument("--camera-size", type=int, nargs=2, metavar=("H", "W"), default=[224, 224])
    parser.add_argument("--concat-multi-camera", choices=["horizontal", "vertical", "none"], default="horizontal")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-windows", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="runs/package_scan_v6_demo")
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = PackageScanV6Dataset(
        root=args.root,
        num_frames=args.num_frames,
        global_sample_stride=args.global_sample_stride,
        action_video_freq_ratio=args.action_video_freq_ratio,
        camera_size=args.camera_size,
        concat_multi_camera=args.concat_multi_camera,
        max_windows=args.max_windows,
        return_camera_stack=True,
    )
    if len(dataset) == 0:
        raise RuntimeError("Package Scan v6 dataset produced zero valid windows.")
    sample_index = min(max(args.index, 0), len(dataset) - 1)
    sample = dataset[sample_index]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    batch = next(iter(loader))

    output_dir = Path(args.output_dir)
    preview_path = output_dir / f"sample_{sample_index:04d}.png"
    if not args.no_preview:
        _save_video_preview(sample["video"], preview_path)

    report = {
        "summary": dataset.summary(),
        "sample_index": sample_index,
        "sample": {
            "video": _tensor_summary(sample["video"]),
            "video_by_camera": _tensor_summary(sample["video_by_camera"]),
            "action": _tensor_summary(sample["action"]),
            "proprio": _tensor_summary(sample["proprio"]),
            "episode_index": sample["episode_index"],
            "start_index": sample["start_index"],
            "video_frame_index": sample["video_frame_index"].tolist(),
            "prompt": sample["prompt"],
            "camera_keys": sample["camera_keys"],
        },
        "batch": {
            key: _tensor_summary(value)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        },
    }
    if not args.no_preview:
        report["preview_path"] = str(preview_path)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
