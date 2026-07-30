#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

from codewam.data.joint_cache import JointWindowConfig
from codewam.data.joint_cache_export import (
    JointCacheExportConfig,
    export_joint_window_cache,
    finalize_exported_joint_cache,
)


def _artifact_mapping(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        try:
            family, path = value.split("=", 1)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Artifacts must use FAMILY=/path/to/codebook.pt."
            ) from exc
        if family in result:
            raise argparse.ArgumentTypeError(f"Duplicate artifact `{family}`.")
        result[family] = path
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export deduplicated unpooled Wan latents, source-rate controls and "
            "verified Q2/Q3/Q5 windows from an official DROID manifest."
        )
    )
    parser.add_argument("--source-manifest")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--endpoint-audit")
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--chart-name", default="droid")
    parser.add_argument("--vae-path")
    parser.add_argument("--fastwam-src")
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
    )
    parser.add_argument("--nominal-fps", type=float, default=15.0)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument(
        "--device",
        default=(
            f"cuda:{os.environ['LOCAL_RANK']}"
            if "LOCAL_RANK" in os.environ
            else "cuda"
        ),
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--rank", type=int, default=int(os.getenv("RANK", "0")))
    parser.add_argument(
        "--world-size",
        type=int,
        default=int(os.getenv("WORLD_SIZE", "1")),
    )
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--state-latent-ticks", type=int, default=8)
    parser.add_argument("--proprio-history-steps", type=int, default=16)
    parser.add_argument("--past-action-steps", type=int, default=16)
    parser.add_argument("--window-stride-ticks", type=int, default=1)
    parser.add_argument("--max-source-shards", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.finalize_only:
        report = finalize_exported_joint_cache(args.output_dir)
    else:
        required = {
            "--source-manifest": args.source_manifest,
            "--data-dir": args.data_dir,
            "--endpoint-audit": args.endpoint_audit,
            "--vae-path": args.vae_path,
            "--fastwam-src": args.fastwam_src,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise SystemExit(f"Missing required export options: {missing}.")
        report = export_joint_window_cache(
            JointCacheExportConfig(
                source_manifest=args.source_manifest,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                endpoint_audit=args.endpoint_audit,
                artifact_paths=_artifact_mapping(args.artifact),
                chart_name=args.chart_name,
                vae_path=args.vae_path,
                fastwam_src=args.fastwam_src,
                rank=args.rank,
                world_size=args.world_size,
                cameras=tuple(
                    dict.fromkeys(
                        args.camera
                        or (
                            "exterior_image_1_left",
                            "wrist_image_left",
                        )
                    )
                ),
                nominal_fps=args.nominal_fps,
                image_height=args.image_height,
                image_width=args.image_width,
                device=args.device,
                dtype=args.dtype,
                window=JointWindowConfig(
                    action_horizon=args.action_horizon,
                    state_latent_ticks=args.state_latent_ticks,
                    proprio_history_steps=args.proprio_history_steps,
                    past_action_steps=args.past_action_steps,
                    window_stride_ticks=args.window_stride_ticks,
                ),
                resume=not args.no_resume,
                max_source_shards=args.max_source_shards,
            )
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
