#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from codewam.codebook_eval.droid_pooled_export import (
    DroidPooledExportConfig,
    export_droid_pooled_features,
    finalize_droid_pooled_export,
)


def _default_device() -> str:
    local_rank = os.environ.get("LOCAL_RANK")
    return f"cuda:{local_rank}" if local_rank is not None else "cuda"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export canonical DROID keep-range segments to pooled Wan features."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export")
    export.add_argument("--source-manifest", required=True)
    export.add_argument("--data-dir", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--vae-path", required=True)
    export.add_argument("--fastwam-src", required=True)
    export.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    export.add_argument(
        "--world-size",
        type=int,
        default=int(os.environ.get("WORLD_SIZE", "1")),
    )
    export.add_argument(
        "--cameras",
        nargs="+",
        default=("exterior_image_1_left", "wrist_image_left"),
    )
    export.add_argument("--nominal-fps", type=float, default=15.0)
    export.add_argument("--image-height", type=int, default=224)
    export.add_argument("--image-width", type=int, default=224)
    export.add_argument("--device", default=_default_device())
    export.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    export.add_argument("--no-resume", action="store_true")

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--source-manifest", required=True)
    finalize.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    if args.command == "export":
        report = export_droid_pooled_features(
            DroidPooledExportConfig(
                source_manifest=args.source_manifest,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                vae_path=args.vae_path,
                fastwam_src=args.fastwam_src,
                rank=args.rank,
                world_size=args.world_size,
                cameras=tuple(args.cameras),
                nominal_fps=args.nominal_fps,
                image_height=args.image_height,
                image_width=args.image_width,
                device=args.device,
                dtype=args.dtype,
                resume=not args.no_resume,
            )
        )
        print(
            f"Rank {report['rank']}/{report['world_size']}: "
            f"outputs={len(report['outputs'])} "
            f"elapsed={report['elapsed_seconds']:.1f}s"
        )
    else:
        report = finalize_droid_pooled_export(
            args.source_manifest,
            args.output_dir,
        )
        print(
            f"Finalized pooled manifest: "
            f"episodes={report['pooled_manifest']['episodes']} "
            f"ticks={report['latent_ticks']} "
            f"path={report['pooled_manifest']['path']}"
        )


if __name__ == "__main__":
    main()
