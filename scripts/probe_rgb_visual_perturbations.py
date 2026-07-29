#!/usr/bin/env python3
from __future__ import annotations

import argparse

from codewam.codebook_eval.visual_perturbations import (
    probe_rgb_visual_perturbations,
)


def _parse_artifact(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Artifact must use FAMILY=PATH.")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError(
            "Artifact must use nonempty FAMILY=PATH."
        )
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-encode controlled RGB perturbations through Wan-VAE and "
            "measure frozen RQ sensitivity."
        )
    )
    parser.add_argument("--source", choices=("droid", "libero"), required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        type=_parse_artifact,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--fastwam-src", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--max-samples", type=int, default=24)
    parser.add_argument("--center-block-size", type=int, default=1024)
    parser.add_argument("--droid-pooled-manifest")
    parser.add_argument("--droid-source-manifest")
    parser.add_argument("--droid-data-dir")
    parser.add_argument("--droid-split", choices=("val", "test"), default="test")
    parser.add_argument("--droid-camera", default="wrist_image_left")
    parser.add_argument("--libero-root")
    parser.add_argument(
        "--libero-suites",
        nargs="+",
        default=(
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
        ),
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    labels = [label for label, _ in args.artifact]
    if len(labels) != len(set(labels)):
        parser.error("Artifact labels must be unique.")

    report = probe_rgb_visual_perturbations(
        source=args.source,
        artifacts=dict(args.artifact),
        output_dir=args.output_dir,
        vae_path=args.vae_path,
        fastwam_src=args.fastwam_src,
        device=args.device,
        dtype=args.dtype,
        image_height=args.image_height,
        image_width=args.image_width,
        max_samples=args.max_samples,
        droid_pooled_manifest=args.droid_pooled_manifest,
        droid_source_manifest=args.droid_source_manifest,
        droid_data_dir=args.droid_data_dir,
        droid_split=args.droid_split,
        droid_camera=args.droid_camera,
        libero_root=args.libero_root,
        libero_suites=tuple(args.libero_suites),
        center_block_size=args.center_block_size,
        resume=not args.no_resume,
    )
    print(
        f"Visual perturbation complete: source={report['source']} "
        f"clips={len(report['clips'])} "
        f"rows={len(report['condition_rows'])}"
    )
    for row in report["condition_rows"]:
        if row["name"] in {
            "uniform_brightness_085",
            "endpoint_translate_x_positive_8",
            "endpoint_scale_110",
        }:
            print(
                f"{row['family']} {row['name']}: "
                f"full_change={row['prefix_change_fraction'][-1]:.3f} "
                f"relative={row['relative_to_natural_next_displacement']:.3f}"
            )


if __name__ == "__main__":
    main()
