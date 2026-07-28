#!/usr/bin/env python3
from __future__ import annotations

import argparse

from codewam.codebook_eval.wan_causality import (
    WanCausalityAuditConfig,
    run_wan_causality_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that real Wan-VAE latent ticks are invariant to unseen "
            "future frames."
        )
    )
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--fastwam-src", required=True)
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=("exterior_image_1_left", "wrist_image_left"),
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--latent-ticks", type=int, default=6)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--atol", type=float, default=1.0e-2)
    parser.add_argument("--rtol", type=float, default=1.0e-2)
    args = parser.parse_args()

    report = run_wan_causality_audit(
        WanCausalityAuditConfig(
            source_manifest=args.source_manifest,
            data_dir=args.data_dir,
            output_path=args.output,
            vae_path=args.vae_path,
            fastwam_src=args.fastwam_src,
            cameras=tuple(args.cameras),
            split=args.split,
            latent_ticks=args.latent_ticks,
            image_height=args.image_height,
            image_width=args.image_width,
            device=args.device,
            dtype=args.dtype,
            atol=args.atol,
            rtol=args.rtol,
        )
    )
    comparison = report["comparison"]
    maximum_error = max(
        float(row["max_abs_error"])
        for row in comparison["rows"]
    )
    print(
        f"Wan causal-prefix audit: passed={report['passed']} "
        f"ticks={comparison['latent_ticks']} max_abs_error={maximum_error:.6g}"
    )
    print(f"Report: {args.output}")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

