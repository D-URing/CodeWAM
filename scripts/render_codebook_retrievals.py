#!/usr/bin/env python3
from __future__ import annotations

import argparse

from codewam.codebook_eval.retrieval import (
    render_codebook_retrieval_montages,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render exact DROID RGB histories for frozen RQ representative "
            "samples from held-out evaluation reports."
        )
    )
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--pooled-manifest", required=True)
    parser.add_argument("--droid-data-dir", required=True)
    parser.add_argument(
        "--evaluation-report",
        action="append",
        required=True,
        help=(
            "Held-out evaluation_report.json; repeat for separate Q2/Q3/Q5 "
            "reports."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera", default="wrist_image_left")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("test",),
        choices=("val", "test"),
    )
    parser.add_argument("--levels", nargs="+", type=int, default=(1,))
    parser.add_argument("--representatives-per-code", type=int, default=3)
    parser.add_argument("--thumbnail-width", type=int, default=128)
    parser.add_argument("--thumbnail-height", type=int, default=72)
    parser.add_argument("--difference-gain", type=float, default=3.0)
    parser.add_argument(
        "--diversity-by",
        choices=("none", "parent", "scene"),
        default="scene",
        help=(
            "Keep nearest anchors from distinct scenes by default; use parent "
            "for episode diversity or none for raw nearest neighbors."
        ),
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    report = render_codebook_retrieval_montages(
        source_manifest_path=args.source_manifest,
        pooled_manifest_path=args.pooled_manifest,
        droid_data_dir=args.droid_data_dir,
        evaluation_report_paths=args.evaluation_report,
        output_dir=args.output_dir,
        camera=args.camera,
        splits=args.splits,
        levels=args.levels,
        representatives_per_code=args.representatives_per_code,
        thumbnail_size=(
            args.thumbnail_width,
            args.thumbnail_height,
        ),
        difference_gain=args.difference_gain,
        diversity_by=args.diversity_by,
        resume=not args.no_resume,
    )
    totals = report["totals"]
    print(
        "Rendered "
        f"{totals['clips']} clips from "
        f"{totals['unique_source_episodes']} source episodes into "
        f"{totals['montages']} montages."
    )
    for montage in report["montages"]:
        print(
            f"{montage['family']} {montage['split']} "
            f"L{montage['level']}: {montage['path']}"
        )
    print(f"Wrote retrieval report to {args.output_dir}.")


if __name__ == "__main__":
    main()
