#!/usr/bin/env python3
from __future__ import annotations

import argparse

from codewam.codebook_eval.temporal_sensitivity import (
    probe_codebook_temporal_sensitivity,
)


def _parse_artifact(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Artifact must use FAMILY=PATH.")
    family, path = value.split("=", 1)
    if not family or not path:
        raise argparse.ArgumentTypeError(
            "Artifact family and path must both be nonempty."
        )
    return family, path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe frozen RQ code sensitivity to history swaps, time reversal, "
            "and static-current temporal counterfactuals."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pooled-shards", nargs="+", required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        type=_parse_artifact,
        help="Frozen RQ artifact as FAMILY=PATH; repeat per family.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="droid-1.0.1")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("val", "test"),
        default=("test",),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--center-block-size", type=int, default=1024)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    families = [family for family, _ in args.artifact]
    if len(families) != len(set(families)):
        parser.error("Artifact family labels must be unique.")

    report = probe_codebook_temporal_sensitivity(
        manifest_path=args.manifest,
        pooled_shards=args.pooled_shards,
        artifacts=dict(args.artifact),
        output_dir=args.output_dir,
        dataset=args.dataset,
        splits=args.splits,
        device=args.device,
        cpu_threads=args.cpu_threads,
        batch_size=args.batch_size,
        center_block_size=args.center_block_size,
        resume=not args.no_resume,
    )
    for row in report["summary_rows"]:
        print(
            f"{row['family']} {row['split']} "
            f"{row['perturbation']}: "
            f"L1_change={row['l1_code_change_fraction']:.4f} "
            f"full_prefix_change="
            f"{row['full_prefix_change_fraction']:.4f} "
            f"mean_level_change="
            f"{row['mean_changed_level_fraction']:.4f} "
            f"cross_penalty="
            f"{row['full_cross_reconstruction_penalty']:.4f}"
        )
    print(f"Wrote temporal sensitivity report to {args.output_dir}.")


if __name__ == "__main__":
    main()
