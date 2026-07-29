#!/usr/bin/env python3
from __future__ import annotations

import argparse

from codewam.codebook_eval.family_association import (
    probe_codebook_family_contributions,
)


def _parse_artifact(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Artifact must use LABEL=PATH.")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError(
            "Artifact must use nonempty LABEL=PATH."
        )
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare aligned single, paired and joint frozen RQ family "
            "association with a train-only additive categorical ridge probe."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pooled-shards", nargs="+", required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        type=_parse_artifact,
        help="Frozen family artifact as LABEL=PATH; repeat two or more times.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=("val", "test"))
    parser.add_argument("--future-offset", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=8.0)
    parser.add_argument("--max-pair-cells", type=int, default=2_000_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--center-block-size", type=int, default=1024)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    labels = [label for label, _ in args.artifact]
    if len(labels) != len(set(labels)):
        parser.error("Artifact labels must be unique.")

    report = probe_codebook_family_contributions(
        manifest_path=args.manifest,
        pooled_shards=args.pooled_shards,
        artifacts=dict(args.artifact),
        output_dir=args.output_dir,
        splits=tuple(args.splits),
        future_offset=args.future_offset,
        ridge=args.ridge,
        max_pair_cells=args.max_pair_cells,
        device=args.device,
        cpu_threads=args.cpu_threads,
        batch_size=args.batch_size,
        center_block_size=args.center_block_size,
        resume=not args.no_resume,
    )
    for row in report["summary_rows"]:
        contributions = ", ".join(
            f"{family}={gain:.4f}"
            for family, gain in row["incremental_gain_by_family"].items()
        )
        print(
            f"{row['split']} {row['target']} L{row['prefix_depth']}: "
            f"full={row['full_normalized_mse_reduction']:.4f} "
            f"best_single={row['best_single_model']}:"
            f"{row['best_single_normalized_mse_reduction']:.4f} "
            f"delta={row['full_gain_over_best_single']:.4f} "
            f"leave-one-out=[{contributions}]"
        )
    print(f"Wrote family contribution report to {args.output_dir}.")


if __name__ == "__main__":
    main()
