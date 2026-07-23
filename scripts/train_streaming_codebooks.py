#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from codewam.codebook_eval.pipeline import (
    create_synthetic_streaming_fixture,
    train_streaming_codebooks,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train canonical causal Q2/Q3/Q5 streaming RQ codebooks."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train from pooled episode shards.")
    train_parser.add_argument("--config", required=True)

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Generate a synthetic pooled cache and run all three codebooks.",
    )
    smoke_parser.add_argument(
        "--output",
        default="runs/codebook_eval/streaming_smoke",
    )

    args = parser.parse_args()
    if args.command == "train":
        rows = train_streaming_codebooks(args.config)
    else:
        config_path = create_synthetic_streaming_fixture(Path(args.output))
        rows = train_streaming_codebooks(config_path)

    for row in rows:
        reductions = ", ".join(
            f"{value:.3f}" for value in row["residual_reduction_by_level"]
        )
        print(
            f"{row['family']}: N={row['normalization_count']} D={row['dim']} "
            f"K={row['k']} L={row['levels']} reductions=[{reductions}]"
        )


if __name__ == "__main__":
    main()
