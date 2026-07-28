#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from datetime import timedelta
from pathlib import Path

import torch

from codewam.codebook_eval.pipeline import (
    create_synthetic_streaming_fixture,
    train_streaming_codebooks,
)


def _initialize_distributed() -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False
    if not torch.distributed.is_available():
        raise RuntimeError("This Torch build has no distributed support.")
    if torch.distributed.is_initialized():
        return True
    backend = os.environ.get(
        "CODEWAM_DISTRIBUTED_BACKEND",
        "nccl" if torch.cuda.is_available() else "gloo",
    )
    if backend == "nccl":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend=backend,
        timeout=timedelta(hours=2),
    )
    return True


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
    distributed = _initialize_distributed()
    try:
        if distributed and args.command != "train":
            raise ValueError("Distributed execution only supports the `train` command.")
        if args.command == "train":
            rows = train_streaming_codebooks(args.config)
        else:
            config_path = create_synthetic_streaming_fixture(Path(args.output))
            rows = train_streaming_codebooks(config_path)

        if not distributed or torch.distributed.get_rank() == 0:
            for row in rows:
                reductions = ", ".join(
                    f"{value:.3f}" for value in row["residual_reduction_by_level"]
                )
                print(
                    f"{row['family']}: N={row['normalization_count']} D={row['dim']} "
                    f"K={row['k']} L={row['levels']} reductions=[{reductions}]"
                )
    finally:
        if distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
