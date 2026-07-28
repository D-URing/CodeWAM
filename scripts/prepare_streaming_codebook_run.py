#!/usr/bin/env python3
from __future__ import annotations

import argparse

from codewam.codebook_eval.run_setup import prepare_droid_streaming_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compile a finalized DROID pooled export into locked streaming "
            "RQ train/evaluation configs."
        )
    )
    parser.add_argument("--pooled-export-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-dir")
    parser.add_argument("--pool", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument(
        "--cameras",
        nargs="+",
        help="Ordered subset of cameras from the pooled export; defaults to all.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    result = prepare_droid_streaming_run(
        args.pooled_export_dir,
        args.output_dir,
        config_dir=args.config_dir,
        pool=args.pool,
        k=args.k,
        levels=args.levels,
        device=args.device,
        seed=args.seed,
        camera_ids=args.cameras,
    )
    print(
        f"Prepared {result['dataset']}: shards={result['pooled_shards']} "
        f"g={result['pool']} K={result['k']} L={result['levels']}"
        f" cameras={','.join(result['camera_ids'])}"
    )
    print(f"Train config: {result['train_config']}")
    print(f"Evaluation config: {result['evaluation_config']}")


if __name__ == "__main__":
    main()
