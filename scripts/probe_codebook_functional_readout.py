#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict

from codewam.codebook_eval.functional_readout import (
    probe_codebook_functional_readout,
)


def _parse_artifact(value: str) -> tuple[str, str, str]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError(
            "Artifact must use RUN:FAMILY=PATH."
        )
    identity, path = value.split("=", 1)
    run, family = identity.split(":", 1)
    if not run or not family or not path:
        raise argparse.ArgumentTypeError(
            "Artifact must use nonempty RUN:FAMILY=PATH."
        )
    return run, family, path


def _parse_depth(value: str) -> tuple[str, int]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Depth must use FAMILY=DEPTH.")
    family, depth = value.split("=", 1)
    try:
        parsed = int(depth)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Depth must be an integer."
        ) from exc
    if not family or parsed <= 0:
        raise argparse.ArgumentTypeError(
            "Depth must use nonempty FAMILY and positive DEPTH."
        )
    return family, parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare proprio/H/C/H+C action readouts across independent "
            "frozen RQ codebook seeds."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pooled-shards", nargs="+", required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        type=_parse_artifact,
    )
    parser.add_argument(
        "--code-depth",
        action="append",
        required=True,
        type=_parse_depth,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--subset-seed", type=int, default=20260729)
    parser.add_argument(
        "--alpha",
        nargs="+",
        type=float,
        default=(1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--center-block-size", type=int, default=1024)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    runs: dict[str, dict[str, str]] = defaultdict(dict)
    for run, family, path in args.artifact:
        if family in runs[run]:
            parser.error(f"Duplicate artifact {run}:{family}.")
        runs[run][family] = path
    depths = {}
    for family, depth in args.code_depth:
        if family in depths:
            parser.error(f"Duplicate code depth for {family}.")
        depths[family] = depth

    report = probe_codebook_functional_readout(
        manifest_path=args.manifest,
        pooled_shards=args.pooled_shards,
        runs=dict(runs),
        output_dir=args.output_dir,
        code_depths=depths,
        train_fraction=args.train_fraction,
        subset_seed=args.subset_seed,
        alpha_candidates=tuple(args.alpha),
        device=args.device,
        cpu_threads=args.cpu_threads,
        batch_size=args.batch_size,
        center_block_size=args.center_block_size,
        resume=not args.no_resume,
    )
    print(
        f"train={report['train_vectors']:,} "
        f"alpha={report['selected_alpha']:g}"
    )
    for summary in report["summaries"]:
        values = ", ".join(
            f"{row['run']}={100.0 * row['p3_minus_p1']:+.3f}pp"
            for row in summary["seed_rows"]
        )
        print(
            f"{summary['split']}: {summary['increment_status']} {values}"
        )


if __name__ == "__main__":
    main()
