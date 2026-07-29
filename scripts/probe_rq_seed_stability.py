#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict

from codewam.codebook_eval.seed_stability import (
    probe_rq_seed_stability,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare RQ partitions from independent K-Means seeds on shared "
            "held-out descriptors."
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
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=("val", "test"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--center-block-size", type=int, default=1024)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    runs: dict[str, dict[str, str]] = defaultdict(dict)
    for run, family, path in args.artifact:
        if family in runs[run]:
            parser.error(f"Duplicate artifact {run}:{family}.")
        runs[run][family] = path
    report = probe_rq_seed_stability(
        manifest_path=args.manifest,
        pooled_shards=args.pooled_shards,
        runs=dict(runs),
        output_dir=args.output_dir,
        reference_run=args.reference_run,
        splits=tuple(args.splits),
        device=args.device,
        cpu_threads=args.cpu_threads,
        batch_size=args.batch_size,
        center_block_size=args.center_block_size,
        resume=not args.no_resume,
    )
    for row in report["pair_rows"]:
        nmi = "/".join(
            f"{value['normalized_mutual_information']:.3f}"
            for value in row["levels"]
        )
        print(
            f"{row['split']} {row['family']} "
            f"{row['left_run']} vs {row['right_run']}: "
            f"prefix={row['mapped_prefix_agreement'][-1]:.3f} "
            f"NMI={nmi}"
        )


if __name__ == "__main__":
    main()
