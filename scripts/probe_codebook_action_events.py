#!/usr/bin/env python3
from __future__ import annotations

import argparse

from codewam.codebook_eval.action_events import (
    probe_codebook_action_events,
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
            "Measure frozen RQ association with held-out Cartesian and "
            "gripper events."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pooled-shards", nargs="+", required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        type=_parse_artifact,
        help="Frozen codebook as LABEL=PATH; repeat for each family.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=("val", "test"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--center-block-size", type=int, default=1024)
    parser.add_argument("--min-train-count", type=int, default=8)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    labels = [label for label, _ in args.artifact]
    if len(labels) != len(set(labels)):
        parser.error("Artifact labels must be unique.")

    report = probe_codebook_action_events(
        manifest_path=args.manifest,
        pooled_shards=args.pooled_shards,
        artifacts=dict(args.artifact),
        output_dir=args.output_dir,
        splits=tuple(args.splits),
        device=args.device,
        cpu_threads=args.cpu_threads,
        batch_size=args.batch_size,
        center_block_size=args.center_block_size,
        min_train_count=args.min_train_count,
        resume=not args.no_resume,
    )
    for row in report["rows"]:
        if row["prefix_depth"] != row["levels"]:
            continue
        print(
            f"{row['label']} {row['split']} {row['event']} "
            f"L{row['prefix_depth']}: "
            f"gain={row['normalized_accuracy_gain']:.3f} "
            f"balanced={row['balanced_accuracy']:.3f} "
            f"exact={row['exact_prefix_coverage']:.3f}"
        )


if __name__ == "__main__":
    main()
