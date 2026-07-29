#!/usr/bin/env python3
from __future__ import annotations

import argparse

from codewam.codebook_eval.concentration import GROUPING_DEFINITIONS
from codewam.codebook_eval.workflow import (
    run_streaming_codebook_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run or resume one streaming RQ candidate through train, "
            "held-out evaluation, association and context concentration."
        )
    )
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--evaluation-config", required=True)
    parser.add_argument("--min-train-count", type=int, default=8)
    parser.add_argument(
        "--groupings",
        nargs="+",
        choices=tuple(GROUPING_DEFINITIONS),
        default=("scene", "institution", "task"),
    )
    args = parser.parse_args()

    summary = run_streaming_codebook_candidate(
        args.train_config,
        args.evaluation_config,
        min_train_count=args.min_train_count,
        groupings=tuple(args.groupings),
    )
    counts = summary["row_counts"]
    print(
        f"Candidate complete: families={','.join(summary['families'])} "
        f"train={counts['train']} heldout={counts['heldout']} "
        f"association={counts['association']} "
        f"concentration={counts['concentration']}"
    )
    print(
        f"Workflow summary: "
        f"{summary['output_dir']}/candidate_workflow.json"
    )


if __name__ == "__main__":
    main()
