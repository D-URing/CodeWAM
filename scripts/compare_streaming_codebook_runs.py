#!/usr/bin/env python3
from __future__ import annotations

import argparse

from codewam.codebook_eval.comparison import compare_streaming_runs


def _parse_run(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Run must use LABEL=PATH.")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Run must use nonempty LABEL=PATH.")
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile comparable train/held-out streaming RQ metrics."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=_parse_run,
        help="Candidate as LABEL=RUN_DIR; repeat for each run.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--families",
        nargs="+",
        help="Optional family filter, for example Q3.",
    )
    args = parser.parse_args()

    report = compare_streaming_runs(
        args.run,
        args.output_dir,
        families=args.families,
    )
    print(
        f"Wrote {len(report['rows'])} comparison rows to "
        f"{args.output_dir}."
    )


if __name__ == "__main__":
    main()
