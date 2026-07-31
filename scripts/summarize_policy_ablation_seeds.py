#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from codewam.experiments import summarize_policy_ablation_reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and summarize paired C0/C1/C2 seed reports."
    )
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--expected-seed", action="append", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_policy_ablation_reports(
        args.report,
        expected_seeds=args.expected_seed or (7, 19, 31),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "seeds": summary["seeds"],
                "test_flow": summary["comparisons"]["test"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
