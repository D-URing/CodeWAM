#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from codewam.data.droid_endpoint import audit_droid_endpoints
from codewam.data.droid_rlds import iter_droid_rlds_episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify official DROID RLDS observation/action endpoint semantics "
            "without loading a model."
        )
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-episodes", type=int, default=32)
    parser.add_argument(
        "--camera",
        default="wrist_image_left",
        help="Only this camera is decoded while the low-dimensional audit runs.",
    )
    parser.add_argument("--minimum-alignment-margin", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = iter_droid_rlds_episodes(
        args.data_dir,
        cameras=(args.camera,),
        max_episodes=args.max_episodes,
        split="train",
    )
    report = audit_droid_endpoints(
        episodes,
        minimum_alignment_margin=args.minimum_alignment_margin,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"endpoint audit {report['verdict']}: "
        f"episodes={report['episodes']} steps={report['steps']} "
        f"report={output}"
    )
    if report["verdict"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
