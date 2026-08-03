#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

from codewam.data.action_target_export import (
    DroidActionTargetExportConfig,
    export_droid_action_targets,
    finalize_droid_action_targets,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export immutable source-rate DROID action/action_dict sidecars "
            "without decoding RGB or running Wan/RQ."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export")
    export.add_argument("--source-manifest", required=True)
    export.add_argument("--data-dir", required=True)
    export.add_argument("--joint-cache-dir", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
    export.add_argument(
        "--world-size",
        type=int,
        default=int(os.environ.get("WORLD_SIZE", "1")),
    )
    export.add_argument("--no-resume", action="store_true")

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--source-manifest", required=True)
    finalize.add_argument("--joint-cache-dir", required=True)
    finalize.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    if args.command == "export":
        report = export_droid_action_targets(
            DroidActionTargetExportConfig(
                source_manifest=args.source_manifest,
                data_dir=args.data_dir,
                joint_cache_dir=args.joint_cache_dir,
                output_dir=args.output_dir,
                rank=args.rank,
                world_size=args.world_size,
                resume=not args.no_resume,
            )
        )
        print(
            f"Rank {report['rank']}/{report['world_size']}: "
            f"shards={len(report['files'])} "
            f"elapsed={report['elapsed_seconds']:.1f}s"
        )
        return
    summary = finalize_droid_action_targets(
        source_manifest_path=args.source_manifest,
        joint_cache_dir=args.joint_cache_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
