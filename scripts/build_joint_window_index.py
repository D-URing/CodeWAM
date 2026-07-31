#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from codewam.data.joint_cache import add_compact_joint_window_index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add an mmap-friendly window index to a finalized joint cache."
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild an existing compact index.",
    )
    args = parser.parse_args()
    summary = add_compact_joint_window_index(
        args.cache_dir,
        force=args.force,
    )
    row = summary["indices"]["window_records"]
    print(
        json.dumps(
            {
                "cache_dir": args.cache_dir,
                "windows": summary["windows"],
                "index": row,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
