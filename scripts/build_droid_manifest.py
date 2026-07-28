#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from codewam.codebook_eval.manifest import SplitConfig
from codewam.data.droid_manifest import (
    DEFAULT_SPLIT_FRACTIONS,
    build_droid_manifest,
    manifest_distribution,
    shard_aware_balanced_sample,
    write_json_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Join official DROID metadata, RLDS shard locations, and keep ranges "
            "into a scene-isolated CodeWAM manifest."
        )
    )
    parser.add_argument("--metadata-index", required=True)
    parser.add_argument("--rlds-index", required=True)
    parser.add_argument("--keep-ranges", required=True)
    parser.add_argument("--language-annotations")
    parser.add_argument("--gcs-metadata")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--include-quality-flags", action="store_true")
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--candidate-multiplier", type=float, default=1.25)
    parser.add_argument("--split-salt", default="codewam-droid-1.0.1-scene-v1")
    parser.add_argument("--sample-salt", default="codewam-droid-10k-balanced-v1")
    args = parser.parse_args()

    if args.sample_size < 0:
        raise ValueError("`sample-size` must be non-negative.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = build_droid_manifest(
        metadata_index=args.metadata_index,
        rlds_index=args.rlds_index,
        keep_ranges=args.keep_ranges,
        language_annotations=args.language_annotations,
        gcs_metadata=args.gcs_metadata,
        successful_only=not args.include_failures,
        exclude_quality_flags=not args.include_quality_flags,
    )
    full_manifest = result.manifest.assign_splits(
        SplitConfig(
            group_by="scene",
            stratify_by_task=False,
            salt=args.split_salt,
        )
    )
    full_manifest_path = output_dir / "droid_scene_manifest.jsonl"
    full_report_path = output_dir / "droid_scene_manifest_report.json"
    full_manifest.write_jsonl(full_manifest_path)
    full_report = {
        **result.report,
        "split": {
            "group_by": "scene",
            "fractions": DEFAULT_SPLIT_FRACTIONS,
            "salt": args.split_salt,
        },
        "assigned_manifest": manifest_distribution(full_manifest),
        "output": str(full_manifest_path),
    }
    write_json_report(full_report_path, full_report)
    print(
        f"Full manifest: episodes={len(full_manifest)} "
        f"fingerprint={full_manifest.fingerprint()} path={full_manifest_path}"
    )

    if args.sample_size:
        sample_result = shard_aware_balanced_sample(
            full_manifest,
            args.sample_size,
            salt=args.sample_salt,
            candidate_multiplier=args.candidate_multiplier,
        )
        sampled = sample_result.manifest
        sample_name = f"droid_{args.sample_size}_manifest.jsonl"
        sample_path = output_dir / sample_name
        sample_report_path = output_dir / f"droid_{args.sample_size}_manifest_report.json"
        sampled.write_jsonl(sample_path)
        write_json_report(
            sample_report_path,
            {
                "schema": "codewam.droid-balanced-sample.v1",
                "source_manifest": str(full_manifest_path),
                "source_manifest_fingerprint": full_manifest.fingerprint(),
                "sample_salt": args.sample_salt,
                "split_fractions": DEFAULT_SPLIT_FRACTIONS,
                "shard_selection": sample_result.report,
                "output": str(sample_path),
            },
        )
        print(
            f"Balanced sample: episodes={len(sampled)} "
            f"fingerprint={sampled.fingerprint()} path={sample_path}"
        )


if __name__ == "__main__":
    main()
