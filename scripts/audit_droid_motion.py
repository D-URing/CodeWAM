#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from codewam.codebook_eval.droid_motion_audit import (
    DroidMotionAuditConfig,
    aggregate_motion_audit,
    select_droid_motion_audit_manifest,
)
from codewam.codebook_eval.manifest import EpisodeManifest
from codewam.data.droid_manifest import write_json_report
from codewam.data.droid_rlds import iter_manifest_droid_rlds_episodes


def _format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare DROID visual/action motion inside and outside official keep ranges."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes-per-institution", type=int, default=2)
    parser.add_argument("--split", default="train")
    parser.add_argument("--thumbnail-size", type=int, default=32)
    parser.add_argument("--minimum-idle-steps", type=int, default=16)
    parser.add_argument("--minimum-active-segment-steps", type=int, default=41)
    parser.add_argument("--salt", default="codewam-droid-motion-audit-v1")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=("exterior_image_1_left", "wrist_image_left"),
    )
    args = parser.parse_args()

    config = DroidMotionAuditConfig(
        cameras=tuple(args.cameras),
        episodes_per_institution=args.episodes_per_institution,
        split=args.split,
        thumbnail_size=args.thumbnail_size,
        minimum_idle_steps=args.minimum_idle_steps,
        minimum_active_segment_steps=args.minimum_active_segment_steps,
        salt=args.salt,
    )
    source_manifest_path = Path(args.manifest)
    source_manifest = EpisodeManifest.read_jsonl(source_manifest_path)
    audit_manifest = select_droid_motion_audit_manifest(source_manifest, config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_manifest_path = output_dir / "motion_audit_manifest.jsonl"
    report_path = output_dir / "motion_audit_report.json"
    audit_manifest.write_jsonl(audit_manifest_path)

    episodes = iter_manifest_droid_rlds_episodes(
        args.data_dir,
        audit_manifest,
        cameras=config.cameras,
    )
    report = aggregate_motion_audit(audit_manifest, episodes, config)
    report["source_manifest"] = {
        "path": str(source_manifest_path),
        "fingerprint": source_manifest.fingerprint(),
    }
    report["audit_manifest"] = {
        "path": str(audit_manifest_path),
        "fingerprint": audit_manifest.fingerprint(),
    }
    write_json_report(report_path, report)

    print(
        f"Motion audit: episodes={len(audit_manifest)} "
        f"institutions={len(report['selection']['institutions'])} "
        f"path={report_path}"
    )
    for name, metrics in sorted(report["aggregate"].items()):
        print(
            f"{name}: inside/outside median="
            f"{_format_ratio(metrics['inside_outside_median_ratio'])} "
            f"episode-median={_format_ratio(metrics['episode_median_ratio'])}"
        )


if __name__ == "__main__":
    main()
