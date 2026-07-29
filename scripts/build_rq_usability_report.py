#!/usr/bin/env python3
from __future__ import annotations

import argparse

from codewam.codebook_eval.usability import build_rq_usability_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the gated RQ-KMeans visual usability report from frozen "
            "machine-readable evidence."
        )
    )
    parser.add_argument("--comparison-report", required=True)
    parser.add_argument("--family-association-report", required=True)
    parser.add_argument("--retrieval-report", required=True)
    parser.add_argument(
        "--temporal-report",
        action="append",
        required=True,
    )
    parser.add_argument("--causal-audit", required=True)
    parser.add_argument("--visual-report", action="append")
    parser.add_argument("--action-event-report")
    parser.add_argument("--seed-stability-report")
    parser.add_argument("--capacity-comparison-report")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    report = build_rq_usability_report(
        comparison_report=args.comparison_report,
        family_association_report=args.family_association_report,
        retrieval_report=args.retrieval_report,
        temporal_reports=tuple(args.temporal_report),
        causal_audit=args.causal_audit,
        visual_reports=tuple(args.visual_report or ()),
        action_event_report=args.action_event_report,
        seed_stability_report=args.seed_stability_report,
        capacity_comparison_report=args.capacity_comparison_report,
        output_dir=args.output_dir,
        resume=not args.no_resume,
    )
    decision = report["decision"]
    print(
        f"RQ usability verdict: {decision['verdict']} "
        f"blockers={','.join(decision['blocking_gates']) or 'none'} "
        f"limiters={','.join(decision['semantic_limiters']) or 'none'}"
    )
    for gate in report["gates"]:
        print(f"{gate['name']}: {gate['status']}")


if __name__ == "__main__":
    main()
