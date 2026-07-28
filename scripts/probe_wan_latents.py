#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from codewam.codebook_eval.latent_probe import (
    probe_config_from_mapping,
    run_latent_probe,
)
from codewam.codebook_eval.wan_probe_export import (
    export_config_from_mapping,
    export_droid_wan_probe,
)


def _load_mapping(path: str | Path) -> dict[str, Any]:
    config = OmegaConf.load(path)
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("Wan latent probe config must be a mapping.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export real Wan-VAE pooled latents and run the DROID small-sample probe."
    )
    parser.add_argument(
        "command",
        choices=("export", "analyze", "run"),
        help="Run only latent export, only analysis, or both stages.",
    )
    parser.add_argument("--config", required=True, help="Path to the probe YAML config.")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Override export.max_episodes (useful for a one-episode export audit).",
    )
    args = parser.parse_args()

    payload = _load_mapping(args.config)
    if args.command in {"export", "run"}:
        if "export" not in payload:
            raise ValueError("Probe config has no `export` section.")
        export_mapping = dict(payload["export"])
        if args.max_episodes is not None:
            export_mapping["max_episodes"] = args.max_episodes
        result = export_droid_wan_probe(export_config_from_mapping(export_mapping))
        exported = sum(
            row.get("status") == "exported" for row in result.get("episodes", ())
        )
        reused = sum(row.get("status") == "reused" for row in result.get("episodes", ()))
        print(
            f"Wan export complete: exported={exported}, reused={reused}, "
            f"elapsed={result['elapsed_seconds']:.1f}s"
        )

    if args.command in {"analyze", "run"}:
        if "analysis" not in payload:
            raise ValueError("Probe config has no `analysis` section.")
        result = run_latent_probe(probe_config_from_mapping(payload["analysis"]))
        print(
            "Probe analysis complete: "
            f"episodes={result['episode_counts']}, "
            f"kmeans_runs={len(result['kmeans_runs'])}, "
            f"rq_runs={len(result['rq_runs'])}"
        )
        print(f"Report: {Path(payload['analysis']['output_dir']) / 'report.md'}")


if __name__ == "__main__":
    main()
