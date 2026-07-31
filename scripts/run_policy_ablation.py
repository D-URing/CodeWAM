#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from codewam.experiments import PolicyAblationRunConfig, run_policy_ablation
from codewam.models import CodeWAMConfig


def _artifact_mapping(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        try:
            family, path = value.split("=", 1)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Artifacts must use FAMILY=/path/to/codebook.pt."
            ) from exc
        if family in result:
            raise argparse.ArgumentTypeError(f"Duplicate artifact `{family}`.")
        result[family] = path
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the immutable equal-budget CodeWAM C0/C1/C2 policy ablation."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/policy/droid_c012_v1.yaml",
        type=Path,
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--language-cache-dir")
    parser.add_argument("--normalization-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--device")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--eval-windows", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> PolicyAblationRunConfig:
    payload = OmegaConf.to_container(
        OmegaConf.load(args.config),
        resolve=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("Policy-ablation config root must be a mapping.")
    values: dict[str, Any] = dict(payload)
    model_values = values.pop("model")
    if not isinstance(model_values, dict):
        raise ValueError("Policy-ablation model config must be a mapping.")
    artifacts = _artifact_mapping(args.artifact)
    if artifacts:
        values["artifact_paths"] = artifacts
    for name, value in (
        ("cache_dir", args.cache_dir),
        ("language_cache_dir", args.language_cache_dir),
        ("normalization_dir", args.normalization_dir),
        ("output_dir", args.output_dir),
        ("device", args.device),
        ("max_steps", args.max_steps),
        ("eval_windows", args.eval_windows),
        ("seed", args.seed),
    ):
        if value is not None:
            values[name] = value
    missing = [
        name
        for name in (
            "cache_dir",
            "language_cache_dir",
            "normalization_dir",
            "output_dir",
            "artifact_paths",
        )
        if not values.get(name)
    ]
    if missing:
        raise SystemExit(
            f"Policy ablation needs config values or CLI overrides for {missing}."
        )
    return PolicyAblationRunConfig(
        **values,
        model=CodeWAMConfig(**model_values),
    )


def main() -> None:
    config = _load_config(_parse_args())
    report = run_policy_ablation(config)
    if int(os.getenv("RANK", "0")) == 0:
        summary = {
            "schema": report["schema"],
            "protocol_hash": report["protocol_hash"],
            "optimizer_steps": {
                variant: row["optimizer_steps"]
                for variant, row in report["training"].items()
            },
            "test": {
                variant: values["test"]
                for variant, values in report["evaluation"].items()
            },
            "test_comparisons": report["paired_episode_comparisons"]["test"],
            "report": str(Path(config.output_dir) / "report.json"),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
