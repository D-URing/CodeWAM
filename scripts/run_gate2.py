#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from codewam.experiments import Gate2RunConfig, run_gate2
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
            "Run the preregistered persistence/no-action/true-action/"
            "shuffled-action CodeWAM Gate 2 protocol."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/gate2/droid_joint_v1.yaml",
        type=Path,
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--device")
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> Gate2RunConfig:
    payload = OmegaConf.to_container(
        OmegaConf.load(args.config),
        resolve=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("Gate2 config root must be a mapping.")
    values: dict[str, Any] = dict(payload)
    model_values = values.pop("model")
    if not isinstance(model_values, dict):
        raise ValueError("Gate2 model config must be a mapping.")
    artifacts = _artifact_mapping(args.artifact)
    if artifacts:
        values["artifact_paths"] = artifacts
    for name, value in (
        ("cache_dir", args.cache_dir),
        ("output_dir", args.output_dir),
        ("device", args.device),
        ("max_steps", args.max_steps),
    ):
        if value is not None:
            values[name] = value
    missing = [
        name
        for name in ("cache_dir", "output_dir", "artifact_paths")
        if not values.get(name)
    ]
    if missing:
        raise SystemExit(
            f"Gate2 needs config values or CLI overrides for {missing}."
        )
    return Gate2RunConfig(
        **values,
        model=CodeWAMConfig(**model_values),
    )


def main() -> None:
    args = _parse_args()
    config = _load_config(args)
    report = run_gate2(config)
    if int(os.getenv("RANK", "0")) == 0:
        summary = {
            "schema": report["schema"],
            "protocol_hash": report["protocol_hash"],
            "split_windows": report["split_windows"],
            "optimizer_steps": {
                name: value["optimizer_steps"]
                for name, value in report["training"].items()
            },
            "gate": report["gate"],
            "report": str(Path(config.output_dir) / "report.json"),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
