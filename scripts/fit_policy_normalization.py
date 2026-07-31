#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import torch

from codewam.codebook_eval.shards import file_sha256
from codewam.data import (
    JointWindowCache,
    PolicyNormalizer,
    create_policy_normalization_contract,
    encode_droid_actions,
    encode_droid_proprio,
    moments_from_sums,
    write_policy_normalization,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit immutable train-only CodeWAM policy normalization from each "
            "referenced source step exactly once."
        )
    )
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def _aggregate(
    values: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    values = values.double()
    if values.ndim != 2 or not torch.isfinite(values).all():
        raise RuntimeError("Policy-normalization source values are invalid.")
    return (
        int(values.shape[0]),
        values.sum(dim=0),
        values.square().sum(dim=0),
    )


def _aggregate_shard(
    work: tuple[str, str, tuple[str, ...]],
) -> dict[str, Any]:
    cache_dir, relative, expected_ids = work
    torch.set_num_threads(1)
    payload = torch.load(
        Path(cache_dir) / relative,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    expected = set(expected_ids)
    selected = {
        str(row["episode_id"]): row
        for row in payload["episodes"]
        if str(row["episode_id"]) in expected
    }
    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        raise RuntimeError(f"Normalization shard misses episodes: {missing[:8]}.")
    action_rows = []
    proprio_rows = []
    for episode_id in expected_ids:
        episode = selected[episode_id]
        if str(episode["split"]) != "train":
            raise RuntimeError("Normalization received a non-train episode.")
        actions = episode["source_actions"].float()
        valid = episode["source_action_valid"].bool()
        proprio = episode["source_proprio"].float()
        if (
            actions.ndim != 2
            or actions.shape[1] != 7
            or tuple(valid.shape) != tuple(actions.shape[:1])
            or proprio.ndim != 2
            or proprio.shape[1] != 14
        ):
            raise RuntimeError("Normalization source dimensions changed.")
        action_rows.append(encode_droid_actions(actions[valid]))
        proprio_rows.append(encode_droid_proprio(proprio))
    actions = torch.cat(action_rows, dim=0)
    proprio = torch.cat(proprio_rows, dim=0)
    action_count, action_sum, action_squared = _aggregate(actions)
    proprio_count, proprio_sum, proprio_squared = _aggregate(proprio)
    return {
        "segments": len(expected_ids),
        "action_count": action_count,
        "action_sum": action_sum.tolist(),
        "action_squared": action_squared.tolist(),
        "proprio_count": proprio_count,
        "proprio_sum": proprio_sum.tolist(),
        "proprio_squared": proprio_squared.tolist(),
    }


def _episode_rows(cache_dir: Path) -> list[dict[str, Any]]:
    rows = []
    with (cache_dir / "episodes.jsonl").open(encoding="utf-8") as source:
        for line in source:
            rows.append(json.loads(line))
    return rows


def _work_items(
    cache: JointWindowCache,
) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    referenced = set(cache.referenced_episode_ids)
    rows = _episode_rows(cache.cache_dir)
    locators = {str(row["episode_id"]): row for row in rows}
    unknown = sorted(referenced - set(locators))
    if unknown:
        raise RuntimeError(f"Normalization sees unknown segments: {unknown[:8]}.")
    grouped: dict[str, list[str]] = {}
    seen = set()
    for row in rows:
        episode_id = str(row["episode_id"])
        if episode_id not in referenced or str(row["split"]) != "train":
            continue
        seen.add(episode_id)
        grouped.setdefault(str(row["episode_shard"]), []).append(episode_id)
    expected_train = {
        episode_id
        for episode_id in referenced
        if str(locators[episode_id]["split"]) == "train"
    }
    if seen != expected_train:
        raise RuntimeError("Normalization could not resolve every train segment.")
    return tuple(
        (str(cache.cache_dir), relative, tuple(sorted(episode_ids)))
        for relative, episode_ids in sorted(grouped.items())
    )


def _sum_results(
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not results:
        raise RuntimeError("Normalization found no referenced train segments.")
    combined = {
        "segments": 0,
        "action_count": 0,
        "action_sum": torch.zeros(10, dtype=torch.float64),
        "action_squared": torch.zeros(10, dtype=torch.float64),
        "proprio_count": 0,
        "proprio_sum": torch.zeros(17, dtype=torch.float64),
        "proprio_squared": torch.zeros(17, dtype=torch.float64),
    }
    for result in results:
        combined["segments"] += int(result["segments"])
        combined["action_count"] += int(result["action_count"])
        combined["proprio_count"] += int(result["proprio_count"])
        for name in (
            "action_sum",
            "action_squared",
            "proprio_sum",
            "proprio_squared",
        ):
            combined[name] += torch.tensor(result[name], dtype=torch.float64)
    return combined


def main() -> None:
    args = _parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive.")
    cache = JointWindowCache(args.cache_dir)
    if (
        int(cache.contract["action_dim"]) != 7
        or int(cache.contract["proprio_dim"]) != 14
    ):
        raise RuntimeError("DROID policy normalization needs raw dimensions 7/14.")
    contract = create_policy_normalization_contract(
        joint_cache_contract_hash=cache.contract["contract_hash"],
        joint_cache_summary_sha256=file_sha256(args.cache_dir / "summary.json"),
        implementation_sha256={
            "policy_normalization": file_sha256(
                Path(__file__).parents[1]
                / "codewam"
                / "data"
                / "policy_normalization.py"
            ),
            "fit_policy_normalization": file_sha256(Path(__file__)),
        },
    )
    summary_path = args.output_dir / "summary.json"
    if summary_path.is_file():
        existing = PolicyNormalizer(
            args.output_dir,
            expected_joint_cache_contract_hash=cache.contract["contract_hash"],
        )
        if existing.contract != contract:
            raise RuntimeError("Existing normalization uses another implementation.")
        print(summary_path.read_text(encoding="utf-8"), end="")
        return
    work = _work_items(cache)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(_aggregate_shard, work, chunksize=1))
    combined = _sum_results(results)
    action_mean, action_std = moments_from_sums(
        combined["action_count"],
        combined["action_sum"],
        combined["action_squared"],
    )
    proprio_mean, proprio_std = moments_from_sums(
        combined["proprio_count"],
        combined["proprio_sum"],
        combined["proprio_squared"],
    )
    summary = write_policy_normalization(
        args.output_dir,
        contract=contract,
        action_mean=action_mean,
        action_std=action_std,
        proprio_mean=proprio_mean,
        proprio_std=proprio_std,
        action_rows=combined["action_count"],
        proprio_rows=combined["proprio_count"],
        source_segments=combined["segments"],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
