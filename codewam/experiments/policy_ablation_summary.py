from __future__ import annotations

import copy
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .policy_ablation import (
    POLICY_ABLATION_PROTOCOL_SCHEMA,
    POLICY_ABLATION_SCHEMA,
)


POLICY_ABLATION_MULTI_SEED_SCHEMA = "codewam.policy-ablation-multi-seed.v1"
POLICY_METRICS = (
    "flow_mse",
    "sample_normalized_mae",
    "sample_xyz_mae",
    "sample_angle_coordinate_mae_degrees",
    "sample_gripper_mae",
)
POLICY_COMPARISONS = (
    ("C1-vs-C0", "C1", "C0"),
    ("C2-vs-C1", "C2", "C1"),
    ("C2-vs-C0", "C2", "C0"),
)


def _protocol_identity(protocol: Mapping[str, Any]) -> dict[str, Any]:
    identity = copy.deepcopy(dict(protocol))
    identity.pop("protocol_hash", None)
    identity.pop("evaluation_subsets", None)
    run_config = dict(identity["run_config"])
    run_config.pop("seed", None)
    identity["run_config"] = run_config
    return identity


def _sample_std(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_policy_ablation_reports(
    report_paths: Sequence[str | Path],
    *,
    expected_seeds: Sequence[int] = (7, 19, 31),
) -> dict[str, Any]:
    if not report_paths:
        raise ValueError("Policy multi-seed summary needs at least one report.")
    expected = tuple(sorted(int(seed) for seed in expected_seeds))
    if len(expected) != len(set(expected)):
        raise ValueError("Expected policy-ablation seeds must be unique.")

    runs: dict[int, dict[str, Any]] = {}
    reference_identity: dict[str, Any] | None = None
    reference_cache: str | None = None
    for value in report_paths:
        report_path = Path(value)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        protocol_path = report_path.with_name("protocol.json")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if report.get("schema") != POLICY_ABLATION_SCHEMA:
            raise ValueError(f"Unsupported policy report `{report_path}`.")
        if protocol.get("schema") != POLICY_ABLATION_PROTOCOL_SCHEMA:
            raise ValueError(f"Unsupported policy protocol `{protocol_path}`.")
        if report.get("protocol_hash") != protocol.get("protocol_hash"):
            raise RuntimeError(
                f"Policy report/protocol hash mismatch at `{report_path}`."
            )
        seed = int(protocol["run_config"]["seed"])
        if seed in runs:
            raise ValueError(f"Duplicate policy-ablation seed `{seed}`.")
        identity = _protocol_identity(protocol)
        if reference_identity is None:
            reference_identity = identity
            reference_cache = str(protocol["cache"]["contract_hash"])
        elif identity != reference_identity:
            raise RuntimeError(
                "Policy seed reports differ beyond seed and evaluation subset."
            )
        steps = {
            int(row["optimizer_steps"])
            for row in report["training"].values()
        }
        if len(steps) != 1:
            raise RuntimeError(f"Policy seed {seed} used unequal variant budgets.")
        runs[seed] = {
            "report": str(report_path),
            "protocol_hash": str(report["protocol_hash"]),
            "optimizer_steps": steps.pop(),
            "evaluation": report["evaluation"],
            "episode_comparisons": report["paired_episode_comparisons"],
        }
    actual = tuple(sorted(runs))
    if actual != expected:
        raise ValueError(
            f"Policy seeds differ: expected {list(expected)}, got {list(actual)}."
        )

    metrics: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    episode_comparisons: dict[str, Any] = {}
    for split in ("val", "test"):
        metrics[split] = {}
        for metric in POLICY_METRICS:
            metrics[split][metric] = {}
            for variant in ("C0", "C1", "C2"):
                values = [
                    float(runs[seed]["evaluation"][variant][split][metric])
                    for seed in actual
                ]
                metrics[split][metric][variant] = {
                    "seed_values": {
                        str(seed): values[index]
                        for index, seed in enumerate(actual)
                    },
                    "mean": statistics.mean(values),
                    "sample_std": _sample_std(values),
                }
        comparisons[split] = {}
        for name, candidate, baseline in POLICY_COMPARISONS:
            comparisons[split][name] = {}
            for metric in POLICY_METRICS:
                deltas = [
                    float(runs[seed]["evaluation"][candidate][split][metric])
                    - float(runs[seed]["evaluation"][baseline][split][metric])
                    for seed in actual
                ]
                comparisons[split][name][metric] = {
                    "seed_deltas_candidate_minus_baseline": {
                        str(seed): deltas[index]
                        for index, seed in enumerate(actual)
                    },
                    "mean_delta": statistics.mean(deltas),
                    "sample_std": _sample_std(deltas),
                    "favorable_seed_count": sum(value < 0 for value in deltas),
                    "all_seeds_favor_candidate": all(
                        value < 0 for value in deltas
                    ),
                }
        episode_comparisons[split] = {
            name: {
                str(seed): runs[seed]["episode_comparisons"][split][name]
                for seed in actual
            }
            for name, _, _ in POLICY_COMPARISONS
        }

    return {
        "schema": POLICY_ABLATION_MULTI_SEED_SCHEMA,
        "cache_contract_hash": reference_cache,
        "seeds": list(actual),
        "runs": {
            str(seed): {
                "report": runs[seed]["report"],
                "protocol_hash": runs[seed]["protocol_hash"],
                "optimizer_steps": runs[seed]["optimizer_steps"],
            }
            for seed in actual
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "within_seed_episode_bootstrap": episode_comparisons,
        "interpretation": {
            "status": "exploratory",
            "lower_is_better": list(POLICY_METRICS),
            "rule": (
                "Do not claim stable superiority unless direction agrees across "
                "seeds and both fixed-noise flow and sampled-action metrics."
            ),
        },
        "aggregation_note": (
            "Across-seed means and sample standard deviations are descriptive. "
            "Episode-block confidence intervals remain within-seed results."
        ),
    }
