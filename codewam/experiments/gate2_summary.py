from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .gate2 import GATE2_PROTOCOL_SCHEMA, GATE2_SCHEMA


GATE2_MULTI_SEED_SCHEMA = "codewam.gate2-multi-seed.v1"


def _protocol_identity(protocol: Mapping[str, Any]) -> dict[str, Any]:
    identity = copy.deepcopy(dict(protocol))
    identity.pop("protocol_hash", None)
    identity.pop("permutation", None)
    run_config = dict(identity["run_config"])
    run_config.pop("seed", None)
    identity["run_config"] = run_config
    return identity


def summarize_gate2_reports(
    report_paths: Sequence[str | Path],
    *,
    expected_seeds: Sequence[int] = (7, 19, 31),
) -> dict[str, Any]:
    if not report_paths:
        raise ValueError("Gate2 multi-seed summary needs at least one report.")
    expected = tuple(sorted(int(seed) for seed in expected_seeds))
    if len(expected) != len(set(expected)):
        raise ValueError("Expected Gate2 seeds must be unique.")

    runs: dict[int, dict[str, Any]] = {}
    reference_identity: dict[str, Any] | None = None
    reference_cache: str | None = None
    for value in report_paths:
        report_path = Path(value)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        protocol_path = report_path.with_name("protocol.json")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if report.get("schema") != GATE2_SCHEMA:
            raise ValueError(f"Unsupported Gate2 report `{report_path}`.")
        if protocol.get("schema") != GATE2_PROTOCOL_SCHEMA:
            raise ValueError(f"Unsupported Gate2 protocol `{protocol_path}`.")
        if report.get("protocol_hash") != protocol.get("protocol_hash"):
            raise RuntimeError(
                f"Gate2 report/protocol hash mismatch at `{report_path}`."
            )
        if report.get("cache_contract_hash") != protocol.get(
            "cache_contract_hash"
        ):
            raise RuntimeError(
                f"Gate2 report/cache hash mismatch at `{report_path}`."
            )
        seed = int(protocol["run_config"]["seed"])
        if seed in runs:
            raise ValueError(f"Duplicate Gate2 seed `{seed}`.")
        identity = _protocol_identity(protocol)
        if reference_identity is None:
            reference_identity = identity
            reference_cache = str(report["cache_contract_hash"])
        elif identity != reference_identity:
            raise RuntimeError(
                "Gate2 seed reports differ in more than seed and permutation."
            )
        runs[seed] = {
            "report": str(report_path),
            "protocol_hash": str(report["protocol_hash"]),
            "verdict": str(report["gate"]["verdict"]),
            "comparisons": dict(report["paired_episode_comparisons"]),
        }
    actual = tuple(sorted(runs))
    if actual != expected:
        raise ValueError(
            f"Gate2 seeds differ: expected {list(expected)}, got {list(actual)}."
        )

    comparison_sets = [set(run["comparisons"]) for run in runs.values()]
    if any(values != comparison_sets[0] for values in comparison_sets[1:]):
        raise RuntimeError("Gate2 seed reports contain different comparisons.")
    comparison_names = tuple(sorted(comparison_sets[0]))
    if not comparison_names:
        raise RuntimeError("Gate2 reports share no paired comparisons.")
    comparisons = {}
    for name in comparison_names:
        rows = {seed: runs[seed]["comparisons"][name] for seed in actual}
        means = [
            float(rows[seed]["mean_delta_true_minus_baseline"])
            for seed in actual
        ]
        comparisons[name] = {
            "seed_mean_deltas": {
                str(seed): means[index]
                for index, seed in enumerate(actual)
            },
            "mean_of_seed_means": sum(means) / len(means),
            "minimum_seed_mean": min(means),
            "maximum_seed_mean": max(means),
            "all_seed_means_favor_true": all(value < 0.0 for value in means),
            "seed_ci95": {
                str(seed): list(rows[seed]["ci95"])
                for seed in actual
            },
            "episodes": {
                str(seed): int(rows[seed]["episodes"])
                for seed in actual
            },
        }

    verdicts = {seed: runs[seed]["verdict"] for seed in actual}
    if all(value == "pass" for value in verdicts.values()):
        verdict = "pass"
        reason = "All preregistered seeds independently passed Gate 2."
    elif any(value == "invalid" for value in verdicts.values()):
        verdict = "invalid"
        reason = "At least one preregistered seed was statistically invalid."
    elif any(value == "fail" for value in verdicts.values()):
        verdict = "fail"
        reason = "At least one preregistered seed failed Gate 2."
    else:
        verdict = "inconclusive"
        reason = "The preregistered seed verdicts were not uniformly decisive."
    return {
        "schema": GATE2_MULTI_SEED_SCHEMA,
        "cache_contract_hash": reference_cache,
        "seeds": list(actual),
        "runs": {
            str(seed): {
                key: value
                for key, value in runs[seed].items()
                if key != "comparisons"
            }
            for seed in actual
        },
        "comparisons": comparisons,
        "gate": {
            "verdict": verdict,
            "reason": reason,
            "seed_verdicts": {
                str(seed): verdicts[seed] for seed in actual
            },
            "rule": "all preregistered seeds must independently pass",
        },
        "aggregation_note": (
            "Across-seed means are descriptive; statistical decisions remain "
            "the episode-block confidence intervals within each seed."
        ),
    }
