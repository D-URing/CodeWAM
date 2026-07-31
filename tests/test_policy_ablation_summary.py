from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codewam.experiments.policy_ablation import (
    POLICY_ABLATION_PROTOCOL_SCHEMA,
    POLICY_ABLATION_SCHEMA,
)
from codewam.experiments.policy_ablation_summary import (
    POLICY_METRICS,
    summarize_policy_ablation_reports,
)


def _write_run(
    root: Path,
    *,
    seed: int,
    learning_rate: float = 2e-4,
) -> Path:
    run_dir = root / f"seed-{seed}"
    run_dir.mkdir(parents=True)
    protocol_hash = f"protocol-{seed}"
    protocol = {
        "schema": POLICY_ABLATION_PROTOCOL_SCHEMA,
        "protocol_hash": protocol_hash,
        "cache": {"contract_hash": "cache"},
        "evaluation_subsets": {
            "val": {"window_ids_hash": f"val-{seed}"},
            "test": {"window_ids_hash": f"test-{seed}"},
        },
        "run_config": {
            "seed": seed,
            "learning_rate": learning_rate,
        },
        "implementation_sha256": {"policy": "implementation"},
    }
    evaluation = {}
    for variant_index, variant in enumerate(("C0", "C1", "C2")):
        evaluation[variant] = {}
        for split in ("val", "test"):
            evaluation[variant][split] = {
                metric: 1.0 + variant_index * 0.1 + seed / 1000
                for metric in POLICY_METRICS
            }
    episode = {
        split: {
            name: {
                "episodes": 10,
                "mean_delta_candidate_minus_baseline": 0.1,
                "ci95": [0.01, 0.2],
            }
            for name in ("C1-vs-C0", "C2-vs-C1", "C2-vs-C0")
        }
        for split in ("val", "test")
    }
    report = {
        "schema": POLICY_ABLATION_SCHEMA,
        "protocol_hash": protocol_hash,
        "training": {
            variant: {"optimizer_steps": 200}
            for variant in ("C0", "C1", "C2")
        },
        "evaluation": evaluation,
        "paired_episode_comparisons": episode,
    }
    (run_dir / "protocol.json").write_text(
        json.dumps(protocol),
        encoding="utf-8",
    )
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


class PolicyAblationSummaryTests(unittest.TestCase):
    def test_matching_seeds_produce_descriptive_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [_write_run(root, seed=seed) for seed in (7, 19, 31)]
            summary = summarize_policy_ablation_reports(paths)

        self.assertEqual(summary["seeds"], [7, 19, 31])
        comparison = summary["comparisons"]["test"]["C1-vs-C0"][
            "flow_mse"
        ]
        self.assertAlmostEqual(comparison["mean_delta"], 0.1)
        self.assertEqual(comparison["favorable_seed_count"], 0)

    def test_non_seed_protocol_difference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [
                _write_run(
                    root,
                    seed=seed,
                    learning_rate=1e-3 if seed == 31 else 2e-4,
                )
                for seed in (7, 19, 31)
            ]
            with self.assertRaisesRegex(RuntimeError, "differ"):
                summarize_policy_ablation_reports(paths)


if __name__ == "__main__":
    unittest.main()
