from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codewam.experiments.gate2 import (
    GATE2_PROTOCOL_SCHEMA,
    GATE2_SCHEMA,
)
from codewam.experiments.gate2_summary import summarize_gate2_reports


def _write_run(
    root: Path,
    *,
    seed: int,
    verdict: str = "pass",
    learning_rate: float = 3e-4,
) -> Path:
    run_dir = root / f"seed-{seed}"
    run_dir.mkdir(parents=True)
    protocol_hash = f"protocol-{seed}"
    protocol = {
        "schema": GATE2_PROTOCOL_SCHEMA,
        "protocol_hash": protocol_hash,
        "cache_contract_hash": "cache",
        "permutation": {
            "seed": seed,
            "permutation_hash": f"permutation-{seed}",
        },
        "run_config": {
            "seed": seed,
            "learning_rate": learning_rate,
            "epochs": 10,
        },
        "implementation_sha256": {"gate2": "implementation"},
    }
    report = {
        "schema": GATE2_SCHEMA,
        "protocol_hash": protocol_hash,
        "cache_contract_hash": "cache",
        "gate": {"verdict": verdict},
        "paired_episode_comparisons": {
            "TRUE-vs-NOACT": {
                "episodes": 100,
                "mean_delta_true_minus_baseline": -0.1 - seed / 1000,
                "ci95": [-0.2, -0.01],
            },
            "TRUE-vs-SHUFFLE": {
                "episodes": 100,
                "mean_delta_true_minus_baseline": -0.2 - seed / 1000,
                "ci95": [-0.3, -0.02],
            },
        },
    }
    (run_dir / "protocol.json").write_text(
        json.dumps(protocol),
        encoding="utf-8",
    )
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return report_path


class Gate2SummaryTests(unittest.TestCase):
    def test_three_matching_passes_produce_one_conservative_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [_write_run(root, seed=seed) for seed in (7, 19, 31)]
            summary = summarize_gate2_reports(paths)

        self.assertEqual(summary["gate"]["verdict"], "pass")
        self.assertEqual(summary["seeds"], [7, 19, 31])
        self.assertTrue(
            summary["comparisons"]["TRUE-vs-NOACT"][
                "all_seed_means_favor_true"
            ]
        )

    def test_non_seed_protocol_difference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [
                _write_run(
                    root,
                    seed=seed,
                    learning_rate=1e-3 if seed == 31 else 3e-4,
                )
                for seed in (7, 19, 31)
            ]
            with self.assertRaisesRegex(RuntimeError, "differ"):
                summarize_gate2_reports(paths)


if __name__ == "__main__":
    unittest.main()
