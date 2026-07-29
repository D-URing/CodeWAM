from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from codewam.codebook_eval.comparison import compare_streaming_runs


def _write_run(root: Path) -> None:
    train = [
        {
            "family": "Q3",
            "stride": 3,
            "pool": 4,
            "camera_ids": ["exterior"],
            "k": 2,
            "levels": 2,
            "dim": 12,
            "normalization_count": 100,
            "iterations_per_level": [3, 2],
            "residual_total_reduction": 0.4,
        }
    ]
    heldout = {
        "contract_hash": "contract",
        "rows": [
            {
                "family": "Q3",
                "split": "val",
                "stride": 3,
                "pool": 4,
                "camera_ids": ["exterior"],
                "dimension": 12,
                "k": 2,
                "levels": 2,
                "vectors": 20,
                "episodes": 4,
                "residual_total_reduction": 0.35,
                "residual_reduction_by_level": [0.2, 0.1875],
                "code_usage": [
                    {
                        "active_codes": 2,
                        "dead_fraction": 0.0,
                        "perplexity_fraction": 0.9,
                        "maximum_cluster_fraction": 0.6,
                    },
                    {
                        "active_codes": 2,
                        "dead_fraction": 0.0,
                        "perplexity_fraction": 0.8,
                        "maximum_cluster_fraction": 0.7,
                    },
                ],
                "temporal": [
                    {
                        "same_next_fraction": 0.8,
                        "change_next_fraction": 0.2,
                        "transition_perplexity": 1.5,
                    },
                    {
                        "same_next_fraction": 0.6,
                        "change_next_fraction": 0.4,
                        "transition_perplexity": 2.5,
                    },
                ],
                "joint_usage": {
                    "active_capacity_fraction": 0.75,
                    "perplexity_fraction": 0.5,
                    "maximum_tuple_fraction": 0.3,
                },
            }
        ],
    }
    (root / "heldout").mkdir(parents=True)
    (root / "train_summary.json").write_text(
        json.dumps(train),
        encoding="utf-8",
    )
    (root / "heldout/evaluation_report.json").write_text(
        json.dumps(heldout),
        encoding="utf-8",
    )


class StreamingComparisonTests(unittest.TestCase):
    def test_comparison_writes_machine_and_human_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            output = root / "comparison"
            _write_run(run)

            report = compare_streaming_runs(
                [("exterior-g4", run)],
                output,
            )
            markdown = (output / "comparison_report.md").read_text(
                encoding="utf-8"
            )

        self.assertEqual(len(report["rows"]), 1)
        row = report["rows"][0]
        self.assertAlmostEqual(row["generalization_gap"], 0.05)
        self.assertEqual(row["active_codes_by_level"], [2, 2])
        self.assertIn("exterior-g4", markdown)
        self.assertIn("35.00%", markdown)

    def test_comparison_rejects_train_heldout_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            _write_run(run)
            report_path = run / "heldout/evaluation_report.json"
            heldout = json.loads(report_path.read_text(encoding="utf-8"))
            heldout["rows"][0]["pool"] = 2
            report_path.write_text(json.dumps(heldout), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "metadata differ"):
                compare_streaming_runs([("broken", run)], root / "output")


if __name__ == "__main__":
    unittest.main()
