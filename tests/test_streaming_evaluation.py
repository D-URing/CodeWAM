from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from codewam.codebook_eval.evaluation import evaluate_frozen_codebooks
from codewam.codebook_eval.pipeline import (
    create_synthetic_streaming_fixture,
    train_streaming_codebooks,
)


class StreamingEvaluationTests(unittest.TestCase):
    def test_frozen_artifacts_evaluate_and_resume_on_val_and_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_config_path = create_synthetic_streaming_fixture(root)
            training_config = OmegaConf.load(training_config_path)
            training_config.training.device = "cpu"
            training_config.training.max_iters = 2
            training_config.training.patience = 1
            OmegaConf.save(training_config, training_config_path)
            trained = train_streaming_codebooks(training_config_path)

            evaluation_config = OmegaConf.create(
                {
                    "output_dir": str(root / "heldout"),
                    "input": {
                        "pooled_shards": [str(root / "pooled/*.pt")],
                        "manifest": str(root / "manifest.jsonl"),
                        "group_by": "scene",
                    },
                    "metadata": {"dataset": "synthetic"},
                    "artifacts": {
                        row["family"]: row["artifact"] for row in trained
                    },
                    "evaluation": {
                        "splits": ["val", "test"],
                        "device": "cpu",
                        "cpu_threads": 1,
                        "batch_size": 32,
                        "center_block_size": 8,
                        "resume": True,
                    },
                }
            )
            evaluation_config_path = root / "heldout.yaml"
            OmegaConf.save(evaluation_config, evaluation_config_path)

            first = evaluate_frozen_codebooks(evaluation_config_path)
            resumed = evaluate_frozen_codebooks(evaluation_config_path)

        self.assertEqual(first, resumed)
        self.assertEqual(len(first["rows"]), 6)
        self.assertEqual(
            {(row["family"], row["split"]) for row in first["rows"]},
            {
                (family, split)
                for family in ("Q2", "Q3", "Q5")
                for split in ("val", "test")
            },
        )
        for row in first["rows"]:
            self.assertGreater(row["vectors"], 0)
            self.assertEqual(len(row["residual_mse"]), 4)
            self.assertEqual(len(row["code_usage"]), 3)
            self.assertEqual(len(row["temporal"]), 3)
            self.assertEqual(len(row["representatives"]), 3)
            self.assertGreater(row["temporal"][0]["adjacent_pairs"], 0)
            self.assertTrue(
                any(
                    code["samples"]
                    for level in row["representatives"]
                    for code in level["codes"]
                )
            )
            self.assertGreater(row["joint_usage"]["active_tuples"], 0)


if __name__ == "__main__":
    unittest.main()
