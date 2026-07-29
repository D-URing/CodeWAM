from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf

from codewam.codebook_eval.concentration import (
    _categorical_association_metrics,
    probe_frozen_codebook_concentration,
)
from codewam.codebook_eval.pipeline import (
    create_synthetic_streaming_fixture,
    train_streaming_codebooks,
)


class CodebookConcentrationTests(unittest.TestCase):
    def test_categorical_metrics_distinguish_pure_and_independent_codes(
        self,
    ) -> None:
        pure = _categorical_association_metrics(
            Counter({(0, "a"): 50, (1, "b"): 50}),
            capacity=2,
        )
        independent = _categorical_association_metrics(
            Counter(
                {
                    (0, "a"): 25,
                    (0, "b"): 25,
                    (1, "a"): 25,
                    (1, "b"): 25,
                }
            ),
            capacity=2,
        )

        self.assertAlmostEqual(pure["group_information_gain"], 1.0)
        self.assertAlmostEqual(pure["normalized_mutual_information"], 1.0)
        self.assertAlmostEqual(pure["normalized_purity_gain"], 1.0)
        self.assertAlmostEqual(
            independent["group_information_gain"],
            0.0,
        )
        self.assertAlmostEqual(
            independent["normalized_mutual_information"],
            0.0,
        )
        self.assertAlmostEqual(
            independent["normalized_purity_gain"],
            0.0,
        )

    def test_concentration_report_is_heldout_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = create_synthetic_streaming_fixture(root)
            config = OmegaConf.load(config_path)
            config.descriptor.strides = [3]
            config.training.device = "cpu"
            config.training.cpu_threads = 1
            config.training.max_iters = 2
            config.training.patience = 1
            config.training.k = 4
            config.training.reservoir_size = 128
            OmegaConf.save(config, config_path)
            trained = train_streaming_codebooks(config_path)
            artifact = trained[0]["artifact"]
            output = root / "concentration"

            first = probe_frozen_codebook_concentration(
                manifest_path=root / "manifest.jsonl",
                pooled_shards=(root / "pooled/*.pt",),
                artifacts={"synthetic-q3": artifact},
                output_dir=output,
                device="cpu",
                cpu_threads=1,
                batch_size=32,
                center_block_size=4,
            )
            resumed = probe_frozen_codebook_concentration(
                manifest_path=root / "manifest.jsonl",
                pooled_shards=(root / "pooled/*.pt",),
                artifacts={"synthetic-q3": artifact},
                output_dir=output,
                device="cpu",
                cpu_threads=1,
                batch_size=32,
                center_block_size=4,
            )

        self.assertEqual(first, resumed)
        self.assertEqual(len(first["rows"]), 18)
        self.assertEqual(
            {row["split"] for row in first["rows"]},
            {"val", "test"},
        )
        self.assertEqual(
            {row["grouping"] for row in first["rows"]},
            {"scene", "institution", "task"},
        )
        self.assertEqual(
            {row["prefix_depth"] for row in first["rows"]},
            {1, 2, 3},
        )
        self.assertEqual(first["weighting"], "descriptor-tick")


if __name__ == "__main__":
    unittest.main()
