from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf

from codewam.codebook_eval.concentration import (
    _categorical_association_metrics,
    _cross_parent_folds,
    _cross_parent_prediction_metrics,
    probe_frozen_codebook_concentration,
)
from codewam.codebook_eval.manifest import EpisodeRecord
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

    def test_cross_parent_prediction_requires_repeated_codes(self) -> None:
        predictive = _cross_parent_prediction_metrics(
            Counter({(0, "a"): 20, (1, "b"): 20}),
            Counter({(0, "a"): 10, (1, "b"): 10}),
        )
        unseen = _cross_parent_prediction_metrics(
            Counter({(0, "a"): 20, (1, "b"): 20}),
            Counter({(2, "a"): 10, (3, "b"): 10}),
        )

        self.assertTrue(predictive["cross_parent_available"])
        self.assertAlmostEqual(
            predictive["cross_parent_normalized_accuracy_gain"],
            1.0,
        )
        self.assertAlmostEqual(
            predictive["cross_parent_exact_code_coverage"],
            1.0,
        )
        self.assertAlmostEqual(
            unseen["cross_parent_exact_code_coverage"],
            0.0,
        )

    def test_cross_parent_folds_keep_segments_and_splits_isolated(
        self,
    ) -> None:
        records = {
            episode_id: EpisodeRecord(
                dataset="synthetic",
                episode_id=episode_id,
                num_steps=10,
                source_uri=f"memory://{episode_id}",
                scene_id=scene,
                institution_id="lab",
                task_ids=(task,),
                split=split,
                metadata={"parent_episode_id": parent},
            )
            for episode_id, parent, scene, task, split in (
                ("val-a-0", "val-a", "shared", "task", "val"),
                ("val-a-1", "val-a", "shared", "task", "val"),
                ("val-b-0", "val-b", "shared", "task", "val"),
                ("test-a-0", "test-a", "shared", "task", "test"),
                ("test-b-0", "test-b", "shared", "task", "test"),
                ("test-c-0", "test-c", "single", "other", "test"),
            )
        }
        labels = {
            episode_id: {
                "scene": record.scene_id or "__missing__",
            }
            for episode_id, record in records.items()
        }

        folds, summaries = _cross_parent_folds(
            records,
            labels,
            ("scene",),
        )

        self.assertEqual(
            folds["val-a-0"]["scene"],
            folds["val-a-1"]["scene"],
        )
        self.assertNotEqual(
            folds["val-a-0"]["scene"],
            folds["val-b-0"]["scene"],
        )
        self.assertEqual(folds["test-c-0"]["scene"], "ineligible")
        self.assertEqual(
            summaries["scene"]["val"]["eligible_groups"],
            1,
        )
        self.assertEqual(
            summaries["scene"]["test"]["eligible_groups"],
            1,
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
        self.assertTrue(
            all(
                "cross_parent_normalized_accuracy_gain" in row
                for row in first["rows"]
            )
        )


if __name__ == "__main__":
    unittest.main()
