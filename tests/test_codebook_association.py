from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from codewam.codebook_eval.association import (
    _ConditionalMeans,
    _episode_probe_values,
    probe_frozen_codebook_associations,
)
from codewam.codebook_eval.pipeline import (
    create_synthetic_streaming_fixture,
    train_streaming_codebooks,
)
from codewam.codebook_eval.shards import PooledFeatureEpisode
from codewam.codebook_eval.streaming import CausalDescriptorSpec


class CodebookAssociationTests(unittest.TestCase):
    def test_probe_targets_are_future_outcomes_not_descriptor_inputs(self) -> None:
        ticks = 9
        pooled = torch.arange(
            ticks * 1 * 2 * 2 * 2,
            dtype=torch.float32,
        ).reshape(ticks, 1, 2, 2, 2)
        episode = PooledFeatureEpisode(
            episode_id="episode",
            split="train",
            timestamps=torch.arange(ticks, dtype=torch.float64),
            pooled_g4=torch.nn.functional.interpolate(
                pooled.reshape(ticks, 2, 2, 2),
                size=(4, 4),
                mode="nearest",
            ).reshape(ticks, 1, 2, 4, 4),
            camera_ids=("camera",),
            action=torch.arange(ticks, dtype=torch.float32).unsqueeze(1),
            proprio=2.0 * torch.arange(
                ticks,
                dtype=torch.float32,
            ).unsqueeze(1),
        )

        vectors, targets = _episode_probe_values(
            episode,
            CausalDescriptorSpec(stride=2, pool=2),
        )

        self.assertEqual(vectors.shape[0], 3)
        self.assertEqual(targets["current_action"][0].item(), 4.0)
        self.assertEqual(targets["future_proprio_change"][0].item(), 4.0)
        selected = episode.pooled(2).reshape(ticks, -1)
        torch.testing.assert_close(
            vectors[0, -selected.shape[1] :],
            selected[4],
        )

    def test_conditional_means_back_off_to_a_supported_prefix(self) -> None:
        table = _ConditionalMeans(k=2, levels=2, dimension=1)
        table.update(
            torch.tensor([[0, 0], [0, 0], [1, 1]]),
            torch.tensor([[1.0], [1.0], [3.0]]),
        )

        predictions, chosen = table.predict(
            torch.tensor([[0, 1], [1, 0]]),
            depth=2,
            min_train_count=2,
        )

        torch.testing.assert_close(predictions[:, 0], torch.tensor([1.0, 5.0 / 3.0]))
        self.assertEqual(chosen.tolist(), [1, 0])

    def test_association_report_trains_only_on_train_and_resumes(self) -> None:
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
            output = root / "association"

            first = probe_frozen_codebook_associations(
                manifest_path=root / "manifest.jsonl",
                pooled_shards=(root / "pooled/*.pt",),
                artifacts={"synthetic-q3": artifact},
                output_dir=output,
                device="cpu",
                batch_size=32,
                center_block_size=4,
                min_train_count=1,
            )
            resumed = probe_frozen_codebook_associations(
                manifest_path=root / "manifest.jsonl",
                pooled_shards=(root / "pooled/*.pt",),
                artifacts={"synthetic-q3": artifact},
                output_dir=output,
                device="cpu",
                batch_size=32,
                center_block_size=4,
                min_train_count=1,
            )

        self.assertEqual(first, resumed)
        self.assertEqual(len(first["rows"]), 18)
        self.assertEqual(
            {row["split"] for row in first["rows"]},
            {"val", "test"},
        )
        self.assertEqual(
            {row["target"] for row in first["rows"]},
            {
                "current_action",
                "future_proprio_change",
                "future_latent_moment_change",
            },
        )


if __name__ == "__main__":
    unittest.main()
