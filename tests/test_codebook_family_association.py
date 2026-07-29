from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from codewam.codebook_eval.family_association import (
    _AdditiveCodeStatistics,
    _episode_aligned_probe_values,
    probe_codebook_family_contributions,
)
from codewam.codebook_eval.pipeline import (
    create_synthetic_streaming_fixture,
    train_streaming_codebooks,
)
from codewam.codebook_eval.shards import PooledFeatureEpisode
from codewam.codebook_eval.streaming import (
    CausalDescriptorSpec,
    FrozenRQArtifact,
    NormalizationStats,
)


def _artifact(stride: int, dimension: int) -> FrozenRQArtifact:
    return FrozenRQArtifact(
        family=f"Q{stride}",
        descriptor=CausalDescriptorSpec(
            stride=stride,
            pool=2,
            camera_ids=("camera",),
        ),
        normalization=NormalizationStats(
            count=2,
            mean=torch.zeros(dimension),
            std=torch.ones(dimension),
        ),
        centers=(torch.zeros((2, dimension)),),
        metadata={
            "dataset": "synthetic",
            "dataset_revision": "v1",
            "wan_model_id": "wan",
            "wan_revision": "v1",
            "preprocess_revision": "v1",
            "manifest_fingerprint": "fingerprint",
            "source_checksums": ["checksum"],
            "config": {},
            "config_hash": "config",
            "implementation_sha256": {},
        },
    )


class CodebookFamilyAssociationTests(unittest.TestCase):
    def test_aligned_values_share_current_and_future_targets(self) -> None:
        ticks = 15
        pooled = torch.arange(
            ticks * 1 * 2 * 4 * 4,
            dtype=torch.float32,
        ).reshape(ticks, 1, 2, 4, 4)
        episode = PooledFeatureEpisode(
            episode_id="episode",
            split="train",
            timestamps=torch.arange(ticks, dtype=torch.float64),
            pooled_g4=pooled,
            camera_ids=("camera",),
            action=torch.arange(ticks, dtype=torch.float32).unsqueeze(1),
            proprio=2.0
            * torch.arange(ticks, dtype=torch.float32).unsqueeze(1),
        )
        dimension = 3 * 1 * 2 * 2 * 2
        artifacts = {
            "Q2": _artifact(2, dimension),
            "Q3": _artifact(3, dimension),
        }

        vectors, targets = _episode_aligned_probe_values(
            episode,
            artifacts,
            future_offset=1,
        )

        self.assertEqual(vectors["Q2"].shape[0], 8)
        self.assertEqual(vectors["Q3"].shape[0], 8)
        self.assertEqual(targets["current_action"][0].item(), 6.0)
        self.assertEqual(
            targets["common_future_proprio_change"][0].item(),
            2.0,
        )

    def test_additive_fit_recovers_complementary_family_effects(self) -> None:
        statistics = _AdditiveCodeStatistics(
            {"Q2": 2, "Q3": 2},
            {"target": 1},
        )
        q2 = torch.tensor([0, 0, 1, 1] * 16)
        q3 = torch.tensor([0, 1, 0, 1] * 16)
        target = (q2.float() + 2.0 * q3.float()).unsqueeze(1)
        statistics.update({"Q2": q2, "Q3": q3}, {"target": target})

        coefficients = statistics.fit(
            ("Q2", "Q3"),
            ridge=1e-3,
            device=torch.device("cpu"),
        )["target"]
        mean, _, _ = statistics.global_statistics("target")
        prediction = mean.expand(target.shape[0], -1).clone()
        prediction += coefficients["Q2"][q2]
        prediction += coefficients["Q3"][q3]

        torch.testing.assert_close(prediction, target, atol=1e-3, rtol=1e-3)

    def test_family_report_uses_aligned_train_fit_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = create_synthetic_streaming_fixture(root)
            config = OmegaConf.load(config_path)
            config.training.device = "cpu"
            config.training.cpu_threads = 1
            config.training.max_iters = 1
            config.training.patience = 1
            config.training.k = 2
            config.training.levels = 1
            config.training.reservoir_size = 64
            OmegaConf.save(config, config_path)
            trained = train_streaming_codebooks(config_path)
            artifacts = {
                row["family"]: row["artifact"] for row in trained
            }
            output = root / "family-association"

            first = probe_codebook_family_contributions(
                manifest_path=root / "manifest.jsonl",
                pooled_shards=(root / "pooled/*.pt",),
                artifacts=artifacts,
                output_dir=output,
                device="cpu",
                cpu_threads=1,
                batch_size=32,
                center_block_size=4,
            )
            resumed = probe_codebook_family_contributions(
                manifest_path=root / "manifest.jsonl",
                pooled_shards=(root / "pooled/*.pt",),
                artifacts=artifacts,
                output_dir=output,
                device="cpu",
                cpu_threads=1,
                batch_size=32,
                center_block_size=4,
            )

        self.assertEqual(first, resumed)
        self.assertEqual(first["families"], ["Q2", "Q3", "Q5"])
        self.assertEqual(len(first["rows"]), 42)
        self.assertEqual(len(first["summary_rows"]), 6)
        self.assertEqual(
            {row["split"] for row in first["summary_rows"]},
            {"val", "test"},
        )


if __name__ == "__main__":
    unittest.main()
