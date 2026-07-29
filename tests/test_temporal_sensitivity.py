from __future__ import annotations

import unittest

import torch

from codewam.codebook_eval.shards import PooledFeatureEpisode
from codewam.codebook_eval.streaming import (
    CausalDescriptorSpec,
    FrozenRQArtifact,
    NormalizationStats,
)
from codewam.codebook_eval.temporal_sensitivity import (
    PERTURBATION_NAMES,
    _descriptor_perturbations,
    _probe_artifact,
)


class TemporalSensitivityTests(unittest.TestCase):
    def test_descriptor_counterfactuals_preserve_expected_endpoint(self) -> None:
        vectors = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])

        perturbed = _descriptor_perturbations(vectors)

        torch.testing.assert_close(
            perturbed["history_swap"],
            torch.tensor([[2.0, 3.0, 0.0, 1.0, 4.0, 5.0]]),
        )
        torch.testing.assert_close(
            perturbed["reverse_time"],
            torch.tensor([[4.0, 5.0, 2.0, 3.0, 0.0, 1.0]]),
        )
        torch.testing.assert_close(
            perturbed["static_current"],
            torch.tensor([[4.0, 5.0, 4.0, 5.0, 4.0, 5.0]]),
        )

    def test_frozen_codes_change_for_exact_temporal_counterfactuals(self) -> None:
        pooled = torch.arange(5, dtype=torch.float32).view(5, 1, 1, 1, 1)
        episode = PooledFeatureEpisode(
            episode_id="episode",
            split="test",
            timestamps=torch.arange(5, dtype=torch.float64),
            pooled_g4=pooled.expand(5, 1, 1, 4, 4).contiguous(),
            camera_ids=("wrist_image_left",),
        )
        block = 16

        def descriptor(values: tuple[float, float, float]) -> torch.Tensor:
            return torch.cat(
                [torch.full((block,), value) for value in values]
            )

        centers = torch.stack(
            [
                descriptor((0.0, 2.0, 4.0)),
                descriptor((2.0, 0.0, 4.0)),
                descriptor((4.0, 2.0, 0.0)),
                descriptor((4.0, 4.0, 4.0)),
            ]
        )
        artifact = FrozenRQArtifact(
            family="Q2",
            descriptor=CausalDescriptorSpec(
                stride=2,
                pool=4,
                camera_ids=("wrist_image_left",),
            ),
            normalization=NormalizationStats(
                count=1,
                mean=torch.zeros(block * 3),
                std=torch.ones(block * 3),
            ),
            centers=(centers,),
            metadata={
                "dataset": "synthetic",
                "dataset_revision": "synthetic-v1",
                "manifest_fingerprint": "manifest",
                "wan_model_id": "wan",
                "wan_revision": "revision",
                "preprocess_revision": "preprocess",
                "config_hash": "config",
                "source_checksums": ["source"],
            },
        )

        row = _probe_artifact(
            artifact,
            episode_factory=lambda: iter((episode,)),
            split="test",
            batch_size=8,
            center_block_size=4,
            device=torch.device("cpu"),
        )

        self.assertEqual(row["vectors"], 1)
        self.assertEqual(
            [value["name"] for value in row["perturbations"]],
            list(PERTURBATION_NAMES),
        )
        for perturbation in row["perturbations"]:
            self.assertEqual(
                perturbation["level_code_change"][0]["change_fraction"],
                1.0,
            )
            self.assertEqual(
                perturbation["prefix_code_change"][0]["change_fraction"],
                1.0,
            )
            self.assertGreater(
                perturbation["normalized_descriptor_displacement_mse"],
                0.0,
            )
            self.assertGreater(
                perturbation["cross_reconstruction_penalty"][0],
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
