from __future__ import annotations

import unittest

import torch

from codewam.data.frozen_assignment import (
    FrozenArtifactChart,
    FrozenCausalCodeAssigner,
)
from tests.model_fixtures import synthetic_artifacts


def make_chart() -> FrozenArtifactChart:
    artifacts = synthetic_artifacts("droid")
    return FrozenArtifactChart(
        name="droid",
        artifacts=artifacts,
        artifact_sha256=("a" * 64, "b" * 64, "c" * 64),
        artifact_paths=("Q2.pt", "Q3.pt", "Q5.pt"),
    )


class FrozenAssignmentTests(unittest.TestCase):
    def test_multiscale_sources_and_availability_are_exact(self) -> None:
        torch.manual_seed(19)
        latents = torch.randn((13, 1, 4, 4, 4))
        source_indices = 100 + 4 * torch.arange(13, dtype=torch.long)
        assignment = FrozenCausalCodeAssigner(make_chart()).assign(
            latents,
            latent_source_indices=source_indices,
            camera_ids=("wrist",),
        )

        self.assertEqual(tuple(assignment.code_ids.shape), (13, 3, 3))
        self.assertEqual(
            assignment.available.sum(dim=0).tolist(),
            [9, 7, 3],
        )
        self.assertEqual(
            assignment.descriptor_source_indices[10, 0].tolist(),
            [124, 132, 140],
        )
        self.assertEqual(
            assignment.descriptor_source_indices[10, 1].tolist(),
            [116, 128, 140],
        )
        self.assertEqual(
            assignment.descriptor_source_indices[10, 2].tolist(),
            [100, 120, 140],
        )
        self.assertTrue(
            torch.all(
                assignment.code_ids[assignment.available] >= 0
            )
        )

    def test_future_perturbation_cannot_change_past_assignments(self) -> None:
        torch.manual_seed(23)
        latents = torch.randn((14, 1, 4, 4, 4))
        source_indices = 4 * torch.arange(14, dtype=torch.long)
        assigner = FrozenCausalCodeAssigner(make_chart())
        original = assigner.assign(
            latents,
            latent_source_indices=source_indices,
            camera_ids=("wrist",),
        )
        changed = latents.clone()
        changed[13] += 10_000.0
        perturbed = assigner.assign(
            changed,
            latent_source_indices=source_indices,
            camera_ids=("wrist",),
        )

        torch.testing.assert_close(
            original.code_ids[:13],
            perturbed.code_ids[:13],
        )
        torch.testing.assert_close(
            original.available,
            perturbed.available,
        )

    def test_missing_artifact_camera_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "lack Q2 cameras"):
            FrozenCausalCodeAssigner(make_chart()).assign(
                torch.zeros((12, 1, 4, 4, 4)),
                latent_source_indices=torch.arange(12),
                camera_ids=("exterior",),
            )


if __name__ == "__main__":
    unittest.main()
