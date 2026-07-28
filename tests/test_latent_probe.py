from __future__ import annotations

import unittest

import torch

from codewam.codebook_eval.latent_probe import (
    build_probe_samples,
    motion_metrics,
    spearman_correlation,
)
from codewam.codebook_eval.shards import PooledFeatureEpisode


def moving_episode(episode_id: str, split: str, offset: float) -> PooledFeatureEpisode:
    ticks = 18
    values = torch.arange(ticks, dtype=torch.float32).square() + float(offset)
    pooled = values.view(ticks, 1, 1, 1, 1).expand(ticks, 1, 2, 4, 4).half()
    thumbnails = torch.zeros((ticks, 1, 3, 8, 8), dtype=torch.uint8)
    intensity = (
        torch.arange(ticks, dtype=torch.float32).square().mul(0.8).round().byte()
    )
    thumbnails[:, 0, :, :, :] = intensity.view(ticks, 1, 1, 1)
    return PooledFeatureEpisode(
        episode_id=episode_id,
        split=split,
        timestamps=torch.arange(ticks, dtype=torch.float64) / 4.0,
        pooled_g4=pooled,
        camera_ids=("exterior",),
        action=values.view(ticks, 1),
        proprio=values.view(ticks, 1),
        metadata={"probe_thumbnails": thumbnails},
    )


class LatentProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.episodes = (
            moving_episode("train-a", "train", 0.0),
            moving_episode("train-b", "train", 20.0),
            moving_episode("val", "val", 40.0),
            moving_episode("test", "test", 60.0),
        )

    def test_descriptor_uses_three_observed_states(self) -> None:
        samples = build_probe_samples(
            self.episodes,
            camera="exterior",
            pool=1,
            representation="descriptor",
            stride=2,
        )
        self.assertEqual(samples.dimension, 6)
        first = samples.vectors[0]
        torch.testing.assert_close(
            first,
            torch.tensor([0.0, 0.0, 4.0, 4.0, 16.0, 16.0]),
        )
        self.assertEqual(int(samples.time_indices[0]), 4)

    def test_residual_is_diagnostic_difference_not_descriptor(self) -> None:
        residual = build_probe_samples(
            self.episodes,
            camera="exterior",
            pool=1,
            representation="residual",
            stride=3,
        )
        self.assertEqual(residual.dimension, 2)
        torch.testing.assert_close(residual.vectors[0], torch.tensor([9.0, 9.0]))

    def test_motion_metrics_detect_monotonic_signal(self) -> None:
        rows = motion_metrics(
            self.episodes,
            cameras=("exterior",),
            pools=(1,),
            strides=(2,),
        )
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["spearman_latent_image"], 0.5)
        self.assertLessEqual(
            rows[0]["adjacent_distance_median"],
            rows[0]["stride_distance_median"],
        )

    def test_spearman_rejects_inverse_order(self) -> None:
        values = torch.arange(10, dtype=torch.float32)
        self.assertAlmostEqual(spearman_correlation(values, values), 1.0)
        self.assertAlmostEqual(spearman_correlation(values, -values), -1.0)


if __name__ == "__main__":
    unittest.main()
