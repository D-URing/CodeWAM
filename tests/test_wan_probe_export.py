from __future__ import annotations

import unittest

import torch

from codewam.codebook_eval.wan_probe_export import (
    _construct_wan_vae,
    _preprocess_video,
    latent_frame_indices,
)
from codewam.data.droid_rlds import probe_split


class WanProbeExportTests(unittest.TestCase):
    def test_wan_constructor_materializes_unregistered_statistics(self) -> None:
        class FakeWanVAE(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.mean = torch.tensor([1.0])
                self.std = torch.tensor([2.0])

        model = _construct_wan_vae(FakeWanVAE)
        self.assertFalse(model.mean.is_meta)
        self.assertFalse(model.std.is_meta)

        with torch.device("meta"):
            with self.assertRaisesRegex(RuntimeError, "was not materialized"):
                _construct_wan_vae(FakeWanVAE)

    def test_latent_ticks_are_aligned_to_causal_chunk_end(self) -> None:
        torch.testing.assert_close(latent_frame_indices(1, 1), torch.tensor([0]))
        torch.testing.assert_close(latent_frame_indices(5, 2), torch.tensor([0, 4]))
        torch.testing.assert_close(
            latent_frame_indices(10, 3),
            torch.tensor([0, 4, 8]),
        )
        with self.assertRaisesRegex(ValueError, "temporal shape mismatch"):
            latent_frame_indices(9, 2)

    def test_preprocess_has_wan_range_and_layout(self) -> None:
        frames = torch.tensor(
            [
                [[[0, 127, 255], [255, 127, 0]]],
                [[[255, 255, 255], [0, 0, 0]]],
            ],
            dtype=torch.uint8,
        )
        video = _preprocess_video(frames, height=16, width=16, dtype=torch.float32)
        self.assertEqual(tuple(video.shape), (3, 2, 16, 16))
        self.assertGreaterEqual(float(video.min()), -1.0)
        self.assertLessEqual(float(video.max()), 1.0)

    def test_probe_split_is_episode_level_and_balanced_per_cycle(self) -> None:
        splits = [probe_split(index) for index in range(12)]
        self.assertEqual(splits.count("train"), 8)
        self.assertEqual(splits.count("val"), 2)
        self.assertEqual(splits.count("test"), 2)


if __name__ == "__main__":
    unittest.main()
