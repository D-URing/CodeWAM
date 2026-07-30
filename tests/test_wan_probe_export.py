from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from codewam.codebook_eval.wan_probe_export import (
    _construct_wan_vae,
    _encode_wan_views_streaming,
    _preprocess_video,
    _reused_episode_row,
    latent_frame_indices,
)
from codewam.codebook_eval.shards import (
    PooledFeatureEpisode,
    write_pooled_feature_shard,
)
from codewam.data.droid_rlds import DroidRLDSEpisode, probe_split


class WanProbeExportTests(unittest.TestCase):
    def test_resume_validates_contract_and_preserves_manifest_fields(self) -> None:
        source = DroidRLDSEpisode(
            episode_id="episode-0",
            index=0,
            frames={
                "camera": torch.zeros((5, 16, 16, 3), dtype=torch.uint8),
            },
            action=torch.zeros((5, 7)),
            proprio=torch.zeros((5, 14)),
            language_instruction="test",
            source_file="source.h5",
            recording_folder="recordings",
        )
        pooled = PooledFeatureEpisode(
            episode_id=source.episode_id,
            split="val",
            timestamps=torch.tensor([0.0, 1.0]),
            pooled_g4=torch.zeros((2, 1, 48, 4, 4)),
            camera_ids=("camera",),
            metadata={
                "source_index": 0,
                "source_frame_count": 5,
                "source_file": "source.h5",
                "recording_folder": "recordings",
                "latent_shape": [48, 2, 1, 1],
            },
        )
        shard_metadata = {
            "dataset_revision": "dataset",
            "wan_model_id": "wan",
            "wan_revision": "revision",
            "preprocess_revision": "preprocess",
            "source_checksums": ["source:checksum"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode-00000.pt"
            info = write_pooled_feature_shard(path, [pooled], shard_metadata)
            row = _reused_episode_row(
                path,
                source,
                expected_metadata=shard_metadata,
                expected_cameras=("camera",),
            )
            self.assertEqual(row["sha256"], info.sha256)
            self.assertEqual(row["latent_shape"], [48, 2, 1, 1])
            self.assertEqual(row["latent_ticks"], 2)
            self.assertEqual(row["split"], "val")
            with self.assertRaisesRegex(RuntimeError, "export contract"):
                _reused_episode_row(
                    path,
                    source,
                    expected_metadata={**shard_metadata, "wan_revision": "stale"},
                    expected_cameras=("camera",),
                )

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

    def test_chunked_preprocess_matches_full_batch_definition(self) -> None:
        torch.manual_seed(41)
        frames = torch.randint(
            0,
            256,
            (19, 18, 30, 3),
            dtype=torch.uint8,
        )
        values = frames.permute(0, 3, 1, 2).float()
        expected = F.interpolate(
            values,
            size=(16, 32),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        expected = (
            (expected / 127.5 - 1.0)
            .to(dtype=torch.bfloat16)
            .permute(1, 0, 2, 3)
            .contiguous()
        )

        observed = _preprocess_video(
            frames,
            height=16,
            width=32,
            dtype=torch.bfloat16,
            frame_batch_size=3,
        )

        torch.testing.assert_close(observed, expected, rtol=0, atol=0)

    def test_streaming_encode_processes_one_camera_at_a_time(self) -> None:
        class RecordingVAE:
            def __init__(self) -> None:
                self.batch_sizes: list[int] = []

            def encode(
                self,
                videos: list[torch.Tensor],
                *,
                device: torch.device,
                tiled: bool,
            ) -> torch.Tensor:
                del tiled
                self.batch_sizes.append(len(videos))
                ticks = 1 + (int(videos[0].shape[1]) - 1) // 4
                marker = videos[0][0, 0, 0, 0].float()
                return marker.expand(1, 48, ticks, 1, 1).to(device)

        vae = RecordingVAE()
        first = torch.zeros((9, 8, 8, 3), dtype=torch.uint8)
        second = torch.full((9, 8, 8, 3), 255, dtype=torch.uint8)

        latent = _encode_wan_views_streaming(
            (first, second),
            vae=vae,
            device=torch.device("cpu"),
            height=16,
            width=16,
            dtype=torch.float32,
        )

        self.assertEqual(vae.batch_sizes, [1, 1])
        self.assertEqual(tuple(latent.shape), (2, 48, 3, 1, 1))
        self.assertEqual(float(latent[0, 0, 0, 0, 0]), -1.0)
        self.assertEqual(float(latent[1, 0, 0, 0, 0]), 1.0)

    def test_probe_split_is_episode_level_and_balanced_per_cycle(self) -> None:
        splits = [probe_split(index) for index in range(12)]
        self.assertEqual(splits.count("train"), 8)
        self.assertEqual(splits.count("val"), 2)
        self.assertEqual(splits.count("test"), 2)


if __name__ == "__main__":
    unittest.main()
