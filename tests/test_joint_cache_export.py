from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from codewam.codebook_eval.manifest import EpisodeRecord
from codewam.data.droid_rlds import DroidRLDSSegment
from codewam.data.frozen_assignment import (
    FrozenArtifactChart,
    FrozenCausalCodeAssigner,
)
from codewam.data.joint_cache import JointWindowConfig, build_joint_windows
from codewam.data.joint_cache_export import (
    JointCacheExportConfig,
    _load_rank_vae,
    _resolve_fastwam_src,
    encode_joint_segment,
)
from tests.model_fixtures import synthetic_artifacts


class FakeWanVAE:
    def encode(
        self,
        videos: list[torch.Tensor],
        *,
        device: torch.device,
        tiled: bool,
    ) -> torch.Tensor:
        del tiled
        frames = int(videos[0].shape[1])
        ticks = 1 + (frames - 1) // 4
        values = torch.arange(
            len(videos) * 48 * ticks,
            dtype=torch.float32,
            device=device,
        )
        return values.reshape(len(videos), 48, ticks, 1, 1) / 1000.0


class JointCacheExportTests(unittest.TestCase):
    def test_multi_rank_vae_load_uses_a_node_local_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = JointCacheExportConfig(
                source_manifest="manifest.jsonl",
                data_dir="data",
                output_dir="output",
                endpoint_audit="endpoint.json",
                artifact_paths={},
                chart_name="droid",
                vae_path="vae.pt",
                fastwam_src="FastWAM",
                rank=2,
                world_size=4,
                device="cpu",
                dtype="float32",
            )
            contract_hash = "a" * 64
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "LOCAL_WORLD_SIZE": "4",
                        "CODEWAM_VAE_LOAD_LOCK_DIR": temporary,
                    },
                ),
                mock.patch(
                    "codewam.data.joint_cache_export._load_wan_vae",
                    return_value=mock.sentinel.vae,
                ) as load,
                mock.patch(
                    "codewam.data.joint_cache_export._release_process_memory",
                ) as release,
            ):
                result = _load_rank_vae(
                    config,
                    contract_hash=contract_hash,
                )

            self.assertIs(result, mock.sentinel.vae)
            load.assert_called_once_with(config)
            release.assert_called_once_with(torch.device("cpu"))
            self.assertTrue(
                (
                    Path(temporary)
                    / f"codewam-vae-load-{contract_hash}.lock"
                ).is_file()
            )

    def test_fastwam_repository_root_resolves_to_src_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "src" / "fastwam"
            package.mkdir(parents=True)
            expected = (root / "src").resolve()
            self.assertEqual(_resolve_fastwam_src(root), expected)
            self.assertEqual(_resolve_fastwam_src(root / "src"), expected)

    def test_encode_segment_preserves_wan_and_rlds_endpoint_alignment(self) -> None:
        torch.manual_seed(43)
        steps = 65
        chart = FrozenArtifactChart(
            name="droid",
            artifacts=synthetic_artifacts(
                "droid",
                descriptor_dim=3 * 48,
            ),
            artifact_sha256=("a" * 64, "b" * 64, "c" * 64),
            artifact_paths=("Q2.pt", "Q3.pt", "Q5.pt"),
        )
        is_first = torch.zeros(steps, dtype=torch.bool)
        is_first[0] = True
        is_last = torch.zeros(steps, dtype=torch.bool)
        is_last[-1] = True
        segment = DroidRLDSSegment(
            episode_id="episode",
            range_index=0,
            start=0,
            stop=steps,
            frames={
                "wrist": torch.randint(
                    0,
                    256,
                    (steps, 12, 16, 3),
                    dtype=torch.uint8,
                )
            },
            action=torch.randn((steps, 7)),
            proprio=torch.randn((steps, 14)),
            language_instruction="move the object",
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_last.clone(),
            split="train",
            manifest_key="droid-v1:episode",
            source_shard="droid-train.tfrecord-00000-of-01024",
            record_index=0,
        )
        record = EpisodeRecord(
            dataset="droid-v1",
            episode_id="episode",
            num_steps=steps,
            source_uri="droid-train.tfrecord-00000-of-01024",
            source_checksum="source-checksum",
            split="train",
            metadata={"success": True, "keep_ranges": [[0, steps]]},
        )
        window_config = JointWindowConfig()
        config = JointCacheExportConfig(
            source_manifest="manifest.jsonl",
            data_dir="data",
            output_dir="output",
            endpoint_audit="endpoint.json",
            artifact_paths={},
            chart_name="droid",
            vae_path="vae.pt",
            fastwam_src="FastWAM",
            cameras=("wrist",),
            image_height=16,
            image_width=16,
            device="cpu",
            dtype="float32",
            window=window_config,
        )

        episode = encode_joint_segment(
            segment,
            record=record,
            vae=FakeWanVAE(),
            chart=chart,
            assigner=FrozenCausalCodeAssigner(chart),
            config=config,
        )
        windows = build_joint_windows(
            episode,
            config=window_config,
            artifact_sha256=chart.artifact_sha256,
        )

        self.assertEqual(tuple(episode.latents.shape), (17, 1, 48, 1, 1))
        self.assertEqual(
            episode.latent_source_indices.tolist(),
            list(range(0, steps, 4)),
        )
        self.assertTrue(episode.source_action_valid[:-1].all())
        self.assertFalse(episode.source_action_valid[-1])
        self.assertEqual(len(windows), 3)
        for window in windows:
            self.assertEqual(
                window.future_observation_source_index,
                window.decision_source_index + 16,
            )
            self.assertEqual(window.descriptor_overlap, (1, 0, 0))
            self.assertEqual(
                window.future_descriptor_sources[0][0],
                window.current_descriptor_sources[0][2],
            )


if __name__ == "__main__":
    unittest.main()
