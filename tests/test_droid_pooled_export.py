from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from codewam.codebook_eval.droid_pooled_export import (
    DroidPooledExportConfig,
    _contract_parameters,
    _cuda_device_index,
    _preserve_first_export_evidence,
    _rank_report_payload,
    encode_droid_segment,
    finalize_droid_pooled_export,
)
from codewam.codebook_eval.manifest import EpisodeManifest, EpisodeRecord
from codewam.codebook_eval.shards import write_pooled_feature_shard
from codewam.data.droid_rlds import DroidRLDSSegment


class FakeWanVAE:
    def encode(self, videos, device, tiled):
        self.device = device
        self.tiled = tiled
        ticks = 1 + (videos[0].shape[1] - 1) // 4
        return torch.arange(
            len(videos) * 48 * ticks * 8 * 8,
            dtype=torch.float32,
        ).reshape(len(videos), 48, ticks, 8, 8)


def make_segment() -> DroidRLDSSegment:
    return DroidRLDSSegment(
        episode_id="episode",
        range_index=0,
        start=10,
        stop=19,
        frames={
            "camera": torch.arange(
                9 * 16 * 16 * 3,
                dtype=torch.int32,
            )
            .remainder(256)
            .to(torch.uint8)
            .reshape(9, 16, 16, 3),
        },
        action=torch.arange(63, dtype=torch.float32).reshape(9, 7),
        proprio=torch.arange(126, dtype=torch.float32).reshape(9, 14),
        language_instruction="move",
        action_components={
            "joint_velocity": torch.arange(
                63,
                dtype=torch.float32,
            ).reshape(9, 7),
        },
        split="train",
        manifest_key="droid-1.0.1:episode",
        source_shard="shard",
        record_index=0,
    )


def make_config(output_dir: str = "output") -> DroidPooledExportConfig:
    return DroidPooledExportConfig(
        source_manifest="manifest.jsonl",
        data_dir="data",
        output_dir=output_dir,
        vae_path="vae.pt",
        fastwam_src="fastwam",
        cameras=("camera",),
        nominal_fps=15.0,
        image_height=16,
        image_width=16,
        device="cpu",
        dtype="float32",
    )


class DroidPooledExportTests(unittest.TestCase):
    def test_cuda_device_index_uses_integer_index(self) -> None:
        self.assertEqual(_cuda_device_index(torch.device("cuda:3")), 3)

    def test_resume_preserves_first_export_evidence(self) -> None:
        identity = {
            "source_shard": "shard",
            "source_episodes": 1,
            "segments": 2,
            "ticks": 10,
            "path": "pooled.pt",
            "sha256": "abc",
            "bytes": 123,
        }
        reused = {**identity, "status": "reused"}
        previous = {
            **identity,
            "status": "exported",
            "elapsed_seconds": 4.5,
            "peak_cuda_memory_gib": 2.0,
        }

        merged = _preserve_first_export_evidence(reused, previous)

        self.assertEqual(merged["status"], "reused")
        self.assertEqual(merged["first_export_status"], "exported")
        self.assertEqual(merged["elapsed_seconds"], 4.5)
        self.assertEqual(merged["peak_cuda_memory_gib"], 2.0)

    def test_rank_progress_preserves_partial_runtime_and_first_export(self) -> None:
        rows = [
            {
                "source_shard": "shard",
                "first_export_status": "exported",
            }
        ]
        progress = _rank_report_payload(
            contract_hash="contract",
            rank=1,
            world_size=4,
            assignment={
                "source_shards": 2,
                "source_episodes": 3,
                "source_bytes": 4,
            },
            rows=rows,
            elapsed_seconds=5.0,
            prior_cumulative_seconds=7.0,
            first_export_elapsed_seconds=None,
            complete=False,
        )
        completed = _rank_report_payload(
            contract_hash="contract",
            rank=1,
            world_size=4,
            assignment=progress["assignment"],
            rows=rows,
            elapsed_seconds=5.0,
            prior_cumulative_seconds=7.0,
            first_export_elapsed_seconds=None,
            complete=True,
        )

        self.assertFalse(progress["complete"])
        self.assertEqual(progress["cumulative_elapsed_seconds"], 12.0)
        self.assertIsNone(progress["first_export_elapsed_seconds"])
        self.assertTrue(completed["complete"])
        self.assertEqual(completed["first_export_elapsed_seconds"], 12.0)

    def test_contract_hashes_all_export_implementations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fastwam = root / "fastwam-src"
            for relative in (
                "fastwam/models/wan22/helpers/io.py",
                "fastwam/models/wan22/helpers/state_dict_converters.py",
                "fastwam/models/wan22/wan_video_vae.py",
            ):
                path = fastwam / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")
            vae_path = root / "vae.pt"
            vae_path.write_bytes(b"vae")
            manifest = EpisodeManifest.from_records(
                (
                    EpisodeRecord(
                        dataset="droid-1.0.1",
                        episode_id="episode",
                        num_steps=9,
                        source_uri="gs://droid/trajectory.h5",
                        institution_id="institution",
                        building_id="building",
                        scene_id="scene",
                        split="train",
                    ),
                )
            )
            config = replace(
                make_config(),
                fastwam_src=str(fastwam),
                vae_path=str(vae_path),
            )

            contract = _contract_parameters(
                config,
                manifest,
                manifest_sha256="manifest",
                vae_path=vae_path,
            )

        self.assertEqual(
            sorted(contract["implementation_sha256"]),
            ["state_dict_converter", "state_dict_io", "wan_vae"],
        )
        self.assertEqual(
            sorted(contract["codewam_dependency_sha256"]),
            [
                "droid_manifest",
                "droid_rlds",
                "manifest",
                "shards",
                "wan_probe_export",
            ],
        )

    def test_segment_encoding_preserves_absolute_time_and_action_components(self) -> None:
        pooled = encode_droid_segment(make_segment(), FakeWanVAE(), make_config())

        self.assertEqual(pooled.episode_id, "episode@10:19")
        self.assertEqual(tuple(pooled.pooled_g4.shape), (3, 1, 48, 4, 4))
        torch.testing.assert_close(
            pooled.timestamps,
            torch.tensor([10, 14, 18], dtype=torch.float64) / 15.0,
        )
        torch.testing.assert_close(
            pooled.action.float(),
            make_segment().action[[0, 4, 8]],
        )
        torch.testing.assert_close(
            pooled.action_components["joint_velocity"].float(),
            make_segment().action_components["joint_velocity"][[0, 4, 8]],
        )
        torch.testing.assert_close(
            pooled.metadata["absolute_latent_frame_indices"],
            torch.tensor([10, 14, 18]),
        )

    def test_finalize_builds_scene_isolated_segment_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.jsonl"
            output_dir = root / "output"
            pooled_dir = output_dir / "pooled"
            pooled_dir.mkdir(parents=True)
            source_record = EpisodeRecord(
                dataset="droid-1.0.1",
                episode_id="episode",
                num_steps=9,
                source_uri="gs://droid/trajectory.h5",
                institution_id="institution",
                building_id="building",
                scene_id="scene",
                task_ids=("move",),
                camera_ids=("camera",),
                source_checksum="crc32c:source",
                split="train",
                metadata={
                    "rlds_shard_name": "shard",
                    "rlds_shard_index": 3,
                    "rlds_shard_bytes": 100,
                    "rlds_record_index": 0,
                    "recording_folderpath": "gs://droid/recordings",
                    "keep_ranges": [[0, 9]],
                    "eligible_steps": 9,
                },
            )
            source = EpisodeManifest.from_records((source_record,))
            source.write_jsonl(source_path)
            contract = {
                "source_manifest_fingerprint": source.fingerprint(),
                "contract_hash": "contract",
            }
            (output_dir / "contract.json").write_text(
                json.dumps(contract),
                encoding="utf-8",
            )

            segment = make_segment()
            segment = replace(
                segment,
                start=0,
                stop=9,
                manifest_key=source_record.key,
                source_shard="shard",
            )
            pooled = encode_droid_segment(segment, FakeWanVAE(), make_config())
            pooled_path = pooled_dir / "droid-rlds-00003.pt"
            write_pooled_feature_shard(
                pooled_path,
                (pooled,),
                metadata={
                    "dataset_revision": "droid-1.0.1",
                    "wan_model_id": "test-wan",
                    "wan_revision": "test-revision",
                    "preprocess_revision": "test-preprocess",
                    "source_checksums": ["crc32c:source"],
                    "export_contract_hash": "contract",
                    "source_shard_name": "shard",
                    "source_shard_bytes": 100,
                },
            )

            report = finalize_droid_pooled_export(source_path, output_dir)
            finalized = EpisodeManifest.read_jsonl(
                report["pooled_manifest"]["path"]
            )

        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized.records[0].episode_id, "episode@0:9")
        self.assertEqual(finalized.records[0].scene_id, "scene")
        self.assertEqual(finalized.records[0].num_steps, 3)


if __name__ == "__main__":
    unittest.main()
