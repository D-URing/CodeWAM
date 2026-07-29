from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image

from codewam.codebook_eval.manifest import EpisodeManifest, EpisodeRecord
from codewam.codebook_eval.retrieval import (
    _select_representative_samples,
    render_codebook_retrieval_montages,
)
from codewam.codebook_eval.shards import (
    PooledFeatureEpisode,
    file_sha256,
    write_pooled_feature_shard,
)
from codewam.data.droid_manifest import write_json_report


class CodebookRetrievalTests(unittest.TestCase):
    def test_representatives_can_require_scene_diversity(self) -> None:
        def record(
            episode_id: str,
            *,
            parent: str,
            scene: str,
        ) -> EpisodeRecord:
            return EpisodeRecord(
                dataset="droid-1.0.1",
                episode_id=episode_id,
                num_steps=10,
                source_uri=f"{episode_id}.pt",
                scene_id=scene,
                building_id="building",
                institution_id="institution",
                split="test",
                metadata={"parent_manifest_key": parent},
            )

        pooled = {
            "first": record("first", parent="parent-a", scene="scene-a"),
            "second": record("second", parent="parent-b", scene="scene-a"),
            "third": record("third", parent="parent-c", scene="scene-b"),
        }
        samples = [
            {"episode_id": episode_id}
            for episode_id in ("first", "second", "third")
        ]

        scene_diverse = _select_representative_samples(
            samples,
            pooled_by_id=pooled,
            limit=2,
            diversity_by="scene",
        )
        parent_diverse = _select_representative_samples(
            samples,
            pooled_by_id=pooled,
            limit=2,
            diversity_by="parent",
        )

        self.assertEqual(
            [(rank, value["episode_id"]) for rank, value in scene_diverse],
            [(1, "first"), (3, "third")],
        )
        self.assertEqual(
            [(rank, value["episode_id"]) for rank, value in parent_diverse],
            [(1, "first"), (2, "second")],
        )

    def test_exact_descriptor_frames_render_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_manifest_path = root / "source.jsonl"
            pooled_manifest_path = root / "pooled_manifest.jsonl"
            pooled_path = root / "pooled" / "shard.pt"
            evaluation_path = root / "evaluation_report.json"
            droid_data_dir = root / "droid"
            output_dir = root / "retrievals"
            droid_data_dir.mkdir()
            write_json_report(droid_data_dir / "dataset_info.json", {})

            source_record = EpisodeRecord(
                dataset="droid-1.0.1",
                episode_id="episode",
                num_steps=40,
                source_uri="gs://droid/trajectory.h5",
                scene_id="scene",
                building_id="building",
                institution_id="institution",
                task_ids=("move the object",),
                camera_ids=("wrist_image_left",),
                source_checksum="crc32c:source",
                split="test",
                metadata={
                    "rlds_shard_name": "source.tfrecord",
                    "rlds_shard_bytes": 1234,
                    "rlds_record_index": 2,
                    "recording_folderpath": "gs://droid/recordings",
                    "keep_ranges": [[0, 40]],
                    "eligible_steps": 40,
                },
            )
            source_manifest = EpisodeManifest.from_records((source_record,))
            source_manifest.write_jsonl(source_manifest_path)

            pooled_episode = PooledFeatureEpisode(
                episode_id="episode@0:40",
                split="test",
                timestamps=torch.tensor(
                    [10, 14, 18, 22, 26],
                    dtype=torch.float64,
                )
                / 15.0,
                pooled_g4=torch.zeros((5, 1, 2, 4, 4)),
                camera_ids=("wrist_image_left",),
                metadata={
                    "parent_manifest_key": source_record.key,
                    "source_shard": "source.tfrecord",
                    "record_index": 2,
                    "source_range": [0, 40],
                    "absolute_latent_frame_indices": torch.tensor(
                        [10, 14, 18, 22, 26]
                    ),
                    "nominal_fps": 15.0,
                },
            )
            write_pooled_feature_shard(
                pooled_path,
                (pooled_episode,),
                metadata={
                    "dataset_revision": "droid-1.0.1",
                    "wan_model_id": "wan",
                    "wan_revision": "revision",
                    "preprocess_revision": "preprocess",
                    "source_checksums": ["crc32c:source"],
                },
            )
            pooled_record = EpisodeRecord(
                dataset="droid-1.0.1",
                episode_id=pooled_episode.episode_id,
                num_steps=pooled_episode.ticks,
                source_uri=f"{pooled_path}#{pooled_episode.episode_id}",
                scene_id="scene",
                building_id="building",
                institution_id="institution",
                task_ids=("move the object",),
                camera_ids=("wrist_image_left",),
                source_checksum=f"sha256:{file_sha256(pooled_path)}",
                split="test",
                metadata={
                    "parent_manifest_key": source_record.key,
                    "pooled_shard": str(pooled_path),
                    "source_shard": "source.tfrecord",
                    "source_range": [0, 40],
                },
            )
            pooled_manifest = EpisodeManifest.from_records((pooled_record,))
            pooled_manifest.write_jsonl(pooled_manifest_path)
            write_json_report(
                evaluation_path,
                {
                    "schema": "codewam.heldout-rq-evaluation.v1",
                    "dataset": "droid-1.0.1",
                    "manifest_fingerprint": pooled_manifest.fingerprint(),
                    "rows": [
                        {
                            "family": "Q2",
                            "stride": 2,
                            "split": "test",
                            "k": 1,
                            "levels": 1,
                            "representatives": [
                                {
                                    "level": 1,
                                    "codes": [
                                        {
                                            "code": 0,
                                            "samples": [
                                                {
                                                    "episode_id": (
                                                        pooled_episode.episode_id
                                                    ),
                                                    "time_index": 4,
                                                    "timestamp": 26.0 / 15.0,
                                                    "distance_mse": 0.25,
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                },
            )

            def fake_reader(
                data_dir: Path,
                manifest: EpisodeManifest,
                requests: dict[str, set[int]],
                *,
                camera: str,
            ) -> dict[tuple[str, int], torch.Tensor]:
                self.assertEqual(Path(data_dir), droid_data_dir)
                self.assertEqual(len(manifest), 1)
                self.assertEqual(
                    requests,
                    {source_record.key: {10, 18, 26}},
                )
                self.assertEqual(camera, "wrist_image_left")
                return {
                    (source_record.key, index): torch.full(
                        (18, 32, 3),
                        index,
                        dtype=torch.uint8,
                    )
                    for index in (10, 18, 26)
                }

            with mock.patch(
                "codewam.codebook_eval.retrieval."
                "read_manifest_droid_rlds_frames",
                side_effect=fake_reader,
            ) as reader:
                first = render_codebook_retrieval_montages(
                    source_manifest_path=source_manifest_path,
                    pooled_manifest_path=pooled_manifest_path,
                    droid_data_dir=droid_data_dir,
                    evaluation_report_paths=(evaluation_path,),
                    output_dir=output_dir,
                    representatives_per_code=1,
                    thumbnail_size=(32, 18),
                )
            self.assertEqual(reader.call_count, 1)
            clip = first["clips"][0]
            self.assertEqual(clip["source_frame_indices"], [10, 18, 26])
            self.assertEqual(clip["latent_tick_indices"], [0, 2, 4])
            self.assertEqual(clip["source_anchor_rank"], 1)
            self.assertAlmostEqual(
                clip["rgb_motion"][
                    "first_last_mean_absolute_rgb_difference"
                ],
                16.0 / 255.0,
            )
            montage_path = Path(first["montages"][0]["path"])
            self.assertTrue(montage_path.is_file())
            self.assertEqual(
                first["montages"][0]["selection_summary"]["examples"],
                1,
            )
            self.assertEqual(
                first["montages"][0]["selection_summary"][
                    "codes_with_full_diversity"
                ],
                1,
            )
            with Image.open(montage_path) as image:
                self.assertEqual(image.size, (240, 102))

            with mock.patch(
                "codewam.codebook_eval.retrieval."
                "read_manifest_droid_rlds_frames",
                side_effect=AssertionError("resume decoded RGB again"),
            ):
                resumed = render_codebook_retrieval_montages(
                    source_manifest_path=source_manifest_path,
                    pooled_manifest_path=pooled_manifest_path,
                    droid_data_dir=droid_data_dir,
                    evaluation_report_paths=(evaluation_path,),
                    output_dir=output_dir,
                    representatives_per_code=1,
                    thumbnail_size=(32, 18),
                )
            self.assertEqual(first, resumed)


if __name__ == "__main__":
    unittest.main()
