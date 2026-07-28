from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from codewam.codebook_eval.manifest import EpisodeManifest, EpisodeRecord
from codewam.data.droid_manifest import (
    balanced_scene_sample,
    build_droid_manifest,
    canonical_droid_episode_path,
    shard_aware_balanced_sample,
)


def write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, mode="wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


class DroidManifestBuildTests(unittest.TestCase):
    def test_official_sources_join_with_explicit_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata_path = root / "metadata.jsonl.gz"
            rlds_path = root / "rlds.jsonl.gz"
            keep_path = root / "keep.json"
            language_path = root / "language.json"
            gcs_path = root / "gcs_metadata.txt"

            metadata_rows: list[dict] = []
            rlds_rows: list[dict] = []
            keep_rows: dict[str, list[list[int]]] = {}
            language_rows: dict[str, dict[str, str]] = {}
            shard_names: set[str] = set()

            for index in range(12):
                success = index != 2
                outcome = "success" if success else "failure"
                episode_path = (
                    f"Lab/{outcome}/2024-01-01/"
                    f"Mon_Jan__1_00:{index:02d}:00_2024"
                )
                episode_id = f"Lab+collector-{index % 3}+episode-{index:02d}"
                shard_index = index % 2
                shard_name = (
                    f"droid_101-train.tfrecord-{shard_index:05d}-of-02048"
                )
                shard_names.add(shard_name)
                file_path = (
                    "gs://xembodiment_data/r2d2/r2d2-data-full/"
                    f"{episode_path}/trajectory.h5"
                )
                recording_path = (
                    "gs://xembodiment_data/r2d2/r2d2-data-full/"
                    f"{episode_path}/recordings/MP4"
                )
                metadata_rows.append(
                    {
                        "schema": "codewam.droid-raw-metadata.v1",
                        "dataset_revision": "droid-1.0.1",
                        "episode_id": episode_id,
                        "episode_path": episode_path,
                        "institution_id": "Lab",
                        "building_id": f"building-{index % 2}",
                        "scene_id": f"scene-{index // 2}",
                        "collector_id": f"collector-{index % 3}",
                        "success": success,
                        "metadata_success": success,
                        "quality_flags": (
                            ["synthetic_quality_flag"] if index == 1 else []
                        ),
                        "task": f"task {index % 4}",
                        "num_steps": 25 if index == 3 else 24,
                        "camera_ids": {
                            "wrist": "wrist",
                            "exterior_1": "ext1",
                            "exterior_2": "ext2",
                        },
                        "metadata_object": f"gs://gresearch/{episode_path}/metadata.json",
                        "metadata_sha256": f"{index:064x}",
                    }
                )
                rlds_rows.append(
                    {
                        "schema": "codewam.droid-rlds-shard-index.v1",
                        "dataset_revision": "droid-1.0.1",
                        "shard_index": shard_index,
                        "record_index": index,
                        "shard_name": shard_name,
                        "file_path": file_path,
                        "recording_folderpath": recording_path,
                        "episode_key": f"{recording_path}--{file_path}",
                        "episode_path": episode_path,
                        "num_steps": 24,
                    }
                )
                keep_rows[f"{recording_path}--{file_path}"] = [[0, 8], [12, 20]]
                language_rows[episode_id] = {
                    "language_instruction1": f"perform task {index % 4}",
                    "language_instruction2": f"do task {index % 4}",
                }

            sixth_key = next(
                key
                for key in keep_rows
                if "00:06:00_2024" in key
            )
            keep_rows[sixth_key] = []
            metadata_rows[5]["episode_id"] = metadata_rows[4]["episode_id"]
            duplicate = dict(metadata_rows[0])
            duplicate["episode_id"] = "duplicate-id"
            duplicate["metadata_sha256"] = "f" * 64
            metadata_rows.append(duplicate)

            missing_path = "Lab/success/2024-01-02/Tue_Jan__2_00:00:00_2024"
            missing_file = (
                "gs://xembodiment_data/r2d2/r2d2-data-full/"
                f"{missing_path}/trajectory.h5"
            )
            missing_recording = (
                "gs://xembodiment_data/r2d2/r2d2-data-full/"
                f"{missing_path}/recordings/MP4"
            )
            keep_rows[f"{missing_recording}--{missing_file}"] = [[0, 16]]
            rlds_rows.append(
                {
                    "schema": "codewam.droid-rlds-shard-index.v1",
                    "dataset_revision": "droid-1.0.1",
                    "shard_index": 0,
                    "record_index": 99,
                    "shard_name": sorted(shard_names)[0],
                    "file_path": missing_file,
                    "recording_folderpath": missing_recording,
                    "episode_key": f"{missing_recording}--{missing_file}",
                    "episode_path": missing_path,
                    "num_steps": 20,
                }
            )

            write_jsonl_gz(metadata_path, metadata_rows)
            write_jsonl_gz(rlds_path, rlds_rows)
            keep_path.write_text(json.dumps(keep_rows), encoding="utf-8")
            language_path.write_text(json.dumps(language_rows), encoding="utf-8")
            gcs_path.write_text(
                "\n".join(
                    f"gs://gresearch/robotics/droid/1.0.1/{name}:\n"
                    f"    Content-Length:         {1000 + index}\n"
                    f"    Hash (crc32c): checksum-{index}"
                    for index, name in enumerate(sorted(shard_names))
                )
                + "\n",
                encoding="utf-8",
            )

            result = build_droid_manifest(
                metadata_index=metadata_path,
                rlds_index=rlds_path,
                keep_ranges=keep_path,
                language_annotations=language_path,
                gcs_metadata=gcs_path,
            )

        self.assertEqual(len(result.manifest), 6)
        self.assertEqual(
            result.report["excluded"],
            {
                "ambiguous_raw_metadata_id": 2,
                "ambiguous_raw_metadata_path": 1,
                "failure_episode": 1,
                "missing_raw_metadata": 1,
                "no_eligible_keep_ranges": 1,
                "raw_metadata_quality_flag": 1,
            },
        )
        self.assertEqual(result.report["raw_vs_rlds_length_mismatches"], 1)
        self.assertEqual(
            result.report["raw_vs_rlds_length_delta_counts"],
            {"-1": 1, "0": 5},
        )
        record = next(item for item in result.manifest if item.metadata["raw_num_steps"] == 25)
        self.assertEqual(record.num_steps, 24)
        self.assertEqual(record.metadata["eligible_steps"], 16)
        self.assertEqual(record.metadata["keep_ranges"], [[0, 8], [12, 20]])
        self.assertTrue(record.source_checksum.startswith("crc32c:checksum-"))
        self.assertEqual(len(record.camera_ids), 3)
        self.assertTrue(record.task_ids)

    def test_canonical_path_accepts_rlds_uris(self) -> None:
        value = (
            "/nfs/data/r2d2-data-full/RAIL/success/2023-04-17/"
            "Mon_Apr_17_14:48:05_2023/trajectory.h5"
        )
        self.assertEqual(
            canonical_droid_episode_path(value),
            "RAIL/success/2023-04-17/Mon_Apr_17_14:48:05_2023",
        )


class BalancedSceneSampleTests(unittest.TestCase):
    def build_manifest(self) -> EpisodeManifest:
        records: list[EpisodeRecord] = []
        split_scenes = {"train": 4, "val": 2, "test": 2}
        for split, scene_count in split_scenes.items():
            for scene in range(scene_count):
                for episode in range(5):
                    records.append(
                        EpisodeRecord(
                            dataset="droid-1.0.1",
                            episode_id=f"{split}-s{scene}-e{episode}",
                            num_steps=32,
                            source_uri=f"memory://{split}/{scene}/{episode}",
                            institution_id=f"site-{scene % 2}",
                            building_id=f"building-{scene % 2}",
                            scene_id=f"{split}-scene-{scene}",
                            task_ids=(f"task-{episode % 3}",),
                            camera_ids=("exterior", "wrist"),
                            split=split,
                            metadata={
                                "collector_id": f"collector-{episode % 2}",
                                "rlds_shard_name": f"shard-{episode}",
                                "rlds_shard_bytes": 100,
                            },
                        )
                    )
        return EpisodeManifest.from_records(records)

    def test_sample_is_exact_deterministic_and_scene_balanced(self) -> None:
        manifest = self.build_manifest()
        first = balanced_scene_sample(manifest, 20, salt="unit-test")
        second = balanced_scene_sample(
            EpisodeManifest.from_records(reversed(manifest.records)),
            20,
            salt="unit-test",
        )

        self.assertEqual(first.fingerprint(), second.fingerprint())
        self.assertEqual(Counter(record.split for record in first), {
            "train": 16,
            "val": 2,
            "test": 2,
        })
        first.assert_group_isolation("scene")
        train_scene_counts = Counter(
            record.scene_id for record in first if record.split == "train"
        )
        self.assertEqual(set(train_scene_counts.values()), {4})
        self.assertEqual(
            Counter(
                record.institution_id
                for record in first
                if record.split == "train"
            ),
            {"site-0": 8, "site-1": 8},
        )

    def test_shard_aware_sample_limits_source_reads(self) -> None:
        result = shard_aware_balanced_sample(
            self.build_manifest(),
            20,
            salt="shard-test",
            candidate_multiplier=1.0,
        )

        self.assertEqual(len(result.manifest), 20)
        self.assertEqual(result.report["source_shards_available"], 5)
        self.assertEqual(result.report["source_shards_selected"], 4)
        self.assertEqual(result.report["selected_source_bytes"], 400)
        self.assertEqual(
            Counter(record.split for record in result.manifest),
            {"train": 16, "val": 2, "test": 2},
        )
        self.assertEqual(
            result.report["institution_candidate_targets"]["train"],
            {"site-0": 8, "site-1": 8},
        )


if __name__ == "__main__":
    unittest.main()
