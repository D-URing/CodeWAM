from __future__ import annotations

import unittest

import torch

from codewam.codebook_eval.manifest import EpisodeManifest, EpisodeRecord
from codewam.data.droid_rlds import (
    DroidRLDSEpisode,
    plan_droid_rank_assignments,
    read_manifest_droid_rlds_frames,
)


def make_record(
    shard: str,
    shard_bytes: int,
    record_index: int,
    episode_index: int,
) -> EpisodeRecord:
    return EpisodeRecord(
        dataset="droid-1.0.1",
        episode_id=f"episode-{episode_index}",
        num_steps=10,
        source_uri=f"gs://droid/episode-{episode_index}/trajectory.h5",
        institution_id="institution",
        building_id="building",
        scene_id=f"scene-{episode_index}",
        split="train",
        metadata={
            "rlds_shard_name": shard,
            "rlds_shard_bytes": shard_bytes,
            "rlds_record_index": record_index,
            "recording_folderpath": (
                f"gs://droid/episode-{episode_index}/recordings/MP4"
            ),
            "keep_ranges": [[1, 4], [6, 9]],
            "eligible_steps": 6,
        },
    )


class DroidRankAssignmentTests(unittest.TestCase):
    def build_manifest(self) -> EpisodeManifest:
        records = []
        episode_index = 0
        for shard_index, shard_bytes in enumerate((100, 90, 80, 70, 60)):
            for record_index in range(1 + shard_index % 2):
                records.append(
                    make_record(
                        f"shard-{shard_index}",
                        shard_bytes,
                        record_index,
                        episode_index,
                    )
                )
                episode_index += 1
        return EpisodeManifest.from_records(records)

    def test_whole_shards_are_deterministically_balanced_across_ranks(self) -> None:
        manifest = self.build_manifest()
        first = plan_droid_rank_assignments(manifest, world_size=2)
        second = plan_droid_rank_assignments(
            EpisodeManifest.from_records(reversed(manifest.records)),
            world_size=2,
        )

        self.assertEqual(first, second)
        assigned_shards = [
            shard.shard_name
            for assignment in first
            for shard in assignment.shards
        ]
        self.assertEqual(len(assigned_shards), len(set(assigned_shards)))
        self.assertEqual(set(assigned_shards), {f"shard-{index}" for index in range(5)})
        self.assertEqual(sum(value.episodes for value in first), len(manifest))
        loads = [value.source_bytes for value in first]
        self.assertLessEqual(max(loads) - min(loads), 100)

    def test_duplicate_shard_position_is_rejected(self) -> None:
        first = make_record("shard", 100, 0, 0)
        second = make_record("shard", 100, 0, 1)
        manifest = EpisodeManifest.from_records((first, second))
        with self.assertRaisesRegex(ValueError, "Duplicate DROID shard position"):
            plan_droid_rank_assignments(manifest, world_size=1)

    def test_sparse_frame_requests_validate_before_tensorflow_io(self) -> None:
        manifest = EpisodeManifest.from_records(
            (make_record("shard", 100, 0, 0),)
        )
        key = manifest.records[0].key

        self.assertEqual(
            read_manifest_droid_rlds_frames(
                "/missing",
                manifest,
                {},
            ),
            {},
        )
        with self.assertRaisesRegex(KeyError, "Unknown DROID manifest keys"):
            read_manifest_droid_rlds_frames(
                "/missing",
                manifest,
                {"droid-1.0.1:missing": [0]},
            )
        with self.assertRaisesRegex(IndexError, r"outside \[0, 10\)"):
            read_manifest_droid_rlds_frames(
                "/missing",
                manifest,
                {key: [10]},
            )


class DroidEligibleSegmentTests(unittest.TestCase):
    def make_episode(
        self,
        keep_ranges: tuple[tuple[int, int], ...],
    ) -> DroidRLDSEpisode:
        values = torch.arange(10, dtype=torch.uint8)
        frames = values.view(10, 1, 1, 1).expand(10, 2, 2, 3).contiguous()
        return DroidRLDSEpisode(
            episode_id="episode",
            index=3,
            frames={"exterior": frames},
            action=torch.arange(70, dtype=torch.float32).view(10, 7),
            proprio=torch.arange(140, dtype=torch.float32).view(10, 14),
            language_instruction="move",
            source_file="trajectory.h5",
            recording_folder="recordings",
            action_components={
                "cartesian_velocity": torch.arange(
                    60,
                    dtype=torch.float32,
                ).view(10, 6),
            },
            split="train",
            keep_ranges=keep_ranges,
            manifest_key="droid-1.0.1:episode",
            source_shard="shard",
            record_index=3,
        )

    def test_keep_ranges_remain_independent_segments(self) -> None:
        episode = self.make_episode(((1, 4), (6, 9)))
        segments = list(episode.iter_eligible_segments())

        self.assertEqual([value.segment_id for value in segments], [
            "episode@1:4",
            "episode@6:9",
        ])
        self.assertEqual([value.steps for value in segments], [3, 3])
        self.assertEqual(int(segments[0].frames["exterior"][0, 0, 0, 0]), 1)
        self.assertEqual(int(segments[1].frames["exterior"][0, 0, 0, 0]), 6)
        self.assertEqual(sum(value.steps for value in segments), 6)
        self.assertTrue(all(value.manifest_key == episode.manifest_key for value in segments))
        self.assertEqual(
            float(segments[1].action_components["cartesian_velocity"][0, 0]),
            36.0,
        )

    def test_overlapping_keep_ranges_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid keep range"):
            self.make_episode(((1, 5), (4, 8)))


if __name__ == "__main__":
    unittest.main()
