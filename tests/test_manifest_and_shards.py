from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from codewam.codebook_eval.manifest import (
    EpisodeManifest,
    EpisodeRecord,
    SplitConfig,
)
from codewam.codebook_eval.shards import (
    PooledFeatureEpisode,
    iter_pooled_feature_episodes,
    write_pooled_feature_shard,
)


def make_episode(
    episode_id: str,
    split: str = "train",
    ticks: int = 8,
    offset: float = 0.0,
) -> PooledFeatureEpisode:
    values = torch.arange(ticks, dtype=torch.float32) + float(offset)
    pooled = values.view(ticks, 1, 1, 1, 1).expand(ticks, 2, 3, 4, 4).half()
    return PooledFeatureEpisode(
        episode_id=episode_id,
        split=split,
        timestamps=torch.arange(ticks, dtype=torch.float64) / 4.0,
        pooled_g4=pooled,
        camera_ids=("exterior", "wrist"),
        action=torch.zeros((ticks, 7)),
        proprio=torch.ones((ticks, 7)),
        metadata={"source": "synthetic"},
    )


class EpisodeManifestTests(unittest.TestCase):
    def build_manifest(self) -> EpisodeManifest:
        records = []
        for scene_index in range(40):
            for episode_index in range(2):
                records.append(
                    EpisodeRecord(
                        dataset="synthetic",
                        episode_id=f"scene-{scene_index}-episode-{episode_index}",
                        num_steps=20 + episode_index,
                        source_uri=f"memory://scene-{scene_index}/{episode_index}",
                        institution_id=f"site-{scene_index % 4}",
                        building_id=f"building-{scene_index % 8}",
                        scene_id=f"scene-{scene_index}",
                        task_ids=(f"task-{scene_index % 3}",),
                        camera_ids=("exterior", "wrist"),
                    )
                )
        return EpisodeManifest.from_records(records)

    def test_scene_split_is_deterministic_and_isolated(self) -> None:
        manifest = self.build_manifest()
        config = SplitConfig(salt="unit-test")
        first = manifest.assign_splits(config)
        reversed_manifest = EpisodeManifest.from_records(reversed(manifest.records))
        second = reversed_manifest.assign_splits(config)

        first_mapping = {record.key: record.split for record in first}
        second_mapping = {record.key: record.split for record in second}
        self.assertEqual(first_mapping, second_mapping)
        first.assert_group_isolation("scene")
        self.assertGreater(len(first.select("train")), 0)
        self.assertGreater(len(first.select("val")), 0)
        self.assertGreater(len(first.select("test")), 0)

        by_scene: dict[str, set[str]] = {}
        for record in first:
            by_scene.setdefault(record.scene_id or "", set()).add(record.split or "")
        self.assertTrue(all(len(splits) == 1 for splits in by_scene.values()))

    def test_jsonl_round_trip_preserves_fingerprint(self) -> None:
        manifest = self.build_manifest().assign_splits(SplitConfig(salt="round-trip"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.jsonl"
            manifest.write_jsonl(path)
            loaded = EpisodeManifest.read_jsonl(path)
        self.assertEqual(manifest.fingerprint(), loaded.fingerprint())
        self.assertEqual(manifest.stats(), loaded.stats())

    def test_duplicate_episode_key_is_rejected(self) -> None:
        record = EpisodeRecord(
            dataset="synthetic",
            episode_id="duplicate",
            num_steps=4,
            source_uri="memory://duplicate",
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            EpisodeManifest((record, record))


class PooledFeatureShardTests(unittest.TestCase):
    def test_shard_round_trip_and_split_filter(self) -> None:
        train = make_episode("train-episode", split="train", offset=0.0)
        test = make_episode("test-episode", split="test", offset=100.0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pooled-00000.pt"
            info = write_pooled_feature_shard(
                path,
                [train, test],
                metadata={
                    "dataset_revision": "synthetic-v1",
                    "wan_model_id": "test-wan",
                    "wan_revision": "test-wan-revision",
                    "preprocess_revision": "test-preprocess",
                    "source_checksums": ["synthetic-checksum"],
                },
            )
            loaded = list(iter_pooled_feature_episodes([path], split="train"))

        self.assertEqual(info.episodes, 2)
        self.assertEqual(info.ticks, train.ticks + test.ticks)
        self.assertEqual(len(info.sha256), 64)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].episode_id, train.episode_id)
        torch.testing.assert_close(loaded[0].pooled_g4, train.pooled_g4)
        self.assertEqual(tuple(loaded[0].pooled(2).shape), (8, 2, 3, 2, 2))
        torch.testing.assert_close(
            loaded[0].pooled(1).float().flatten(),
            train.pooled_g4.float().mean(dim=(-1, -2)).flatten(),
        )

    def test_non_monotonic_timestamps_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            PooledFeatureEpisode(
                episode_id="bad-time",
                split="train",
                timestamps=torch.tensor([0.0, 1.0, 0.5]),
                pooled_g4=torch.zeros((3, 1, 1, 4, 4)),
                camera_ids=("exterior",),
            )

    def test_non_finite_pooled_feature_is_rejected(self) -> None:
        pooled = torch.zeros((3, 1, 1, 4, 4))
        pooled[1, 0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            PooledFeatureEpisode(
                episode_id="bad-feature",
                split="train",
                timestamps=torch.arange(3, dtype=torch.float32),
                pooled_g4=pooled,
                camera_ids=("exterior",),
            )


if __name__ == "__main__":
    unittest.main()
