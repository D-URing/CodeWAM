from __future__ import annotations

import json
import unittest

import torch

from codewam.codebook_eval.droid_motion_audit import (
    DroidMotionAuditConfig,
    aggregate_motion_audit,
    episode_motion_measurements,
    select_droid_motion_audit_manifest,
)
from codewam.codebook_eval.manifest import EpisodeManifest, EpisodeRecord
from codewam.data.droid_rlds import DroidRLDSEpisode


def make_record(institution: str, scene: int, episode: int) -> EpisodeRecord:
    return EpisodeRecord(
        dataset="droid-1.0.1",
        episode_id=f"{institution}-{episode}",
        num_steps=64,
        source_uri=f"gs://droid/{institution}/{episode}/trajectory.h5",
        institution_id=institution,
        building_id="building",
        scene_id=f"scene-{scene}",
        split="train",
        metadata={
            "collector_id": f"collector-{episode}",
            "keep_ranges": [[16, 64]],
            "eligible_steps": 48,
        },
    )


class DroidMotionAuditTests(unittest.TestCase):
    def test_selection_is_institution_balanced_and_scene_distinct(self) -> None:
        manifest = EpisodeManifest.from_records(
            make_record(institution, episode, episode)
            for institution in ("a", "b")
            for episode in range(3)
        )
        config = DroidMotionAuditConfig(
            cameras=("camera",),
            episodes_per_institution=2,
        )
        selected = select_droid_motion_audit_manifest(manifest, config)

        self.assertEqual(len(selected), 4)
        for institution in ("a", "b"):
            records = [
                record for record in selected if record.institution_id == institution
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual(len({record.scene_id for record in records}), 2)

    def test_motion_metrics_separate_kept_motion_from_idle_frames(self) -> None:
        frames = torch.zeros((12, 4, 4, 3), dtype=torch.uint8)
        for index in range(4, 12):
            frames[index] = 20 * (index - 4)
        action = torch.zeros((12, 7))
        action[4:] = 2.0
        proprio = torch.zeros((12, 14))
        proprio[4:] = torch.arange(8).view(8, 1)
        episode = DroidRLDSEpisode(
            episode_id="institution-0",
            index=0,
            frames={"camera": frames},
            action=action,
            proprio=proprio,
            language_instruction="move",
            source_file="trajectory.h5",
            recording_folder="recordings",
            action_components={
                "cartesian_velocity": action[:, :6].clone(),
                "gripper_position": action[:, :1].clone(),
            },
            split="train",
            keep_ranges=((4, 12),),
            manifest_key="droid-1.0.1:institution-0",
        )

        row = episode_motion_measurements(episode, thumbnail_size=2)
        self.assertEqual(
            row["transition_counts"],
            {"inside": 7, "outside": 3, "boundary": 1},
        )
        self.assertGreater(
            row["metrics"]["image_motion/camera"][
                "inside_outside_median_ratio"
            ],
            1.0,
        )
        self.assertGreater(
            row["metrics"]["action_velocity/cartesian"][
                "inside_outside_median_ratio"
            ],
            1.0,
        )

        record = EpisodeRecord(
            dataset="droid-1.0.1",
            episode_id=episode.episode_id,
            num_steps=12,
            source_uri="trajectory.h5",
            institution_id="institution",
            building_id="building",
            scene_id="scene",
            split="train",
            metadata={
                "keep_ranges": [[4, 12]],
                "eligible_steps": 8,
                "collector_id": "collector",
            },
        )
        manifest = EpisodeManifest.from_records((record,))
        report = aggregate_motion_audit(
            manifest,
            (episode,),
            DroidMotionAuditConfig(
                cameras=("camera",),
                episodes_per_institution=1,
                minimum_idle_steps=0,
                minimum_active_segment_steps=1,
            ),
        )
        self.assertEqual(report["selection"]["episodes"], 1)
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
