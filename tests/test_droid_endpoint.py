from __future__ import annotations

import unittest

import torch

from codewam.data.droid_endpoint import (
    DROID_ENDPOINT_POLICY,
    audit_droid_endpoints,
)
from codewam.data.droid_rlds import DroidRLDSEpisode


def make_endpoint_episode() -> DroidRLDSEpisode:
    steps = 12
    joint_velocity = torch.zeros((steps, 7), dtype=torch.float32)
    cartesian_velocity = torch.zeros((steps, 6), dtype=torch.float32)
    for index in range(steps - 1):
        joint_velocity[index, index % 7] = 1.0 + index / 10.0
        cartesian_velocity[index, index % 3] = 1.0 + index / 20.0
    joint_position = torch.zeros((steps, 7), dtype=torch.float32)
    cartesian_position = torch.zeros((steps, 6), dtype=torch.float32)
    for index in range(steps - 1):
        joint_position[index + 1] = (
            joint_position[index] + joint_velocity[index]
        )
        cartesian_position[index + 1, :3] = (
            cartesian_position[index, :3]
            + cartesian_velocity[index, :3]
        )
    proprio = torch.cat(
        (
            cartesian_position,
            joint_position,
            torch.zeros((steps, 1)),
        ),
        dim=1,
    )
    is_first = torch.zeros(steps, dtype=torch.bool)
    is_last = torch.zeros(steps, dtype=torch.bool)
    is_terminal = torch.zeros(steps, dtype=torch.bool)
    is_first[0] = True
    is_last[-1] = True
    is_terminal[-1] = True
    frames = torch.zeros((steps, 2, 2, 3), dtype=torch.uint8)
    return DroidRLDSEpisode(
        episode_id="endpoint",
        index=0,
        frames={"wrist": frames},
        action=torch.cat(
            (joint_velocity[:, :6], torch.zeros((steps, 1))),
            dim=1,
        ),
        proprio=proprio,
        language_instruction="move",
        source_file="trajectory.h5",
        recording_folder="recordings",
        action_components={
            "joint_velocity": joint_velocity,
            "cartesian_velocity": cartesian_velocity,
        },
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        split="train",
        keep_ranges=((0, steps),),
    )


class DroidEndpointTests(unittest.TestCase):
    def test_current_action_aligns_with_successor_observation(self) -> None:
        report = audit_droid_endpoints((make_endpoint_episode(),))

        self.assertEqual(report["verdict"], "pass")
        self.assertEqual(report["endpoint_policy"], DROID_ENDPOINT_POLICY)
        self.assertGreater(
            report["joint_velocity_alignment"]["current_minus_shifted"],
            0.5,
        )
        self.assertGreater(
            report["cartesian_velocity_alignment"]["current_minus_shifted"],
            0.5,
        )

    def test_terminal_action_is_not_a_transition(self) -> None:
        episode = make_endpoint_episode()
        self.assertEqual(episode.action_valid.tolist(), [True] * 11 + [False])
        segment = next(episode.iter_eligible_segments())
        self.assertEqual(segment.action_valid.tolist(), [True] * 11 + [False])

    def test_partial_rlds_flags_are_rejected(self) -> None:
        episode = make_endpoint_episode()
        with self.assertRaisesRegex(ValueError, "incomplete RLDS flags"):
            DroidRLDSEpisode(
                episode_id="bad-flags",
                index=0,
                frames=episode.frames,
                action=episode.action,
                proprio=episode.proprio,
                language_instruction="move",
                source_file="trajectory.h5",
                recording_folder="recordings",
                is_first=episode.is_first,
            )


if __name__ == "__main__":
    unittest.main()
