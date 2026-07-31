from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

import codewam.data
from codewam.codebook_eval.manifest import EpisodeRecord
from codewam.data.roles import (
    TrajectoryRole,
    build_supervision_masks,
    role_supervision,
    trajectory_role,
)
from codewam.models import ActionBatch, CodeMeasurements, CodeWAMConfig, StateInputs


class ModelContractTests(unittest.TestCase):
    def test_data_public_exports_preserve_existing_and_role_apis(self) -> None:
        self.assertIn("PackageScanV6Dataset", codewam.data.__all__)
        self.assertIn("TrajectoryRole", codewam.data.__all__)

    def test_canonical_config_instantiates_without_legacy_runtime(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "model"
            / "codewam_v1.yaml"
        )
        config = instantiate(OmegaConf.load(path))
        self.assertIsInstance(config, CodeWAMConfig)
        self.assertEqual(config, CodeWAMConfig())

    def test_public_v1_import_does_not_load_fastwam(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "from codewam import CodeWAMConfig, CodeWAMV1, build_codewam_v1; "
                    "assert 'fastwam' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_joint_cache_import_does_not_load_optional_video_decoder(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "from codewam.data.joint_cache import JointWindowCache; "
                    "assert JointWindowCache is not None; "
                    "assert 'av' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_state_and_code_measurements_reject_misaligned_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "6D"):
            StateInputs(
                latents=torch.zeros(2, 3, 4),
                proprio_history=torch.zeros(2, 1, 6),
                past_actions=torch.zeros(2, 0, 7),
            )
        with self.assertRaisesRegex(ValueError, "availability"):
            CodeMeasurements(
                code_ids=torch.zeros((2, 3, 3), dtype=torch.long),
                available=torch.ones((2, 2), dtype=torch.bool),
                chart_names=("a", "a"),
            )
        with self.assertRaisesRegex(ValueError, "horizon"):
            ActionBatch(values=torch.zeros((2, 0, 7)))

    def test_config_rejects_invalid_training_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "layer counts"):
            CodeWAMConfig(state_spatial_layers=0)
        with self.assertRaisesRegex(ValueError, "dropout"):
            CodeWAMConfig(dropout=1.0)
        with self.assertRaisesRegex(ValueError, "lambda_code"):
            CodeWAMConfig(lambda_code=float("nan"))

    def test_data_roles_keep_world_and_policy_supervision_separate(self) -> None:
        masks = build_supervision_masks(
            (
                TrajectoryRole.EXPERT,
                TrajectoryRole.FAILURE,
                TrajectoryRole.RECOVERY,
                TrajectoryRole.ACTION_FREE_VIDEO,
            )
        )
        self.assertEqual(masks.action.tolist(), [True, False, True, False])
        self.assertEqual(masks.dynamics.tolist(), [True, True, True, False])
        self.assertEqual(masks.temporal.tolist(), [True, True, True, True])
        self.assertTrue(role_supervision(TrajectoryRole.FAILURE).codebook_fit)

    def test_action_availability_gates_action_dependent_objectives(self) -> None:
        masks = build_supervision_masks(
            (
                TrajectoryRole.EXPERT,
                TrajectoryRole.FAILURE,
                TrajectoryRole.UNLABELED_INTERACTION,
            ),
            action_available=(False, True, False),
        )
        self.assertEqual(masks.temporal.tolist(), [True, True, True])
        self.assertEqual(masks.action.tolist(), [False, False, False])
        self.assertEqual(masks.dynamics.tolist(), [False, True, False])

    def test_episode_role_requires_explicit_or_narrow_fallback_metadata(self) -> None:
        base = dict(
            dataset="test",
            episode_id="episode",
            num_steps=3,
            source_uri="/tmp/episode",
        )
        self.assertEqual(
            trajectory_role(EpisodeRecord(**base, metadata={"success": False})),
            TrajectoryRole.FAILURE,
        )
        self.assertEqual(
            trajectory_role(
                EpisodeRecord(
                    **base,
                    metadata={"trajectory_role": "unlabeled_interaction"},
                )
            ),
            TrajectoryRole.UNLABELED_INTERACTION,
        )
        with self.assertRaisesRegex(ValueError, "explicit"):
            trajectory_role(EpisodeRecord(**base))


if __name__ == "__main__":
    unittest.main()
