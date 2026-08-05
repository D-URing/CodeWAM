from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from codewam.experiments.policy_ablation import (
    PolicyAblationRunConfig,
    _flow_inputs,
    fixed_eval_subset,
    paired_episode_bootstrap,
)
from codewam.models import CodeWAMConfig


def _config(**changes) -> PolicyAblationRunConfig:
    values = {
        "cache_dir": "cache",
        "language_cache_dir": "language",
        "normalization_dir": "normalization",
        "output_dir": "output",
        "artifact_paths": {"Q2": "q2", "Q3": "q3", "Q5": "q5"},
        "model": CodeWAMConfig(
            variant="C2",
            latent_channels=4,
            proprio_dim=17,
            action_dim=10,
            language_dim=8,
            dim=16,
            heads=4,
            max_time=2,
            max_cameras=1,
            max_spatial_tokens=4,
            max_action_horizon=2,
            lambda_code=0.1,
        ),
    }
    values.update(changes)
    return PolicyAblationRunConfig(**values)


class PolicyAblationTests(unittest.TestCase):
    def test_config_requires_c2_shared_base(self) -> None:
        config = _config()
        self.assertEqual(config.model.variant, "C2")
        with self.assertRaisesRegex(ValueError, "base model"):
            _config(model=CodeWAMConfig(variant="C1"))

    def test_config_selects_v2_explicitly(self) -> None:
        self.assertEqual(_config(architecture="v2").architecture, "v2")
        with self.assertRaisesRegex(ValueError, "architecture"):
            _config(architecture="latest")

    def test_fixed_eval_subset_is_order_independent(self) -> None:
        cache = SimpleNamespace(
            windows=tuple(
                SimpleNamespace(window_id=f"window-{index}")
                for index in range(10)
            )
        )
        first = fixed_eval_subset(
            cache,
            tuple(range(10)),
            split="test",
            seed=7,
            maximum=4,
        )
        second = fixed_eval_subset(
            cache,
            tuple(reversed(range(10))),
            split="test",
            seed=7,
            maximum=4,
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)

    def test_flow_noise_is_deterministic_and_phase_separated(self) -> None:
        actions = torch.zeros((2, 3, 4))
        first = _flow_inputs(actions, seed=11, phase="train", step=5, rank=0)
        second = _flow_inputs(actions, seed=11, phase="train", step=5, rank=0)
        other = _flow_inputs(actions, seed=11, phase="eval", step=5, rank=0)
        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])
        self.assertFalse(torch.equal(first[0], other[0]))

    def test_paired_bootstrap_uses_episode_means(self) -> None:
        candidate = {
            "a": {"sum": 1.0, "count": 2},
            "b": {"sum": 2.0, "count": 2},
        }
        baseline = {
            "a": {"sum": 2.0, "count": 2},
            "b": {"sum": 4.0, "count": 2},
        }
        report = paired_episode_bootstrap(
            candidate,
            baseline,
            samples=200,
            seed=3,
        )
        self.assertEqual(report["episodes"], 2)
        self.assertAlmostEqual(
            report["mean_delta_candidate_minus_baseline"],
            -0.75,
        )
        self.assertEqual(report["episode_win_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
