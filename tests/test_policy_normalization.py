from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from codewam.data import (
    PolicyNormalizer,
    create_policy_normalization_contract,
    decode_droid_actions,
    encode_droid_actions,
    encode_droid_proprio,
    moments_from_sums,
    write_policy_normalization,
)
from codewam.models import (
    ActionBatch,
    CodeWAMBatch,
    FutureCodeTargets,
    PolicyCondition,
    StateInputs,
    SupervisionMasks,
    TransitionSchedule,
)


class PolicyNormalizationTests(unittest.TestCase):
    def test_action_representation_round_trip_handles_angle_wrap(self) -> None:
        raw = torch.tensor(
            [[0.1, 0.2, 0.3, -3.13, 0.5, 3.13, 0.8]],
            dtype=torch.float32,
        )
        encoded = encode_droid_actions(raw)
        decoded = decode_droid_actions(encoded)
        torch.testing.assert_close(decoded, raw)
        self.assertEqual(tuple(encoded.shape), (1, 10))

    def test_proprio_representation_preserves_joint_state(self) -> None:
        raw = torch.arange(28, dtype=torch.float32).reshape(2, 14) / 10
        encoded = encode_droid_proprio(raw)
        self.assertEqual(tuple(encoded.shape), (2, 17))
        torch.testing.assert_close(encoded[:, 9:], raw[:, 6:])

    def test_normalizer_round_trip_and_contract_binding(self) -> None:
        contract = create_policy_normalization_contract(
            joint_cache_contract_hash="joint",
            joint_cache_summary_sha256="summary",
            implementation_sha256={"normalization": "implementation"},
        )
        with tempfile.TemporaryDirectory() as temporary:
            write_policy_normalization(
                temporary,
                contract=contract,
                action_mean=torch.linspace(-0.2, 0.2, 10),
                action_std=torch.linspace(0.5, 1.4, 10),
                proprio_mean=torch.linspace(-0.3, 0.3, 17),
                proprio_std=torch.linspace(0.5, 2.1, 17),
                action_rows=100,
                proprio_rows=101,
                source_segments=2,
            )
            normalizer = PolicyNormalizer(
                temporary,
                expected_joint_cache_contract_hash="joint",
            )
            raw = torch.tensor(
                [[0.1, 0.2, 0.3, -2.9, 0.5, 2.9, 0.8]],
                dtype=torch.float32,
            )
            decoded = normalizer.denormalize_actions(
                normalizer.normalize_actions(raw)
            )
            torch.testing.assert_close(decoded, raw, atol=1e-6, rtol=1e-6)
            schedule = TransitionSchedule(
                action_prefix_lengths=torch.tensor([[1, 2, 2]]),
                delta_times=torch.tensor([[0.1, 0.2, 0.2]]),
            )
            latent_time = torch.tensor([[-0.1, 0.0]])
            proprio_time = torch.tensor([[-0.1, 0.0]])
            action_time = torch.tensor([[-0.1]])
            batch = CodeWAMBatch(
                state=StateInputs(
                    latents=torch.zeros((1, 2, 1, 4, 2, 2)),
                    proprio_history=torch.zeros((1, 2, 14)),
                    past_actions=torch.zeros((1, 1, 7)),
                    latent_time_offsets=latent_time,
                    proprio_time_offsets=proprio_time,
                    past_action_time_offsets=action_time,
                ),
                policy=PolicyCondition(language=torch.zeros((1, 1, 8))),
                actions=ActionBatch(values=torch.zeros((1, 2, 7))),
                supervision=SupervisionMasks(
                    temporal=torch.ones(1, dtype=torch.bool),
                    action=torch.ones(1, dtype=torch.bool),
                    dynamics=torch.ones(1, dtype=torch.bool),
                ),
                future_codes=FutureCodeTargets(
                    code_ids=torch.zeros((1, 3, 3), dtype=torch.long),
                    available=torch.ones((1, 3), dtype=torch.bool),
                    schedule=schedule,
                ),
            )
            transformed = normalizer.transform_batch(batch)
            self.assertIs(transformed.state.latent_time_offsets, latent_time)
            self.assertIs(transformed.state.proprio_time_offsets, proprio_time)
            self.assertIs(transformed.state.past_action_time_offsets, action_time)
            self.assertIs(transformed.future_codes.schedule, schedule)
            with self.assertRaisesRegex(RuntimeError, "different joint cache"):
                PolicyNormalizer(
                    temporary,
                    expected_joint_cache_contract_hash="other",
                )

    def test_moments_use_population_variance(self) -> None:
        rows = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
        mean, std = moments_from_sums(
            2,
            rows.double().sum(0),
            rows.double().square().sum(0),
        )
        torch.testing.assert_close(mean, torch.tensor([2.0, 4.0]))
        torch.testing.assert_close(std, torch.tensor([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
