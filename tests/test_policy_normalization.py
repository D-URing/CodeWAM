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
