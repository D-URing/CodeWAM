from __future__ import annotations

import unittest

import torch

from codewam.models import (
    ActionBatch,
    CodeMeasurements,
    CodeDynamicsDecoder,
    FutureCodePrediction,
    FutureCodeTargets,
    WorldBelief,
    decode_prefix_ids,
    encode_prefix_ids,
    future_code_metrics,
    persistence_code_metrics,
    transition_family_masks,
)


class CodeDynamicsTests(unittest.TestCase):
    def test_prefix_ids_round_trip_with_nonuniform_sizes(self) -> None:
        sizes = (2, 3, 5)
        values = torch.tensor(
            [[0, 0, 0], [1, 2, 4], [1, 0, 3]],
            dtype=torch.long,
        )
        encoded = encode_prefix_ids(values, sizes)
        self.assertEqual(encoded.tolist(), [0, 29, 18])
        torch.testing.assert_close(decode_prefix_ids(encoded, sizes), values)
        with self.assertRaisesRegex(ValueError, "outside"):
            decode_prefix_ids(torch.tensor([30]), sizes)

    def test_prediction_contract_rejects_wrong_head_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected 3"):
            FutureCodePrediction(
                mode="independent",
                logits=(torch.randn(2, 4),),
                families=("Q2",),
                codebook_sizes=((4, 4, 4),),
            )

    def test_independent_and_prefix_heads_share_target_contract(self) -> None:
        torch.manual_seed(11)
        belief = WorldBelief(tokens=torch.randn(2, 3, 16))
        actions = ActionBatch(values=torch.randn(2, 3, 7))
        targets = FutureCodeTargets(
            code_ids=torch.randint(0, 4, (2, 3, 3)),
            available=torch.ones((2, 3), dtype=torch.bool),
        )
        for mode, expected_heads in (("independent", 9), ("prefix", 3)):
            decoder = CodeDynamicsDecoder(
                dim=16,
                heads=4,
                action_dim=7,
                max_horizon=4,
                families=("Q2", "Q3", "Q5"),
                codebook_sizes=((4, 4, 4),) * 3,
                mode=mode,
                layers=1,
                action_layers=1,
            )
            prediction = decoder(belief, actions)
            self.assertEqual(len(prediction.logits), expected_heads)
            self.assertEqual(tuple(prediction.predicted_ids().shape), (2, 3, 3))
            loss = decoder.loss(
                prediction,
                targets,
                sample_valid=torch.tensor([True, False]),
            )
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(any(value.grad is not None for value in decoder.parameters()))

    def test_unavailable_negative_labels_are_ignored_and_metrics_are_finite(self) -> None:
        decoder = CodeDynamicsDecoder(
            dim=16,
            heads=4,
            action_dim=7,
            max_horizon=4,
            families=("Q2", "Q3", "Q5"),
            codebook_sizes=((4, 4, 4),) * 3,
            mode="prefix",
            layers=1,
            action_layers=1,
        )
        prediction = decoder(
            WorldBelief(tokens=torch.randn(2, 3, 16)),
            ActionBatch(values=torch.randn(2, 3, 7)),
        )
        targets = FutureCodeTargets(
            code_ids=torch.tensor(
                [
                    [[0, 1, 2], [-1, -1, -1], [3, 2, 1]],
                    [[1, 1, 1], [2, 2, 2], [0, 0, 0]],
                ],
                dtype=torch.long,
            ),
            available=torch.tensor(
                [[True, False, True], [True, True, True]],
                dtype=torch.bool,
            ),
        )
        sample_valid = torch.tensor([True, True])
        loss = decoder.loss(prediction, targets, sample_valid=sample_valid)
        self.assertTrue(torch.isfinite(loss))
        metrics = future_code_metrics(
            prediction,
            targets,
            sample_valid=sample_valid,
            calibration_bins=5,
        )
        self.assertEqual(metrics["count"], 5)
        self.assertEqual(metrics["family_count"], 5)
        self.assertEqual(metrics["classification_unit"], "family_prefix")
        for key in (
            "nll",
            "classification_nll",
            "family_prefix_nll",
            "classification_accuracy",
            "brier",
            "entropy",
            "ece",
        ):
            self.assertTrue(torch.isfinite(torch.tensor(metrics[key])))

    def test_normalized_nll_is_comparable_across_output_factorizations(self) -> None:
        targets = FutureCodeTargets(
            code_ids=torch.zeros((1, 1, 3), dtype=torch.long),
            available=torch.ones((1, 1), dtype=torch.bool),
        )
        level_logits = torch.tensor([[2.0, 0.0]])
        independent = FutureCodePrediction(
            mode="independent",
            logits=(level_logits, level_logits, level_logits),
            families=("Q2",),
            codebook_sizes=((2, 2, 2),),
        )
        expected_level_nll = torch.nn.functional.cross_entropy(
            level_logits,
            torch.zeros(1, dtype=torch.long),
        )
        target_probability = torch.exp(-3.0 * expected_level_nll)
        tuple_logits = torch.full(
            (1, 8),
            float(torch.log((1.0 - target_probability) / 7.0)),
        )
        tuple_logits[0, 0] = float(torch.log(target_probability))
        prefix = FutureCodePrediction(
            mode="prefix",
            logits=(tuple_logits,),
            families=("Q2",),
            codebook_sizes=((2, 2, 2),),
        )
        independent_metrics = future_code_metrics(
            independent,
            targets,
            sample_valid=torch.ones(1, dtype=torch.bool),
        )
        prefix_metrics = future_code_metrics(
            prefix,
            targets,
            sample_valid=torch.ones(1, dtype=torch.bool),
        )
        self.assertAlmostEqual(
            independent_metrics["nll"],
            prefix_metrics["nll"],
            places=5,
        )
        self.assertAlmostEqual(
            independent_metrics["family_prefix_nll"],
            prefix_metrics["family_prefix_nll"],
            places=5,
        )

    def test_fully_padded_action_sample_remains_finite(self) -> None:
        decoder = CodeDynamicsDecoder(
            dim=16,
            heads=4,
            action_dim=7,
            max_horizon=4,
            families=("Q2", "Q3", "Q5"),
            codebook_sizes=((4, 4, 4),) * 3,
            mode="independent",
            layers=1,
            action_layers=1,
        )
        prediction = decoder(
            WorldBelief(tokens=torch.randn(2, 3, 16)),
            ActionBatch(
                values=torch.randn(2, 3, 7),
                valid=torch.tensor(
                    [[False, False, False], [True, True, False]]
                ),
            ),
        )
        self.assertTrue(
            all(torch.isfinite(logits).all() for logits in prediction.logits)
        )

    def test_persistence_and_changed_family_metrics_are_separated(self) -> None:
        current_ids = torch.tensor(
            [
                [[0, 1, 2], [1, 1, 1], [2, 2, 2]],
                [[3, 3, 3], [0, 0, 0], [1, 1, 1]],
            ],
            dtype=torch.long,
        )
        future_ids = current_ids.clone()
        future_ids[0, 1, 2] = 2
        available = torch.tensor(
            [[True, True, True], [True, False, True]],
            dtype=torch.bool,
        )
        current_ids[1, 1] = -1
        future_ids[1, 1] = -1
        current = CodeMeasurements(
            code_ids=current_ids,
            available=available,
            chart_names=("droid", "droid"),
        )
        targets = FutureCodeTargets(
            code_ids=future_ids,
            available=available,
        )
        masks = transition_family_masks(current, targets)
        self.assertEqual(int(masks["common"].sum()), 5)
        self.assertEqual(int(masks["changed"].sum()), 1)
        persistence = persistence_code_metrics(
            current,
            targets,
            sample_valid=torch.ones(2, dtype=torch.bool),
        )
        self.assertEqual(persistence["family_count"], 5)
        self.assertAlmostEqual(persistence["family_prefix_accuracy"], 0.8)
        self.assertAlmostEqual(persistence["changed_family_fraction"], 0.2)

        decoder = CodeDynamicsDecoder(
            dim=16,
            heads=4,
            action_dim=7,
            max_horizon=4,
            families=("Q2", "Q3", "Q5"),
            codebook_sizes=((4, 4, 4),) * 3,
            mode="independent",
            layers=1,
            action_layers=1,
        )
        prediction = decoder(
            WorldBelief(tokens=torch.randn(2, 3, 16)),
            ActionBatch(values=torch.randn(2, 3, 7)),
        )
        changed = future_code_metrics(
            prediction,
            targets,
            sample_valid=torch.ones(2, dtype=torch.bool),
            family_valid=masks["changed"],
        )
        self.assertEqual(changed["family_count"], 1)
        self.assertEqual(changed["classification_count"], 3)


if __name__ == "__main__":
    unittest.main()
