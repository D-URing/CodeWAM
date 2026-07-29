from __future__ import annotations

import unittest

import torch

from codewam.codebook_eval.action_events import (
    _CategoricalPrefixTable,
    _ClassificationAccumulator,
    _direction_labels,
    _event_labels,
    _fit_event_thresholds,
)


class ActionEventTests(unittest.TestCase):
    def test_train_thresholds_and_labels_are_deterministic(self) -> None:
        values = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, -0.2],
                [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.3],
                [4.0, 0.0, 0.0, 0.0, -4.0, 0.0, 0.0],
            ]
        )
        thresholds = _fit_event_thresholds(values)
        labels = _event_labels(values, thresholds)

        self.assertEqual(
            labels["translation_magnitude_quartile"].tolist(),
            [0, 1, 2, 3],
        )
        self.assertEqual(
            labels["translation_direction"].tolist(),
            [0, 2, 2, 2],
        )
        self.assertEqual(
            labels["rotation_direction"].tolist(),
            [0, 3, 4, 3],
        )
        self.assertEqual(
            labels["gripper_change"].tolist(),
            [1, 0, 2, 1],
        )

    def test_direction_uses_signed_dominant_axis(self) -> None:
        vectors = torch.tensor(
            [
                [0.1, 0.0, 0.0],
                [-2.0, 1.0, 0.0],
                [0.0, 3.0, 1.0],
                [0.0, 0.0, -4.0],
            ]
        )
        self.assertEqual(
            _direction_labels(vectors, 0.5).tolist(),
            [0, 1, 4, 5],
        )

    def test_prefix_table_backs_off_to_supported_depth(self) -> None:
        table = _CategoricalPrefixTable(k=2, levels=2, classes=2)
        train_codes = torch.tensor(
            [[0, 0], [0, 0], [0, 1], [1, 0], [1, 0], [1, 1]]
        )
        train_labels = torch.tensor([0, 0, 1, 1, 1, 0])
        table.update(train_codes, train_labels)

        predictions, depth = table.predict(
            torch.tensor([[0, 0], [0, 1], [1, 1]]),
            depth=2,
            min_train_count=2,
        )

        self.assertEqual(predictions.tolist(), [0, 0, 1])
        self.assertEqual(depth.tolist(), [2, 1, 1])

    def test_classification_reports_gain_coverage_and_nmi(self) -> None:
        accumulator = _ClassificationAccumulator(
            classes=2,
            depth=1,
            train_global_label=0,
        )
        accumulator.update(
            prefix_keys=torch.tensor([0, 0, 1, 1]),
            labels=torch.tensor([0, 0, 1, 1]),
            predictions=torch.tensor([0, 0, 1, 1]),
            chosen_depth=torch.ones(4, dtype=torch.long),
            prefix_capacity=2,
        )

        row = accumulator.row()

        self.assertEqual(row["accuracy"], 1.0)
        self.assertEqual(row["balanced_accuracy"], 1.0)
        self.assertEqual(row["exact_prefix_coverage"], 1.0)
        self.assertAlmostEqual(
            row["normalized_mutual_information"],
            1.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
