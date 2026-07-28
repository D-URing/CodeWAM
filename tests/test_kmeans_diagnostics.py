from __future__ import annotations

import unittest

import torch

from codewam.codebook_eval.kmeans_diagnostics import (
    DiagnosticKMeansConfig,
    adjusted_rand_index,
    fit_diagnostic_kmeans,
    fit_diagnostic_rq,
    kmeans_plus_plus_gpu,
    usage_summary,
)


class DiagnosticKMeansTests(unittest.TestCase):
    @staticmethod
    def clusters() -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(41)
        centers = torch.tensor([[-4.0, -1.0], [0.5, 4.0], [4.0, -0.5]])
        values = torch.cat(
            [
                center + 0.15 * torch.randn((100, 2), generator=generator)
                for center in centers
            ]
        )
        return values[:240], values[240:]

    def test_gpu_compatible_kmeans_plus_plus_is_deterministic_on_cpu(self) -> None:
        train, _ = self.clusters()
        first = kmeans_plus_plus_gpu(train, k=3, seed=7)
        second = kmeans_plus_plus_gpu(train, k=3, seed=7)
        torch.testing.assert_close(first, second)

    def test_history_records_early_stop_and_validation(self) -> None:
        train, validation = self.clusters()
        result = fit_diagnostic_kmeans(
            train,
            validation,
            DiagnosticKMeansConfig(
                k=3,
                max_iters=30,
                min_iters=2,
                tol=1e-5,
                patience=2,
                seed=3,
                device="cpu",
            ),
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.stop_reason, "inertia_plateau")
        self.assertLess(result.iterations, 30)
        self.assertIsNotNone(result.validation_inertia)
        self.assertIsNotNone(result.history[-1].assignment_change)
        self.assertEqual(result.history[-1].empty_clusters, 0)
        self.assertLess(result.train_inertia, 0.2)

    def test_rq_reduces_train_validation_and_test_residual(self) -> None:
        generator = torch.Generator().manual_seed(17)
        values = torch.randn((360, 7), generator=generator)
        result = fit_diagnostic_rq(
            values[:240],
            values[240:300],
            values[300:],
            DiagnosticKMeansConfig(
                k=8,
                max_iters=10,
                min_iters=2,
                tol=0.0,
                patience=2,
                seed=5,
                device="cpu",
            ),
            levels=3,
        )
        for sequence in (
            result.train_residual_mse,
            result.validation_residual_mse,
            result.test_residual_mse,
        ):
            self.assertIsNotNone(sequence)
            assert sequence is not None
            self.assertEqual(len(sequence), 4)
            for before, after in zip(sequence, sequence[1:]):
                self.assertLessEqual(after, before + 1e-7)
        self.assertEqual(tuple(result.test_codes.shape), (60, 3))

    def test_usage_and_ari_are_permutation_invariant(self) -> None:
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        permuted = torch.tensor([2, 2, 0, 0, 1, 1])
        self.assertAlmostEqual(adjusted_rand_index(labels, permuted), 1.0)
        usage = usage_summary(labels, k=4)
        self.assertEqual(usage["used"], 3)
        self.assertEqual(usage["dead"], 1)


if __name__ == "__main__":
    unittest.main()
