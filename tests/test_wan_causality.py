from __future__ import annotations

import unittest

import torch

from codewam.codebook_eval.wan_causality import compare_causal_prefixes


class CausalFakeVAE:
    def encode(self, videos, device, tiled):
        del device, tiled
        indices = torch.arange(
            0,
            videos[0].shape[1],
            4,
            dtype=torch.long,
        )
        return torch.stack(
            [video[:, indices] for video in videos],
            dim=0,
        )


class FutureLeakingFakeVAE:
    def encode(self, videos, device, tiled):
        del device, tiled
        outputs = []
        for video in videos:
            indices = torch.arange(
                0,
                video.shape[1],
                4,
                dtype=torch.long,
            )
            outputs.append(
                video[:, indices]
                + video[:, -1:].expand(-1, indices.numel(), -1, -1)
            )
        return torch.stack(outputs, dim=0)


def make_videos() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20260729)
    return tuple(
        torch.randn((3, 21, 4, 4), generator=generator)
        for _ in range(2)
    )


class WanCausalityTests(unittest.TestCase):
    def test_causal_encoder_is_prefix_invariant(self) -> None:
        report = compare_causal_prefixes(
            make_videos(),
            CausalFakeVAE(),
            device="cpu",
            latent_ticks=6,
            atol=0.0,
            rtol=0.0,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["latent_shape"], [2, 3, 6, 4, 4])
        self.assertTrue(
            all(row["max_abs_error"] == 0.0 for row in report["rows"])
        )

    def test_future_leakage_fails_prefix_invariance(self) -> None:
        report = compare_causal_prefixes(
            make_videos(),
            FutureLeakingFakeVAE(),
            device="cpu",
            latent_ticks=6,
            atol=0.0,
            rtol=0.0,
        )

        self.assertFalse(report["passed"])
        self.assertGreater(report["rows"][1]["mismatch_fraction"], 0.0)
        self.assertTrue(report["rows"][-1]["passed"])


if __name__ == "__main__":
    unittest.main()
