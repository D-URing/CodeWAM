from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from codewam.models import (
    CodeMeasurements,
    ContinuousStateEncoder,
    FrozenCodebookAdapter,
    StateInputs,
    TemporalLatentPredictor,
    temporal_pretraining_loss,
)
from tests.model_fixtures import synthetic_artifacts


class ContinuousStateTests(unittest.TestCase):
    def _encoder(self) -> ContinuousStateEncoder:
        return ContinuousStateEncoder(
            latent_channels=4,
            dim=16,
            heads=4,
            patch_size=2,
            spatial_layers=1,
            temporal_layers=1,
            max_time=8,
            max_cameras=4,
            max_spatial_tokens=64,
            dropout=0.0,
        )

    def test_future_latents_cannot_change_earlier_state_tokens(self) -> None:
        torch.manual_seed(3)
        encoder = self._encoder().eval()
        latents = torch.randn(2, 5, 2, 4, 8, 8)
        state = StateInputs(
            latents=latents,
            proprio_history=torch.randn(2, 2, 6),
            past_actions=torch.randn(2, 1, 7),
        )
        original, _ = encoder.forward_sequence(state)
        changed = latents.clone()
        changed[:, 4] = torch.randn_like(changed[:, 4]) * 100.0
        perturbed, _ = encoder.forward_sequence(
            StateInputs(
                latents=changed,
                proprio_history=state.proprio_history,
                past_actions=state.past_actions,
            )
        )
        torch.testing.assert_close(original[:, :4], perturbed[:, :4])
        self.assertGreater(
            float((original[:, 4] - perturbed[:, 4]).abs().max().detach()),
            1e-4,
        )

    def test_stage0_target_is_detached_and_future_is_causally_hidden(self) -> None:
        torch.manual_seed(5)
        encoder = self._encoder()
        predictor = TemporalLatentPredictor(
            dim=16,
            latent_channels=4,
            patch_size=2,
            heads=4,
            layers=1,
        )
        latents = torch.randn(2, 5, 2, 4, 8, 8, requires_grad=True)
        state = StateInputs(
            latents=latents,
            proprio_history=torch.randn(2, 2, 6),
            past_actions=torch.randn(2, 1, 7),
        )
        loss = temporal_pretraining_loss(
            encoder,
            predictor,
            state,
            context_index=2,
            target_index=4,
        )
        loss.backward()
        self.assertGreater(float(latents.grad[:, :3].abs().sum()), 0.0)
        self.assertEqual(float(latents.grad[:, 3:].abs().sum()), 0.0)
        self.assertTrue(any(value.grad is not None for value in predictor.parameters()))

    def test_stage0_ignores_unavailable_future_views(self) -> None:
        torch.manual_seed(7)
        encoder = self._encoder().eval()
        predictor = TemporalLatentPredictor(
            dim=16,
            latent_channels=4,
            patch_size=2,
            heads=4,
            layers=1,
        ).eval()
        latents = torch.randn(1, 5, 2, 4, 8, 8)
        latent_valid = torch.ones((1, 5, 2), dtype=torch.bool)
        latent_valid[:, 4, 1] = False
        state = StateInputs(
            latents=latents,
            proprio_history=torch.randn(1, 2, 6),
            past_actions=torch.randn(1, 1, 7),
            latent_valid=latent_valid,
        )
        expected = temporal_pretraining_loss(
            encoder,
            predictor,
            state,
            context_index=2,
            target_index=4,
        )
        changed = latents.clone()
        changed[:, 4, 1] += 10_000.0
        actual = temporal_pretraining_loss(
            encoder,
            predictor,
            StateInputs(
                latents=changed,
                proprio_history=state.proprio_history,
                past_actions=state.past_actions,
                latent_valid=latent_valid,
            ),
            context_index=2,
            target_index=4,
        )
        torch.testing.assert_close(expected, actual)


class FrozenCodebookAdapterTests(unittest.TestCase):
    def test_mixed_family_provenance_is_rejected(self) -> None:
        artifacts = list(synthetic_artifacts("droid"))
        artifacts[1] = replace(
            artifacts[1],
            metadata={
                **artifacts[1].metadata,
                "wan_revision": "different-wan-revision",
            },
        )
        with self.assertRaisesRegex(ValueError, "mixes provenance"):
            FrozenCodebookAdapter(
                {"droid": artifacts},
                dim=16,
            )

    def test_state_dict_rejects_a_different_chart_provenance(self) -> None:
        source = FrozenCodebookAdapter(
            {"droid": synthetic_artifacts("droid")},
            dim=16,
        )
        target = FrozenCodebookAdapter(
            {"libero": synthetic_artifacts("libero")},
            dim=16,
        )
        with self.assertRaisesRegex(RuntimeError, "provenance"):
            target.load_state_dict(source.state_dict())

    def test_duplicate_family_artifacts_are_rejected(self) -> None:
        artifacts = synthetic_artifacts("droid")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            FrozenCodebookAdapter(
                {"droid": (*artifacts, artifacts[0])},
                dim=16,
            )

    def test_mixed_charts_use_local_centers_without_id_alignment(self) -> None:
        adapter = FrozenCodebookAdapter(
            {
                "droid": synthetic_artifacts("droid", offset=0.0),
                "libero": synthetic_artifacts("libero", offset=10.0),
            },
            dim=16,
        )
        ids = torch.zeros((2, 3, 3), dtype=torch.long)
        measurements = CodeMeasurements(
            code_ids=ids,
            available=torch.ones((2, 3), dtype=torch.bool),
            chart_names=("droid", "libero"),
        )
        tokens = adapter(measurements)
        self.assertEqual(tuple(tokens.tokens.shape), (2, 9, 16))
        self.assertFalse(torch.equal(tokens.tokens[0], tokens.tokens[1]))

    def test_grouped_lookup_preserves_sample_order_and_projection_gradients(self) -> None:
        adapter = FrozenCodebookAdapter(
            {
                "droid": synthetic_artifacts("droid", offset=0.0),
                "libero": synthetic_artifacts("libero", offset=10.0),
            },
            dim=16,
        )
        measurements = CodeMeasurements(
            code_ids=torch.tensor(
                [
                    [[0, 1, 2], [1, 2, 3], [2, 3, 0]],
                    [[3, 2, 1], [2, 1, 0], [1, 0, 3]],
                    [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                ],
                dtype=torch.long,
            ),
            available=torch.ones((3, 3), dtype=torch.bool),
            chart_names=("droid", "libero", "droid"),
        )
        tokens = adapter(measurements).tokens
        self.assertEqual(tuple(tokens.shape), (3, 9, 16))
        tokens.square().mean().backward()
        self.assertTrue(
            all(
                projection.weight.grad is not None
                for projection in adapter.projections.values()
            )
        )

    def test_unavailable_family_uses_missing_tokens_and_centers_stay_frozen(self) -> None:
        adapter = FrozenCodebookAdapter(
            {"droid": synthetic_artifacts("droid")},
            dim=16,
        )
        center_names = tuple(adapter._buffer_names.values())
        before = {name: getattr(adapter, name).clone() for name in center_names}
        measurements = CodeMeasurements(
            code_ids=torch.tensor(
                [[[-1, -1, -1], [0, 1, 2], [3, 2, 1]]],
                dtype=torch.long,
            ),
            available=torch.tensor([[False, True, True]]),
            chart_names=("droid",),
        )
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
        loss = adapter(measurements).tokens.square().mean()
        loss.backward()
        optimizer.step()
        for name in center_names:
            center = getattr(adapter, name)
            self.assertFalse(center.requires_grad)
            torch.testing.assert_close(center, before[name])

    def test_center_distance_uses_chart_local_rq_reconstruction(self) -> None:
        adapter = FrozenCodebookAdapter(
            {"droid": synthetic_artifacts("droid")},
            dim=16,
        )
        target = torch.zeros((1, 3, 3), dtype=torch.long)
        self.assertEqual(
            float(
                adapter.normalized_center_mse(
                    target,
                    target,
                    available=torch.ones((1, 3), dtype=torch.bool),
                    chart_names=("droid",),
                )
            ),
            0.0,
        )
        predicted = target.clone()
        predicted[0, 1, 2] = 1
        self.assertGreater(
            float(
                adapter.normalized_center_mse(
                    predicted,
                    target,
                    available=torch.ones((1, 3), dtype=torch.bool),
                    chart_names=("droid",),
                )
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
