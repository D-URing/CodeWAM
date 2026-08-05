from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from codewam.models import (
    ActionBatch,
    CodeMeasurements,
    FrozenCodebookAdapter,
    FutureCodeTargets,
    StateInputs,
    TransitionSchedule,
    build_codewam_v2,
)

from .model_fixtures import small_batch, small_config, synthetic_artifacts


class HierarchicalCodebookTests(unittest.TestCase):
    def test_rq_prefix_tokens_are_cumulative_center_reconstructions(self) -> None:
        artifacts = synthetic_artifacts("droid", descriptor_dim=12)
        adapter = FrozenCodebookAdapter(
            {"droid": artifacts},
            dim=12,
            layout="hierarchical",
        )
        with torch.no_grad():
            adapter.chart_embedding.zero_()
            adapter.family_embedding.zero_()
            adapter.level_embedding.zero_()
            for projection in adapter.projections.values():
                projection.weight.copy_(torch.eye(12))
                projection.bias.zero_()
        measurements = CodeMeasurements(
            code_ids=torch.tensor([[[1, 2, 3], [0, 0, 0], [0, 0, 0]]]),
            available=torch.ones((1, 3), dtype=torch.bool),
            chart_names=("droid",),
        )

        state = adapter(measurements)
        expected = torch.stack(
            (
                artifacts[0].centers[0][1],
                artifacts[0].centers[0][1] + artifacts[0].centers[1][2],
                artifacts[0].centers[0][1]
                + artifacts[0].centers[1][2]
                + artifacts[0].centers[2][3],
            )
        )

        torch.testing.assert_close(state.prefix_tokens[0, 0], expected)
        self.assertEqual(tuple(state.tokens.shape), (1, 3, 12))


class CodeWAMV2Tests(unittest.TestCase):
    def _model(self, variant: str = "C2"):
        return build_codewam_v2(
            small_config(variant=variant, dynamics_mode="prefix"),
            {"droid": synthetic_artifacts("droid")},
        )

    def test_forward_backward_and_inference_are_complete(self) -> None:
        torch.manual_seed(3)
        model = self._model("C2")
        batch = small_batch()
        noise = torch.randn_like(batch.actions.values)
        flow_time = torch.tensor([0.25, 0.75])

        output = model(batch, noise=noise, flow_time=flow_time)

        self.assertTrue(torch.isfinite(output.total))
        self.assertGreater(float(output.action.detach()), 0.0)
        self.assertGreater(float(output.code.detach()), 0.0)
        self.assertIsNotNone(output.future)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        output.total.backward()
        self.assertIsNotNone(model.world_builder.queries.grad)
        self.assertIsNotNone(model.action_flow.output.weight.grad)
        self.assertIsNotNone(model.transition.heads[0].weight.grad)
        optimizer.step()
        self.assertTrue(
            all(
                torch.isfinite(parameter).all()
                for parameter in model.parameters()
            )
        )

        actions = model.infer_actions(
            state=batch.state,
            policy=batch.policy,
            codes=batch.codes,
            horizon=3,
            steps=2,
            initial_noise=torch.zeros_like(batch.actions.values),
        )
        self.assertEqual(tuple(actions.shape), (2, 3, 7))
        self.assertTrue(torch.isfinite(actions).all())

    def test_c0_is_structurally_independent_of_code_measurements(self) -> None:
        torch.manual_seed(5)
        model = self._model("C0").eval()
        batch = small_batch()
        changed_codes = replace(
            batch.codes,
            code_ids=(batch.codes.code_ids + 1).remainder(4),
        )
        noised = torch.randn_like(batch.actions.values)
        flow_time = torch.tensor([0.2, 0.6])

        _, first = model.policy_velocity(
            state=batch.state,
            codes=batch.codes,
            policy=batch.policy,
            noised_actions=noised,
            flow_time=flow_time,
        )
        _, second = model.policy_velocity(
            state=batch.state,
            codes=changed_codes,
            policy=batch.policy,
            noised_actions=noised,
            flow_time=flow_time,
        )

        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_future_labels_cannot_change_policy_computation(self) -> None:
        torch.manual_seed(7)
        model = self._model("C2").eval()
        batch = small_batch()
        changed_targets = FutureCodeTargets(
            code_ids=(batch.future_codes.code_ids + 1).remainder(4),
            available=batch.future_codes.available,
            schedule=batch.future_codes.schedule,
        )
        changed_batch = replace(batch, future_codes=changed_targets)
        noise = torch.randn_like(batch.actions.values)
        flow_time = torch.tensor([0.3, 0.7])

        first = model(batch, noise=noise, flow_time=flow_time)
        second = model(changed_batch, noise=noise, flow_time=flow_time)

        torch.testing.assert_close(
            first.flow.velocity,
            second.flow.velocity,
            rtol=0.0,
            atol=0.0,
        )

    def test_zero_initialized_code_gate_preserves_global_state(self) -> None:
        torch.manual_seed(11)
        model = self._model("C1").eval()
        batch = small_batch()
        changed_codes = replace(
            batch.codes,
            code_ids=(batch.codes.code_ids + 1).remainder(4),
        )

        first = model.build_world_state(batch.state, batch.codes)
        second = model.build_world_state(batch.state, changed_codes)

        torch.testing.assert_close(
            first.belief.tokens,
            second.belief.tokens,
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(torch.equal(first.codes.tokens, second.codes.tokens))

    def test_each_clock_reads_only_its_declared_action_prefix(self) -> None:
        torch.manual_seed(13)
        model = self._model("C2").eval()
        batch = small_batch(chart_names=("droid",))
        world = model.build_world_state(batch.state, batch.codes)
        schedule = TransitionSchedule(
            action_prefix_lengths=torch.tensor([[1, 2, 3]]),
            delta_times=torch.tensor([[0.1, 0.2, 0.3]]),
        )
        changed_values = batch.actions.values.clone()
        changed_values[:, 2] += 50.0
        changed_actions = ActionBatch(values=changed_values, valid=batch.actions.valid)

        first = model.transition(
            world.belief,
            world.codes,
            batch.actions,
            schedule,
        )
        second = model.transition(
            world.belief,
            world.codes,
            changed_actions,
            schedule,
        )

        torch.testing.assert_close(
            first.logits[0], second.logits[0], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            first.logits[1], second.logits[1], rtol=0.0, atol=0.0
        )
        self.assertFalse(torch.equal(first.logits[2], second.logits[2]))

    def test_relative_time_is_part_of_world_state(self) -> None:
        torch.manual_seed(17)
        model = self._model("C0").eval()
        batch = small_batch()
        state = batch.state
        shifted = StateInputs(
            latents=state.latents,
            proprio_history=state.proprio_history,
            past_actions=state.past_actions,
            latent_valid=state.latent_valid,
            proprio_valid=state.proprio_valid,
            past_action_valid=state.past_action_valid,
            latent_time_offsets=state.latent_time_offsets * 2.0,
            proprio_time_offsets=state.proprio_time_offsets * 2.0,
            past_action_time_offsets=state.past_action_time_offsets * 2.0,
        )

        first = model.build_world_state(state, None)
        second = model.build_world_state(shifted, None)

        self.assertFalse(torch.equal(first.belief.tokens, second.belief.tokens))


if __name__ == "__main__":
    unittest.main()
