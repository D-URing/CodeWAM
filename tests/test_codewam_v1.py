from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

import torch

from codewam.models import (
    ActionBatch,
    FutureCodeTargets,
    build_codewam_v1,
)
from tests.model_fixtures import (
    small_batch,
    small_config,
    synthetic_artifacts,
)


def _has_grad(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in module.parameters()
    )


class CodeWAMV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(13)
        self.artifacts = {"droid": synthetic_artifacts("droid")}
        self.batch = small_batch()
        self.noise = torch.randn_like(self.batch.actions.values)
        self.flow_time = torch.tensor([0.2, 0.7])

    def test_c0_c1_c2_share_one_batch_and_optimizer_contract(self) -> None:
        for variant in ("C0", "C1", "C2"):
            model = build_codewam_v1(
                small_config(variant=variant),
                self.artifacts,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            output = model(
                self.batch,
                noise=self.noise,
                flow_time=self.flow_time,
            )
            self.assertTrue(torch.isfinite(output.total))
            self.assertEqual(output.future is not None, variant == "C2")
            output.total.backward()
            optimizer.step()

    def test_future_labels_cannot_change_policy_velocity(self) -> None:
        model = build_codewam_v1(small_config(variant="C2"), self.artifacts).eval()
        noised = torch.randn_like(self.batch.actions.values)
        _, expected = model.policy_velocity(
            state=self.batch.state,
            codes=self.batch.codes,
            policy=self.batch.policy,
            noised_actions=noised,
            flow_time=self.flow_time,
        )
        permuted = FutureCodeTargets(
            code_ids=self.batch.future_codes.code_ids.flip(0),
            available=self.batch.future_codes.available,
        )
        changed_batch = replace(self.batch, future_codes=permuted)
        _, actual = model.policy_velocity(
            state=changed_batch.state,
            codes=changed_batch.codes,
            policy=changed_batch.policy,
            noised_actions=noised,
            flow_time=self.flow_time,
        )
        torch.testing.assert_close(expected, actual)

    def test_gt_action_only_changes_dynamics_when_policy_input_is_fixed(self) -> None:
        model = build_codewam_v1(small_config(variant="C2"), self.artifacts).eval()
        noised = torch.randn_like(self.batch.actions.values)
        belief, policy_velocity = model.policy_velocity(
            state=self.batch.state,
            codes=self.batch.codes,
            policy=self.batch.policy,
            noised_actions=noised,
            flow_time=self.flow_time,
        )
        first = model.code_dynamics(belief, self.batch.actions)
        permuted_actions = ActionBatch(values=self.batch.actions.values.flip(0))
        second = model.code_dynamics(belief, permuted_actions)
        _, repeated_policy_velocity = model.policy_velocity(
            state=self.batch.state,
            codes=self.batch.codes,
            policy=self.batch.policy,
            noised_actions=noised,
            flow_time=self.flow_time,
        )
        torch.testing.assert_close(policy_velocity, repeated_policy_velocity)
        self.assertTrue(
            any(
                not torch.equal(left, right)
                for left, right in zip(first.logits, second.logits)
            )
        )

    def test_gradient_routes_meet_module_boundaries(self) -> None:
        model = build_codewam_v1(small_config(variant="C2"), self.artifacts)
        belief = model.build_belief(self.batch.state, self.batch.codes)
        prediction = model.code_dynamics(belief, self.batch.actions)
        code_loss = model.code_dynamics.loss(
            prediction,
            self.batch.future_codes,
            sample_valid=self.batch.supervision.dynamics,
        )
        code_loss.backward()
        self.assertTrue(_has_grad(model.continuous_state))
        self.assertTrue(_has_grad(model.frozen_codebook))
        self.assertTrue(_has_grad(model.belief_core))
        self.assertTrue(_has_grad(model.code_dynamics))
        self.assertFalse(_has_grad(model.action_flow))

        model.zero_grad(set_to_none=True)
        belief = model.build_belief(self.batch.state, self.batch.codes)
        flow = model.action_flow.flow_matching_loss(
            self.batch.actions,
            belief=belief,
            policy=self.batch.policy,
            state=self.batch.state,
            sample_valid=self.batch.supervision.action,
            noise=self.noise,
            flow_time=self.flow_time,
        )
        flow.loss.backward()
        self.assertTrue(_has_grad(model.continuous_state))
        self.assertTrue(_has_grad(model.frozen_codebook))
        self.assertTrue(_has_grad(model.belief_core))
        self.assertTrue(_has_grad(model.action_flow))
        self.assertFalse(_has_grad(model.code_dynamics))

    def test_failure_samples_are_masked_from_action_imitation(self) -> None:
        batch = small_batch(action_supervision=torch.tensor([True, False]))
        model = build_codewam_v1(small_config(variant="C1"), self.artifacts).eval()
        noise = torch.randn_like(batch.actions.values)
        flow_time = torch.tensor([0.3, 0.6])
        first = model(
            batch,
            noise=noise,
            flow_time=flow_time,
        ).action
        changed_actions = batch.actions.values.clone()
        changed_actions[1] += 1_000.0
        changed_batch = replace(
            batch,
            actions=ActionBatch(values=changed_actions),
        )
        second = model(
            changed_batch,
            noise=noise,
            flow_time=flow_time,
        ).action
        torch.testing.assert_close(first, second)

    def test_action_flow_handles_a_fully_padded_unsupervised_sample(self) -> None:
        actions = ActionBatch(
            values=self.batch.actions.values,
            valid=torch.tensor(
                [[True, True, True], [False, False, False]],
                dtype=torch.bool,
            ),
        )
        batch = replace(
            self.batch,
            actions=actions,
            supervision=replace(
                self.batch.supervision,
                action=torch.tensor([True, False]),
                dynamics=torch.tensor([True, False]),
            ),
        )
        model = build_codewam_v1(small_config(variant="C2"), self.artifacts)
        output = model(
            batch,
            noise=self.noise,
            flow_time=self.flow_time,
        )
        self.assertTrue(torch.isfinite(output.total))
        output.total.backward()

    def test_basic_inference_never_calls_code_dynamics(self) -> None:
        model = build_codewam_v1(small_config(variant="C2"), self.artifacts).eval()
        with mock.patch.object(
            model.code_dynamics,
            "forward",
            side_effect=AssertionError("dynamics must not run"),
        ):
            actions = model.infer_actions(
                state=self.batch.state,
                policy=self.batch.policy,
                codes=self.batch.codes,
                horizon=3,
                steps=2,
                initial_noise=torch.zeros_like(self.batch.actions.values),
            )
        self.assertEqual(tuple(actions.shape), (2, 3, 7))

    def test_prefix_mode_and_state_dict_round_trip(self) -> None:
        config = small_config(variant="C2", dynamics_mode="prefix")
        model = build_codewam_v1(config, self.artifacts).eval()
        output = model(
            self.batch,
            noise=self.noise,
            flow_time=self.flow_time,
        )
        self.assertEqual(len(output.future.logits), 3)
        clone = build_codewam_v1(config, self.artifacts).eval()
        clone.load_state_dict(model.state_dict())
        first = model(
            self.batch,
            noise=self.noise,
            flow_time=self.flow_time,
        )
        second = clone(
            self.batch,
            noise=self.noise,
            flow_time=self.flow_time,
        )
        torch.testing.assert_close(first.total, second.total)
        torch.testing.assert_close(first.flow.velocity, second.flow.velocity)


if __name__ == "__main__":
    unittest.main()
