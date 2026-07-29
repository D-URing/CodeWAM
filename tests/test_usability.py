from __future__ import annotations

import unittest

from codewam.codebook_eval.usability import (
    _event_gate,
    _overall_decision,
    _quantizer_health,
    _stability_gate,
    _validate_artifact_identity,
    _visual_gates,
)


def comparison_row(family: str, split: str) -> dict:
    return {
        "family": family,
        "split": split,
        "k": 8,
        "levels": 3,
        "perplexity_fraction_by_level": [0.9, 0.9, 0.9],
        "maximum_cluster_fraction_by_level": [0.2, 0.2, 0.2],
        "heldout_residual_total_reduction": 0.3,
        "heldout_residual_reduction_by_level": [0.18, 0.09, 0.06],
        "generalization_gap": 0.01,
        "active_codes_by_level": [8, 8, 8],
    }


class UsabilityTests(unittest.TestCase):
    def test_provenance_rejects_mixed_frozen_artifacts(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "changed=.*Q3"):
            _validate_artifact_identity(
                {"Q2": "a", "Q3": "b", "Q5": "c"},
                {"Q2": "a", "Q3": "other", "Q5": "c"},
                source="fixture",
            )

    def test_quantizer_health_passes_complete_balanced_fixture(self) -> None:
        comparison = {
            "rows": [
                comparison_row(family, split)
                for family in ("Q2", "Q3", "Q5")
                for split in ("val", "test")
            ]
        }

        gate = _quantizer_health(comparison)

        self.assertEqual(gate["status"], "pass")
        self.assertTrue(
            gate["evidence"]["all_levels_use_all_centers"]
        )

    def test_overall_decision_preserves_semantic_limiters(self) -> None:
        gates = []
        for name in (
            "causal_reproduction",
            "quantizer_health",
            "rq_hierarchy",
            "family_complementarity",
            "context_leakage",
            "photometric_robustness",
            "geometry_sensitivity",
            "action_event_semantics",
            "seed_stability",
            "cross_domain_stress",
        ):
            gates.append(
                {
                    "name": name,
                    "status": (
                        "conditional"
                        if name == "geometry_sensitivity"
                        else "pass"
                    ),
                }
            )

        decision = _overall_decision(gates)

        self.assertEqual(decision["verdict"], "conditional_pass")
        self.assertEqual(
            decision["deployment_scope"],
            "DROID-in-domain research only",
        )
        self.assertEqual(
            decision["semantic_limiters"],
            ["geometry_sensitivity"],
        )
        self.assertIn(
            "code-only precision control",
            decision["not_approved"],
        )

    def test_overall_decision_blocks_unstable_seed_partitions(self) -> None:
        names = (
            "causal_reproduction",
            "quantizer_health",
            "rq_hierarchy",
            "family_complementarity",
            "context_leakage",
            "photometric_robustness",
            "geometry_sensitivity",
            "action_event_semantics",
            "seed_stability",
            "cross_domain_stress",
        )
        gates = [
            {
                "name": name,
                "status": "fail" if name == "seed_stability" else "pass",
            }
            for name in names
        ]

        decision = _overall_decision(gates)

        self.assertEqual(decision["verdict"], "not_ready")
        self.assertEqual(decision["blocking_gates"], ["seed_stability"])
        self.assertEqual(
            decision["deployment_scope"],
            "fixed-artifact DROID research only; universal tokenizer claims blocked",
        )
        self.assertIn(
            "DROID C0/C1/C2 world-action experiments with all available codes",
            decision["approved_use"],
        )
        self.assertTrue(
            any(
                "bind every downstream run to one artifact seed" in followup
                for followup in decision["required_followups"]
            )
        )

    def test_overall_decision_blocks_scoped_use_when_causality_fails(self) -> None:
        names = (
            "causal_reproduction",
            "quantizer_health",
            "rq_hierarchy",
            "family_complementarity",
            "context_leakage",
            "photometric_robustness",
            "geometry_sensitivity",
            "action_event_semantics",
            "seed_stability",
            "cross_domain_stress",
        )
        gates = [
            {
                "name": name,
                "status": "fail" if name == "causal_reproduction" else "pass",
            }
            for name in names
        ]

        decision = _overall_decision(gates)

        self.assertEqual(decision["verdict"], "not_ready")
        self.assertEqual(
            decision["deployment_scope"],
            "blocked pending in-domain artifact gates",
        )
        self.assertEqual(decision["approved_use"], [])

    def test_visual_gates_use_quantized_geometry_response(self) -> None:
        condition_rows = []
        for family in ("Q2", "Q3", "Q5"):
            for name in (
                "uniform_brightness_085",
                "uniform_brightness_115",
                "uniform_contrast_085",
                "uniform_contrast_115",
            ):
                condition_rows.append(
                    {
                        "family": family,
                        "name": name,
                        "category": "photometric_nuisance",
                        "prefix_change_fraction": [0.1, 0.3, 0.5],
                        "relative_to_natural_next_displacement": 0.2,
                        "quantized_prefix_relative_to_natural_next": [
                            0.1,
                            0.3,
                            0.5,
                        ],
                        "mean_quantized_prefix_displacement_mse": [
                            0.01,
                            0.02,
                            0.03,
                        ],
                    }
                )
            for name in (
                "endpoint_translate_x_negative_4",
                "endpoint_translate_x_positive_4",
                "endpoint_translate_x_negative_8",
                "endpoint_translate_x_positive_8",
                "endpoint_translate_y_negative_8",
                "endpoint_translate_y_positive_8",
                "endpoint_scale_090",
                "endpoint_scale_110",
            ):
                strong = not name.endswith("_4")
                condition_rows.append(
                    {
                        "family": family,
                        "name": name,
                        "category": "endpoint_geometry",
                        "prefix_change_fraction": [0.1, 0.2, 0.4],
                        "relative_to_natural_next_displacement": 0.5,
                        "quantized_prefix_relative_to_natural_next": [
                            0.1,
                            0.3,
                            0.5,
                        ],
                        "mean_quantized_prefix_displacement_mse": [
                            0.01,
                            0.02,
                            0.20 if strong else 0.10,
                        ],
                    }
                )
        direction_rows = [
            {
                "family": family,
                "left_condition": left,
                "right_condition": right,
                "prefix_distinct_fraction": [0.1, 0.2, 0.4],
            }
            for family in ("Q2", "Q3", "Q5")
            for left, right in (
                (
                    "endpoint_translate_x_negative_8",
                    "endpoint_translate_x_positive_8",
                ),
                (
                    "endpoint_translate_y_negative_8",
                    "endpoint_translate_y_positive_8",
                ),
                ("endpoint_scale_090", "endpoint_scale_110"),
            )
        ]

        photometric, geometry, cross_domain = _visual_gates(
            [
                {
                    "source": "droid",
                    "condition_rows": condition_rows,
                    "direction_rows": direction_rows,
                }
            ]
        )

        self.assertEqual(photometric["status"], "pass")
        self.assertEqual(geometry["status"], "pass")
        self.assertEqual(cross_domain["status"], "not_run")
        self.assertEqual(
            geometry["evidence"]["dose_response_pass_fraction"],
            1.0,
        )
        bad_rows = []
        for row in condition_rows:
            copied = dict(row)
            if (
                row["family"] == "Q2"
                and row["category"] == "photometric_nuisance"
            ):
                copied["quantized_prefix_relative_to_natural_next"] = [
                    0.1,
                    0.3,
                    2.5,
                ]
            bad_rows.append(copied)

        worst_photometric, _, _ = _visual_gates(
            [
                {
                    "source": "droid",
                    "condition_rows": condition_rows,
                    "direction_rows": direction_rows,
                },
                {
                    "source": "droid",
                    "condition_rows": bad_rows,
                    "direction_rows": direction_rows,
                },
            ]
        )

        self.assertEqual(worst_photometric["status"], "fail")
        self.assertEqual(
            worst_photometric["evidence"][
                "maximum_family_split_quantized_relative_to_natural_next"
            ],
            2.5,
        )

    def test_stability_gate_requires_joint_prefix_partition(self) -> None:
        stability = {
            "distortion_rows": [
                {
                    "run": "__across_runs__",
                    "coefficient_of_variation": 0.01,
                    "maximum_relative_range": 0.02,
                }
            ],
            "pair_rows": [
                {
                    "levels": [
                        {
                            "normalized_mutual_information": 0.8,
                            "adjusted_rand_index": 0.7,
                        },
                        {
                            "normalized_mutual_information": 0.7,
                            "adjusted_rand_index": 0.6,
                        },
                        {
                            "normalized_mutual_information": 0.6,
                            "adjusted_rand_index": 0.5,
                        },
                    ],
                    "prefix_partitions": [
                        {
                            "depth": 1,
                            "normalized_mutual_information": 0.8,
                            "adjusted_rand_index": 0.7,
                        },
                        {
                            "depth": 2,
                            "normalized_mutual_information": 0.7,
                            "adjusted_rand_index": 0.6,
                        },
                        {
                            "depth": 3,
                            "normalized_mutual_information": 0.55,
                            "adjusted_rand_index": 0.45,
                        },
                    ],
                    "mapped_prefix_agreement": [0.8, 0.6, 0.5],
                }
            ],
        }

        gate = _stability_gate(stability)

        self.assertEqual(gate["status"], "pass")
        self.assertEqual(
            gate["evidence"]["minimum_full_prefix_partition_nmi"],
            0.55,
        )
        self.assertEqual(
            gate["evidence"]["minimum_full_prefix_partition_ari"],
            0.45,
        )

    def test_libero_gate_requires_geometry_under_domain_shift(self) -> None:
        condition_rows = []
        for family in ("Q2", "Q3", "Q5"):
            for name in (
                "uniform_brightness_085",
                "uniform_brightness_115",
            ):
                condition_rows.append(
                    {
                        "family": family,
                        "name": name,
                        "category": "photometric_nuisance",
                        "quantized_prefix_relative_to_natural_next": [
                            0.1,
                            0.2,
                            0.4,
                        ],
                    }
                )
            for name in (
                "endpoint_translate_x_negative_8",
                "endpoint_translate_x_positive_8",
                "endpoint_translate_y_negative_8",
                "endpoint_translate_y_positive_8",
                "endpoint_scale_090",
                "endpoint_scale_110",
            ):
                condition_rows.append(
                    {
                        "family": family,
                        "name": name,
                        "category": "endpoint_geometry",
                        "prefix_change_fraction": [0.1, 0.2, 0.3],
                    }
                )
        direction_rows = [
            {
                "family": family,
                "left_condition": left,
                "right_condition": right,
                "prefix_distinct_fraction": [0.1, 0.2, 0.3],
            }
            for family in ("Q2", "Q3", "Q5")
            for left, right in (
                (
                    "endpoint_translate_x_negative_8",
                    "endpoint_translate_x_positive_8",
                ),
                (
                    "endpoint_translate_y_negative_8",
                    "endpoint_translate_y_positive_8",
                ),
                ("endpoint_scale_090", "endpoint_scale_110"),
            )
        ]
        usage_rows = [
            {
                "family": family,
                "level": level,
                "active_codes": 7,
                "capacity": 8,
                "perplexity_fraction": 0.7,
            }
            for family in ("Q2", "Q3", "Q5")
            for level in (1, 2, 3)
        ]

        _, _, cross_domain = _visual_gates(
            [
                {
                    "source": "libero",
                    "condition_rows": condition_rows,
                    "direction_rows": direction_rows,
                    "identity_usage_rows": usage_rows,
                }
            ]
        )

        self.assertEqual(cross_domain["status"], "pass")
        self.assertEqual(
            cross_domain["evidence"][
                "mean_endpoint_full_prefix_change_fraction"
            ],
            0.3,
        )

    def test_event_gate_does_not_hide_missing_gripper_signal(self) -> None:
        rows = []
        for family in ("Q2", "Q3", "Q5"):
            for split in ("val", "test"):
                for event, gain in (
                    ("translation_magnitude_quartile", 0.15),
                    ("translation_direction", 0.08),
                    ("rotation_magnitude_quartile", 0.14),
                    ("rotation_direction", 0.05),
                    ("gripper_change", 0.0),
                ):
                    rows.append(
                        {
                            "family": family,
                            "split": split,
                            "event": event,
                            "prefix_depth": 3,
                            "levels": 3,
                            "normalized_accuracy_gain": gain,
                            "any_code_coverage": 1.0,
                        }
                    )

        gate = _event_gate({"rows": rows})

        self.assertEqual(gate["status"], "conditional")
        self.assertEqual(
            gate["evidence"]["best_gripper_event_gain_by_split"],
            {"val": 0.0, "test": 0.0},
        )


if __name__ == "__main__":
    unittest.main()
