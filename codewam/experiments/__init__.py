"""Reproducible experiment protocols for CodeWAM."""

from .gate2 import (
    GATE2_SCHEMA,
    FixedActionPermutation,
    Gate2RunConfig,
    build_fixed_action_permutation,
    run_gate2,
)
from .gate2_summary import (
    GATE2_MULTI_SEED_SCHEMA,
    summarize_gate2_reports,
)
from .policy_ablation import (
    POLICY_ABLATION_SCHEMA,
    PolicyAblationRunConfig,
    fixed_eval_subset,
    paired_episode_bootstrap,
    run_policy_ablation,
)
from .policy_ablation_summary import (
    POLICY_ABLATION_MULTI_SEED_SCHEMA,
    summarize_policy_ablation_reports,
)

__all__ = [
    "GATE2_SCHEMA",
    "GATE2_MULTI_SEED_SCHEMA",
    "FixedActionPermutation",
    "Gate2RunConfig",
    "POLICY_ABLATION_SCHEMA",
    "POLICY_ABLATION_MULTI_SEED_SCHEMA",
    "PolicyAblationRunConfig",
    "build_fixed_action_permutation",
    "fixed_eval_subset",
    "paired_episode_bootstrap",
    "run_gate2",
    "run_policy_ablation",
    "summarize_policy_ablation_reports",
    "summarize_gate2_reports",
]
