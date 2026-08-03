from typing import TYPE_CHECKING, Any

from .droid_manifest import (
    DroidBalancedSampleResult,
    DroidManifestBuildResult,
    balanced_scene_sample,
    build_droid_manifest,
    droid_temporal_distribution,
    shard_aware_balanced_sample,
)
from .droid_rlds import (
    DROID_ACTION_COMPONENT_DIMS,
    DroidRankAssignment,
    DroidRLDSActionEpisode,
    DroidRLDSEpisode,
    DroidRLDSSegment,
    DroidShardWork,
    iter_manifest_droid_action_episodes,
    iter_manifest_droid_rlds_episodes,
    plan_droid_rank_assignments,
)
from .action_targets import (
    DROID_ACTION_TARGET_CACHE_SCHEMA,
    DROID_FLAT_ACTION_COMPONENTS,
    DroidActionTargetSegment,
    FrozenDroidActionTargetCache,
    action_target_mapping_statistics,
    create_droid_action_target_contract,
    validate_action_targets_against_joint_episodes,
    validate_droid_action_target_contract,
    validate_droid_action_target_shard,
    write_droid_action_target_contract,
    write_droid_action_target_index,
    write_droid_action_target_shard,
)
from .droid_endpoint import (
    DROID_ENDPOINT_AUDIT_SCHEMA,
    DROID_ENDPOINT_POLICY,
    audit_droid_endpoints,
)
from .frozen_assignment import (
    DEFAULT_CODE_FAMILIES,
    FrozenArtifactChart,
    FrozenCausalCodeAssigner,
    FrozenCodeAssignment,
    load_frozen_artifact_chart,
)
from .joint_cache import (
    JOINT_CACHE_SCHEMA,
    JointEpisode,
    JointModelBatch,
    JointWindowCache,
    JointWindowConfig,
    JointWindowRecord,
    JointWindowSample,
    add_compact_joint_window_index,
    build_joint_windows,
    collate_joint_windows,
    create_joint_cache_contract,
    finalize_joint_cache,
    validate_joint_cache_contract,
    write_joint_cache_contract,
    write_joint_episode_shard,
)
from .language_cache import (
    LANGUAGE_CACHE_SCHEMA,
    FrozenLanguageCache,
    LanguageConditionedJointWindowCache,
    create_language_cache_contract,
    normalize_language_instruction,
    validate_language_cache_contract,
    write_frozen_language_cache,
)
from .policy_normalization import (
    DROID_ACTION_SEMANTICS,
    DROID_POLICY_REPRESENTATION,
    POLICY_NORMALIZATION_SCHEMA,
    PolicyNormalizer,
    create_policy_normalization_contract,
    decode_droid_actions,
    encode_droid_actions,
    encode_droid_proprio,
    moments_from_sums,
    validate_policy_normalization_contract,
    write_policy_normalization,
)

if TYPE_CHECKING:
    from .package_scan_v6 import PackageScanV6Dataset

__all__ = [
    "DroidBalancedSampleResult",
    "DroidManifestBuildResult",
    "DroidActionTargetSegment",
    "DroidRankAssignment",
    "DroidRLDSActionEpisode",
    "DroidRLDSEpisode",
    "DroidRLDSSegment",
    "DroidShardWork",
    "DROID_ACTION_COMPONENT_DIMS",
    "DROID_ACTION_TARGET_CACHE_SCHEMA",
    "DROID_ENDPOINT_AUDIT_SCHEMA",
    "DROID_ENDPOINT_POLICY",
    "DEFAULT_CODE_FAMILIES",
    "FrozenArtifactChart",
    "FrozenCausalCodeAssigner",
    "FrozenCodeAssignment",
    "JOINT_CACHE_SCHEMA",
    "LANGUAGE_CACHE_SCHEMA",
    "POLICY_NORMALIZATION_SCHEMA",
    "DROID_ACTION_SEMANTICS",
    "DROID_POLICY_REPRESENTATION",
    "DROID_FLAT_ACTION_COMPONENTS",
    "FrozenLanguageCache",
    "FrozenDroidActionTargetCache",
    "JointEpisode",
    "JointModelBatch",
    "JointWindowCache",
    "JointWindowConfig",
    "JointWindowRecord",
    "JointWindowSample",
    "LanguageConditionedJointWindowCache",
    "PolicyNormalizer",
    "PackageScanV6Dataset",
    "add_compact_joint_window_index",
    "action_target_mapping_statistics",
    "audit_droid_endpoints",
    "balanced_scene_sample",
    "build_joint_windows",
    "build_droid_manifest",
    "collate_joint_windows",
    "create_joint_cache_contract",
    "create_droid_action_target_contract",
    "create_language_cache_contract",
    "create_policy_normalization_contract",
    "decode_droid_actions",
    "droid_temporal_distribution",
    "iter_manifest_droid_rlds_episodes",
    "iter_manifest_droid_action_episodes",
    "finalize_joint_cache",
    "encode_droid_actions",
    "encode_droid_proprio",
    "load_frozen_artifact_chart",
    "plan_droid_rank_assignments",
    "normalize_language_instruction",
    "moments_from_sums",
    "shard_aware_balanced_sample",
    "validate_joint_cache_contract",
    "validate_action_targets_against_joint_episodes",
    "validate_droid_action_target_contract",
    "validate_droid_action_target_shard",
    "validate_language_cache_contract",
    "validate_policy_normalization_contract",
    "write_frozen_language_cache",
    "write_droid_action_target_contract",
    "write_droid_action_target_index",
    "write_droid_action_target_shard",
    "write_policy_normalization",
    "write_joint_cache_contract",
    "write_joint_episode_shard",
]
from .roles import (
    ROLE_SUPERVISION,
    RoleSupervision,
    TrajectoryRole,
    build_supervision_masks,
    codebook_fit_records,
    role_supervision,
    trajectory_role,
)

__all__ += [
    "ROLE_SUPERVISION",
    "RoleSupervision",
    "TrajectoryRole",
    "build_supervision_masks",
    "codebook_fit_records",
    "role_supervision",
    "trajectory_role",
]


def __getattr__(name: str) -> Any:
    if name == "PackageScanV6Dataset":
        from .package_scan_v6 import PackageScanV6Dataset

        return PackageScanV6Dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
