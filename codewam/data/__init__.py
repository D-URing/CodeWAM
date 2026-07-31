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
    DroidRankAssignment,
    DroidRLDSEpisode,
    DroidRLDSSegment,
    DroidShardWork,
    iter_manifest_droid_rlds_episodes,
    plan_droid_rank_assignments,
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

if TYPE_CHECKING:
    from .package_scan_v6 import PackageScanV6Dataset

__all__ = [
    "DroidBalancedSampleResult",
    "DroidManifestBuildResult",
    "DroidRankAssignment",
    "DroidRLDSEpisode",
    "DroidRLDSSegment",
    "DroidShardWork",
    "DROID_ENDPOINT_AUDIT_SCHEMA",
    "DROID_ENDPOINT_POLICY",
    "DEFAULT_CODE_FAMILIES",
    "FrozenArtifactChart",
    "FrozenCausalCodeAssigner",
    "FrozenCodeAssignment",
    "JOINT_CACHE_SCHEMA",
    "JointEpisode",
    "JointModelBatch",
    "JointWindowCache",
    "JointWindowConfig",
    "JointWindowRecord",
    "JointWindowSample",
    "PackageScanV6Dataset",
    "add_compact_joint_window_index",
    "audit_droid_endpoints",
    "balanced_scene_sample",
    "build_joint_windows",
    "build_droid_manifest",
    "collate_joint_windows",
    "create_joint_cache_contract",
    "droid_temporal_distribution",
    "iter_manifest_droid_rlds_episodes",
    "finalize_joint_cache",
    "load_frozen_artifact_chart",
    "plan_droid_rank_assignments",
    "shard_aware_balanced_sample",
    "validate_joint_cache_contract",
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
