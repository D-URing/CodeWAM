"""Independent CodeWAM model components.

This package must remain free of FastWAM imports. Wan-VAE and language encoding
are external frozen preprocessing stages that provide tensors to these modules.
"""

from .contracts import (
    ActionBatch,
    CodeMeasurements,
    CodeTokens,
    CodeWAMBatch,
    ContinuousState,
    FutureCodeTargets,
    MultiClockCodeState,
    PolicyCondition,
    StateInputs,
    StructuredWorldState,
    SupervisionMasks,
    TransitionSchedule,
    WorldBelief,
)
from .action_flow import ActionFlowDecoder, FlowMatchingOutput
from .belief_core import WorldBeliefCore
from .code_dynamics import (
    CodeDynamicsDecoder,
    FutureCodePrediction,
    decode_prefix_ids,
    encode_prefix_ids,
    future_code_metrics,
    persistence_code_metrics,
    transition_family_masks,
)
from .codewam_v1 import (
    CodeWAMLossOutput,
    CodeWAMV1,
    build_codewam_v1,
)
from .codewam_v2 import CodeWAMV2, build_codewam_v2
from .config import CodeWAMConfig
from .continuous_state import (
    ContinuousStateEncoder,
    TemporalLatentPredictor,
    temporal_pretraining_loss,
)
from .frozen_codebook import FrozenCodebookAdapter
from .multiclock_dynamics import MultiClockTransitionModel
from .world_state import StructuredWorldBuilder

__all__ = [
    "ActionBatch",
    "ActionFlowDecoder",
    "CodeMeasurements",
    "CodeDynamicsDecoder",
    "CodeTokens",
    "CodeWAMBatch",
    "CodeWAMConfig",
    "CodeWAMLossOutput",
    "CodeWAMV1",
    "CodeWAMV2",
    "ContinuousState",
    "ContinuousStateEncoder",
    "FlowMatchingOutput",
    "FrozenCodebookAdapter",
    "FutureCodePrediction",
    "FutureCodeTargets",
    "MultiClockCodeState",
    "MultiClockTransitionModel",
    "PolicyCondition",
    "StateInputs",
    "StructuredWorldBuilder",
    "StructuredWorldState",
    "SupervisionMasks",
    "TemporalLatentPredictor",
    "TransitionSchedule",
    "WorldBeliefCore",
    "WorldBelief",
    "build_codewam_v1",
    "build_codewam_v2",
    "decode_prefix_ids",
    "encode_prefix_ids",
    "future_code_metrics",
    "persistence_code_metrics",
    "temporal_pretraining_loss",
    "transition_family_masks",
]
