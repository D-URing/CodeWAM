"""Offline codebook evaluation utilities for CodeWAM."""

from .association import probe_frozen_codebook_associations
from .comparison import compare_streaming_runs
from .concentration import probe_frozen_codebook_concentration
from .evaluation import evaluate_frozen_codebooks
from .manifest import EpisodeManifest, EpisodeRecord, SplitConfig
from .pipeline import train_streaming_codebooks
from .shards import PooledFeatureEpisode, iter_pooled_feature_episodes, write_pooled_feature_shard
from .streaming import (
    CausalDescriptorSource,
    CausalDescriptorSpec,
    FrozenRQArtifact,
    NormalizationStats,
    StreamingKMeans,
    StreamingKMeansConfig,
    StreamingRQTrainer,
    fit_normalization,
)

__all__ = [
    "CausalDescriptorSource",
    "CausalDescriptorSpec",
    "EpisodeManifest",
    "EpisodeRecord",
    "FrozenRQArtifact",
    "NormalizationStats",
    "PooledFeatureEpisode",
    "SplitConfig",
    "StreamingKMeans",
    "StreamingKMeansConfig",
    "StreamingRQTrainer",
    "compare_streaming_runs",
    "evaluate_frozen_codebooks",
    "fit_normalization",
    "iter_pooled_feature_episodes",
    "probe_frozen_codebook_associations",
    "probe_frozen_codebook_concentration",
    "train_streaming_codebooks",
    "write_pooled_feature_shard",
]
