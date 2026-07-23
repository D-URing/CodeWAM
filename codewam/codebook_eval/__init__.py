"""Offline codebook evaluation utilities for CodeWAM."""

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
    "fit_normalization",
    "iter_pooled_feature_episodes",
    "train_streaming_codebooks",
    "write_pooled_feature_shard",
]
