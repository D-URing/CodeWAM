from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from codewam.codebook_eval.clustering import kmeans
from codewam.codebook_eval.shards import PooledFeatureEpisode
from codewam.codebook_eval.streaming import (
    CausalDescriptorSource,
    CausalDescriptorSpec,
    FrozenRQArtifact,
    RunningMoments,
    StreamingKMeans,
    StreamingKMeansConfig,
    StreamingRQTrainer,
    UniformReservoir,
    encode_residual_quantizer,
    fit_normalization,
)


def scalar_episode(
    episode_id: str,
    offset: float = 0.0,
    split: str = "train",
    ticks: int = 10,
    invalid_ticks: tuple[int, ...] = (),
) -> PooledFeatureEpisode:
    values = torch.arange(ticks, dtype=torch.float32) + float(offset)
    valid = torch.ones((ticks, 1), dtype=torch.bool)
    if invalid_ticks:
        valid[list(invalid_ticks)] = False
    return PooledFeatureEpisode(
        episode_id=episode_id,
        split=split,
        timestamps=torch.arange(ticks, dtype=torch.float64) * 0.25,
        pooled_g4=values.view(ticks, 1, 1, 1, 1).expand(ticks, 1, 1, 4, 4).half(),
        camera_ids=("exterior",),
        valid_mask=valid,
    )


def tensor_batch_factory(values: torch.Tensor, batch_size: int):
    def batches():
        for start in range(0, values.shape[0], batch_size):
            yield values[start : start + batch_size]

    return batches


class CausalDescriptorTests(unittest.TestCase):
    def test_q2_uses_only_two_past_offsets_and_current(self) -> None:
        episodes = (
            scalar_episode("episode-a", offset=0.0, ticks=9),
            scalar_episode("episode-b", offset=100.0, ticks=7),
        )
        source = CausalDescriptorSource(
            episode_factory=lambda: iter(episodes),
            spec=CausalDescriptorSpec(stride=2, pool=1, max_gap_factor=None),
            batch_size=3,
            split="train",
        )
        batches = list(source)
        vectors = torch.cat([batch.vectors for batch in batches])
        episode_ids = sum((batch.episode_ids for batch in batches), ())
        time_indices = torch.cat([batch.time_indices for batch in batches])

        expected_a = torch.tensor(
            [
                [0.0, 2.0, 4.0],
                [1.0, 3.0, 5.0],
                [2.0, 4.0, 6.0],
                [3.0, 5.0, 7.0],
                [4.0, 6.0, 8.0],
            ]
        )
        expected_b = torch.tensor(
            [
                [100.0, 102.0, 104.0],
                [101.0, 103.0, 105.0],
                [102.0, 104.0, 106.0],
            ]
        )
        torch.testing.assert_close(vectors.float(), torch.cat([expected_a, expected_b]))
        self.assertEqual(episode_ids, ("episode-a",) * 5 + ("episode-b",) * 3)
        torch.testing.assert_close(time_indices, torch.tensor([4, 5, 6, 7, 8, 4, 5, 6]))

    def test_unavailable_past_tick_removes_only_dependent_descriptors(self) -> None:
        episode = scalar_episode("masked", ticks=9, invalid_ticks=(2,))
        source = CausalDescriptorSource(
            episode_factory=lambda: iter((episode,)),
            spec=CausalDescriptorSpec(stride=2, pool=1, max_gap_factor=None),
            batch_size=8,
            split="train",
        )
        batch = next(iter(source))
        self.assertEqual(batch.time_indices.tolist(), [5, 7, 8])

    def test_train_only_normalization_matches_direct_statistics(self) -> None:
        episodes = (scalar_episode("a", ticks=9), scalar_episode("b", offset=20, ticks=8))
        source = CausalDescriptorSource(
            episode_factory=lambda: iter(episodes),
            spec=CausalDescriptorSpec(stride=2, pool=1, max_gap_factor=None),
            batch_size=2,
            split="train",
        )
        all_vectors = torch.cat([batch.vectors for batch in source])
        stats = fit_normalization(source)
        all_vectors = all_vectors.float()
        torch.testing.assert_close(stats.mean, all_vectors.mean(dim=0))
        torch.testing.assert_close(
            stats.std,
            all_vectors.var(dim=0, unbiased=False).sqrt(),
        )
        normalized = torch.cat(
            list(source.vector_batch_factory(normalization=stats, device="cpu")())
        )
        torch.testing.assert_close(normalized.mean(dim=0), torch.zeros(3), atol=1e-6, rtol=0)
        torch.testing.assert_close(
            normalized.var(dim=0, unbiased=False),
            torch.ones(3),
            atol=1e-5,
            rtol=0,
        )

        val_source = CausalDescriptorSource(
            episode_factory=lambda: iter((scalar_episode("val", split="val"),)),
            spec=CausalDescriptorSpec(stride=2, pool=1),
            split="val",
        )
        with self.assertRaisesRegex(ValueError, "train-only"):
            fit_normalization(val_source)

    def test_running_moments_merge_is_partition_invariant(self) -> None:
        generator = torch.Generator().manual_seed(11)
        values = torch.randn((101, 7), generator=generator)
        whole = RunningMoments()
        whole.update(values)
        partitioned = RunningMoments()
        for chunk in values.split([17, 23, 61]):
            local = RunningMoments()
            local.update(chunk)
            partitioned.merge(local)
        whole_stats = whole.finalize()
        partitioned_stats = partitioned.finalize()
        torch.testing.assert_close(whole_stats.mean, partitioned_stats.mean)
        torch.testing.assert_close(whole_stats.std, partitioned_stats.std)


class StreamingClusteringTests(unittest.TestCase):
    @staticmethod
    def synthetic_clusters() -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(23)
        expected = torch.tensor([[-4.0, -1.0], [0.5, 4.0], [4.0, -0.5]])
        values = torch.cat(
            [
                center + 0.2 * torch.randn((80, 2), generator=generator)
                for center in expected
            ],
            dim=0,
        )
        return values, expected

    def test_uniform_reservoir_is_chunk_partition_invariant(self) -> None:
        values = torch.arange(1000, dtype=torch.float32).view(200, 5)
        whole = UniformReservoir(max_samples=31, seed=13)
        whole.update(values)
        partitioned = UniformReservoir(max_samples=31, seed=13)
        for chunk in values.split([7, 19, 3, 81, 90]):
            partitioned.update(chunk)
        self.assertEqual(whole.seen, values.shape[0])
        self.assertEqual(partitioned.seen, values.shape[0])
        torch.testing.assert_close(whole.result(), partitioned.result())

    def test_streaming_kmeans_is_batch_partition_invariant(self) -> None:
        values, initial = self.synthetic_clusters()
        config = StreamingKMeansConfig(
            k=3,
            max_iters=8,
            tol=0.0,
            seed=5,
            device="cpu",
        )
        small_batches = StreamingKMeans(config).fit(
            tensor_batch_factory(values, batch_size=13),
            initial_centers=initial,
        )
        large_batches = StreamingKMeans(config).fit(
            tensor_batch_factory(values, batch_size=71),
            initial_centers=initial,
        )
        torch.testing.assert_close(
            small_batches.centers,
            large_batches.centers,
            atol=2e-6,
            rtol=0,
        )
        torch.testing.assert_close(small_batches.counts, large_batches.counts)
        self.assertAlmostEqual(small_batches.inertia, large_batches.inertia, places=6)

    def test_streaming_kmeans_matches_legacy_full_batch_lloyd(self) -> None:
        values, _ = self.synthetic_clusters()
        seed = next(
            candidate
            for candidate in range(100)
            if len(
                set(
                    (
                        torch.randperm(
                            values.shape[0],
                            generator=torch.Generator().manual_seed(candidate),
                        )[:3]
                        // 80
                    ).tolist()
                )
            )
            == 3
        )
        generator = torch.Generator().manual_seed(seed)
        initial_indices = torch.randperm(values.shape[0], generator=generator)[:3]
        initial = values[initial_indices]
        reference = kmeans(values, k=3, iters=8, seed=seed, tol=0.0)
        streaming = StreamingKMeans(
            StreamingKMeansConfig(
                k=3,
                max_iters=8,
                tol=0.0,
                seed=seed,
                device="cpu",
            )
        ).fit(
            tensor_batch_factory(values, batch_size=29),
            initial_centers=initial,
        )

        torch.testing.assert_close(
            streaming.centers,
            reference.centers,
            atol=2e-6,
            rtol=0,
        )
        self.assertAlmostEqual(streaming.inertia, reference.inertia, places=6)

    def test_kmeans_checkpoint_resume_matches_uninterrupted_run(self) -> None:
        values, initial = self.synthetic_clusters()
        factory = tensor_batch_factory(values, batch_size=19)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "kmeans.pt"
            first_stage = StreamingKMeans(
                StreamingKMeansConfig(k=3, max_iters=2, tol=0.0, device="cpu")
            )
            first_stage.fit(
                factory,
                initial_centers=initial,
                checkpoint_path=checkpoint,
            )
            resumed = StreamingKMeans(
                StreamingKMeansConfig(k=3, max_iters=7, tol=0.0, device="cpu")
            ).fit(factory, checkpoint_path=checkpoint, resume=True)
            uninterrupted = StreamingKMeans(
                StreamingKMeansConfig(k=3, max_iters=7, tol=0.0, device="cpu")
            ).fit(factory, initial_centers=initial)

        torch.testing.assert_close(resumed.centers, uninterrupted.centers)
        torch.testing.assert_close(resumed.counts, uninterrupted.counts)
        self.assertAlmostEqual(resumed.inertia, uninterrupted.inertia, places=7)
        self.assertEqual(resumed.iterations, 7)

    def test_converged_checkpoint_does_not_run_extra_iterations(self) -> None:
        values, initial = self.synthetic_clusters()
        factory = tensor_batch_factory(values, batch_size=23)
        config = StreamingKMeansConfig(
            k=3,
            max_iters=20,
            tol=1.0,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "converged.pt"
            first = StreamingKMeans(config).fit(
                factory,
                initial_centers=initial,
                checkpoint_path=checkpoint,
            )
            resumed = StreamingKMeans(config).fit(
                factory,
                checkpoint_path=checkpoint,
                resume=True,
            )
        self.assertTrue(first.converged)
        self.assertEqual(resumed.history, first.history)
        torch.testing.assert_close(resumed.centers, first.centers)

    def test_three_level_rq_reduces_residual_and_freezes_artifact(self) -> None:
        generator = torch.Generator().manual_seed(37)
        values = torch.randn((256, 6), generator=generator)
        factory = tensor_batch_factory(values, batch_size=31)
        config = StreamingKMeansConfig(
            k=4,
            max_iters=7,
            tol=0.0,
            seed=9,
            reservoir_size=128,
            device="cpu",
        )
        result = StreamingRQTrainer(config, levels=3).fit(factory)
        self.assertEqual(len(result.centers), 3)
        self.assertEqual(len(result.residual_mse), 4)
        for before, after in zip(result.residual_mse, result.residual_mse[1:]):
            self.assertLessEqual(after, before + 1e-7)

        codes, quantized, residual = encode_residual_quantizer(values, result.centers)
        self.assertEqual(tuple(codes.shape), (values.shape[0], 3))
        torch.testing.assert_close(values, quantized + residual)
        self.assertAlmostEqual(
            float(residual.square().mean().item()),
            result.residual_mse[-1],
            places=6,
        )

        moments = RunningMoments()
        moments.update(values)
        artifact = FrozenRQArtifact(
            family="Q2",
            descriptor=CausalDescriptorSpec(stride=2, pool=1),
            normalization=moments.finalize(),
            centers=result.centers,
            metadata={
                "manifest_fingerprint": "synthetic",
                "dataset_revision": "synthetic-v1",
                "wan_model_id": "test-wan",
                "wan_revision": "test-wan-revision",
                "preprocess_revision": "unit-test",
                "config_hash": "synthetic-config",
                "source_checksums": ["synthetic-source"],
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "q2.pt"
            artifact.save(path)
            loaded = FrozenRQArtifact.load(path)
        self.assertEqual(loaded.family, "Q2")
        self.assertEqual(loaded.descriptor, artifact.descriptor)
        self.assertEqual(loaded.metadata, artifact.metadata)
        for expected, actual in zip(artifact.centers, loaded.centers):
            torch.testing.assert_close(expected, actual)


if __name__ == "__main__":
    unittest.main()
