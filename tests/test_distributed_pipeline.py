from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
from omegaconf import OmegaConf

from codewam.codebook_eval.pipeline import (
    create_synthetic_streaming_fixture,
    partition_shard_paths,
    train_streaming_codebooks,
)
from codewam.codebook_eval.streaming import FrozenRQArtifact


def _distributed_train_worker(
    rank: int,
    world_size: int,
    init_path: str,
    config_path: str,
) -> None:
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        train_streaming_codebooks(config_path)
    finally:
        torch.distributed.destroy_process_group()


@unittest.skipUnless(
    torch.distributed.is_available(),
    "Torch distributed is unavailable",
)
class DistributedPipelineTests(unittest.TestCase):
    def test_shards_are_lpt_partitioned_without_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for index, size in enumerate((10, 20, 30, 40, 50)):
                path = root / f"shard-{index}.pt"
                path.write_bytes(bytes(size))
                paths.append(path)

            assignments = partition_shard_paths(tuple(paths), world_size=2)
            self.assertEqual(
                {path for rank_paths in assignments for path in rank_paths},
                set(paths),
            )
            self.assertFalse(set(assignments[0]) & set(assignments[1]))
            loads = [
                sum(path.stat().st_size for path in rank)
                for rank in assignments
            ]
            self.assertLessEqual(abs(loads[0] - loads[1]), 20)

    def test_two_rank_rq_matches_single_process_with_shared_initialization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = create_synthetic_streaming_fixture(root)
            config = OmegaConf.load(config_path)
            config.training.device = "cpu"
            config.training.cpu_threads = 1
            config.training.max_iters = 3
            config.training.patience = 1
            config.training.reservoir_size = 128
            config.output_dir = str(root / "distributed")
            OmegaConf.save(config, config_path)

            torch.multiprocessing.spawn(
                _distributed_train_worker,
                args=(2, str(root / "gloo-init"), str(config_path)),
                nprocs=2,
                join=True,
            )
            first_summary_bytes = (
                root / "distributed" / "train_summary.json"
            ).read_bytes()
            torch.multiprocessing.spawn(
                _distributed_train_worker,
                args=(2, str(root / "gloo-resume"), str(config_path)),
                nprocs=2,
                join=True,
            )
            self.assertEqual(
                (
                    root / "distributed" / "train_summary.json"
                ).read_bytes(),
                first_summary_bytes,
            )

            single_config_path = root / "single.yaml"
            config.output_dir = str(root / "single")
            OmegaConf.save(config, single_config_path)
            train_streaming_codebooks(single_config_path)

            distributed_summary = OmegaConf.load(
                root / "distributed" / "train_summary.json"
            )
            single_summary = OmegaConf.load(
                root / "single" / "train_summary.json"
            )
            self.assertEqual(
                [row.normalization_count for row in distributed_summary],
                [row.normalization_count for row in single_summary],
            )
            for family in ("Q2", "Q3", "Q5"):
                distributed = FrozenRQArtifact.load(
                    root / "distributed" / family / "codebook.pt"
                )
                single = FrozenRQArtifact.load(
                    root / "single" / family / "codebook.pt"
                )
                for observed, expected in zip(
                    distributed.centers,
                    single.centers,
                ):
                    torch.testing.assert_close(
                        observed,
                        expected,
                        atol=2e-5,
                        rtol=0,
                    )


if __name__ == "__main__":
    unittest.main()
