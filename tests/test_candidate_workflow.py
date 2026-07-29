from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from codewam.codebook_eval.pipeline import (
    create_synthetic_streaming_fixture,
)
from codewam.codebook_eval.workflow import (
    run_streaming_codebook_candidate,
)


class CandidateWorkflowTests(unittest.TestCase):
    def test_candidate_runs_all_frozen_stages_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_path = create_synthetic_streaming_fixture(root)
            train = OmegaConf.load(train_path)
            train.descriptor.strides = [3]
            train.training.device = "cpu"
            train.training.cpu_threads = 1
            train.training.max_iters = 2
            train.training.patience = 1
            train.training.k = 4
            train.training.reservoir_size = 128
            OmegaConf.save(train, train_path)

            run_dir = Path(str(train.output_dir))
            evaluation_path = root / "evaluate.yaml"
            evaluation = OmegaConf.create(
                {
                    "output_dir": str(run_dir / "heldout"),
                    "input": {
                        "pooled_shards": list(
                            train.input.pooled_shards
                        ),
                        "manifest": str(train.input.manifest),
                        "group_by": "scene",
                    },
                    "metadata": {
                        "dataset": str(train.metadata.dataset),
                    },
                    "artifacts": {
                        "Q3": str(run_dir / "Q3/codebook.pt"),
                    },
                    "evaluation": {
                        "splits": ["val", "test"],
                        "device": "cpu",
                        "cpu_threads": 1,
                        "batch_size": 32,
                        "center_block_size": 4,
                        "representatives_per_code": 1,
                        "resume": True,
                    },
                }
            )
            OmegaConf.save(evaluation, evaluation_path)

            first = run_streaming_codebook_candidate(
                train_path,
                evaluation_path,
                min_train_count=1,
            )
            resumed = run_streaming_codebook_candidate(
                train_path,
                evaluation_path,
                min_train_count=1,
            )

        self.assertEqual(first, resumed)
        self.assertEqual(first["families"], ["Q3"])
        self.assertEqual(
            first["row_counts"],
            {
                "train": 1,
                "heldout": 2,
                "association": 18,
                "concentration": 18,
            },
        )
        self.assertTrue(
            all(value["sha256"] for value in first["reports"].values())
        )


if __name__ == "__main__":
    unittest.main()
