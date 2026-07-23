from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from codewam.codebook_eval.pipeline import (
    create_synthetic_streaming_fixture,
    train_streaming_codebooks,
)
from codewam.codebook_eval.streaming import FrozenRQArtifact


class StreamingPipelineTests(unittest.TestCase):
    def test_one_command_trains_and_resumes_all_three_families(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = create_synthetic_streaming_fixture(root)
            config = OmegaConf.load(config_path)
            config.training.device = "cpu"
            config.training.max_iters = 2
            OmegaConf.save(config, config_path)

            first = train_streaming_codebooks(config_path)
            resumed = train_streaming_codebooks(config_path)

            self.assertEqual([row["family"] for row in first], ["Q2", "Q3", "Q5"])
            self.assertEqual(first, resumed)
            for row in first:
                artifact = FrozenRQArtifact.load(row["artifact"])
                self.assertEqual(artifact.family, row["family"])
                self.assertEqual(len(artifact.centers), 3)
                self.assertGreater(row["normalization_count"], 0)


if __name__ == "__main__":
    unittest.main()
