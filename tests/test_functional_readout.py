from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from codewam.codebook_eval.functional_readout import (
    _CodeStatistics,
    _ContinuousStatistics,
    _episode_functional_values,
    _fit_code_model,
    _fit_continuous_model,
    _scene_train_subset,
    probe_codebook_functional_readout,
)
from codewam.codebook_eval.manifest import (
    EpisodeManifest,
    EpisodeRecord,
)
from codewam.codebook_eval.shards import PooledFeatureEpisode
from codewam.codebook_eval.pipeline import (
    create_synthetic_streaming_fixture,
    train_streaming_codebooks,
)
from codewam.codebook_eval.streaming import (
    CausalDescriptorSpec,
    FrozenRQArtifact,
    NormalizationStats,
)


def _artifact(stride: int, state_dimension: int) -> FrozenRQArtifact:
    dimension = state_dimension * 3
    return FrozenRQArtifact(
        family=f"Q{stride}",
        descriptor=CausalDescriptorSpec(
            stride=stride,
            pool=4,
            camera_ids=("wrist",),
        ),
        normalization=NormalizationStats(
            count=2,
            mean=torch.zeros(dimension),
            std=torch.ones(dimension),
        ),
        centers=(torch.zeros((2, dimension)),),
        metadata={
            "dataset_revision": "v1",
            "wan_model_id": "wan",
            "wan_revision": "v1",
            "preprocess_revision": "v1",
            "manifest_fingerprint": "manifest",
            "source_checksums": ["source"],
            "config_hash": "config",
        },
    )


class FunctionalReadoutTests(unittest.TestCase):
    def test_continuous_input_is_exact_union_of_causal_states(self) -> None:
        ticks = 16
        pooled = torch.arange(
            ticks * 2,
            dtype=torch.float32,
        ).reshape(ticks, 1, 2, 1, 1).expand(-1, -1, -1, 4, 4)
        episode = PooledFeatureEpisode(
            episode_id="episode",
            split="train",
            timestamps=torch.arange(ticks, dtype=torch.float64),
            pooled_g4=pooled,
            camera_ids=("wrist",),
            action=torch.arange(ticks, dtype=torch.float32).unsqueeze(1),
            proprio=torch.arange(ticks, dtype=torch.float32).unsqueeze(1),
        )
        artifacts = {
            "Q2": _artifact(2, 2 * 4 * 4),
            "Q3": _artifact(3, 2 * 4 * 4),
            "Q5": _artifact(5, 2 * 4 * 4),
        }

        values = _episode_functional_values(episode, artifacts)

        self.assertIsNotNone(values)
        assert values is not None
        self.assertEqual(values.base.shape[0], 6)
        self.assertEqual(
            values.base.shape[1],
            1 + 7 * 2 * 4 * 4,
        )
        expected_offsets = (-10, -6, -5, -4, -3, -2, 0)
        first_current = 10
        state_dimension = 2 * 4 * 4
        for position, offset in enumerate(expected_offsets):
            start = 1 + position * state_dimension
            expected = pooled[first_current + offset].reshape(-1)
            torch.testing.assert_close(
                values.base[0, start : start + state_dimension],
                expected,
            )

    def test_scene_subset_is_nested_and_deterministic(self) -> None:
        records = []
        for index in range(20):
            records.append(
                EpisodeRecord(
                    dataset="fixture",
                    episode_id=f"episode-{index}",
                    num_steps=20,
                    source_uri=f"fixture://{index}",
                    scene_id=f"scene-{index // 2}",
                    building_id="building",
                    institution_id="institution",
                    split="train",
                )
            )
        manifest = EpisodeManifest(records)

        small_ids, small_scenes = _scene_train_subset(
            manifest,
            fraction=0.2,
            seed=7,
        )
        large_ids, large_scenes = _scene_train_subset(
            manifest,
            fraction=0.5,
            seed=7,
        )

        self.assertTrue(set(small_scenes) < set(large_scenes))
        self.assertTrue(small_ids < large_ids)
        repeated = _scene_train_subset(
            manifest,
            fraction=0.2,
            seed=7,
        )
        self.assertEqual((small_ids, small_scenes), repeated)

    def test_code_model_adds_nonlinear_partition_to_continuous_input(
        self,
    ) -> None:
        device = torch.device("cpu")
        x = torch.tensor(
            [[-1.0], [-0.5], [0.5], [1.0]] * 32,
            dtype=torch.float64,
        )
        key = torch.tensor([0, 1, 1, 0] * 32)
        y = (2.0 * (key == 1).double()).unsqueeze(1)
        continuous = _ContinuousStatistics(
            dimension=1,
            target_dimension=1,
            device=device,
        )
        continuous.update(x, y)
        code = _CodeStatistics(
            capacities={"Q2": 2},
            continuous_dimension=1,
            target_dimension=1,
            device=device,
        )
        code.update(keys={"Q2": key}, x=x, y=y)

        h_only = _fit_continuous_model(
            continuous,
            name="P1",
            x_indices=torch.tensor([0]),
            alpha=1e-3,
        )
        h_plus_c = _fit_code_model(
            continuous,
            code,
            name="P3",
            x_indices=torch.tensor([0]),
            alpha=1e-3,
        )

        h_error = (h_only.predict(x) - y).square().mean()
        hc_error = (
            h_plus_c.predict(x, keys={"Q2": key}) - y
        ).square().mean()
        self.assertLess(float(hc_error), float(h_error) * 0.01)

    def test_probe_runs_three_artifact_seeds_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = create_synthetic_streaming_fixture(root)
            config = OmegaConf.load(config_path)
            config.training.device = "cpu"
            config.training.cpu_threads = 1
            config.training.max_iters = 1
            config.training.patience = 1
            config.training.tol = 1e-3
            config.training.k = 2
            config.training.levels = 1
            config.training.reservoir_size = 64
            OmegaConf.save(config, config_path)
            trained = train_streaming_codebooks(config_path)
            original = {
                row["family"]: Path(row["artifact"])
                for row in trained
            }
            runs = {}
            for seed in (7, 19, 31):
                label = f"seed{seed}"
                runs[label] = {}
                for family, artifact in original.items():
                    destination = root / label / family
                    shutil.copytree(artifact.parent, destination)
                    contract_path = destination / "contract.json"
                    contract = json.loads(
                        contract_path.read_text(encoding="utf-8")
                    )
                    contract["seed"] = seed
                    contract_path.write_text(
                        json.dumps(contract),
                        encoding="utf-8",
                    )
                    runs[label][family] = destination / "codebook.pt"
            output = root / "functional"

            first = probe_codebook_functional_readout(
                manifest_path=root / "manifest.jsonl",
                pooled_shards=(root / "pooled/*.pt",),
                runs=runs,
                output_dir=output,
                code_depths={"Q2": 1, "Q3": 1, "Q5": 1},
                alpha_candidates=(1e-2,),
                device="cpu",
                cpu_threads=1,
                batch_size=32,
                center_block_size=4,
            )
            resumed = probe_codebook_functional_readout(
                manifest_path=root / "manifest.jsonl",
                pooled_shards=(root / "pooled/*.pt",),
                runs=runs,
                output_dir=output,
                code_depths={"Q2": 1, "Q3": 1, "Q5": 1},
                alpha_candidates=(1e-2,),
                device="cpu",
                cpu_threads=1,
                batch_size=32,
                center_block_size=4,
            )

        self.assertEqual(first, resumed)
        self.assertEqual(
            first["run_seeds"],
            {"seed19": 19, "seed31": 31, "seed7": 7},
        )
        self.assertEqual(
            {row["model"] for row in first["rows"]},
            {"P0", "P1", "P2", "P3"},
        )
        self.assertEqual(
            {summary["split"] for summary in first["summaries"]},
            {"val", "test"},
        )


if __name__ == "__main__":
    unittest.main()
