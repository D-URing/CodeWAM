from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

import codewam.experiments.gate2 as gate2_module
from codewam.data.frozen_assignment import (
    FrozenArtifactChart,
    FrozenCausalCodeAssigner,
    load_frozen_artifact_chart,
)
from codewam.data.joint_cache import (
    JointEpisode,
    JointWindowCache,
    JointWindowConfig,
    JointWindowRecord,
    build_joint_windows,
    create_joint_cache_contract,
    finalize_joint_cache,
    write_joint_cache_contract,
    write_joint_episode_shard,
)
from codewam.experiments.gate2 import (
    Gate2RunConfig,
    _IndexSampler,
    _rank_indices,
    build_fixed_action_permutation,
    run_gate2,
)
from tests.model_fixtures import small_config, synthetic_artifacts


def _episode(
    identity: str,
    split: str,
    chart: FrozenArtifactChart,
) -> JointEpisode:
    generator = torch.Generator().manual_seed(
        100 + sum(ord(value) for value in identity)
    )
    ticks = 20
    source_steps = 80
    latents = torch.randn(
        (ticks, 1, 4, 4, 4),
        generator=generator,
    )
    source_indices = 4 * torch.arange(ticks, dtype=torch.long)
    assignment = FrozenCausalCodeAssigner(chart).assign(
        latents,
        latent_source_indices=source_indices,
        camera_ids=("wrist",),
    )
    action_valid = torch.ones(source_steps, dtype=torch.bool)
    action_valid[-1] = False
    return JointEpisode(
        episode_id=f"{identity}@0:80",
        parent_episode_id=identity,
        manifest_key=f"droid:{identity}",
        range_index=0,
        range_start=0,
        range_stop=source_steps,
        split=split,
        chart_name=chart.name,
        role="expert",
        camera_ids=("wrist",),
        latents=latents,
        latent_source_indices=source_indices,
        latent_valid=torch.ones((ticks, 1), dtype=torch.bool),
        source_actions=torch.randn(
            (source_steps, 7),
            generator=generator,
        ),
        source_proprio=torch.randn(
            (source_steps, 6),
            generator=generator,
        ),
        source_action_valid=action_valid,
        code_ids=assignment.code_ids,
        code_available=assignment.available,
        descriptor_source_indices=assignment.descriptor_source_indices,
        families=assignment.families,
        language_instruction="",
    )


def _write_fixture(root: Path) -> tuple[Path, dict[str, str]]:
    artifact_dir = root / "artifacts"
    artifact_paths = {}
    for artifact in synthetic_artifacts(
        "droid",
        descriptor_dim=3 * 4,
    ):
        path = artifact_dir / artifact.family / "codebook.pt"
        artifact.save(path)
        artifact_paths[artifact.family] = str(path)
    chart = load_frozen_artifact_chart("droid", artifact_paths)
    cache_dir = root / "cache"
    window = JointWindowConfig()
    contract = create_joint_cache_contract(
        dataset_revision="droid-v1",
        source_manifest_fingerprint="manifest",
        source_manifest_sha256="1" * 64,
        endpoint_audit_sha256="2" * 64,
        chart=chart,
        camera_ids=("wrist",),
        wan_model_id="synthetic-wan",
        wan_revision="synthetic-wan-revision",
        preprocess_revision="synthetic-preprocess",
        nominal_fps=15.0,
        action_dim=7,
        proprio_dim=6,
        latent_channels=4,
        window=window,
    )
    episodes = tuple(
        replace(
            _episode(f"{split}-{index}", split, chart),
            parent_episode_id=f"{split}-parent",
        )
        for split in ("train", "val", "test")
        for index in range(2)
    )
    windows = tuple(
        window_record
        for episode in episodes
        for window_record in build_joint_windows(
            episode,
            config=window,
            artifact_sha256=chart.artifact_sha256,
        )
    )
    write_joint_cache_contract(cache_dir, contract)
    write_joint_episode_shard(
        cache_dir,
        "fixture",
        episodes,
        windows,
        contract_hash=contract["contract_hash"],
    )
    finalize_joint_cache(cache_dir)
    return cache_dir, artifact_paths


class Gate2Tests(unittest.TestCase):
    def test_index_sampler_can_change_epochs_without_rebuilding_loader(self) -> None:
        sampler = _IndexSampler((1, 2, 3))
        self.assertEqual(list(sampler), [1, 2, 3])
        sampler.set_indices((4, 5))
        self.assertEqual(list(sampler), [4, 5])
        self.assertEqual(len(sampler), 2)

    def test_action_permutation_is_fixed_group_local_and_deranged(self) -> None:
        def row(index: int, split: str, episode: str) -> JointWindowRecord:
            return JointWindowRecord(
                window_id=f"window-{index}",
                episode_id=episode,
                parent_episode_id=episode,
                split=split,
                chart_name="droid",
                role="expert",
                families=("Q2", "Q3", "Q5"),
                state_latent_start=0,
                state_latent_stop=1,
                current_latent_index=0,
                future_latent_index=1,
                proprio_start=1,
                proprio_stop=2,
                past_action_start=0,
                past_action_stop=1,
                action_start=1,
                action_stop=17,
                decision_source_index=1,
                future_observation_source_index=17,
                current_code_ids=((0, 0, 0),) * 3,
                future_code_ids=((1, 1, 1),) * 3,
                code_available=(True,) * 3,
                current_descriptor_sources=((0, 1, 2),) * 3,
                future_descriptor_sources=((2, 3, 4),) * 3,
                descriptor_overlap=(1, 1, 1),
                artifact_sha256=("a", "b", "c"),
            )

        windows = tuple(
            row(index, "train" if index < 4 else "test", f"episode-{index}")
            for index in range(8)
        )
        first = build_fixed_action_permutation(windows, seed=7)
        second = build_fixed_action_permutation(windows, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first.singleton_groups, 0)
        for source, donor in enumerate(first.donor_indices):
            self.assertNotEqual(source, donor)
            self.assertEqual(windows[source].split, windows[donor].split)
            self.assertEqual(
                windows[source].action_stop - windows[source].action_start,
                windows[donor].action_stop - windows[donor].action_start,
            )

    def test_rank_indices_keep_each_shard_contiguous(self) -> None:
        groups = ("a",) * 4 + ("b",) * 3 + ("c",) * 5 + ("d",) * 4
        ranks = [
            _rank_indices(
                tuple(range(len(groups))),
                rank=rank,
                world_size=2,
                seed=7,
                epoch=3,
                training=True,
                group_keys=groups,
            )
            for rank in range(2)
        ]
        self.assertEqual(set(ranks[0]) | set(ranks[1]), set(range(len(groups))))
        self.assertFalse(set(ranks[0]) & set(ranks[1]))
        self.assertEqual(len(ranks[0]), len(ranks[1]))
        for values in ranks:
            ordered_groups = [groups[index] for index in values]
            runs = [
                name
                for index, name in enumerate(ordered_groups)
                if index == 0 or name != ordered_groups[index - 1]
            ]
            self.assertEqual(len(runs), len(set(runs)))

    def test_one_command_run_keeps_noact_action_encoder_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache_dir, artifact_paths = _write_fixture(root)
            model = replace(
                small_config(variant="C2"),
                max_time=8,
                max_action_horizon=16,
            )
            with mock.patch.object(
                gate2_module,
                "_make_loader",
                wraps=gate2_module._make_loader,
            ) as make_loader:
                report = run_gate2(
                    Gate2RunConfig(
                        cache_dir=str(cache_dir),
                        output_dir=str(root / "gate2"),
                        artifact_paths=artifact_paths,
                        batch_size=2,
                        eval_batch_size=4,
                        epochs=1,
                        max_steps=1,
                        device="cpu",
                        amp_dtype="float32",
                        calibration_bins=5,
                        bootstrap_samples=20,
                        minimum_gate_episodes=3,
                        model=model,
                    )
                )
                loader_calls = make_loader.call_count
            initialization = torch.load(
                root / "gate2" / "initialization.pt",
                map_location="cpu",
                weights_only=False,
            )["model"]
            noact = torch.load(
                root / "gate2" / "noact" / "final.pt",
                map_location="cpu",
                weights_only=False,
            )["model"]
            protocol = json.loads(
                (root / "gate2" / "protocol.json").read_text(encoding="utf-8")
            )
            cache = JointWindowCache(cache_dir)
            for index in (0, len(cache) - 1):
                actions, valid = cache.action_chunk(index)
                torch.testing.assert_close(
                    actions,
                    cache[index].actions,
                )
                torch.testing.assert_close(valid, cache[index].action_valid)

        action_key = "codewam.code_dynamics.action_projection.weight"
        torch.testing.assert_close(noact[action_key], initialization[action_key])
        self.assertEqual(loader_calls, 5)
        self.assertEqual(
            protocol["distributed"],
            {
                "world_size": 1,
                "per_rank_batch_size": 2,
                "effective_batch_size": 2,
                "per_rank_eval_batch_size": 4,
                "effective_eval_batch_size": 4,
            },
        )
        self.assertEqual(
            {
                result["optimizer_steps"]
                for result in report["training"].values()
            },
            {1},
        )
        self.assertEqual(report["conditions"]["TRUE"]["test"]["windows"], 12)
        self.assertEqual(report["action_index"]["rows"], 36)
        self.assertEqual(report["gate"]["verdict"], "invalid")
        self.assertEqual(report["gate"]["minimum_gate_episodes"], 3)
        self.assertLessEqual(
            report["paired_episode_comparisons"]["TRUE-vs-NOACT"]["episodes"],
            1,
        )
        self.assertEqual(
            report["conditions"]["TRUE"]["test"]["strata"]["all"][
                "classification_unit"
            ],
            "rq_level",
        )


if __name__ == "__main__":
    unittest.main()
