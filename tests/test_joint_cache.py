from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from codewam.data.frozen_assignment import (
    FrozenArtifactChart,
    FrozenCausalCodeAssigner,
)
from codewam.data.joint_cache import (
    JointEpisode,
    JointWindowCache,
    JointWindowConfig,
    build_joint_windows,
    collate_joint_windows,
    create_joint_cache_contract,
    finalize_joint_cache,
    write_joint_cache_contract,
    write_joint_episode_shard,
)
from tests.model_fixtures import synthetic_artifacts


def make_chart() -> FrozenArtifactChart:
    return FrozenArtifactChart(
        name="droid",
        artifacts=synthetic_artifacts("droid"),
        artifact_sha256=("a" * 64, "b" * 64, "c" * 64),
        artifact_paths=("Q2.pt", "Q3.pt", "Q5.pt"),
    )


def make_episode(*, with_language: bool = True) -> JointEpisode:
    torch.manual_seed(31)
    latents = torch.randn((20, 1, 4, 4, 4))
    latent_source_indices = 4 * torch.arange(20, dtype=torch.long)
    assignment = FrozenCausalCodeAssigner(make_chart()).assign(
        latents,
        latent_source_indices=latent_source_indices,
        camera_ids=("wrist",),
    )
    source_steps = 80
    action_valid = torch.ones(source_steps, dtype=torch.bool)
    action_valid[-1] = False
    return JointEpisode(
        episode_id="episode@0:80",
        parent_episode_id="episode",
        manifest_key="droid:episode",
        range_index=0,
        range_start=0,
        range_stop=source_steps,
        split="train",
        chart_name="droid",
        role="expert",
        camera_ids=("wrist",),
        latents=latents.half(),
        latent_source_indices=latent_source_indices,
        latent_valid=torch.ones((20, 1), dtype=torch.bool),
        source_actions=torch.randn((source_steps, 7)),
        source_proprio=torch.randn((source_steps, 6)),
        source_action_valid=action_valid,
        code_ids=assignment.code_ids,
        code_available=assignment.available,
        descriptor_source_indices=assignment.descriptor_source_indices,
        families=assignment.families,
        language_instruction="move the object",
        language_tokens=(
            torch.randn((3, 8)) if with_language else None
        ),
        language_valid=(
            torch.tensor([True, True, False]) if with_language else None
        ),
        metadata={"source_shard": "synthetic"},
    )


def make_contract(chart: FrozenArtifactChart) -> dict:
    return create_joint_cache_contract(
        dataset_revision="droid-1.0.1",
        source_manifest_fingerprint="manifest-fingerprint",
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
        window=JointWindowConfig(),
        language_encoder_id="synthetic-language",
        language_encoder_revision="v1",
        language_dim=8,
    )


class JointCacheTests(unittest.TestCase):
    def _write_cache(
        self,
        root: Path,
        *,
        with_language: bool = True,
    ) -> tuple[JointEpisode, tuple, dict]:
        chart = make_chart()
        contract = make_contract(chart)
        episode = make_episode(with_language=with_language)
        windows = build_joint_windows(
            episode,
            config=JointWindowConfig(),
            artifact_sha256=chart.artifact_sha256,
        )
        write_joint_cache_contract(root, contract)
        write_joint_episode_shard(
            root,
            "shard-00000",
            (episode,),
            windows,
            contract_hash=contract["contract_hash"],
        )
        summary = finalize_joint_cache(root)
        return episode, windows, summary

    def test_round_trip_deduplicates_episode_and_materializes_model_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, windows, summary = self._write_cache(root)
            cache = JointWindowCache(root, split="train")
            batch = collate_joint_windows(
                (cache[0], cache[1]),
                language_dim=8,
            )

        self.assertEqual(len(windows), 6)
        self.assertEqual(summary["episodes"], 1)
        self.assertEqual(summary["windows"], 6)
        self.assertEqual(tuple(batch.model.state.latents.shape), (2, 8, 1, 4, 4, 4))
        self.assertEqual(tuple(batch.model.actions.values.shape), (2, 16, 7))
        self.assertEqual(tuple(batch.model.codes.code_ids.shape), (2, 3, 3))
        self.assertEqual(batch.episode_ids, ("episode@0:80",) * 2)
        self.assertEqual(batch.descriptor_overlap.tolist(), [[1, 0, 0]] * 2)
        self.assertTrue(batch.model.supervision.action.all())
        self.assertTrue(batch.model.supervision.dynamics.all())

    def test_missing_language_gates_imitation_but_not_dynamics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_cache(root, with_language=False)
            cache = JointWindowCache(root)
            batch = collate_joint_windows((cache[0],), language_dim=8)

        self.assertFalse(batch.model.supervision.action.item())
        self.assertTrue(batch.model.supervision.dynamics.item())
        self.assertFalse(batch.model.policy.language_valid.any())

    def test_writer_rejects_a_window_whose_code_label_changed(self) -> None:
        chart = make_chart()
        episode = make_episode()
        windows = build_joint_windows(
            episode,
            config=JointWindowConfig(),
            artifact_sha256=chart.artifact_sha256,
        )
        wrong_codes = list(windows[0].future_code_ids)
        wrong_family = list(wrong_codes[0])
        wrong_family[0] = (wrong_family[0] + 1) % 4
        wrong_codes[0] = tuple(wrong_family)
        wrong = replace(windows[0], future_code_ids=tuple(wrong_codes))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = make_contract(chart)
            write_joint_cache_contract(root, contract)
            with self.assertRaisesRegex(RuntimeError, "code label changed"):
                write_joint_episode_shard(
                    root,
                    "bad",
                    (episode,),
                    (wrong,),
                    contract_hash=contract["contract_hash"],
                )

    def test_index_hash_change_is_detected_before_loading_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_cache(root)
            windows_path = root / "windows.jsonl"
            windows_path.write_text(
                windows_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "index hash changed"):
                JointWindowCache(root)


if __name__ == "__main__":
    unittest.main()
