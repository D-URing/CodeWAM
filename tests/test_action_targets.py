from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from codewam.codebook_eval.shards import file_sha256
from codewam.codebook_eval.manifest import EpisodeManifest, EpisodeRecord
from codewam.data.action_target_export import (
    _create_contract,
    finalize_droid_action_targets,
)
from codewam.data.action_targets import (
    DroidActionTargetSegment,
    FrozenDroidActionTargetCache,
    action_target_mapping_statistics,
    create_droid_action_target_contract,
    validate_action_targets_against_joint_episodes,
    validate_droid_action_target_contract,
    write_droid_action_target_contract,
    write_droid_action_target_index,
    write_droid_action_target_shard,
)
from codewam.data.droid_rlds import (
    DROID_ACTION_COMPONENT_DIMS,
    DroidRLDSActionEpisode,
)
from codewam.data.joint_cache import (
    JointWindowConfig,
    build_joint_windows,
    create_joint_cache_contract,
    finalize_joint_cache,
    write_joint_cache_contract,
    write_joint_episode_shard,
)
from tests.test_joint_cache import make_chart, make_episode


def _components(steps: int) -> dict[str, torch.Tensor]:
    rows = {
        name: torch.arange(steps * width, dtype=torch.float32).reshape(
            steps, width
        )
        / 10
        for name, width in DROID_ACTION_COMPONENT_DIMS.items()
    }
    rows["gripper_position"] = torch.linspace(0.0, 1.0, steps).unsqueeze(1)
    return rows


def _segment() -> DroidActionTargetSegment:
    components = _components(3)
    flat = torch.cat(
        (components["cartesian_position"], components["gripper_position"]),
        dim=-1,
    )
    return DroidActionTargetSegment(
        episode_id="parent@2:5",
        parent_episode_id="parent",
        manifest_key="droid:parent",
        range_index=0,
        range_start=2,
        range_stop=5,
        split="train",
        source_shard="source.tfrecord",
        record_index=4,
        flat_action=flat,
        action_components=components,
        action_valid=torch.tensor([True, True, False]),
    )


def _contract() -> dict:
    return create_droid_action_target_contract(
        joint_cache_contract_hash="joint-contract",
        joint_cache_summary_sha256="joint-summary",
        source_manifest_fingerprint="manifest-fingerprint",
        source_manifest_sha256="manifest-sha",
        dataset_revision="droid-1.0.1",
        implementation_sha256={"extractor": "implementation"},
    )


class DroidActionTargetTests(unittest.TestCase):
    def test_contract_is_hashed_and_leaves_target_unselected(self) -> None:
        contract = _contract()
        validate_droid_action_target_contract(contract)
        self.assertEqual(
            contract["target_selection"],
            "unselected-controller-contract-required",
        )
        tampered = dict(contract)
        tampered["storage_dtype"] = "float16"
        with self.assertRaisesRegex(RuntimeError, "contract is invalid"):
            validate_droid_action_target_contract(tampered)

    def test_mapping_and_joint_alignment_are_exact(self) -> None:
        segment = _segment()
        mapping = action_target_mapping_statistics((segment,))
        self.assertEqual(mapping["exact_values"], mapping["values"])
        self.assertEqual(mapping["max_abs_error"], 0.0)
        joint = SimpleNamespace(
            episode_id=segment.episode_id,
            parent_episode_id=segment.parent_episode_id,
            manifest_key=segment.manifest_key,
            range_index=segment.range_index,
            range_start=segment.range_start,
            range_stop=segment.range_stop,
            split=segment.split,
            source_actions=segment.flat_action.clone(),
            source_action_valid=segment.action_valid.clone(),
        )
        alignment = validate_action_targets_against_joint_episodes(
            (segment,),
            (joint,),
        )
        self.assertTrue(alignment["flat_action_exact"])
        joint.source_actions[0, 0] += 1
        with self.assertRaisesRegex(RuntimeError, "flat values differ"):
            validate_action_targets_against_joint_episodes((segment,), (joint,))

    def test_frozen_cache_round_trip_and_contract_binding(self) -> None:
        segment = _segment()
        contract = _contract()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_droid_action_target_contract(root, contract)
            shard_path = root / "shards" / "source.pt"
            info = write_droid_action_target_shard(
                shard_path,
                contract_hash=contract["contract_hash"],
                segments=(segment,),
                metadata={"source_shard": segment.source_shard},
            )
            relative = str(shard_path.relative_to(root))
            write_droid_action_target_index(
                root,
                contract_hash=contract["contract_hash"],
                file_rows=(
                    {
                        **info,
                        "path": relative,
                        "source_steps": segment.source_steps,
                    },
                ),
                segment_rows=(
                    {
                        "episode_id": segment.episode_id,
                        "shard": relative,
                        "offset": 0,
                    },
                ),
                mapping_statistics=action_target_mapping_statistics((segment,)),
            )
            cache = FrozenDroidActionTargetCache(
                root,
                expected_joint_cache_contract_hash="joint-contract",
            )
            loaded = cache.segment(segment.episode_id)
            torch.testing.assert_close(loaded.flat_action, segment.flat_action)
            self.assertEqual(cache.episode_ids, (segment.episode_id,))
            with self.assertRaisesRegex(RuntimeError, "different joint cache"):
                FrozenDroidActionTargetCache(
                    root,
                    expected_joint_cache_contract_hash="other",
                )
            self.assertEqual(file_sha256(shard_path), info["sha256"])

    def test_image_free_episode_preserves_source_ranges(self) -> None:
        components = _components(5)
        episode = DroidRLDSActionEpisode(
            episode_id="parent",
            index=4,
            action=torch.cat(
                (
                    components["cartesian_position"],
                    components["gripper_position"],
                ),
                dim=-1,
            ),
            action_components=components,
            is_first=torch.tensor([True, False, False, False, False]),
            is_last=torch.tensor([False, False, False, False, True]),
            is_terminal=torch.tensor([False, False, False, False, True]),
            split="train",
            keep_ranges=((0, 2), (3, 5)),
            manifest_key="droid:parent",
            source_file="source",
            recording_folder="recording",
            source_shard="source.tfrecord",
            record_index=4,
        )
        self.assertEqual(
            episode.eligible_ranges(),
            ((0, 0, 2), (1, 3, 5)),
        )
        self.assertEqual(episode.action_valid.tolist(), [True] * 4 + [False])

    def test_finalize_binds_source_joint_and_action_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.jsonl"
            joint_root = root / "joint"
            action_root = root / "actions"
            source = EpisodeManifest.from_records(
                (
                    EpisodeRecord(
                        dataset="droid",
                        episode_id="episode",
                        num_steps=80,
                        source_uri="source",
                        split="train",
                        source_checksum="crc32c:source",
                        metadata={
                            "keep_ranges": [[0, 80]],
                            "eligible_steps": 80,
                            "rlds_shard_name": "synthetic.tfrecord",
                            "rlds_shard_bytes": 123,
                            "rlds_record_index": 0,
                        },
                    ),
                )
            )
            source.write_jsonl(source_path)
            chart = make_chart()
            joint_contract = create_joint_cache_contract(
                dataset_revision="droid-1.0.1",
                source_manifest_fingerprint=source.fingerprint(),
                source_manifest_sha256=file_sha256(source_path),
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
            joint_episode = make_episode()
            windows = build_joint_windows(
                joint_episode,
                config=JointWindowConfig(),
                artifact_sha256=chart.artifact_sha256,
            )
            write_joint_cache_contract(joint_root, joint_contract)
            write_joint_episode_shard(
                joint_root,
                "shard-00000",
                (joint_episode,),
                windows,
                contract_hash=joint_contract["contract_hash"],
            )
            finalize_joint_cache(joint_root)

            action_contract = _create_contract(
                source_manifest_path=source_path,
                joint_cache_dir=joint_root,
                joint_contract=joint_contract,
                source_manifest=source,
            )
            write_droid_action_target_contract(action_root, action_contract)
            components = {
                name: torch.zeros((80, width), dtype=torch.float32)
                for name, width in DROID_ACTION_COMPONENT_DIMS.items()
            }
            components["cartesian_position"] = (
                joint_episode.source_actions[:, :6].clone()
            )
            components["gripper_position"] = (
                joint_episode.source_actions[:, 6:].clone()
            )
            target = DroidActionTargetSegment(
                episode_id=joint_episode.episode_id,
                parent_episode_id=joint_episode.parent_episode_id,
                manifest_key=joint_episode.manifest_key,
                range_index=0,
                range_start=0,
                range_stop=80,
                split="train",
                source_shard="synthetic.tfrecord",
                record_index=0,
                flat_action=joint_episode.source_actions.clone(),
                action_components=components,
                action_valid=joint_episode.source_action_valid.clone(),
            )
            joint_shard = joint_root / "episode_shards" / "shard-00000.pt"
            action_shard = action_root / "shards" / "shard-00000.pt"
            write_droid_action_target_shard(
                action_shard,
                contract_hash=action_contract["contract_hash"],
                segments=(target,),
                metadata={
                    "source_shard": "synthetic.tfrecord",
                    "source_shard_bytes": 123,
                    "joint_cache_contract_hash": joint_contract["contract_hash"],
                    "joint_episode_shard": "episode_shards/shard-00000.pt",
                    "joint_episode_shard_sha256": file_sha256(joint_shard),
                },
            )
            summary = finalize_droid_action_targets(
                source_manifest_path=source_path,
                joint_cache_dir=joint_root,
                output_dir=action_root,
            )
            self.assertEqual(summary["episodes"], 1)
            self.assertEqual(summary["source_steps"], 80)
            self.assertEqual(
                summary["flat_action_mapping"]["exact_fraction"],
                1.0,
            )
            loaded = FrozenDroidActionTargetCache(action_root).segment(
                target.episode_id
            )
            torch.testing.assert_close(loaded.flat_action, target.flat_action)


if __name__ == "__main__":
    unittest.main()
