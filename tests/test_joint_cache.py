from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

from codewam.codebook_eval.shards import file_sha256
from codewam.data.frozen_assignment import (
    FrozenArtifactChart,
    FrozenCausalCodeAssigner,
)
from codewam.data.joint_cache import (
    JointEpisode,
    JointWindowCache,
    JointWindowConfig,
    add_compact_joint_window_index,
    build_joint_windows,
    collate_joint_windows,
    create_joint_cache_contract,
    finalize_joint_cache,
    write_joint_cache_contract,
    write_joint_episode_shard,
)
from codewam.data.joint_cache_export import (
    JOINT_CACHE_EXPORT_REPORT_SCHEMA,
    finalize_exported_joint_cache,
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
            metadata={"source_shard_name": "synthetic.tfrecord"},
        )
        summary = finalize_joint_cache(root)
        return episode, windows, summary

    def _write_export_report(
        self,
        root: Path,
        *,
        contract_hash: str,
        planned: int,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        selected = 1
        payload = {
            "schema": JOINT_CACHE_EXPORT_REPORT_SCHEMA,
            "contract_hash": contract_hash,
            "rank": rank,
            "world_size": world_size,
            "source_shards_planned": planned,
            "source_shards_selected": selected,
            "source_shards_exported_or_reused": selected,
            "selection_complete": selected == planned,
            "max_source_shards": None if selected == planned else selected,
            "episodes": 1,
            "windows": 6,
            "elapsed_seconds": 1.0,
            "completed_unix_seconds": 10.0,
            "outputs": [
                {
                    "source_shard": "synthetic.tfrecord",
                    "episodes": 1,
                    "windows": 6,
                }
            ],
        }
        path = root / f"rank-{rank:03d}-of-{world_size:03d}-report.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def test_round_trip_deduplicates_episode_and_materializes_model_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, windows, summary = self._write_cache(root)
            cache = JointWindowCache(root, split="train")
            batch = collate_joint_windows(
                (cache[0], cache[1]),
                language_dim=8,
            )
            compact_actions, compact_valid = cache.action_chunk(0)
            torch.testing.assert_close(compact_actions, cache[0].actions)
            torch.testing.assert_close(compact_valid, cache[0].action_valid)

        self.assertEqual(len(windows), 6)
        self.assertEqual(summary["episodes"], 1)
        self.assertEqual(summary["windows"], 6)
        self.assertIn("window_records", summary["indices"])
        self.assertEqual(pickle.loads(pickle.dumps(windows[0])), windows[0])
        self.assertEqual(tuple(batch.model.state.latents.shape), (2, 8, 1, 4, 4, 4))
        self.assertEqual(tuple(batch.model.actions.values.shape), (2, 16, 7))
        self.assertEqual(tuple(batch.model.codes.code_ids.shape), (2, 3, 3))
        self.assertEqual(batch.episode_ids, ("episode@0:80",) * 2)
        self.assertEqual(batch.parent_episode_ids, ("episode",) * 2)
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

    def test_export_finalize_records_a_complete_rank_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, initial = self._write_cache(root)
            self._write_export_report(
                root,
                contract_hash=initial["contract_hash"],
                planned=1,
            )
            summary = finalize_exported_joint_cache(root)

        self.assertEqual(summary["export_audit"]["status"], "complete")
        self.assertEqual(summary["export_audit"]["world_size"], 1)
        self.assertEqual(
            summary["export_audit"]["source_shards_selected"],
            1,
        )

    def test_export_finalize_rejects_partial_selection_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, initial = self._write_cache(root)
            self._write_export_report(
                root,
                contract_hash=initial["contract_hash"],
                planned=2,
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "selected 1 of 2 planned shards",
            ):
                finalize_exported_joint_cache(root)
            summary = finalize_exported_joint_cache(
                root,
                allow_partial=True,
            )

        self.assertEqual(summary["export_audit"]["status"], "partial")

    def test_export_finalize_rejects_a_missing_rank_even_when_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, initial = self._write_cache(root)
            self._write_export_report(
                root,
                contract_hash=initial["contract_hash"],
                planned=1,
                world_size=2,
                rank=0,
            )
            with self.assertRaisesRegex(RuntimeError, "missing ranks.*1"):
                finalize_exported_joint_cache(
                    root,
                    allow_partial=True,
                )

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
        for relative in (
            "windows.jsonl",
            "window_records.pt",
            "window_actions.pt",
        ):
            with self.subTest(relative=relative):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    self._write_cache(root)
                    path = root / relative
                    with path.open("ab") as handle:
                        handle.write(b"\n")
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "index hash changed",
                    ):
                        JointWindowCache(root)

    def test_loader_streams_jsonl_indices_without_read_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_cache(root)
            original_read_text = Path.read_text

            def reject_jsonl_read_text(
                path: Path,
                *args: object,
                **kwargs: object,
            ) -> str:
                if path.suffix == ".jsonl":
                    raise AssertionError("JSONL indices must be streamed.")
                return original_read_text(path, *args, **kwargs)

            with mock.patch.object(
                Path,
                "read_text",
                new=reject_jsonl_read_text,
            ):
                cache = JointWindowCache(root, split="train")

        self.assertEqual(len(cache), 6)

    def test_compact_loader_does_not_parse_window_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, windows, _ = self._write_cache(root)
            from codewam.data import joint_cache

            original_iter_jsonl = joint_cache._iter_jsonl

            def reject_window_jsonl(path: Path):
                if Path(path).name == "windows.jsonl":
                    raise AssertionError(
                        "Compact cache must not parse windows.jsonl."
                    )
                yield from original_iter_jsonl(path)

            with mock.patch.object(
                joint_cache,
                "_iter_jsonl",
                new=reject_window_jsonl,
            ):
                cache = JointWindowCache(root, split="train")

            expected = tuple(sorted(windows, key=lambda row: row.window_id))
            self.assertEqual(tuple(cache.windows), expected)
            self.assertEqual(cache.split_indices("train"), tuple(range(6)))
            self.assertEqual(cache.split_indices("val"), ())
            self.assertEqual(
                tuple(cache.window_shards),
                ("episode_shards/shard-00000.pt",) * 6,
            )
            self.assertEqual(
                [row[4] for row in cache.permutation_rows()],
                [window.window_id for window in expected],
            )

    def test_legacy_cache_can_be_upgraded_without_reexport(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, windows, summary = self._write_cache(root)
            (root / summary["indices"]["window_records"]["path"]).unlink()
            summary["indices"].pop("window_records")
            (root / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            legacy = JointWindowCache(root)
            expected = tuple(sorted(windows, key=lambda row: row.window_id))
            self.assertEqual(tuple(legacy.windows), expected)
            upgraded = add_compact_joint_window_index(root)
            self.assertIn("window_records", upgraded["indices"])
            compact = JointWindowCache(root)
            self.assertEqual(tuple(compact.windows), expected)

    def test_shard_hash_is_checked_once_across_lru_reloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chart = make_chart()
            contract = make_contract(chart)
            first = make_episode()
            second = replace(
                make_episode(),
                episode_id="episode-2@0:80",
                parent_episode_id="episode",
                manifest_key="droid:episode-2",
            )
            write_joint_cache_contract(root, contract)
            for shard_name, episode in (
                ("shard-a", first),
                ("shard-b", second),
            ):
                windows = build_joint_windows(
                    episode,
                    config=JointWindowConfig(),
                    artifact_sha256=chart.artifact_sha256,
                )
                write_joint_episode_shard(
                    root,
                    shard_name,
                    (episode,),
                    windows,
                    contract_hash=contract["contract_hash"],
                )
            summary = finalize_joint_cache(root)
            self.assertEqual(
                summary["transition_coverage"]["train"]["any_family"][
                    "available_parent_episodes"
                ],
                1,
            )
            with mock.patch(
                "codewam.data.joint_cache.file_sha256",
                wraps=file_sha256,
            ) as digest:
                cache = JointWindowCache(root, max_cached_shards=1)
                digest.reset_mock()
                first_by_shard = {}
                for index, shard in enumerate(cache.window_shards):
                    first_by_shard.setdefault(shard, index)
                shard_indices = list(first_by_shard.values())
                self.assertEqual(len(shard_indices), 2)
                _ = cache[shard_indices[0]]
                _ = cache[shard_indices[1]]
                _ = cache[shard_indices[0]]

        shard_hashes = [
            call
            for call in digest.call_args_list
            if Path(call.args[0]).suffix == ".pt"
        ]
        self.assertEqual(len(shard_hashes), 2)


if __name__ == "__main__":
    unittest.main()
