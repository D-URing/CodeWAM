from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from codewam.codebook_eval.droid_pooled_export import DROID_POOLED_EXPORT_SCHEMA
from codewam.codebook_eval.manifest import EpisodeManifest, EpisodeRecord
from codewam.codebook_eval.run_setup import prepare_droid_streaming_run
from codewam.codebook_eval.shards import file_sha256


def canonical_hash(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RunSetupTests(unittest.TestCase):
    def test_finalized_export_compiles_locked_train_and_evaluation_configs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export_dir = root / "pooled-export"
            pooled_dir = export_dir / "pooled"
            pooled_dir.mkdir(parents=True)
            shard_paths = []
            for index in range(2):
                path = pooled_dir / f"shard-{index:03d}.pt"
                path.write_bytes(f"pooled-{index}".encode("utf-8"))
                shard_paths.append(path)

            records = tuple(
                EpisodeRecord(
                    dataset="droid-1.0.1",
                    episode_id=f"episode-{split}",
                    num_steps=12,
                    source_uri=f"memory://{split}",
                    scene_id=f"scene-{split}",
                    split=split,
                )
                for split in ("train", "val", "test")
            )
            manifest = EpisodeManifest.from_records(records)
            manifest_path = export_dir / "pooled_manifest.jsonl"
            manifest.write_jsonl(manifest_path)

            contract_payload = {
                "schema": DROID_POOLED_EXPORT_SCHEMA,
                "dataset_revisions": ["droid-1.0.1"],
                "vae_model_id": "Wan-AI/Wan2.2-TI2V-5B",
                "vae_sha256": "vae-sha",
                "preprocess_revision": "preprocess-v1",
                "cameras": ["exterior", "wrist"],
            }
            contract = {
                **contract_payload,
                "contract_hash": canonical_hash(contract_payload),
            }
            (export_dir / "contract.json").write_text(
                json.dumps(contract),
                encoding="utf-8",
            )
            summary = {
                "schema": DROID_POOLED_EXPORT_SCHEMA,
                "contract_hash": contract["contract_hash"],
                "pooled_manifest": {
                    "path": str(manifest_path),
                    **manifest.stats(),
                },
                "pooled_shards": len(shard_paths),
                "files": [
                    {
                        "path": str(path),
                        "bytes": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                    for path in reversed(shard_paths)
                ],
            }
            (export_dir / "export_summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )

            result = prepare_droid_streaming_run(
                export_dir,
                root / "rq",
                pool=2,
                k=8,
                levels=3,
                device="cpu",
                camera_ids=("exterior",),
            )
            train = OmegaConf.load(result["train_config"])
            evaluation = OmegaConf.load(result["evaluation_config"])

            self.assertEqual(result["pooled_shards"], 2)
            self.assertEqual(train.metadata.dataset, "droid-1.0.1")
            self.assertEqual(train.descriptor.pool, 2)
            self.assertEqual(list(train.descriptor.camera_ids), ["exterior"])
            self.assertEqual(train.training.k, 8)
            self.assertEqual(
                list(train.metadata.source_checksums),
                [file_sha256(path) for path in shard_paths],
            )
            self.assertEqual(
                evaluation.artifacts.Q3,
                str((root / "rq" / "Q3" / "codebook.pt").resolve()),
            )
            self.assertEqual(
                Path(result["train_config"]).stat().st_mode & 0o777,
                0o644,
            )

    def test_stale_pooled_shard_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export_dir = root / "export"
            pooled_dir = export_dir / "pooled"
            pooled_dir.mkdir(parents=True)
            shard = pooled_dir / "shard.pt"
            shard.write_bytes(b"current")
            manifest = EpisodeManifest.from_records(
                (
                    EpisodeRecord(
                        dataset="droid",
                        episode_id="episode",
                        num_steps=3,
                        source_uri="memory://episode",
                        scene_id="scene",
                        split="train",
                    ),
                )
            )
            manifest_path = export_dir / "pooled_manifest.jsonl"
            manifest.write_jsonl(manifest_path)
            contract_payload = {
                "schema": DROID_POOLED_EXPORT_SCHEMA,
                "dataset_revisions": ["droid"],
                "vae_model_id": "wan",
                "vae_sha256": "vae",
                "preprocess_revision": "prep",
                "cameras": ["exterior", "wrist"],
            }
            contract = {
                **contract_payload,
                "contract_hash": canonical_hash(contract_payload),
            }
            (export_dir / "contract.json").write_text(
                json.dumps(contract),
                encoding="utf-8",
            )
            (export_dir / "export_summary.json").write_text(
                json.dumps(
                    {
                        "schema": DROID_POOLED_EXPORT_SCHEMA,
                        "contract_hash": contract["contract_hash"],
                        "pooled_manifest": {
                            "path": str(manifest_path),
                            **manifest.stats(),
                        },
                        "pooled_shards": 1,
                        "files": [
                            {
                                "path": str(shard),
                                "bytes": shard.stat().st_size,
                                "sha256": "stale",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                prepare_droid_streaming_run(export_dir, root / "rq")


if __name__ == "__main__":
    unittest.main()
