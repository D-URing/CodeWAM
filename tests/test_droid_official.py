from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXPECTED_BYTES = 2_192_615_094
MANIFEST_RELATIVE = Path("droid/droid-100-rlds-1.0.0")
DATA_RELATIVE = Path("droid_100/1.0.0")


class DroidOfficialVerifierTests(unittest.TestCase):
    def test_debug_verifier_accepts_complete_sparse_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "datasets"
            local_root = data_root / DATA_RELATIVE
            manifest_root = data_root / "manifests"
            manifest_dir = manifest_root / MANIFEST_RELATIVE
            local_root.mkdir(parents=True)
            manifest_dir.mkdir(parents=True)

            metadata_size = 128
            features_payload = "{}"
            shard_bytes = EXPECTED_BYTES - metadata_size - len(features_payload)
            split_metadata = json.dumps(
                {"splits": [{"numBytes": str(shard_bytes)}]},
                separators=(",", ":"),
            )
            split_metadata += " " * (metadata_size - len(split_metadata))
            (local_root / "dataset_info.json").write_text(
                split_metadata, encoding="utf-8"
            )
            (local_root / "features.json").write_text(
                features_payload, encoding="utf-8"
            )

            objects = [
                self._object("dataset_info.json", metadata_size),
                self._object("features.json", len(features_payload)),
            ]
            base_size, remainder = divmod(shard_bytes, 31)
            for index in range(31):
                size = base_size + (1 if index < remainder else 0)
                name = f"r2d2_faceblur-train.tfrecord-{index:05d}-of-00031"
                with (local_root / name).open("wb") as handle:
                    handle.truncate(size)
                objects.append(self._object(name, size))

            (manifest_dir / "objects.json").write_text(
                json.dumps(
                    {
                        "bucket": "gresearch",
                        "prefix": "robotics/droid_100/1.0.0/",
                        "objects": sorted(objects, key=lambda item: item["path"]),
                    }
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/droid_official.py",
                    "verify",
                    "--data-root",
                    str(data_root),
                    "--manifest-root",
                    str(manifest_root),
                    "--dataset",
                    "debug",
                    "--skip-hashes",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(
                (manifest_dir / "verification.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["local_objects"], 33)
            self.assertEqual(report["local_bytes"], EXPECTED_BYTES)
            self.assertEqual(report["errors"], [])

    @staticmethod
    def _object(path: str, size: int) -> dict[str, object]:
        return {
            "path": path,
            "size": size,
            "md5_base64": base64.b64encode(hashlib.md5(b"").digest()).decode(),
            "crc32c_base64": "AAAAAA==",
            "generation": "1",
            "updated": "2026-01-01T00:00:00Z",
            "etag": "test",
        }


if __name__ == "__main__":
    unittest.main()
