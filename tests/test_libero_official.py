from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXPECTED_SUITES = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_90": 90,
    "libero_10": 10,
}
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


class LiberoOfficialVerifierTests(unittest.TestCase):
    def test_verifier_accepts_complete_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root = root / "official"
            manifest_path = root / "expected_hdf5.sha256"
            report_path = root / "verification.json"
            checksum_lines: list[str] = []

            for suite, count in EXPECTED_SUITES.items():
                suite_root = dataset_root / suite
                suite_root.mkdir(parents=True)
                for index in range(count):
                    payload = HDF5_MAGIC + f"{suite}:{index}".encode()
                    relative = Path(suite) / f"task_{index:03d}.hdf5"
                    (dataset_root / relative).write_bytes(payload)
                    checksum_lines.append(
                        f"{hashlib.sha256(payload).hexdigest()}  "
                        f"{relative.as_posix()}\n"
                    )

            (dataset_root / ".gitattributes").write_text("test\n", encoding="utf-8")
            (dataset_root / "README.md").write_text("test\n", encoding="utf-8")
            manifest_path.write_text("".join(checksum_lines), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/libero_official.py",
                    "verify",
                    "--root",
                    str(dataset_root),
                    "--sha256-manifest",
                    str(manifest_path),
                    "--report",
                    str(report_path),
                    "--workers",
                    "2",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertEqual(
                report["revision"],
                "f13aa24a3da8c43c7225569f28c562979fa0e35a",
            )
            self.assertEqual(report["total_hdf5"], 130)
            self.assertEqual(report["hdf5_signature_checked"], 130)
            self.assertEqual(report["sha256_checked"], 130)
            self.assertEqual(len(report["sha256_manifest_digest"]), 64)
            self.assertEqual(report["errors"], [])


if __name__ == "__main__":
    unittest.main()
