from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.probe_wan_latents import _load_mapping


class ProbeWanLatentsCliTests(unittest.TestCase):
    def test_analysis_does_not_resolve_unused_export_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.yaml"
            path.write_text(
                "\n".join(
                    [
                        "export:",
                        "  data_dir: ${oc.env:CODEWAM_MISSING_DATA_ROOT}/droid",
                        "  output_dir: pooled",
                        "analysis:",
                        "  pooled_shards:",
                        "    - ${export.output_dir}/episode-*.pt",
                        "  output_dir: analysis",
                    ]
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(
                os.environ,
                {"CODEWAM_MISSING_DATA_ROOT": ""},
                clear=False,
            ):
                del os.environ["CODEWAM_MISSING_DATA_ROOT"]
                analysis = _load_mapping(path, "analysis")

        self.assertEqual(analysis["pooled_shards"], ["pooled/episode-*.pt"])
        self.assertEqual(analysis["output_dir"], "analysis")


if __name__ == "__main__":
    unittest.main()
