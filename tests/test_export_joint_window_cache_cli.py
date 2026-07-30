from __future__ import annotations

import os
import runpy
import sys
import unittest
from unittest import mock


class JointCacheExportCLITests(unittest.TestCase):
    def test_torchrun_local_rank_selects_its_cuda_device(self) -> None:
        arguments = [
            "scripts/export_joint_window_cache.py",
            "--output-dir",
            "unused",
            "--finalize-only",
        ]
        with (
            mock.patch.dict(os.environ, {"LOCAL_RANK": "3"}),
            mock.patch.object(sys, "argv", arguments),
        ):
            module = runpy.run_path(
                "scripts/export_joint_window_cache.py",
                run_name="not_main",
            )
            parsed = module["parse_args"]()
        self.assertEqual(parsed.device, "cuda:3")


if __name__ == "__main__":
    unittest.main()
