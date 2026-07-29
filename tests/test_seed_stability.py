from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from codewam.codebook_eval.seed_stability import (
    _best_overlap_mapping,
    _mapped_codes,
    _nmi_ari,
    _prefix_ids,
    _training_contract_identity,
)


class SeedStabilityTests(unittest.TestCase):
    def test_best_mapping_recovers_permuted_labels(self) -> None:
        contingency = torch.tensor(
            [
                [0, 5, 0],
                [0, 0, 7],
                [9, 0, 0],
            ]
        )

        mapping = _best_overlap_mapping(contingency)

        self.assertEqual(mapping, (2, 0, 1))

    def test_nmi_and_ari_are_one_for_permuted_partition(self) -> None:
        contingency = torch.tensor(
            [
                [0, 5, 0],
                [0, 0, 7],
                [9, 0, 0],
            ]
        )

        nmi, ari = _nmi_ari(contingency)

        self.assertAlmostEqual(nmi, 1.0, places=6)
        self.assertAlmostEqual(ari, 1.0, places=6)

    def test_mapped_codes_apply_each_level_independently(self) -> None:
        codes = torch.tensor([[0, 1], [2, 0]])

        mapped = _mapped_codes(
            codes,
            [(2, 0, 1), (1, 2, 0)],
        )

        torch.testing.assert_close(
            mapped,
            torch.tensor([[2, 2], [1, 1]]),
        )

    def test_prefix_ids_preserve_ordered_rq_tuple(self) -> None:
        codes = torch.tensor([[0, 1, 2], [2, 0, 1]])

        prefixes = _prefix_ids(codes, k=3)

        self.assertEqual(
            [value.tolist() for value in prefixes],
            [[0, 2], [1, 6], [5, 19]],
        )

    def test_training_contract_supplies_verified_seed(self) -> None:
        implementation = {"pipeline": "a", "streaming": "b"}
        source_checksums = ["source-a", "source-b"]
        artifact = SimpleNamespace(
            family="Q2",
            descriptor=SimpleNamespace(
                stride=2,
                pool=4,
                max_gap_factor=1.5,
                camera_ids=("wrist_image_left",),
            ),
            centers=(torch.zeros(8, 6),) * 3,
            metadata={
                "manifest_fingerprint": "manifest",
                "source_checksums": source_checksums,
                "implementation_sha256": implementation,
            },
        )
        contract = {
            "schema": "codewam.family-run-contract.v1",
            "family": "Q2",
            "stride": 2,
            "pool": 4,
            "max_gap_factor": 1.5,
            "camera_ids": ["wrist_image_left"],
            "k": 8,
            "levels": 3,
            "manifest_fingerprint": "manifest",
            "source_checksums": source_checksums,
            "implementation_sha256": implementation,
            "seed": 19,
            "tol": 1e-3,
            "patience": 2,
            "initialization_policy": "test",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "codebook.pt"
            (root / "contract.json").write_text(
                json.dumps(contract),
                encoding="utf-8",
            )

            identity = _training_contract_identity(
                artifact_path,
                artifact,
            )

        self.assertEqual(identity["seed"], 19)
        self.assertEqual(identity["tol"], 1e-3)
        self.assertEqual(identity["patience"], 2)


if __name__ == "__main__":
    unittest.main()
