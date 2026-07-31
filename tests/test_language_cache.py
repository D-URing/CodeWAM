from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from codewam.data.language_cache import (
    FrozenLanguageCache,
    LanguageConditionedJointWindowCache,
    create_language_cache_contract,
    normalize_language_instruction,
    write_frozen_language_cache,
)
from codewam.data.joint_cache import JointWindowSample


def _contract() -> dict:
    return create_language_cache_contract(
        joint_cache_contract_hash="joint-cache",
        joint_cache_summary_sha256="joint-summary",
        source_manifest_fingerprint="manifest-fingerprint",
        source_manifest_sha256="manifest-sha",
        encoder_id="synthetic-t5",
        encoder_revision="revision",
        hidden_size=8,
        max_tokens=4,
        dtype="float16",
        model_files=(
            {"name": "model.safetensors", "sha256": "model-sha", "bytes": 10},
        ),
        implementation_sha256={"language_cache": "implementation"},
    )


def _write(root: Path) -> FrozenLanguageCache:
    write_frozen_language_cache(
        root,
        contract=_contract(),
        instructions=("move object", "open drawer"),
        episode_instruction={"episode-a": 0, "episode-b": 1},
        token_sequences=(
            torch.arange(16, dtype=torch.float16).reshape(2, 8),
            torch.arange(24, dtype=torch.float16).reshape(3, 8),
        ),
    )
    return FrozenLanguageCache(
        root,
        expected_joint_cache_contract_hash="joint-cache",
    )


class _FakeJointCache:
    def __init__(self, sample: JointWindowSample):
        self.sample = sample
        self.contract = {"contract_hash": "joint-cache"}
        self.summary = {}
        self.cache_dir = Path("joint")
        self.windows = (sample.record,)
        self.window_shards = ("shard.pt",)
        self.parent_episode_ids = (sample.record.parent_episode_id,)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.sample


class LanguageCacheTests(unittest.TestCase):
    def test_frozen_cache_round_trip_and_contract_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = _write(Path(temporary))
            expected = torch.arange(16, dtype=torch.float16).reshape(2, 8)
            torch.testing.assert_close(
                cache.tokens_for_parent("episode-a"),
                expected,
            )
            self.assertEqual(cache.hidden_size, 8)
            with self.assertRaisesRegex(KeyError, "episode-c"):
                cache.tokens_for_parent("episode-c")
            with self.assertRaisesRegex(RuntimeError, "different joint cache"):
                FrozenLanguageCache(
                    temporary,
                    expected_joint_cache_contract_hash="other",
                )

    def test_sidecar_attaches_tokens_without_mutating_joint_sample(self) -> None:
        sample = JointWindowSample(
            record=SimpleNamespace(parent_episode_id="episode-a"),
            latents=torch.zeros(1),
            latent_valid=torch.ones(1, dtype=torch.bool),
            proprio_history=torch.zeros(1),
            past_actions=torch.zeros(1),
            actions=torch.zeros(1),
            action_valid=torch.ones(1, dtype=torch.bool),
            current_codes=torch.zeros(1, dtype=torch.long),
            future_codes=torch.zeros(1, dtype=torch.long),
            code_available=torch.ones(1, dtype=torch.bool),
            language_tokens=None,
            language_valid=None,
        )
        with tempfile.TemporaryDirectory() as temporary:
            language = _write(Path(temporary))
            wrapped = LanguageConditionedJointWindowCache(
                _FakeJointCache(sample),
                language,
            )
            actual = wrapped[0]
        self.assertIsNone(sample.language_tokens)
        self.assertEqual(tuple(actual.language_tokens.shape), (2, 8))
        self.assertTrue(actual.language_valid.all())

    def test_text_normalization_is_deterministic(self) -> None:
        self.assertEqual(
            normalize_language_instruction("  open\n the   drawer "),
            "open the drawer",
        )


if __name__ == "__main__":
    unittest.main()
