from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from codewam.codebook_eval.shards import atomic_torch_save, file_sha256

from .joint_cache import JointWindowCache, JointWindowSample


LANGUAGE_CACHE_SCHEMA = "codewam.frozen-language-cache.v1"
LANGUAGE_CACHE_INDEX_SCHEMA = "codewam.frozen-language-index.v1"
LANGUAGE_CACHE_TENSOR_SCHEMA = "codewam.frozen-language-tensors.v1"
LANGUAGE_CACHE_SUMMARY_SCHEMA = "codewam.frozen-language-summary.v1"
LANGUAGE_TEXT_POLICY = "manifest-primary-task-collapse-whitespace-v1"


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def normalize_language_instruction(value: str) -> str:
    return " ".join(str(value).split())


def create_language_cache_contract(
    *,
    joint_cache_contract_hash: str,
    joint_cache_summary_sha256: str,
    source_manifest_fingerprint: str,
    source_manifest_sha256: str,
    encoder_id: str,
    encoder_revision: str,
    hidden_size: int,
    max_tokens: int,
    dtype: str,
    model_files: Sequence[Mapping[str, Any]],
    implementation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    strings = (
        joint_cache_contract_hash,
        joint_cache_summary_sha256,
        source_manifest_fingerprint,
        source_manifest_sha256,
        encoder_id,
        encoder_revision,
    )
    if any(not str(value) for value in strings):
        raise ValueError("Language-cache provenance strings must be nonempty.")
    if hidden_size <= 0 or max_tokens <= 0:
        raise ValueError("Language-cache dimensions must be positive.")
    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError(f"Unsupported language-cache dtype `{dtype}`.")
    normalized_files = []
    for row in model_files:
        name = str(row.get("name", ""))
        sha256 = str(row.get("sha256", ""))
        size = int(row.get("bytes", -1))
        if not name or not sha256 or size < 0:
            raise ValueError("Language model file provenance is incomplete.")
        normalized_files.append(
            {"name": name, "sha256": sha256, "bytes": size}
        )
    if not normalized_files:
        raise ValueError("Language-cache contract needs model file hashes.")
    payload = {
        "schema": LANGUAGE_CACHE_SCHEMA,
        "joint_cache": {
            "contract_hash": joint_cache_contract_hash,
            "summary_sha256": joint_cache_summary_sha256,
        },
        "source_manifest": {
            "fingerprint": source_manifest_fingerprint,
            "sha256": source_manifest_sha256,
        },
        "text_policy": LANGUAGE_TEXT_POLICY,
        "encoder": {
            "id": encoder_id,
            "revision": encoder_revision,
            "hidden_size": int(hidden_size),
            "max_tokens": int(max_tokens),
            "dtype": dtype,
            "output": "last_hidden_state-valid-tokens",
        },
        "model_files": sorted(normalized_files, key=lambda row: row["name"]),
        "implementation_sha256": {
            str(name): str(value)
            for name, value in sorted(implementation_sha256.items())
        },
    }
    return {**payload, "contract_hash": _canonical_hash(payload)}


def validate_language_cache_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema") != LANGUAGE_CACHE_SCHEMA:
        raise ValueError("Unsupported language-cache contract schema.")
    payload = {
        key: value for key, value in contract.items() if key != "contract_hash"
    }
    if contract.get("contract_hash") != _canonical_hash(payload):
        raise RuntimeError("Language-cache contract hash is invalid.")
    encoder = contract.get("encoder")
    joint = contract.get("joint_cache")
    source = contract.get("source_manifest")
    if (
        not isinstance(encoder, dict)
        or int(encoder.get("hidden_size", 0)) <= 0
        or int(encoder.get("max_tokens", 0)) <= 0
        or encoder.get("dtype") not in {"float16", "bfloat16", "float32"}
        or not isinstance(joint, dict)
        or not joint.get("contract_hash")
        or not isinstance(source, dict)
        or not source.get("sha256")
        or contract.get("text_policy") != LANGUAGE_TEXT_POLICY
    ):
        raise ValueError("Language-cache contract fields are invalid.")


def write_frozen_language_cache(
    output_dir: str | Path,
    *,
    contract: Mapping[str, Any],
    instructions: Sequence[str],
    episode_instruction: Mapping[str, int],
    token_sequences: Sequence[torch.Tensor],
) -> dict[str, Any]:
    validate_language_cache_contract(contract)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = tuple(normalize_language_instruction(value) for value in instructions)
    if (
        not normalized
        or any(not value for value in normalized)
        or len(set(normalized)) != len(normalized)
        or len(token_sequences) != len(normalized)
    ):
        raise ValueError("Language-cache instructions must be unique and nonempty.")
    hidden_size = int(contract["encoder"]["hidden_size"])
    max_tokens = int(contract["encoder"]["max_tokens"])
    expected_dtype = getattr(torch, str(contract["encoder"]["dtype"]))
    sequences = []
    lengths = []
    for index, value in enumerate(token_sequences):
        if (
            value.ndim != 2
            or value.shape[1] != hidden_size
            or value.shape[0] <= 0
            or value.shape[0] > max_tokens
            or value.dtype != expected_dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid language token sequence {index}.")
        sequence = value.detach().cpu().contiguous()
        sequences.append(sequence)
        lengths.append(int(sequence.shape[0]))
    episode_rows = []
    for episode_id, instruction_index in sorted(episode_instruction.items()):
        position = int(instruction_index)
        if not episode_id or position < 0 or position >= len(normalized):
            raise ValueError("Language-cache episode mapping is invalid.")
        episode_rows.append([str(episode_id), position])
    if not episode_rows or len({row[0] for row in episode_rows}) != len(episode_rows):
        raise ValueError("Language-cache episode IDs must be unique and nonempty.")

    offsets = torch.zeros(len(sequences) + 1, dtype=torch.long)
    offsets[1:] = torch.tensor(lengths, dtype=torch.long).cumsum(dim=0)
    values = torch.cat(sequences, dim=0)
    contract_path = output_dir / "contract.json"
    index_path = output_dir / "index.json"
    tensor_path = output_dir / "tokens.pt"
    existing = (
        json.loads(contract_path.read_text(encoding="utf-8"))
        if contract_path.is_file()
        else None
    )
    if existing is not None and existing != dict(contract):
        raise RuntimeError("Existing language-cache contract differs.")
    _atomic_json(contract_path, dict(contract))
    _atomic_json(
        index_path,
        {
            "schema": LANGUAGE_CACHE_INDEX_SCHEMA,
            "contract_hash": contract["contract_hash"],
            "instructions": list(normalized),
            "episodes": episode_rows,
        },
    )
    atomic_torch_save(
        {
            "schema": LANGUAGE_CACHE_TENSOR_SCHEMA,
            "contract_hash": contract["contract_hash"],
            "offsets": offsets,
            "values": values,
        },
        tensor_path,
    )
    summary = {
        "schema": LANGUAGE_CACHE_SUMMARY_SCHEMA,
        "contract_hash": contract["contract_hash"],
        "instructions": len(normalized),
        "episodes": len(episode_rows),
        "tokens": int(values.shape[0]),
        "token_length": {
            "minimum": min(lengths),
            "maximum": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
        "indices": {
            "index": {"path": index_path.name, "sha256": file_sha256(index_path)},
            "tokens": {"path": tensor_path.name, "sha256": file_sha256(tensor_path)},
        },
    }
    _atomic_json(output_dir / "summary.json", summary)
    return summary


class FrozenLanguageCache:
    """Read-only token-level language features keyed by parent episode ID."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        expected_joint_cache_contract_hash: str | None = None,
        verify_hashes: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.contract = json.loads(
            (self.cache_dir / "contract.json").read_text(encoding="utf-8")
        )
        validate_language_cache_contract(self.contract)
        joint_hash = str(self.contract["joint_cache"]["contract_hash"])
        if (
            expected_joint_cache_contract_hash is not None
            and joint_hash != expected_joint_cache_contract_hash
        ):
            raise RuntimeError("Language cache belongs to a different joint cache.")
        self.summary = json.loads(
            (self.cache_dir / "summary.json").read_text(encoding="utf-8")
        )
        if (
            self.summary.get("schema") != LANGUAGE_CACHE_SUMMARY_SCHEMA
            or self.summary.get("contract_hash") != self.contract["contract_hash"]
        ):
            raise RuntimeError("Language-cache summary does not match its contract.")
        for row in self.summary["indices"].values():
            path = self.cache_dir / str(row["path"])
            if verify_hashes and file_sha256(path) != row["sha256"]:
                raise RuntimeError(f"Language-cache file hash changed: {path}.")
        index_row = self.summary["indices"]["index"]
        index = json.loads(
            (self.cache_dir / index_row["path"]).read_text(encoding="utf-8")
        )
        if (
            index.get("schema") != LANGUAGE_CACHE_INDEX_SCHEMA
            or index.get("contract_hash") != self.contract["contract_hash"]
        ):
            raise RuntimeError("Language-cache index is invalid.")
        self.instructions = tuple(str(value) for value in index["instructions"])
        self._episode_instruction = {
            str(episode_id): int(position)
            for episode_id, position in index["episodes"]
        }
        if (
            len(self.instructions) != int(self.summary["instructions"])
            or len(self._episode_instruction) != int(self.summary["episodes"])
        ):
            raise RuntimeError("Language-cache index counts changed.")
        tensor_row = self.summary["indices"]["tokens"]
        try:
            payload = torch.load(
                self.cache_dir / tensor_row["path"],
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
        except TypeError:
            payload = torch.load(
                self.cache_dir / tensor_row["path"],
                map_location="cpu",
            )
        if (
            payload.get("schema") != LANGUAGE_CACHE_TENSOR_SCHEMA
            or payload.get("contract_hash") != self.contract["contract_hash"]
        ):
            raise RuntimeError("Language-cache tensors are invalid.")
        self._offsets = payload.get("offsets")
        self._values = payload.get("values")
        hidden_size = int(self.contract["encoder"]["hidden_size"])
        expected_dtype = getattr(torch, str(self.contract["encoder"]["dtype"]))
        if (
            not isinstance(self._offsets, torch.Tensor)
            or self._offsets.dtype != torch.long
            or tuple(self._offsets.shape) != (len(self.instructions) + 1,)
            or int(self._offsets[0]) != 0
            or not torch.all(self._offsets[1:] > self._offsets[:-1])
            or not isinstance(self._values, torch.Tensor)
            or self._values.dtype != expected_dtype
            or self._values.ndim != 2
            or self._values.shape[1] != hidden_size
            or int(self._offsets[-1]) != int(self._values.shape[0])
            or int(self._values.shape[0]) != int(self.summary["tokens"])
        ):
            raise RuntimeError("Language-cache tensor shapes changed.")
        if any(
            position < 0 or position >= len(self.instructions)
            for position in self._episode_instruction.values()
        ):
            raise RuntimeError("Language-cache episode mapping is out of range.")

    @property
    def hidden_size(self) -> int:
        return int(self.contract["encoder"]["hidden_size"])

    @property
    def parent_episode_ids(self) -> tuple[str, ...]:
        return tuple(self._episode_instruction)

    def tokens_for_parent(self, parent_episode_id: str) -> torch.Tensor:
        try:
            position = self._episode_instruction[parent_episode_id]
        except KeyError as exc:
            raise KeyError(
                f"No frozen language tokens for `{parent_episode_id}`."
            ) from exc
        start = int(self._offsets[position])
        stop = int(self._offsets[position + 1])
        return self._values[start:stop]


class LanguageConditionedJointWindowCache:
    """Attach a frozen language sidecar without rewriting joint episode shards."""

    def __init__(
        self,
        joint_cache: JointWindowCache,
        language_cache: FrozenLanguageCache,
    ):
        expected = str(joint_cache.contract["contract_hash"])
        actual = str(language_cache.contract["joint_cache"]["contract_hash"])
        if actual != expected:
            raise RuntimeError("Language and joint cache contracts differ.")
        missing = sorted(
            set(joint_cache.parent_episode_ids)
            - set(language_cache.parent_episode_ids)
        )
        if missing:
            raise RuntimeError(
                f"Language cache misses parent episodes: {missing[:8]}."
            )
        self.joint_cache = joint_cache
        self.language_cache = language_cache
        self.cache_dir = joint_cache.cache_dir
        self.contract = joint_cache.contract
        self.summary = joint_cache.summary
        self.windows = joint_cache.windows
        self.window_shards = joint_cache.window_shards

    def __len__(self) -> int:
        return len(self.joint_cache)

    def __getitem__(self, index: int) -> JointWindowSample:
        sample = self.joint_cache[index]
        tokens = self.language_cache.tokens_for_parent(
            sample.record.parent_episode_id
        )
        return replace(
            sample,
            language_tokens=tokens,
            language_valid=torch.ones(tokens.shape[0], dtype=torch.bool),
        )

    def split_indices(self, split: str) -> tuple[int, ...]:
        return self.joint_cache.split_indices(split)

    def permutation_rows(self):
        return self.joint_cache.permutation_rows()

    def action_chunk(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.joint_cache.action_chunk(index)
