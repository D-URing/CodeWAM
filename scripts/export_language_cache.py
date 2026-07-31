#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from codewam.codebook_eval.manifest import EpisodeManifest
from codewam.codebook_eval.shards import file_sha256
from codewam.data import (
    FrozenLanguageCache,
    JointWindowCache,
    create_language_cache_contract,
    normalize_language_instruction,
    write_frozen_language_cache,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode one frozen token-level language sidecar for a verified "
            "JointWindowCache without rewriting Wan latent shards."
        )
    )
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    return parser.parse_args()


def _manifest_instructions(
    manifest: EpisodeManifest,
) -> dict[str, str]:
    result = {}
    for record in manifest:
        candidates = tuple(record.task_ids) + tuple(
            str(value) for value in record.metadata.get("task_texts", ())
        )
        instruction = next(
            (
                normalized
                for value in candidates
                if (normalized := normalize_language_instruction(value))
            ),
            "",
        )
        if not instruction:
            raise RuntimeError(
                f"Manifest episode `{record.episode_id}` has no task text."
            )
        if record.episode_id in result:
            raise RuntimeError(
                f"Duplicate manifest episode `{record.episode_id}`."
            )
        result[record.episode_id] = instruction
    return result


def _model_files(model_path: Path) -> list[dict[str, object]]:
    names = (
        "config.json",
        "model.safetensors",
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    )
    rows = []
    for name in names:
        path = model_path / name
        if path.is_file():
            rows.append(
                {
                    "name": name,
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    required = {"config.json", "model.safetensors", "spiece.model"}
    if not required.issubset({str(row["name"]) for row in rows}):
        raise FileNotFoundError("T5 model directory is incomplete.")
    return rows


def main() -> None:
    args = _parse_args()
    if args.max_tokens <= 0 or args.batch_size <= 0:
        raise SystemExit("Token and batch limits must be positive.")
    if not args.source_manifest.is_file() or not args.model_path.is_dir():
        raise FileNotFoundError("Language-cache manifest or model path is missing.")
    joint_cache = JointWindowCache(args.cache_dir)
    joint_contract = joint_cache.contract
    source_sha256 = file_sha256(args.source_manifest)
    if source_sha256 != joint_contract["source_manifest_sha256"]:
        raise RuntimeError("Language manifest differs from the joint-cache source.")
    manifest = EpisodeManifest.read_jsonl(args.source_manifest)
    if manifest.fingerprint() != joint_contract["source_manifest_fingerprint"]:
        raise RuntimeError("Language manifest fingerprint differs from joint cache.")
    parent_instructions = _manifest_instructions(manifest)
    missing = sorted(
        set(joint_cache.parent_episode_ids) - set(parent_instructions)
    )
    if missing:
        raise RuntimeError(f"Joint cache has unknown parent episodes: {missing[:8]}.")
    instructions = tuple(
        sorted({parent_instructions[value] for value in joint_cache.parent_episode_ids})
    )
    instruction_index = {value: index for index, value in enumerate(instructions)}
    episode_instruction = {
        parent: instruction_index[parent_instructions[parent]]
        for parent in joint_cache.parent_episode_ids
    }

    try:
        from transformers import AutoTokenizer, T5EncoderModel
    except ImportError as exc:
        raise RuntimeError(
            "Language export requires transformers and sentencepiece."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        use_fast=True,
    )
    model = T5EncoderModel.from_pretrained(
        args.model_path,
        local_files_only=True,
    ).eval()
    hidden_size = int(model.config.d_model)
    contract = create_language_cache_contract(
        joint_cache_contract_hash=str(joint_contract["contract_hash"]),
        joint_cache_summary_sha256=file_sha256(args.cache_dir / "summary.json"),
        source_manifest_fingerprint=manifest.fingerprint(),
        source_manifest_sha256=source_sha256,
        encoder_id=args.model_id,
        encoder_revision=args.model_revision,
        hidden_size=hidden_size,
        max_tokens=args.max_tokens,
        dtype=args.dtype,
        model_files=_model_files(args.model_path),
        implementation_sha256={
            "export_language_cache": file_sha256(Path(__file__)),
            "language_cache": file_sha256(
                Path(__file__).parents[1] / "codewam" / "data" / "language_cache.py"
            ),
        },
    )
    existing_contract = args.output_dir / "contract.json"
    existing_summary = args.output_dir / "summary.json"
    if existing_contract.is_file() and existing_summary.is_file():
        existing = json.loads(existing_contract.read_text(encoding="utf-8"))
        if existing != contract:
            raise RuntimeError("Existing language-cache contract differs.")
        cache = FrozenLanguageCache(
            args.output_dir,
            expected_joint_cache_contract_hash=str(joint_contract["contract_hash"]),
        )
        print(
            json.dumps(
                {**cache.summary, "status": "reused"},
                indent=2,
                sort_keys=True,
            )
        )
        return

    lengths = [
        len(tokenizer(value, add_special_tokens=True)["input_ids"])
        for value in instructions
    ]
    longest = max(lengths)
    if longest > args.max_tokens:
        raise RuntimeError(
            f"Language instruction needs {longest} tokens, above "
            f"--max-tokens {args.max_tokens}."
        )
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA language export requested unavailable CUDA.")
        torch.cuda.set_device(device)
    model.to(device)
    output_dtype = getattr(torch, args.dtype)
    sequences = []
    with torch.inference_mode():
        for start in range(0, len(instructions), args.batch_size):
            texts = instructions[start : start + args.batch_size]
            encoded = tokenizer(
                list(texts),
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=output_dtype,
                enabled=device.type == "cuda" and output_dtype != torch.float32,
            ):
                hidden = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state
            for row in range(hidden.shape[0]):
                length = int(attention_mask[row].sum())
                sequences.append(
                    hidden[row, :length].to(dtype=output_dtype).cpu().contiguous()
                )
    summary = write_frozen_language_cache(
        args.output_dir,
        contract=contract,
        instructions=instructions,
        episode_instruction=episode_instruction,
        token_sequences=sequences,
    )
    print(json.dumps({**summary, "status": "exported"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
