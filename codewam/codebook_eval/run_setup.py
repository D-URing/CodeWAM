from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf

from .droid_pooled_export import DROID_POOLED_EXPORT_SCHEMA
from .manifest import EpisodeManifest
from .shards import file_sha256


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing finalized export file `{path}`.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON mapping in `{path}`.")
    return payload


def _resolve_existing(path: str | Path, relative_to: Path) -> Path:
    value = Path(path)
    candidates = (
        [value]
        if value.is_absolute()
        else [
            Path.cwd() / value,
            relative_to / value,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing finalized export dependency `{value}`.")


def _atomic_write_yaml(path: Path, payload: dict[str, Any]) -> None:
    text = OmegaConf.to_yaml(
        OmegaConf.create(payload),
        resolve=True,
        sort_keys=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_droid_streaming_run(
    pooled_export_dir: str | Path,
    output_dir: str | Path,
    *,
    config_dir: str | Path | None = None,
    pool: int = 4,
    k: int = 16,
    levels: int = 3,
    device: str = "auto",
    seed: int = 7,
    camera_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    if int(pool) not in {1, 2, 4}:
        raise ValueError(f"`pool` must be one of 1, 2, 4; got {pool}.")
    if int(k) <= 1 or int(levels) <= 0:
        raise ValueError("`k` must exceed one and `levels` must be positive.")

    export_dir = Path(pooled_export_dir).resolve()
    output_dir = Path(output_dir).resolve()
    config_dir = (
        Path(config_dir).resolve()
        if config_dir is not None
        else output_dir / "configs"
    )
    contract = _read_json(export_dir / "contract.json")
    summary = _read_json(export_dir / "export_summary.json")

    if contract.get("schema") != DROID_POOLED_EXPORT_SCHEMA:
        raise ValueError("The input directory is not a canonical DROID pooled export.")
    contract_payload = dict(contract)
    observed_contract_hash = str(contract_payload.pop("contract_hash", ""))
    if (
        not observed_contract_hash
        or _canonical_hash(contract_payload) != observed_contract_hash
    ):
        raise RuntimeError("The DROID pooled export contract hash is invalid.")
    if summary.get("schema") != DROID_POOLED_EXPORT_SCHEMA:
        raise ValueError("The DROID pooled export summary schema is invalid.")
    if summary.get("contract_hash") != observed_contract_hash:
        raise RuntimeError("The DROID pooled export summary uses another contract.")

    pooled_manifest_info = summary.get("pooled_manifest")
    if not isinstance(pooled_manifest_info, dict):
        raise ValueError("The export summary has no pooled manifest metadata.")
    manifest_path = _resolve_existing(
        str(pooled_manifest_info.get("path", "")),
        export_dir,
    )
    manifest = EpisodeManifest.read_jsonl(manifest_path)
    manifest.assert_group_isolation("scene")
    if manifest.fingerprint() != pooled_manifest_info.get("fingerprint"):
        raise RuntimeError("The finalized pooled manifest fingerprint is invalid.")
    datasets = sorted({record.dataset for record in manifest})
    if len(datasets) != 1:
        raise ValueError(
            "One streaming run must contain exactly one dataset, "
            f"found {datasets}."
        )

    file_rows = summary.get("files")
    if not isinstance(file_rows, list) or not file_rows:
        raise ValueError("The export summary has no pooled shard rows.")
    if int(summary.get("pooled_shards", -1)) != len(file_rows):
        raise RuntimeError("The pooled shard count differs from the export summary.")

    shard_rows: list[tuple[Path, str]] = []
    for row in file_rows:
        if not isinstance(row, dict):
            raise ValueError("Malformed pooled shard row in the export summary.")
        path = _resolve_existing(str(row.get("path", "")), export_dir)
        expected_bytes = int(row.get("bytes", -1))
        expected_sha256 = str(row.get("sha256", ""))
        if path.stat().st_size != expected_bytes:
            raise RuntimeError(f"Pooled shard size mismatch for `{path}`.")
        observed_sha256 = file_sha256(path)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(f"Pooled shard SHA-256 mismatch for `{path}`.")
        shard_rows.append((path, observed_sha256))
    shard_rows.sort(key=lambda value: str(value[0]))
    shard_paths = [path for path, _ in shard_rows]
    if len(shard_paths) != len(set(shard_paths)):
        raise RuntimeError("The export summary contains duplicate pooled shards.")
    pooled_parents = {path.parent for path in shard_paths}
    if len(pooled_parents) != 1:
        raise ValueError("Pooled shards must share one directory.")
    shard_pattern = str(next(iter(pooled_parents)) / "*.pt")

    dataset_revisions = contract.get("dataset_revisions")
    if not isinstance(dataset_revisions, list) or not dataset_revisions:
        raise ValueError("The DROID export contract has no dataset revisions.")
    export_cameras = contract.get("cameras")
    if not isinstance(export_cameras, list) or not export_cameras:
        raise ValueError("The DROID export contract has no camera order.")
    available_cameras = tuple(str(value) for value in export_cameras)
    selected_cameras = (
        available_cameras
        if camera_ids is None
        else tuple(str(value) for value in camera_ids)
    )
    if not selected_cameras or len(set(selected_cameras)) != len(selected_cameras):
        raise ValueError("Selected descriptor cameras must be nonempty and unique.")
    unknown_cameras = [
        value for value in selected_cameras if value not in available_cameras
    ]
    if unknown_cameras:
        raise ValueError(
            f"Selected cameras are absent from the pooled export: {unknown_cameras}."
        )
    metadata = {
        "dataset": datasets[0],
        "dataset_revision": ",".join(str(value) for value in dataset_revisions),
        "wan_model_id": str(contract.get("vae_model_id", "")),
        "wan_revision": str(contract.get("vae_sha256", "")),
        "preprocess_revision": str(contract.get("preprocess_revision", "")),
        "source_checksums": [checksum for _, checksum in shard_rows],
    }
    empty_metadata = [
        key for key, value in metadata.items()
        if key != "source_checksums" and not value
    ]
    if empty_metadata:
        raise ValueError(f"Empty run metadata fields: {empty_metadata}.")

    train_config = {
        "output_dir": str(output_dir),
        "input": {
            "pooled_shards": [shard_pattern],
            "manifest": str(manifest_path),
            "split": "train",
            "group_by": "scene",
        },
        "metadata": metadata,
        "descriptor": {
            "strides": [2, 3, 5],
            "pool": int(pool),
            "max_gap_factor": 1.5,
            "camera_ids": list(selected_cameras),
        },
        "training": {
            "device": str(device),
            "cpu_threads": 4,
            "batch_size": 8192,
            "k": int(k),
            "levels": int(levels),
            "max_iters": 50,
            "tol": 1.0e-3,
            "patience": 2,
            "seed": int(seed),
            "reservoir_size": 100000,
            "initialization_chunk_size": 8192,
            "center_block_size": 1024,
            "resume": True,
        },
    }
    evaluation_config = {
        "output_dir": str(output_dir / "heldout"),
        "input": {
            "pooled_shards": [shard_pattern],
            "manifest": str(manifest_path),
            "group_by": "scene",
        },
        "metadata": {"dataset": datasets[0]},
        "artifacts": {
            family: str(output_dir / family / "codebook.pt")
            for family in ("Q2", "Q3", "Q5")
        },
        "evaluation": {
            "splits": ["val", "test"],
            "device": str(device),
            "cpu_threads": 4,
            "batch_size": 8192,
            "center_block_size": 1024,
            "representatives_per_code": 3,
            "resume": True,
        },
    }

    train_path = config_dir / f"train_g{pool}_k{k}_l{levels}.yaml"
    evaluation_path = config_dir / f"evaluate_g{pool}_k{k}_l{levels}.yaml"
    _atomic_write_yaml(train_path, train_config)
    _atomic_write_yaml(evaluation_path, evaluation_config)
    return {
        "train_config": str(train_path),
        "evaluation_config": str(evaluation_path),
        "output_dir": str(output_dir),
        "dataset": datasets[0],
        "manifest_fingerprint": manifest.fingerprint(),
        "pooled_shards": len(shard_rows),
        "pool": int(pool),
        "k": int(k),
        "levels": int(levels),
        "camera_ids": list(selected_cameras),
    }
