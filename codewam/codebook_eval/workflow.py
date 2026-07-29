from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from codewam.data.droid_manifest import write_json_report

from .association import probe_frozen_codebook_associations
from .concentration import (
    GROUPING_DEFINITIONS,
    probe_frozen_codebook_concentration,
)
from .evaluation import evaluate_frozen_codebooks
from .pipeline import train_streaming_codebooks
from .shards import expand_shard_paths, file_sha256


CANDIDATE_WORKFLOW_SCHEMA = "codewam.codebook-candidate-workflow.v1"


def _plain_config(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(
        OmegaConf.load(path),
        resolve=True,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"Candidate config must be a mapping: `{path}`.")
    return payload


def _release_runtime_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _validate_configs(
    train_path: Path,
    evaluation_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    train = _plain_config(train_path)
    evaluation = _plain_config(evaluation_path)
    train_input = train.get("input", {})
    evaluation_input = evaluation.get("input", {})
    if not isinstance(train_input, dict) or not isinstance(
        evaluation_input,
        dict,
    ):
        raise ValueError("Candidate train/evaluation inputs must be mappings.")
    if str(train_input.get("manifest", "")) != str(
        evaluation_input.get("manifest", "")
    ):
        raise ValueError(
            "Candidate train and evaluation manifests must be identical."
        )
    train_shards = tuple(
        expand_shard_paths(train_input.get("pooled_shards", ()))
    )
    evaluation_shards = tuple(
        expand_shard_paths(evaluation_input.get("pooled_shards", ()))
    )
    if train_shards != evaluation_shards:
        raise ValueError(
            "Candidate train and evaluation pooled shards must be identical."
        )

    train_output = Path(str(train.get("output_dir", ""))).resolve()
    evaluation_output = Path(
        str(evaluation.get("output_dir", ""))
    ).resolve()
    if evaluation_output != train_output / "heldout":
        raise ValueError(
            "Candidate evaluation output must be TRAIN_OUTPUT/heldout."
        )
    descriptor = train.get("descriptor", {})
    artifacts = evaluation.get("artifacts", {})
    if not isinstance(descriptor, dict) or not isinstance(artifacts, dict):
        raise ValueError(
            "Candidate descriptor and artifact configs must be mappings."
        )
    strides = tuple(int(value) for value in descriptor.get("strides", ()))
    expected_artifacts = {
        f"Q{stride}": (train_output / f"Q{stride}" / "codebook.pt").resolve()
        for stride in strides
    }
    configured_artifacts = {
        str(family): Path(str(path)).resolve()
        for family, path in artifacts.items()
    }
    if not expected_artifacts or configured_artifacts != expected_artifacts:
        raise ValueError(
            "Candidate evaluation artifacts do not match train families."
        )
    train_dataset = str(train.get("metadata", {}).get("dataset", ""))
    evaluation_dataset = str(
        evaluation.get("metadata", {}).get("dataset", "")
    )
    if not train_dataset or train_dataset != evaluation_dataset:
        raise ValueError(
            "Candidate train/evaluation dataset metadata must match."
        )
    return train, evaluation, train_output


def run_streaming_codebook_candidate(
    train_config_path: str | Path,
    evaluation_config_path: str | Path,
    *,
    min_train_count: int = 8,
    groupings: tuple[str, ...] = ("scene", "institution", "task"),
) -> dict[str, Any]:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "The end-to-end candidate workflow is single-process. Use "
            "torchrun for training, then run held-out probes on rank 0."
        )
    if min_train_count <= 0:
        raise ValueError("Candidate min_train_count must be positive.")
    if (
        not groupings
        or len(groupings) != len(set(groupings))
        or any(value not in GROUPING_DEFINITIONS for value in groupings)
    ):
        raise ValueError(
            "Candidate groupings must be unique supported values."
        )
    train_path = Path(train_config_path)
    evaluation_path = Path(evaluation_config_path)
    _, evaluation, output_dir = _validate_configs(
        train_path,
        evaluation_path,
    )
    evaluation_settings = evaluation.get("evaluation", {})
    if not isinstance(evaluation_settings, dict):
        raise ValueError("Candidate evaluation settings must be a mapping.")
    input_settings = evaluation["input"]
    artifact_settings = evaluation["artifacts"]
    splits = tuple(
        str(value)
        for value in evaluation_settings.get("splits", ("val", "test"))
    )
    if (
        not splits
        or len(splits) != len(set(splits))
        or any(value not in {"val", "test"} for value in splits)
    ):
        raise ValueError(
            "Candidate splits must be unique val/test values."
        )
    device = str(evaluation_settings.get("device", "auto"))
    cpu_threads = int(evaluation_settings.get("cpu_threads", 4))
    batch_size = int(evaluation_settings.get("batch_size", 8192))
    center_block_size = int(
        evaluation_settings.get("center_block_size", 1024)
    )

    train_rows = train_streaming_codebooks(train_path)
    _release_runtime_memory()
    heldout = evaluate_frozen_codebooks(evaluation_path)
    _release_runtime_memory()
    association = probe_frozen_codebook_associations(
        manifest_path=input_settings["manifest"],
        pooled_shards=input_settings["pooled_shards"],
        artifacts=artifact_settings,
        output_dir=output_dir / "association",
        splits=splits,
        device=device,
        cpu_threads=cpu_threads,
        batch_size=batch_size,
        center_block_size=center_block_size,
        min_train_count=min_train_count,
        resume=True,
    )
    _release_runtime_memory()
    concentration = probe_frozen_codebook_concentration(
        manifest_path=input_settings["manifest"],
        pooled_shards=input_settings["pooled_shards"],
        artifacts=artifact_settings,
        output_dir=output_dir / "concentration",
        splits=splits,
        groupings=groupings,
        device=device,
        cpu_threads=cpu_threads,
        batch_size=batch_size,
        center_block_size=center_block_size,
        resume=True,
    )
    _release_runtime_memory()

    report_paths = {
        "train": output_dir / "train_summary.json",
        "heldout": output_dir / "heldout/evaluation_report.json",
        "association": output_dir / "association/association_report.json",
        "concentration": (
            output_dir / "concentration/concentration_report.json"
        ),
    }
    missing = [
        name for name, path in report_paths.items() if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            f"Candidate workflow completed without reports: {missing}."
        )
    summary = {
        "schema": CANDIDATE_WORKFLOW_SCHEMA,
        "train_config": {
            "path": str(train_path.resolve()),
            "sha256": file_sha256(train_path),
        },
        "evaluation_config": {
            "path": str(evaluation_path.resolve()),
            "sha256": file_sha256(evaluation_path),
        },
        "output_dir": str(output_dir),
        "families": sorted(str(value) for value in artifact_settings),
        "splits": list(splits),
        "min_train_count": int(min_train_count),
        "groupings": list(groupings),
        "reports": {
            name: {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for name, path in report_paths.items()
        },
        "row_counts": {
            "train": len(train_rows),
            "heldout": len(heldout["rows"]),
            "association": len(association["rows"]),
            "concentration": len(concentration["rows"]),
        },
        "implementation_sha256": file_sha256(Path(__file__)),
    }
    summary_path = output_dir / "candidate_workflow.json"
    if summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous != summary:
            raise RuntimeError(
                "Existing candidate workflow summary differs from the "
                "completed reports."
            )
        return previous
    write_json_report(summary_path, summary)
    return summary
