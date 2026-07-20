from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Iterable

import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def expand_paths(patterns: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        text = str(pattern)
        matched = [Path(p) for p in glob.glob(text)]
        if matched:
            paths.extend(matched)
        else:
            paths.append(Path(text))
    unique = sorted({p.resolve() for p in paths})
    missing = [p for p in unique if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing latent files: {[str(p) for p in missing[:8]]}")
    return unique


def load_latent_shards(paths: Iterable[str | Path], max_windows: int | None = None) -> dict[str, Any]:
    latents = []
    actions = []
    proprios = []
    metas = []
    total = 0

    for path in expand_paths(paths):
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, torch.Tensor):
            z = payload
            payload = {"latents": z}
        elif not isinstance(payload, dict) or "latents" not in payload:
            raise ValueError(f"Latent shard must be a Tensor or dict with `latents`: {path}")

        z = payload["latents"]
        if z.ndim == 4:
            z = z.unsqueeze(0)
        if z.ndim != 5:
            raise ValueError(f"`latents` must be [N,C,T,H,W] or [C,T,H,W], got {tuple(z.shape)} in {path}")

        if max_windows is not None:
            remaining = int(max_windows) - total
            if remaining <= 0:
                break
            z = z[:remaining]
        latents.append(z.float())
        total += int(z.shape[0])

        if "action" in payload:
            actions.append(payload["action"][: z.shape[0]].float())
        if "proprio" in payload:
            proprios.append(payload["proprio"][: z.shape[0]].float())
        if "meta" in payload:
            meta = payload["meta"]
            if isinstance(meta, list):
                metas.extend(meta[: z.shape[0]])
            else:
                metas.append(meta)

    if not latents:
        raise ValueError("No latent shards were loaded.")

    out: dict[str, Any] = {"latents": torch.cat(latents, dim=0), "meta": metas}
    if actions:
        out["action"] = torch.cat(actions, dim=0)
    if proprios:
        out["proprio"] = torch.cat(proprios, dim=0)
    return out


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_summary_tsv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    columns = [
        "run",
        "dataset",
        "device",
        "method",
        "descriptor",
        "stride",
        "k",
        "levels",
        "n_vectors",
        "dim",
        "relative_mse",
        "r2_like",
        "mean_cosine",
        "level1_usage",
        "level1_perplexity_frac",
        "level1_dead_frac",
        "temporal_same_next_frac",
        "temporal_change_next_frac",
    ]
    extra = sorted({key for row in rows for key in row if key not in columns})
    columns.extend(extra)
    lines = ["\t".join(columns)]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("\t".join(values))
    path.write_text("\n".join(lines) + "\n")
