from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from .clustering import assign_codes, kmeans, pack_codebook, train_residual_quantizer
from .descriptors import build_temporal_descriptors
from .io import ensure_dir, load_latent_shards, save_json, write_summary_tsv
from .metrics import action_relevance_metrics, kmeans_metrics, rq_metrics


def _cfg(path: str | Path) -> DictConfig:
    return OmegaConf.load(path)


def _select_vectors(x: torch.Tensor, max_vectors: int | None, seed: int) -> torch.Tensor:
    if max_vectors is None or x.shape[0] <= int(max_vectors):
        return x
    generator = torch.Generator().manual_seed(int(seed))
    idx = torch.randperm(x.shape[0], generator=generator)[: int(max_vectors)]
    return x[idx]


def _train_one_dataset(cfg: DictConfig, dataset_cfg: DictConfig, output_root: Path) -> list[dict[str, Any]]:
    dataset_name = str(dataset_cfg.get("name", "dataset"))
    latent_paths = dataset_cfg.get("latent_paths")
    if not latent_paths:
        raise ValueError(f"Dataset `{dataset_name}` has no `latent_paths`.")

    training = cfg.get("training", {})
    descriptors_cfg = cfg.get("descriptors", {})
    max_windows = dataset_cfg.get("max_windows", cfg.get("max_windows", None))
    payload = load_latent_shards(latent_paths, max_windows=None if max_windows is None else int(max_windows))
    latents = payload["latents"]
    actions = payload.get("action")

    batches = build_temporal_descriptors(
        latents,
        strides=descriptors_cfg.get("strides", [2, 3, 5]),
        pool=int(descriptors_cfg.get("pool", 2)),
        include_current=bool(descriptors_cfg.get("include_current", True)),
        include_future=bool(descriptors_cfg.get("include_future", True)),
        include_delta=bool(descriptors_cfg.get("include_delta", True)),
        normalize=bool(descriptors_cfg.get("normalize", True)),
    )

    device = torch.device(str(training.get("device", "cpu")))
    seed = int(training.get("seed", 0))
    iters = int(training.get("kmeans_iters", 50))
    chunk_size = int(training.get("chunk_size", 8192))
    max_vectors = training.get("max_vectors_per_descriptor", None)
    variants = list(training.get("variants", []))
    if not variants:
        raise ValueError("`training.variants` must contain at least one candidate.")

    rows: list[dict[str, Any]] = []
    for batch in batches:
        train_x = _select_vectors(
            batch.vectors,
            None if max_vectors is None else int(max_vectors),
            seed=seed + batch.stride,
        ).to(device=device)
        eval_batch = batch
        for variant in variants:
            method = str(variant.get("method", "rq")).lower()
            k = int(variant.get("k", 256))
            levels = int(variant.get("levels", 3))
            run_name = f"{dataset_name}_{batch.name}_{method}_k{k}_l{levels}"
            run_dir = ensure_dir(output_root / dataset_name / run_name)

            if method == "kmeans":
                km = kmeans(train_x, k=k, iters=iters, seed=seed, chunk_size=chunk_size)
                centers = km.centers.to(device=eval_batch.vectors.device)
                codes, _ = assign_codes(eval_batch.vectors.float(), centers)
                metrics = kmeans_metrics(eval_batch, km.centers, codes.cpu(), k=k)
                metrics.update(action_relevance_metrics(eval_batch, codes.cpu(), actions, latent_t=latents.shape[2], k=k))
                torch.save(pack_codebook(km, {"dataset": dataset_name, "descriptor": batch.name}), run_dir / "codebook.pt")
            elif method == "rq":
                rq = train_residual_quantizer(
                    train_x,
                    k=k,
                    levels=levels,
                    iters=iters,
                    seed=seed,
                    chunk_size=chunk_size,
                )
                centers = [center.to(device=eval_batch.vectors.device) for center in rq.centers]
                residual = eval_batch.vectors.float()
                quantized = torch.zeros_like(residual)
                codes = []
                for center in centers:
                    code, _ = assign_codes(residual, center, chunk_size=chunk_size)
                    q = center[code]
                    quantized = quantized + q
                    residual = residual - q
                    codes.append(code.cpu())
                eval_rq = rq
                eval_rq.codes = torch.stack(codes, dim=1)
                eval_rq.quantized = quantized.cpu()
                metrics = rq_metrics(eval_batch, eval_rq, k=k)
                metrics.update(
                    action_relevance_metrics(eval_batch, eval_rq.codes, actions, latent_t=latents.shape[2], k=k)
                )
                torch.save(pack_codebook(rq, {"dataset": dataset_name, "descriptor": batch.name}), run_dir / "codebook.pt")
            else:
                raise ValueError(f"Unsupported clustering method: {method}")

            row = {
                "run": run_name,
                "dataset": dataset_name,
                "method": method,
                **metrics,
            }
            save_json(run_dir / "metrics.json", row)
            rows.append(row)
    return rows


def train_from_config(config_path: str | Path) -> list[dict[str, Any]]:
    cfg = _cfg(config_path)
    output_root = ensure_dir(cfg.get("output_dir", "runs/codebook_eval"))
    datasets = list(cfg.get("datasets", []))
    if not datasets:
        raise ValueError("Config must define at least one dataset under `datasets`.")

    rows: list[dict[str, Any]] = []
    for dataset_cfg in datasets:
        if not bool(dataset_cfg.get("enabled", True)):
            continue
        rows.extend(_train_one_dataset(cfg, dataset_cfg, output_root))
    if not rows:
        raise ValueError("No codebook runs were executed. Check `datasets[].enabled` and latent paths.")
    write_summary_tsv(output_root / "summary.tsv", rows)
    save_json(output_root / "summary.json", rows)
    return rows


def make_synthetic_cache(output_dir: str | Path, n: int = 64, c: int = 48, t: int = 9, h: int = 7, w: int = 14) -> Path:
    output_dir = ensure_dir(output_dir)
    generator = torch.Generator().manual_seed(123)
    phase = torch.randint(0, 5, (n,), generator=generator)
    centers = torch.randn(5, c, 1, 1, 1, generator=generator)
    trend = torch.linspace(0, 1, t).view(1, 1, t, 1, 1)
    latents = centers[phase] + 0.35 * trend * torch.randn(n, c, 1, 1, 1, generator=generator)
    latents = latents.expand(-1, -1, -1, h, w).clone()
    latents = latents + 0.05 * torch.randn(n, c, t, h, w, generator=generator)
    action_centers = torch.randn(5, 7, generator=generator)
    actions = action_centers[phase].unsqueeze(1).expand(n, 32, 7).clone()
    actions = actions + 0.05 * torch.randn(n, 32, 7, generator=generator)
    path = output_dir / "synthetic_latents.pt"
    torch.save({"latents": latents, "action": actions, "meta": {"kind": "synthetic-smoke"}}, path)
    return path


def synthetic_smoke(output_dir: str | Path) -> list[dict[str, Any]]:
    output_dir = ensure_dir(output_dir)
    latent_path = make_synthetic_cache(output_dir / "latents")
    cfg = OmegaConf.create(
        {
            "output_dir": str(output_dir / "runs"),
            "datasets": [{"name": "synthetic", "latent_paths": [str(latent_path)]}],
            "descriptors": {
                "strides": [2, 3, 5],
                "pool": 2,
                "include_current": True,
                "include_future": True,
                "include_delta": True,
                "normalize": True,
            },
            "training": {
                "device": "cpu",
                "seed": 7,
                "kmeans_iters": 8,
                "chunk_size": 2048,
                "max_vectors_per_descriptor": 512,
                "variants": [
                    {"method": "kmeans", "k": 8, "levels": 1},
                    {"method": "rq", "k": 8, "levels": 3},
                ],
            },
        }
    )
    config_path = output_dir / "synthetic_config.yaml"
    OmegaConf.save(cfg, config_path)
    return train_from_config(config_path)


def _load_vae(device: str, torch_dtype: torch.dtype):
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs

    _, _, vae_config, _ = _resolve_configs(
        model_id="Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id="Wan-AI/Wan2.1-T2V-1.3B",
        redirect_common_files=False,
    )
    vae_config.download_if_necessary()
    return _load_registered_model(vae_config.path, "wan_video_vae", torch_dtype=torch_dtype, device=device)


@torch.no_grad()
def export_latents(config_path: str | Path) -> None:
    cfg = _cfg(config_path)
    export = cfg.get("export", {})
    if not export:
        raise ValueError("Config has no `export` section.")

    from hydra.utils import instantiate

    device = str(export.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype_name = str(export.get("dtype", "bfloat16"))
    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    vae = _load_vae(device=device, torch_dtype=torch_dtype)
    dataset = instantiate(export["dataset"])

    out_dir = ensure_dir(export.get("output_dir", "runs/codebook_eval/latents"))
    batch_size = int(export.get("batch_size", 1))
    max_windows = export.get("max_windows", None)
    max_windows = len(dataset) if max_windows is None else min(int(max_windows), len(dataset))
    shard_size = int(export.get("shard_size", 64))
    tiled = bool(export.get("tiled", False))
    tile_size = tuple(export.get("tile_size", [30, 52]))
    tile_stride = tuple(export.get("tile_stride", [15, 26]))

    shard_latents = []
    shard_actions = []
    shard_proprios = []
    shard_meta = []
    shard_idx = 0

    def flush() -> None:
        nonlocal shard_idx, shard_latents, shard_actions, shard_proprios, shard_meta
        if not shard_latents:
            return
        payload: dict[str, Any] = {"latents": torch.cat(shard_latents, dim=0).cpu(), "meta": shard_meta}
        if shard_actions:
            payload["action"] = torch.stack(shard_actions, dim=0).cpu()
        if shard_proprios:
            payload["proprio"] = torch.stack(shard_proprios, dim=0).cpu()
        torch.save(payload, out_dir / f"shard_{shard_idx:05d}.pt")
        shard_idx += 1
        shard_latents = []
        shard_actions = []
        shard_proprios = []
        shard_meta = []

    for start in range(0, max_windows, batch_size):
        samples = [dataset[i] for i in range(start, min(start + batch_size, max_windows))]
        videos = torch.stack([sample["video"] for sample in samples], dim=0).to(device=device, dtype=torch_dtype)
        z = vae.encode(videos, device=device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = torch.stack(z, dim=0)
        shard_latents.append(z.detach().float().cpu())
        for offset, sample in enumerate(samples):
            if "action" in sample:
                shard_actions.append(sample["action"].float())
            if "proprio" in sample:
                shard_proprios.append(sample["proprio"].float())
            shard_meta.append({"index": start + offset, "prompt": sample.get("prompt", sample.get("task", ""))})
        if sum(item.shape[0] for item in shard_latents) >= shard_size:
            flush()
    flush()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Offline CodeWAM codebook evaluation.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train all codebook candidates from latent cache.")
    p_train.add_argument("--config", required=True)

    p_export = sub.add_parser("export-latents", help="Export Wan-VAE latent shards from a Hydra dataset.")
    p_export.add_argument("--config", required=True)

    p_all = sub.add_parser("all", help="Export latents, then train codebook candidates.")
    p_all.add_argument("--config", required=True)
    p_all.add_argument("--skip-export", action="store_true")

    p_smoke = sub.add_parser("synthetic-smoke", help="Run a tiny synthetic end-to-end smoke test.")
    p_smoke.add_argument("--output-dir", default="runs/codebook_eval_synthetic")

    args = parser.parse_args(argv)
    if args.cmd == "train":
        train_from_config(args.config)
    elif args.cmd == "export-latents":
        export_latents(args.config)
    elif args.cmd == "all":
        if not args.skip_export:
            export_latents(args.config)
        train_from_config(args.config)
    elif args.cmd == "synthetic-smoke":
        synthetic_smoke(args.output_dir)


if __name__ == "__main__":
    main()
