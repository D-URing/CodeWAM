#!/usr/bin/env python
"""CodeWAM state-codebook probe.

Validates the pre-integration assumptions for an RQ state codebook on frozen
Wan-VAE latents:

P1: visual latents are cleanly discretizable.
P2: current visual state predicts next-frame codes better than copying current codes.
P4: visual state contains information beyond proprio-only prediction.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Subset

from fastwam.utils.config_resolvers import register_default_resolvers

from codewam.codebook import StateCodebook


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run P1/P2/P4 state-codebook probes on Wan-VAE latents.")
    parser.add_argument("--config-dir", default=os.environ.get("CODEWAM_CONFIG_DIR", str(root / "configs")))
    parser.add_argument("--config-name", default=os.environ.get("CODEWAM_CONFIG_NAME", "train"))
    parser.add_argument("--task", default=os.environ.get("CODEWAM_PROBE_TASK", "libero_codewam_2cam224"))
    parser.add_argument("--model-root", default=os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", str(root / "checkpoints")))
    parser.add_argument("--output", default=os.environ.get("CODEWAM_PROBE_OUTPUT", ""))
    parser.add_argument("--n", type=int, default=int(os.environ.get("N", "800")))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "60")))
    parser.add_argument("--dyn-epochs", type=int, default=int(os.environ.get("DYN_EPOCHS", "100")))
    parser.add_argument("--codebook-size", type=int, default=int(os.environ.get("K", "64")))
    parser.add_argument("--dim", type=int, default=int(os.environ.get("DIM", "128")))
    parser.add_argument("--pool", type=int, default=int(os.environ.get("POOL", "2")))
    parser.add_argument("--levels", type=int, default=int(os.environ.get("LEVELS", "3")))
    parser.add_argument("--recon-grid", type=int, default=int(os.environ.get("RECON_GRID", "4")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")))
    parser.add_argument("--train-batch-size", type=int, default=int(os.environ.get("TRAIN_BATCH_SIZE", "256")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "8")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    parser.add_argument("--split-seed", type=int, default=int(os.environ.get("SPLIT_SEED", "1")))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--vae-dtype", default=os.environ.get("VAE_DTYPE", "bfloat16"), choices=["float32", "float16", "bfloat16"])
    return parser.parse_args()


def parse_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def resolve_vae_path(model_root: Path) -> Path:
    candidates = [
        model_root / "Wan-AI" / "Wan2.2-TI2V-5B" / "Wan2.2_VAE.pth",
        model_root / "Wan2.2-TI2V-5B" / "Wan2.2_VAE.pth",
        model_root / "DiffSynth-Studio" / "Wan-Series-Converted-Safetensors" / "Wan2.2_VAE.safetensors",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find Wan VAE. Checked: {[str(p) for p in candidates]}")


def load_vae(args: argparse.Namespace):
    from fastwam.models.wan22.helpers.loader import _load_registered_model

    path = resolve_vae_path(Path(args.model_root))
    print(f"loading VAE from {path}", flush=True)
    vae = _load_registered_model(
        str(path),
        "wan_video_vae",
        torch_dtype=parse_dtype(args.vae_dtype),
        device=args.device,
    )
    vae.eval()
    return vae


@torch.no_grad()
def collect_latents(args: argparse.Namespace, vae):
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(Path(args.config_dir).resolve()), version_base="1.3"):
        cfg = compose(
            config_name=args.config_name,
            overrides=[f"task={args.task}", "num_workers=0", "eval_every=0"],
        )
    ds = instantiate(cfg.data.train)
    total = len(ds)
    idxs = np.random.RandomState(args.seed).choice(total, size=min(args.n, total), replace=False).tolist()
    print(f"dataset={total} sampled={len(idxs)} device={args.device}", flush=True)
    loader = DataLoader(Subset(ds, idxs), batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    latents, proprio = [], []
    seen = 0
    vae_dtype = parse_dtype(args.vae_dtype)
    for batch in loader:
        video = batch["video"].to(args.device, dtype=vae_dtype)
        z = vae.encode(video, device=args.device, tiled=False)
        latents.append(z.float().cpu())
        proprio.append(batch["proprio"][:, 0, :].float())
        seen += video.shape[0]
        if seen % max(args.batch_size * 20, args.batch_size) == 0:
            print(f"  encoded {seen}/{len(idxs)}", flush=True)

    latents = torch.cat(latents)
    proprio = torch.cat(proprio)
    print("latent", tuple(latents.shape), "proprio", tuple(proprio.shape), flush=True)
    return latents, proprio


def train_code_predictor(head, opt, x, targets, epochs, batch_size, levels, codebook_size, device):
    for _ in range(epochs):
        perm = torch.randperm(x.shape[0], device=device)
        for i in range(0, x.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            out = head(x[idx]).view(-1, levels, codebook_size)
            ce = sum(F.cross_entropy(out[:, lvl], targets[idx, lvl]) for lvl in range(levels)) / levels
            opt.zero_grad()
            ce.backward()
            opt.step()


@torch.no_grad()
def eval_top1(head, x, targets, levels, codebook_size):
    out = head(x).view(-1, levels, codebook_size)
    return float(np.mean([
        (out[:, lvl].argmax(1) == targets[:, lvl]).float().mean().item()
        for lvl in range(levels)
    ]))


def run_probe(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vae = load_vae(args)
    latents, proprio = collect_latents(args, vae)
    nwin, channels, latent_t, height, width = latents.shape
    if channels != 48:
        raise ValueError(f"Expected Wan-VAE latent channel C=48, got C={channels}")

    perm = np.random.RandomState(args.split_seed).permutation(nwin)
    train_idx = torch.tensor(perm[:int(nwin * 0.8)])
    test_idx = torch.tensor(perm[int(nwin * 0.8):])

    sc = StateCodebook(
        in_ch=channels,
        dim=args.dim,
        n_levels=args.levels,
        codebook_size=args.codebook_size,
        action_dim=None,
        pool=args.pool,
    ).to(args.device).train()
    recon_head = nn.Linear(args.dim, channels * args.recon_grid * args.recon_grid).to(args.device)
    opt = torch.optim.Adam(list(sc.parameters()) + list(recon_head.parameters()), lr=2e-3)

    def frames_of(win):
        x = latents[win].permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
        return x.to(args.device)

    def recon_target(x):
        return F.adaptive_avg_pool2d(x, args.recon_grid).reshape(x.shape[0], -1)

    def pairs_of(win):
        cur, nxt, pro = [], [], []
        for t in range(latent_t - 1):
            cur.append(latents[win, :, t])
            nxt.append(latents[win, :, t + 1])
            pro.append(proprio[win])
        return torch.cat(cur).to(args.device), torch.cat(nxt).to(args.device), torch.cat(pro).to(args.device)

    x_train = frames_of(train_idx)
    y_train = recon_target(x_train)

    for epoch in range(args.epochs):
        perm_train = torch.randperm(x_train.shape[0], device=args.device)
        tr_rec = tr_vq = 0.0
        for i in range(0, x_train.shape[0], args.train_batch_size):
            idx = perm_train[i:i + args.train_batch_size]
            out = sc.encode(x_train[idx], update=True)
            rec = recon_head(out["z_q"])
            rec_loss = F.mse_loss(rec, y_train[idx])
            loss = rec_loss + out["vq_loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_rec += float(rec_loss.detach())
            tr_vq += float(out["vq_loss"].detach())
        if (epoch + 1) % max(args.epochs // 4, 1) == 0:
            usage = [round(float(x), 3) for x in sc.encode(x_train[:512], update=False)["usage"]]
            print(f"[repr] ep{epoch + 1:3d} rec={tr_rec:.3f} vq={tr_vq:.3f} usage={usage}", flush=True)

    sc.eval()
    for p in sc.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        x_test = frames_of(test_idx)
        out_test = sc.encode(x_test, update=False)
        usage = [float(x) for x in out_test["usage"]]
        perp = [float(x) for x in out_test["perplexity"]]
        z, zq = out_test["z"], out_test["z_q"]
        quant_err = float((z - zq).pow(2).mean() / (z.pow(2).mean() + 1e-8))
        rec = recon_head(zq)
        y_test = recon_target(x_test)
        ss_res = (rec - y_test).pow(2).sum()
        ss_tot = (y_test - y_test.mean(0)).pow(2).sum()
        recon_r2 = float(1 - ss_res / ss_tot)

        cur_train, nxt_train, pro_train = pairs_of(train_idx)
        cur_test, nxt_test, pro_test = pairs_of(test_idx)
        z_cur_train = sc.encode(cur_train, update=False)["z"]
        z_cur_test = sc.encode(cur_test, update=False)["z"]
        code_cur_test = sc.encode(cur_test, update=False)["codes"]
        code_nxt_train = sc.encode(nxt_train, update=False)["codes"]
        code_nxt_test = sc.encode(nxt_test, update=False)["codes"]

    copy_acc = float((code_cur_test == code_nxt_test).float().mean())
    proprio_dim = int(pro_train.shape[1])

    vis_head = nn.Sequential(nn.Linear(args.dim, 512), nn.SiLU(), nn.Linear(512, args.codebook_size * args.levels)).to(args.device)
    train_code_predictor(
        vis_head,
        torch.optim.Adam(vis_head.parameters(), lr=2e-3),
        z_cur_train,
        code_nxt_train,
        args.dyn_epochs,
        args.train_batch_size,
        args.levels,
        args.codebook_size,
        args.device,
    )
    vis_acc = eval_top1(vis_head, z_cur_test, code_nxt_test, args.levels, args.codebook_size)

    pro_head = nn.Sequential(nn.Linear(proprio_dim, 512), nn.SiLU(), nn.Linear(512, args.codebook_size * args.levels)).to(args.device)
    train_code_predictor(
        pro_head,
        torch.optim.Adam(pro_head.parameters(), lr=2e-3),
        pro_train,
        code_nxt_train,
        args.dyn_epochs,
        args.train_batch_size,
        args.levels,
        args.codebook_size,
        args.device,
    )
    pro_acc = eval_top1(pro_head, pro_test, code_nxt_test, args.levels, args.codebook_size)

    both_train = torch.cat([z_cur_train, pro_train], dim=1)
    both_test = torch.cat([z_cur_test, pro_test], dim=1)
    both_head = nn.Sequential(nn.Linear(args.dim + proprio_dim, 512), nn.SiLU(), nn.Linear(512, args.codebook_size * args.levels)).to(args.device)
    train_code_predictor(
        both_head,
        torch.optim.Adam(both_head.parameters(), lr=2e-3),
        both_train,
        code_nxt_train,
        args.dyn_epochs,
        args.train_batch_size,
        args.levels,
        args.codebook_size,
        args.device,
    )
    both_acc = eval_top1(both_head, both_test, code_nxt_test, args.levels, args.codebook_size)

    result = {
        "task": args.task,
        "n_windows": int(nwin),
        "latent_shape": list(latents.shape),
        "proprio_dim": proprio_dim,
        "pool": args.pool,
        "codebook_size": args.codebook_size,
        "dim": args.dim,
        "levels": args.levels,
        "p1_usage": usage,
        "p1_perplexity": perp,
        "p1_quant_relative_error": quant_err,
        "p1_recon_r2": recon_r2,
        "p2_visual_next_code_top1": vis_acc,
        "p2_copy_baseline_top1": copy_acc,
        "p2_visual_minus_copy": vis_acc - copy_acc,
        "p4_proprio_only_top1": pro_acc,
        "p4_visual_plus_proprio_top1": both_acc,
        "p4_visual_minus_proprio": vis_acc - pro_acc,
        "p4_both_minus_proprio": both_acc - pro_acc,
    }
    return result


def print_report(result: dict) -> None:
    print("\n" + "=" * 70)
    print(f"CodeWAM state-codebook probe (task={result['task']}, pool={result['pool']}, K={result['codebook_size']})")
    print("=" * 70)
    print(f"[P1] usage = {[round(u, 3) for u in result['p1_usage']]}  "
          f"perplexity = {[round(p, 1) for p in result['p1_perplexity']]}")
    print(f"     quant relative error = {result['p1_quant_relative_error']:.4f}  "
          f"recon R2 = {result['p1_recon_r2']:.3f}")
    print(f"[P2] next-code top1: visual={result['p2_visual_next_code_top1']:.3f}  "
          f"copy={result['p2_copy_baseline_top1']:.3f}  "
          f"delta={result['p2_visual_minus_copy']:+.3f}")
    print(f"[P4] next-code top1: proprio={result['p4_proprio_only_top1']:.3f}  "
          f"visual={result['p2_visual_next_code_top1']:.3f}  "
          f"visual+proprio={result['p4_visual_plus_proprio_top1']:.3f}")
    print(f"     visual-proprio={result['p4_visual_minus_proprio']:+.3f}  "
          f"both-proprio={result['p4_both_minus_proprio']:+.3f}")
    print("-" * 70)
    print("Read: P1 high usage/perplexity => discretizable; P2 visual >> copy => predictable dynamics;")
    print("      P4 visual(+proprio) > proprio => visual codes carry information beyond proprio shortcuts.")


def main() -> None:
    args = parse_args()
    result = run_probe(args)
    print_report(result)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
