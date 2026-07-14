#!/usr/bin/env python
"""Lightweight native CodeWAM probe on Package Scan v6.

This probe is intentionally small enough to run on a local machine. It does not
claim to be the final Wan-VAE/DiT tokenizer experiment. Instead it exercises the
native experimental protocol:

1. Train a visual RQ tokenizer offline.
2. Freeze the tokenizer.
3. Compare action heads that receive different information.
4. Compare next-code dynamics heads against simple baselines.

The default feature source is resized RGB frames, so the script can run without
CUDA. The same protocol should later be swapped to frozen Wan-VAE or video-DiT
features for the real cluster experiment.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from codewam.codebook import StateCodebook
from codewam.data import PackageScanV6Dataset


@dataclass
class Standardizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, x: torch.Tensor) -> "Standardizer":
        std = x.std(dim=0).clamp(min=1e-6)
        return cls(mean=x.mean(dim=0), std=std)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight Package Scan v6 native CodeWAM probe.")
    parser.add_argument("--root", default="package_scan_v6")
    parser.add_argument("--output", default="runs/native_probe/package_scan_v6_native_probe.json")
    parser.add_argument("--max-windows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--action-video-freq-ratio", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--image-size", type=int, nargs=2, metavar=("H", "W"), default=[96, 192])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--code-dim", type=int, default=128)
    parser.add_argument("--code-levels", type=int, default=3)
    parser.add_argument("--codebook-size", type=int, default=64)
    parser.add_argument("--pool", type=int, default=2)
    parser.add_argument("--recon-grid", type=int, default=4)
    parser.add_argument("--epochs-codebook", type=int, default=12)
    parser.add_argument("--epochs-head", type=int, default=120)
    parser.add_argument("--epochs-dynamics", type=int, default=120)
    parser.add_argument("--lr-codebook", type=float, default=2e-3)
    parser.add_argument("--lr-head", type=float, default=2e-3)
    parser.add_argument("--hidden", type=int, default=256)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def collect_package_scan(args: argparse.Namespace) -> dict[str, Any]:
    dataset = PackageScanV6Dataset(
        root=args.root,
        num_frames=args.num_frames,
        action_video_freq_ratio=args.action_video_freq_ratio,
        max_windows=None,
        return_camera_stack=False,
    )
    total_windows = len(dataset)
    sample_count = min(int(args.max_windows), total_windows)
    sample_indices = np.random.RandomState(args.seed).choice(
        total_windows,
        size=sample_count,
        replace=False,
    ).tolist()
    sampled = Subset(dataset, sample_indices)
    loader = DataLoader(
        sampled,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    cur_images, next_images, actions, proprio = [], [], [], []
    for batch in loader:
        video = batch["video"].float()
        if video.shape[2] < 2:
            raise ValueError("Native probe needs at least two video frames per sample.")
        cur = F.interpolate(video[:, :, 0], size=tuple(args.image_size), mode="bilinear", align_corners=False)
        nxt = F.interpolate(video[:, :, 1], size=tuple(args.image_size), mode="bilinear", align_corners=False)
        cur_images.append(cur)
        next_images.append(nxt)
        actions.append(batch["action"][:, : args.action_horizon].reshape(video.shape[0], -1).float())
        proprio.append(batch["proprio"][:, 0].float())

    return {
        "summary": dataset.summary(),
        "sample_indices": sample_indices,
        "cur_images": torch.cat(cur_images),
        "next_images": torch.cat(next_images),
        "actions": torch.cat(actions),
        "proprio": torch.cat(proprio),
    }


def split_indices(n: int, split_seed: int, train_fraction: float = 0.8) -> tuple[torch.Tensor, torch.Tensor]:
    perm = np.random.RandomState(split_seed).permutation(n)
    n_train = max(1, int(n * train_fraction))
    if n - n_train < 1:
        raise ValueError("Need at least two samples for train/test split.")
    return torch.tensor(perm[:n_train]), torch.tensor(perm[n_train:])


def train_offline_tokenizer(
    images: torch.Tensor,
    train_idx: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[StateCodebook, nn.Module, dict[str, float]]:
    channels = int(images.shape[1])
    tokenizer = StateCodebook(
        in_ch=channels,
        dim=args.code_dim,
        n_levels=args.code_levels,
        codebook_size=args.codebook_size,
        action_dim=None,
        pool=args.pool,
    ).to(args.device).train()
    recon_head = nn.Linear(args.code_dim, channels * args.recon_grid * args.recon_grid).to(args.device)
    opt = torch.optim.Adam(list(tokenizer.parameters()) + list(recon_head.parameters()), lr=args.lr_codebook)

    x_train = images[train_idx].to(args.device)
    y_train = F.adaptive_avg_pool2d(x_train, args.recon_grid).reshape(x_train.shape[0], -1)
    batch_size = min(args.batch_size * 8, max(1, x_train.shape[0]))

    last_rec = 0.0
    last_vq = 0.0
    for _ in range(args.epochs_codebook):
        perm = torch.randperm(x_train.shape[0], device=args.device)
        for start in range(0, x_train.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            out = tokenizer.encode(x_train[idx], update=True)
            rec = recon_head(out["z_q"])
            rec_loss = F.mse_loss(rec, y_train[idx])
            loss = rec_loss + out["vq_loss"]
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_rec = float(rec_loss.detach().item())
            last_vq = float(out["vq_loss"].detach().item())

    tokenizer.eval()
    recon_head.eval()
    for param in tokenizer.parameters():
        param.requires_grad_(False)
    for param in recon_head.parameters():
        param.requires_grad_(False)
    return tokenizer, recon_head, {"last_recon_loss": last_rec, "last_vq_loss": last_vq}


@torch.no_grad()
def encode_frozen_codes(tokenizer: StateCodebook, images: torch.Tensor, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    zq_list, code_list, usage_list, perp_list = [], [], [], []
    batch_size = min(args.batch_size * 8, max(1, images.shape[0]))
    for start in range(0, images.shape[0], batch_size):
        x = images[start:start + batch_size].to(args.device)
        out = tokenizer.encode(x, update=False)
        zq_list.append(out["z_q"].cpu())
        code_list.append(out["codes"].cpu())
        usage_list.append(out["usage"].cpu())
        perp_list.append(out["perplexity"].cpu())
    return {
        "zq": torch.cat(zq_list),
        "codes": torch.cat(code_list),
        "usage": torch.stack(usage_list).mean(dim=0),
        "perplexity": torch.stack(perp_list).mean(dim=0),
    }


def train_action_head(
    name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    x_scaler = Standardizer.fit(x[train_idx])
    y_scaler = Standardizer.fit(y[train_idx])
    x_train = x_scaler.encode(x[train_idx]).to(args.device)
    y_train = y_scaler.encode(y[train_idx]).to(args.device)
    x_test = x_scaler.encode(x[test_idx]).to(args.device)
    y_test = y[test_idx].to(args.device)

    head = MLP(x.shape[1], y.shape[1], hidden=args.hidden).to(args.device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr_head)
    batch_size = min(args.batch_size * 8, max(1, x_train.shape[0]))

    for _ in range(args.epochs_head):
        perm = torch.randperm(x_train.shape[0], device=args.device)
        for start in range(0, x_train.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            pred = head(x_train[idx])
            loss = F.mse_loss(pred, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        pred = y_scaler.decode(head(x_test))
        mse = F.mse_loss(pred, y_test).item()
        mae = F.l1_loss(pred, y_test).item()
        norm_mse = F.mse_loss(y_scaler.encode(pred.cpu()), y_scaler.encode(y[test_idx])).item()
    return {
        "name": name,
        "mse": float(mse),
        "mae": float(mae),
        "standardized_mse": float(norm_mse),
    }


def train_code_classifier(
    name: str,
    x: torch.Tensor,
    target_codes: torch.Tensor,
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    x_scaler = Standardizer.fit(x[train_idx])
    x_train = x_scaler.encode(x[train_idx]).to(args.device)
    x_test = x_scaler.encode(x[test_idx]).to(args.device)
    y_train = target_codes[train_idx].to(args.device)
    y_test = target_codes[test_idx].to(args.device)

    head = MLP(x.shape[1], args.code_levels * args.codebook_size, hidden=args.hidden).to(args.device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr_head)
    batch_size = min(args.batch_size * 8, max(1, x_train.shape[0]))

    for _ in range(args.epochs_dynamics):
        perm = torch.randperm(x_train.shape[0], device=args.device)
        for start in range(0, x_train.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            logits = head(x_train[idx]).view(-1, args.code_levels, args.codebook_size)
            loss = sum(F.cross_entropy(logits[:, lvl], y_train[idx, lvl]) for lvl in range(args.code_levels))
            loss = loss / args.code_levels
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        logits = head(x_test).view(-1, args.code_levels, args.codebook_size)
        accs = [
            (logits[:, lvl].argmax(dim=1) == y_test[:, lvl]).float().mean().item()
            for lvl in range(args.code_levels)
        ]
    return {
        "name": name,
        "top1": float(np.mean(accs)),
        "per_level_top1": [float(acc) for acc in accs],
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    data = collect_package_scan(args)
    n = int(data["cur_images"].shape[0])
    train_idx, test_idx = split_indices(n, args.split_seed)

    tokenizer, _recon_head, tokenizer_stats = train_offline_tokenizer(data["cur_images"], train_idx, args)
    cur = encode_frozen_codes(tokenizer, data["cur_images"], args)
    nxt = encode_frozen_codes(tokenizer, data["next_images"], args)

    zq = cur["zq"].float()
    next_codes = nxt["codes"].long()
    action = data["actions"].float()
    proprio = data["proprio"].float()

    action_results = [
        train_action_head("proprio_only", proprio, action, train_idx, test_idx, args),
        train_action_head("code_only", zq, action, train_idx, test_idx, args),
        train_action_head("code_plus_proprio", torch.cat([zq, proprio], dim=1), action, train_idx, test_idx, args),
    ]

    copy_top1 = float((cur["codes"][test_idx] == next_codes[test_idx]).float().mean().item())
    dynamics_results = [
        {"name": "copy_current_code", "top1": copy_top1},
        train_code_classifier("proprio_only_to_next_code", proprio, next_codes, train_idx, test_idx, args),
        train_code_classifier("code_only_to_next_code", zq, next_codes, train_idx, test_idx, args),
        train_code_classifier(
            "code_plus_action_to_next_code",
            torch.cat([zq, action], dim=1),
            next_codes,
            train_idx,
            test_idx,
            args,
        ),
    ]

    result = {
        "protocol": "offline_tokenizer_then_frozen_downstream",
        "feature_source": "resized_rgb_first_frame",
        "dataset": data["summary"],
        "n_windows": n,
        "sample_indices_head": data["sample_indices"][:16],
        "train_windows": int(train_idx.numel()),
        "test_windows": int(test_idx.numel()),
        "action_horizon": int(args.action_horizon),
        "image_size": list(args.image_size),
        "codebook": {
            "dim": int(args.code_dim),
            "levels": int(args.code_levels),
            "codebook_size": int(args.codebook_size),
            "pool": int(args.pool),
            "usage": [float(x) for x in cur["usage"]],
            "perplexity": [float(x) for x in cur["perplexity"]],
            **tokenizer_stats,
        },
        "action_probe": action_results,
        "dynamics_probe": dynamics_results,
        "args": vars(args),
    }
    return result


def print_report(result: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("CodeWAM Native Probe: offline tokenizer -> frozen downstream")
    print("=" * 72)
    print(f"windows: train={result['train_windows']} test={result['test_windows']} "
          f"feature={result['feature_source']}")
    cb = result["codebook"]
    print(f"codebook: K={cb['codebook_size']} levels={cb['levels']} dim={cb['dim']} pool={cb['pool']}")
    print(f"usage={ [round(v, 3) for v in cb['usage']] } "
          f"perplexity={ [round(v, 1) for v in cb['perplexity']] }")

    print("\n[action probe] lower is better")
    for row in result["action_probe"]:
        print(f"  {row['name']:<18} mse={row['mse']:.4f} "
              f"mae={row['mae']:.4f} standardized_mse={row['standardized_mse']:.4f}")

    print("\n[dynamics probe] higher top1 is better")
    for row in result["dynamics_probe"]:
        top1 = row["top1"]
        print(f"  {row['name']:<30} top1={top1:.4f}")

    print("\nRead:")
    print("  proprio_only is the shortcut baseline.")
    print("  code_only asks whether the visual code itself carries action information.")
    print("  code_plus_proprio should beat proprio_only if the code is useful to policy.")
    print("  code_plus_action_to_next_code should beat copy/proprio baselines if the code supports world dynamics.")


def main() -> None:
    args = parse_args()
    result = run_probe(args)
    print_report(result)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
