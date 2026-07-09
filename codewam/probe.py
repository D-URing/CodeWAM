#!/usr/bin/env python
# =============================================================================
# 世界快速码本 判定实验 P1 + P2 (+P4) —— 真机 package_scan_v6 的 Wan-VAE latent 上
# -----------------------------------------------------------------------------
# 目的(见 project memory `FastWAM×RQ-VAE 世界快速码本方向`)：动大结构前先证/证伪
#   P1 可离散性：池化 Wan-VAE latent 能否被 RQ 干净量化(每级利用率/困惑度/量化误差/重构R²)。
#   P2 动力学可预测：从当前状态码预测下一 latent 帧的码, top1 必须【打过“复制当前码”基线】。
#   P4 因果信息：proprio-only 预测下一码 vs 视觉状态码, 差=视觉码的增量信息(能否破 proprio 捷径)。
#
# 干净协议(修正上一版不公平比较):
#   Phase-1 只用【重构(下采样全空间 latent, 非通道均值)+ vq】塑形码, 不让动力学塑形(否则会
#           人为把码变得可预测, 高估 P2)。
#   Phase-2 冻结 encoder+码本, 用【相同预算】分别训 视觉/proprio/视觉+proprio 三个动力学头, 公平对比。
#   状态编码 pool>1 保留空间网格, 避免全局平均抹掉决策相关的局部视觉细节。
#
# 用法(fastwam 环境, 单卡):
#   DIFFSYNTH_MODEL_BASE_PATH=/root/.../dingxibo/models DIFFSYNTH_SKIP_DOWNLOAD=true \
#   PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python sample_dump/state_codebook_probe.py
#   env: N(窗口数,默认800) EPOCHS(表示相,默认60) DYN_EPOCHS(动力学相,默认100)
#        K(码本,默认64) DIM(默认128) POOL(空间网格,默认2)
# =============================================================================
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Subset

from fastwam.utils.config_resolvers import register_default_resolvers
from codewam.codebook import StateCodebook

ROOT = "/root/paddlejob/share-storage/gpfs/system-public/dingxibo/Embodied_AI/FastWAM"
MODELS = "/root/paddlejob/share-storage/gpfs/system-public/dingxibo/models"
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", MODELS)

N = int(os.environ.get("N", "800"))
EPOCHS = int(os.environ.get("EPOCHS", "60"))
DYN_EPOCHS = int(os.environ.get("DYN_EPOCHS", "100"))
K = int(os.environ.get("K", "64"))
DIM = int(os.environ.get("DIM", "128"))
POOL = int(os.environ.get("POOL", "2"))
LEVELS = 3
DEV = "cuda" if torch.cuda.is_available() else "cpu"
RECON_GRID = 4   # 重构目标: latent 下采样到 4x4 全空间(比通道均值强, 逼码保留空间视觉内容)


def load_vae():
    from fastwam.models.wan22.helpers.loader import _load_registered_model
    candidates = [
        os.path.join(MODELS, "Wan2.2-TI2V-5B", "Wan2.2_VAE.pth"),
        os.path.join(MODELS, "Wan-AI", "Wan2.2-TI2V-5B", "Wan2.2_VAE.pth"),
    ]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        raise FileNotFoundError(f"未找到本地 Wan VAE: {candidates}")
    print(f"loading VAE from {path}", flush=True)
    vae = _load_registered_model(path, "wan_video_vae", torch_dtype=torch.bfloat16, device=DEV)
    vae.eval()
    return vae


@torch.no_grad()
def collect_latents(vae):
    register_default_resolvers()
    with initialize_config_dir(config_dir=os.path.join(ROOT, "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[
            "task=real_robot_joint_2cam224_v6_clean", "num_workers=0", "eval_every=0"])
    ds = instantiate(cfg.data.train)
    total = len(ds)
    idxs = np.random.RandomState(0).choice(total, size=min(N, total), replace=False).tolist()
    print(f"dataset={total} 采样 N={len(idxs)}  device={DEV}", flush=True)
    loader = DataLoader(Subset(ds, idxs), batch_size=8, num_workers=8, shuffle=False)
    LAT, PRO = [], []
    seen = 0
    for batch in loader:
        video = batch["video"].to(DEV, dtype=torch.bfloat16)   # [B,3,9,224,448]
        z = vae.encode(video, device=DEV, tiled=False)          # [B,48,3,14,28]
        LAT.append(z.float().cpu())
        PRO.append(batch["proprio"][:, 0, :].float())
        seen += video.shape[0]
        if seen % 160 == 0:
            print(f"  encoded {seen}/{len(idxs)}", flush=True)
    LAT = torch.cat(LAT); PRO = torch.cat(PRO)
    print("latent", tuple(LAT.shape), "proprio", tuple(PRO.shape), flush=True)
    return LAT, PRO


def train_code_predictor(head, opt, X, targets, epochs, bs=256):
    """通用: 训练一个把特征 X 映射到 (LEVELS×K) logits 的头, 预测 targets:(M,LEVELS)。"""
    for _ in range(epochs):
        pm = torch.randperm(X.shape[0], device=DEV)
        for i in range(0, X.shape[0], bs):
            idx = pm[i:i + bs]
            out = head(X[idx]).view(-1, LEVELS, K)
            ce = sum(F.cross_entropy(out[:, l], targets[idx, l]) for l in range(LEVELS)) / LEVELS
            opt.zero_grad(); ce.backward(); opt.step()


@torch.no_grad()
def eval_top1(head, X, targets):
    out = head(X).view(-1, LEVELS, K)
    return float(np.mean([(out[:, l].argmax(1) == targets[:, l]).float().mean().item()
                          for l in range(LEVELS)]))


def main():
    torch.manual_seed(0); np.random.seed(0)
    vae = load_vae()
    LAT, PRO = collect_latents(vae)
    Nwin, C, Tl, h, w = LAT.shape
    assert C == 48

    perm = np.random.RandomState(1).permutation(Nwin)
    tr = torch.tensor(perm[:int(Nwin * 0.8)]); te = torch.tensor(perm[int(Nwin * 0.8):])

    sc = StateCodebook(in_ch=C, dim=DIM, n_levels=LEVELS, codebook_size=K,
                       action_dim=None, pool=POOL).to(DEV).train()
    recon_head = nn.Linear(DIM, C * RECON_GRID * RECON_GRID).to(DEV)
    opt = torch.optim.Adam(list(sc.parameters()) + list(recon_head.parameters()), lr=2e-3)

    def frames_of(win):
        x = LAT[win].permute(0, 2, 1, 3, 4).reshape(-1, C, h, w)   # [M,48,h,w]
        return x.to(DEV)

    def recon_target(x):
        return F.adaptive_avg_pool2d(x, RECON_GRID).reshape(x.shape[0], -1)

    def pairs_of(win):
        cur, nxt, pro = [], [], []
        for t in range(Tl - 1):
            cur.append(LAT[win, :, t]); nxt.append(LAT[win, :, t + 1]); pro.append(PRO[win])
        return torch.cat(cur).to(DEV), torch.cat(nxt).to(DEV), torch.cat(pro).to(DEV)

    Xtr = frames_of(tr); Ytr = recon_target(Xtr)

    # ---------------- Phase-1: 只用 重构 + vq 塑形码 (不含动力学) ----------------
    bs = 256
    for ep in range(EPOCHS):
        pm = torch.randperm(Xtr.shape[0], device=DEV)
        tr_rec = tr_vq = 0.0
        for i in range(0, Xtr.shape[0], bs):
            idx = pm[i:i + bs]
            out = sc.encode(Xtr[idx], update=True)
            rec = recon_head(out["z_q"])
            rec_loss = F.mse_loss(rec, Ytr[idx])
            loss = rec_loss + out["vq_loss"]
            opt.zero_grad(); loss.backward(); opt.step()
            tr_rec += float(rec_loss); tr_vq += float(out["vq_loss"])
        if (ep + 1) % 15 == 0:
            u = [round(float(x), 3) for x in sc.encode(Xtr[:512], update=False)["usage"]]
            print(f"[repr] ep{ep+1:3d} rec={tr_rec:.3f} vq={tr_vq:.3f} usage={u}", flush=True)

    # ---------------- 冻结 encoder + 码本 ----------------
    sc.eval()
    for p in sc.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        Xte = frames_of(te)
        oute = sc.encode(Xte, update=False)
        usage = [float(x) for x in oute["usage"]]; perp = [float(x) for x in oute["perplexity"]]
        z, zq = oute["z"], oute["z_q"]
        quant_err = float((z - zq).pow(2).mean() / (z.pow(2).mean() + 1e-8))
        rec = recon_head(zq); Yte = recon_target(Xte)
        ss_res = (rec - Yte).pow(2).sum(); ss_tot = (Yte - Yte.mean(0)).pow(2).sum()
        recon_r2 = float(1 - ss_res / ss_tot)

        cur_tr, nxt_tr, pro_tr = pairs_of(tr)
        cur_te, nxt_te, pro_te = pairs_of(te)
        z_cur_tr = sc.encode(cur_tr, update=False)["z"]
        z_cur_te = sc.encode(cur_te, update=False)["z"]
        code_cur_te = sc.encode(cur_te, update=False)["codes"]
        code_nxt_tr = sc.encode(nxt_tr, update=False)["codes"]
        code_nxt_te = sc.encode(nxt_te, update=False)["codes"]

    copy_acc = float((code_cur_te == code_nxt_te).float().mean())

    # ---------------- Phase-2: 相同预算公平训三个动力学头(冻结码) ----------------
    # (b) 视觉状态码 -> 下一码
    vis_head = nn.Sequential(nn.Linear(DIM, 512), nn.SiLU(), nn.Linear(512, K * LEVELS)).to(DEV)
    train_code_predictor(vis_head, torch.optim.Adam(vis_head.parameters(), lr=2e-3),
                         z_cur_tr, code_nxt_tr, DYN_EPOCHS)
    vis_acc = eval_top1(vis_head, z_cur_te, code_nxt_te)
    # (c) proprio-only -> 下一码
    pro_head = nn.Sequential(nn.Linear(7, 512), nn.SiLU(), nn.Linear(512, K * LEVELS)).to(DEV)
    train_code_predictor(pro_head, torch.optim.Adam(pro_head.parameters(), lr=2e-3),
                         pro_tr, code_nxt_tr, DYN_EPOCHS)
    pro_acc = eval_top1(pro_head, pro_te, code_nxt_te)
    # (d) 视觉 + proprio -> 下一码 (视觉是否在 proprio 之上带增量)
    both_tr = torch.cat([z_cur_tr, pro_tr], dim=1); both_te = torch.cat([z_cur_te, pro_te], dim=1)
    both_head = nn.Sequential(nn.Linear(DIM + 7, 512), nn.SiLU(), nn.Linear(512, K * LEVELS)).to(DEV)
    train_code_predictor(both_head, torch.optim.Adam(both_head.parameters(), lr=2e-3),
                         both_tr, code_nxt_tr, DYN_EPOCHS)
    both_acc = eval_top1(both_head, both_te, code_nxt_te)

    print("\n" + "=" * 70)
    print(f"世界快速码本 判定实验 (公平协议, POOL={POOL}, K={K}, held-out 窗口)")
    print("=" * 70)
    print(f"[P1 可离散性] 每级利用率 = {[round(u,3) for u in usage]}  困惑度 = {[round(p,1) for p in perp]}")
    print(f"             量化相对误差 = {quant_err:.4f}   {RECON_GRID}x{RECON_GRID}全空间重构 R² = {recon_r2:.3f}")
    print(f"[P2 动力学]  下一码 top1: 视觉={vis_acc:.3f}  复制基线={copy_acc:.3f}   增量={vis_acc-copy_acc:+.3f}")
    print(f"[P4 因果]    下一码 top1: proprio-only={pro_acc:.3f}  视觉={vis_acc:.3f}  视觉+proprio={both_acc:.3f}")
    print(f"             视觉相对proprio增量={vis_acc-pro_acc:+.3f}   (视觉+proprio)相对proprio={both_acc-pro_acc:+.3f}")
    print("-" * 70)
    print("判读: P1 利用率>~0.5&困惑度>>1=可离散; P2 视觉>>复制=码含可控动力学;")
    print("      P4 (视觉+proprio)>proprio 才说明视觉码带 proprio 之外信息(破捷径的必要条件)。")


if __name__ == "__main__":
    main()
