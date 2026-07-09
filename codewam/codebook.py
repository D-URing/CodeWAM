"""World Fast Codebook —— 以 FastWAM 为 base 融合 RQ 离散状态码本。

设计(见 project memory `FastWAM×RQ-VAE 世界快速码本方向`):
- 共享 RQ 状态码本同时充当三角色:
  A 条件: 把当前观测(Wan-VAE latent 帧)量化成离散状态码, 作为额外 token 供动作专家 cross-attend。
  B 离散世界模型: DynamicsHead 从当前状态码(+动作特征)预测未来帧的状态码(交叉熵), 取代/补充像素扩散想象。
  C 小数据微调: 大数据预训码本, 真机冻结码本只调动作头。

本文件只依赖 torch, 不 import 5B 视频专家, 因此:
- 可被 P1/P2 探针脚本独立调用(冻结 Wan-VAE latent 上评测可离散性 + 动力学可预测性);
- 通过后再由 fastwam.py 引入(A/B), runtime.py 透传 config。

关键工程(直击 ElimWM 头号失败=码本坍塌, 利用率曾仅 0.06~0.20):
- EMA 码本更新(非梯度) + commitment loss(只训 encoder) + 死码重置(dead-code reset)。
- RQ 多级残差, 保留粗→细层次; 可选保留一路连续残差防"抽象丢精度"。
- forward 返回每级利用率/困惑度, 供 P1 直接读数。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    """把单帧 Wan-VAE latent [B, C, h, w] 池化成状态向量 [B, dim]。

    小 conv + 池化。`pool>1` 时保留 pool×pool 空间网格再展平, 以免全局平均把
    决策相关的局部视觉细节(夹爪/物体位置/条码朝向)抹掉。
    """

    def __init__(self, in_ch: int, dim: int = 256, hidden: int = 256, pool: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GroupNorm(8, hidden), nn.SiLU(),
            nn.AdaptiveAvgPool2d(pool), nn.Flatten(),
        )
        self.proj = nn.Linear(hidden * pool * pool, dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4:
            raise ValueError(f"`latent` must be 4D [B,C,h,w], got {tuple(latent.shape)}")
        return self.proj(self.net(latent))


class RQLevel(nn.Module):
    """单级 VQ, EMA 码本 + 死码重置。码本是 buffer(EMA 更新), 不吃梯度。"""

    def __init__(self, dim: int, codebook_size: int, decay: float = 0.99,
                 eps: float = 1e-5, dead_code_threshold: float = 0.1):
        super().__init__()
        self.dim = dim
        self.K = codebook_size
        self.decay = decay
        self.eps = eps
        self.dead_code_threshold = dead_code_threshold
        self.register_buffer("codebook", torch.randn(codebook_size, dim) * 0.1)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", self.codebook.clone())

    @torch.no_grad()
    def _ema_update(self, residual: torch.Tensor, onehot: torch.Tensor):
        # residual:(N,D) fp32, onehot:(N,K) fp32
        cluster = onehot.sum(0)                       # (K,)
        embed_sum = onehot.t() @ residual             # (K,D)
        self.cluster_size.mul_(self.decay).add_(cluster, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
        n = self.cluster_size.sum()
        normalized = (self.cluster_size + self.eps) / (n + self.K * self.eps) * n
        self.codebook.copy_(self.embed_avg / normalized.unsqueeze(1))
        # 死码重置: 长期没被选中的码字, 用当前 batch 随机残差重灌
        dead = self.cluster_size < self.dead_code_threshold
        n_dead = int(dead.sum().item())
        if n_dead > 0 and residual.shape[0] > 0:
            idx = torch.randint(0, residual.shape[0], (n_dead,), device=residual.device)
            self.codebook[dead] = residual[idx].detach()
            self.cluster_size[dead] = 1.0
            self.embed_avg[dead] = residual[idx].detach()

    def forward(self, residual: torch.Tensor, update: bool):
        r = residual.float()
        cb = self.codebook.float()
        d = (r.pow(2).sum(1, keepdim=True) - 2 * r @ cb.t() + cb.pow(2).sum(1).unsqueeze(0))
        idx = d.argmin(1)                              # (N,)
        q = self.codebook[idx].to(residual.dtype)      # (N,D)
        onehot = F.one_hot(idx, self.K).to(r.dtype)    # (N,K)
        if self.training and update:
            self._ema_update(r, onehot)
        # 利用率/困惑度(当前 batch), 纯统计量, 与梯度无关
        with torch.no_grad():
            probs = onehot.detach().mean(0)
            usage = float((probs > 0).float().mean().item())
            perplexity = float(torch.exp(-(probs * (probs + 1e-10).log()).sum()).item())
        return q, idx, onehot, usage, perplexity


class ResidualQuantizer(nn.Module):
    """L 级残差量化。commitment loss 训练 encoder; 码本靠 EMA。straight-through 传梯度。"""

    def __init__(self, dim: int, n_levels: int = 3, codebook_size: int = 256,
                 beta: float = 0.25, decay: float = 0.99):
        super().__init__()
        self.n_levels = n_levels
        self.codebook_size = codebook_size
        self.dim = dim
        self.beta = beta
        self.levels = nn.ModuleList([
            RQLevel(dim, codebook_size, decay=decay) for _ in range(n_levels)
        ])

    def forward(self, z: torch.Tensor, update: bool = True):
        residual = z
        z_q = torch.zeros_like(z)
        codes, usages, perps = [], [], []
        commit = z.new_zeros(())
        for lvl in self.levels:
            q, idx, _, usage, perp = lvl(residual, update=update)
            commit = commit + F.mse_loss(q.detach(), residual)
            z_q = z_q + q
            residual = residual - q.detach()
            codes.append(idx)
            usages.append(usage)
            perps.append(perp)
        z_q_st = z + (z_q - z).detach()                # straight-through
        vq_loss = self.beta * commit
        codes = torch.stack(codes, dim=1)              # (B, L)
        return {
            "z_q": z_q_st,
            "codes": codes,
            "vq_loss": vq_loss,
            "usage": torch.tensor(usages),             # (L,)
            "perplexity": torch.tensor(perps),         # (L,)
        }

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """codes:(B,L) -> z_q:(B,D)。用于从码重建条件向量(推理/条件复用)。"""
        z_q = None
        for l, lvl in enumerate(self.levels):
            e = lvl.codebook[codes[:, l]]
            z_q = e if z_q is None else z_q + e
        return z_q


class DynamicsHead(nn.Module):
    """B: 从当前状态嵌入(+可选动作特征)预测未来帧的状态码。每级一个分类头(交叉熵)。"""

    def __init__(self, dim: int, n_levels: int, codebook_size: int,
                 action_dim: Optional[int] = None, hidden: int = 512):
        super().__init__()
        self.n_levels = n_levels
        self.codebook_size = codebook_size
        in_dim = dim + (action_dim if action_dim else 0)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, codebook_size) for _ in range(n_levels)])

    def forward(self, state_emb: torch.Tensor, action_feat: Optional[torch.Tensor] = None):
        x = state_emb if action_feat is None else torch.cat([state_emb, action_feat], dim=1)
        h = self.trunk(x)
        return [head(h) for head in self.heads]        # list of (B, K)

    def loss(self, logits_list, target_codes: torch.Tensor):
        """logits_list: L×(B,K); target_codes:(B,L)。返回 (ce_loss, per_level_top1_acc)。"""
        ce = torch.zeros((), dtype=torch.float32, device=target_codes.device)
        accs = []
        for l, logits in enumerate(logits_list):
            tgt = target_codes[:, l]
            ce = ce + F.cross_entropy(logits, tgt)
            accs.append(float((logits.argmax(1) == tgt).float().mean().item()))
        return ce / len(logits_list), accs


class StateCodebook(nn.Module):
    """封装: StateEncoder + RQ + 码嵌入投影(A条件) + DynamicsHead(B)。

    - encode(latent_frame) -> dict(z, z_q, codes, vq_loss, usage, perplexity)
    - condition_token(z_q) -> [B, cond_dim] 供拼进动作专家 context (A)
    - predict_future_codes(state_emb, action_feat) + dynamics.loss (B)
    """

    def __init__(self, in_ch: int, dim: int = 256, n_levels: int = 3,
                 codebook_size: int = 256, cond_dim: Optional[int] = None,
                 action_dim: Optional[int] = None, beta: float = 0.25, decay: float = 0.99,
                 pool: int = 1):
        super().__init__()
        self.dim = dim
        self.encoder = StateEncoder(in_ch=in_ch, dim=dim, pool=pool)
        self.rq = ResidualQuantizer(dim=dim, n_levels=n_levels,
                                    codebook_size=codebook_size, beta=beta, decay=decay)
        self.dynamics = DynamicsHead(dim=dim, n_levels=n_levels,
                                     codebook_size=codebook_size, action_dim=action_dim)
        # A: 把 z_q 投到动作专家 context 的维度(cond_dim, 默认=dim)
        self.cond_proj = nn.Linear(dim, cond_dim) if (cond_dim and cond_dim != dim) else nn.Identity()

    def encode(self, latent_frame: torch.Tensor, update: bool = True) -> dict:
        z = self.encoder(latent_frame)
        out = self.rq(z, update=update)
        out["z"] = z
        return out

    def condition_token(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.cond_proj(z_q)                     # (B, cond_dim)

    def predict_future_codes(self, state_emb, action_feat=None):
        return self.dynamics(state_emb, action_feat)
