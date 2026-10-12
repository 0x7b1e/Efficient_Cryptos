"""Cross-sectional objectives migrated from ``rw0710``."""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

SUPPORTED_LOSSES = (
    "mse", "huber", "pairwise_rank", "huber_pairwise_rank", "ic", "mse_ic",
    "spread", "mse_spread", "ic_spread", "mse_ic_spread",
    "mse_spread_disp", "mse_spread_contrast", "mse_ic_spread_contrast",
)


def ic_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p, t = pred - pred.mean(), target - target.mean()
    return -(p * t).sum() / (torch.sqrt((p.square()).sum() * (t.square()).sum()) + eps)


def pairwise_rank_loss(pred: torch.Tensor, target: torch.Tensor,
                       temperature: float = 1.0, max_samples: int = 512) -> torch.Tensor:
    p, y = pred.reshape(-1), target.reshape(-1)
    finite = torch.isfinite(p) & torch.isfinite(y)
    p, y = p[finite], y[finite]
    n = p.numel()
    if n < 2:
        return pred.sum() * 0.0
    if max_samples > 0 and n > max_samples:
        idx = torch.randperm(n, device=p.device)[:max_samples] if pred.requires_grad else torch.argsort(y)[
            torch.linspace(0, n - 1, max_samples, device=p.device).round().long()
        ]
        p, y, n = p[idx], y[idx], max_samples
    pdiff, ydiff = p[:, None] - p[None, :], y[:, None] - y[None, :]
    valid = torch.triu(torch.ones(n, n, dtype=torch.bool, device=p.device), diagonal=1) & ydiff.ne(0)
    if not valid.any():
        return pred.sum() * 0.0
    return F.softplus(-ydiff[valid].sign() * pdiff[valid] / temperature).mean()


def spread_loss(pred: torch.Tensor, target: torch.Tensor, sharpness: float = 5.0,
                short_weight: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    if pred.numel() < 2:
        return pred.sum() * 0.0
    p = (pred - pred.mean()) / (pred.std() + eps)
    return -((torch.softmax(p * sharpness, 0) * target).sum()
             - short_weight * (torch.softmax(-p * sharpness, 0) * target).sum())


def dispersive_loss(embedding: torch.Tensor | None, tau: float = 1.0,
                    max_samples: int = 256) -> torch.Tensor | None:
    if embedding is None:
        return None
    z = embedding.reshape(embedding.shape[0], -1)
    if z.shape[0] < 2:
        return z.sum() * 0.0
    if max_samples > 0 and z.shape[0] > max_samples:
        z = z[torch.randperm(z.shape[0], device=z.device)[:max_samples]]
    z = F.normalize(z.float(), dim=1)
    d2 = torch.cdist(z, z).square()
    mask = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    logits = -d2[mask] / max(tau, 1e-8)
    return torch.logsumexp(logits, 0) - torch.log(logits.new_tensor(float(logits.numel())))


def quantile_contrastive_loss(embedding: torch.Tensor | None, target: torch.Tensor,
                              tau: float = 0.2, bins: int = 5,
                              max_samples: int = 256) -> torch.Tensor | None:
    if embedding is None:
        return None
    z, y = embedding.reshape(embedding.shape[0], -1), target.reshape(-1)
    finite = torch.isfinite(y) & torch.isfinite(z).all(1)
    z, y = z[finite], y[finite]
    if z.shape[0] < 4:
        return z.sum() * 0.0
    if max_samples > 0 and z.shape[0] > max_samples:
        idx = torch.randperm(z.shape[0], device=z.device)[:max_samples]
        z, y = z[idx], y[idx]
    n = z.shape[0]
    effective_bins = min(bins, max(2, n // 2))
    order = torch.argsort(y, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(n, device=y.device)
    labels = (ranks * effective_bins // n).clamp_max(effective_bins - 1)
    z = F.normalize(z.float(), dim=1)
    logits = z @ z.T / max(tau, 1e-8)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    positives = labels[:, None].eq(labels[None, :]) & ~eye
    valid = positives.any(1)
    if not valid.any():
        return z.sum() * 0.0
    return -(torch.logsumexp(logits.masked_fill(~positives, -torch.inf), 1)[valid]
             - torch.logsumexp(logits.masked_fill(eye, -torch.inf), 1)[valid]).mean()


def build_loss(name: str, *, ic_weight: float = 0.1, spread_weight: float = 1.0,
               spread_sharpness: float = 5.0, short_weight: float = 1.0,
               huber_delta: float = 1.0, rank_weight: float = 1.0,
               rank_temperature: float = 1.0, rank_max_samples: int = 512
               ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    name = name.lower()
    mse, huber = nn.MSELoss(), nn.HuberLoss(delta=huber_delta)
    rank = lambda p, y: pairwise_rank_loss(p, y, rank_temperature, rank_max_samples)
    ic = ic_loss
    spread = lambda p, y: spread_loss(p, y, spread_sharpness, short_weight)
    choices = {
        "mse": lambda p, y: mse(p, y),
        "huber": lambda p, y: huber(p, y),
        "pairwise_rank": rank,
        "huber_pairwise_rank": lambda p, y: huber(p, y) + rank_weight * rank(p, y),
        "ic": ic,
        "mse_ic": lambda p, y: mse(p, y) + ic_weight * ic(p, y),
        "spread": spread,
        "mse_spread": lambda p, y: mse(p, y) + spread_weight * spread(p, y),
        "mse_spread_disp": lambda p, y: mse(p, y) + spread_weight * spread(p, y),
        "mse_spread_contrast": lambda p, y: mse(p, y) + spread_weight * spread(p, y),
        "ic_spread": lambda p, y: ic(p, y) + spread_weight * spread(p, y),
        "mse_ic_spread": lambda p, y: mse(p, y) + ic_weight * ic(p, y) + spread_weight * spread(p, y),
        "mse_ic_spread_contrast": lambda p, y: mse(p, y) + ic_weight * ic(p, y) + spread_weight * spread(p, y),
    }
    if name not in choices:
        raise ValueError(f"Unsupported loss {name!r}; choose from {SUPPORTED_LOSSES}")
    return choices[name]
