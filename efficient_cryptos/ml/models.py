"""Sequence models migrated from ``rw0710`` for crypto cross-sections.

Every model accepts ``(batch, sequence, features)`` and emits one score per
symbol. The final scalar ``Linear`` head is deliberately kept explicit so the
trainer can capture its input for dispersive/contrastive representation losses.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

SUPPORTED_MODELS = ("mlp", "gru", "resnet")


class MLPBackbone(nn.Module):
    def __init__(self, seq_len: int, feature_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * feature_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(-1)


class CNNBackbone(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.features = nn.Sequential(nn.Flatten(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(self.conv(x.transpose(1, 2)))).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GroupNorm(1, channels), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=1), nn.GroupNorm(1, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.net(x))


class ResNetBackbone(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float, blocks: int = 3):
        super().__init__()
        self.stem = nn.Conv1d(feature_dim, hidden_dim, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden_dim, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.blocks(self.stem(x.transpose(1, 2)))).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x)).squeeze(-1)


class ResNetAttentionBackbone(ResNetBackbone):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float,
                 blocks: int = 3, layers: int = 2, heads: int = 8):
        super().__init__(feature_dim, hidden_dim, dropout, blocks)
        heads = _valid_heads(hidden_dim, heads)
        layer = nn.TransformerEncoderLayer(
            hidden_dim, heads, hidden_dim * 4, dropout, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.attn = nn.TransformerEncoder(layer, layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.blocks(self.stem(x.transpose(1, 2))).transpose(1, 2)
        return self.attn(z).mean(dim=1)


class GRUBackbone(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden_dim, num_layers=2, batch_first=True,
                          dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.gru(x)
        return self.head(self.norm(z[:, -1])).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.shape[1]]


def _valid_heads(dim: int, requested: int) -> int:
    for n in range(min(dim, requested), 0, -1):
        if dim % n == 0:
            return n
    return 1


class TransformerBackbone(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float,
                 layers: int = 2, heads: int = 8):
        super().__init__()
        heads = _valid_heads(hidden_dim, heads)
        self.input = nn.Linear(feature_dim, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim, heads, hidden_dim * 4, dropout, batch_first=True,
            activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.pos(self.input(x)))
        return self.head(self.norm(z[:, -1])).squeeze(-1)


class BiN(nn.Module):
    """Bilinear normalization used by the migrated TLOB model."""
    def __init__(self, features: int, steps: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma_f = nn.Parameter(torch.ones(1, features, 1))
        self.beta_f = nn.Parameter(torch.zeros(1, features, 1))
        self.gamma_t = nn.Parameter(torch.ones(1, 1, steps))
        self.beta_t = nn.Parameter(torch.zeros(1, 1, steps))
        self.mix = nn.Parameter(torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        by_time = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + self.eps)
        by_feat = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + self.eps)
        w = torch.softmax(self.mix, dim=0)
        return w[0] * (self.gamma_f * by_time + self.beta_f) + w[1] * (
            self.gamma_t * by_feat + self.beta_t
        )


class TLOBLayer(nn.Module):
    """Dual time/feature attention block adapted from rw0710's TLOB."""
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        heads = _valid_heads(dim, heads)
        self.time_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.feature_gate = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.time_attn(x, x, x, need_weights=False)
        x = self.norm1(x + z)
        return self.norm2(x + self.feature_gate(x))


class TLOBBackbone(nn.Module):
    def __init__(self, seq_len: int, feature_dim: int, hidden_dim: int, dropout: float,
                 layers: int = 4, heads: int = 1):
        super().__init__()
        self.bin = BiN(feature_dim, seq_len)
        self.input = nn.Linear(feature_dim, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim, max_len=seq_len)
        self.layers = nn.ModuleList([TLOBLayer(hidden_dim, heads, dropout) for _ in range(layers)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pos(self.input(self.bin(x.transpose(1, 2)).transpose(1, 2)))
        for layer in self.layers:
            z = layer(z)
        return self.head(z.mean(dim=1)).squeeze(-1)


class MoEBackbone(nn.Module):
    """Factor-level mixture of experts migrated from rw0710."""
    def __init__(self, seq_len: int, feature_dim: int, hidden_dim: int, dropout: float,
                 tau: float = 1.0, expert_hidden: int = 8):
        super().__init__()
        h = max(2, expert_hidden)
        self.tau = max(float(tau), 1e-4)
        self.w1 = nn.Parameter(torch.empty(feature_dim, h, seq_len))
        self.b1 = nn.Parameter(torch.zeros(feature_dim, h))
        self.w2 = nn.Parameter(torch.empty(feature_dim, h))
        self.b2 = nn.Parameter(torch.zeros(feature_dim))
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        self.drop = nn.Dropout(dropout)
        self.gate = nn.Sequential(nn.Flatten(), nn.Linear(seq_len * feature_dim, hidden_dim),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, feature_dim))
        self.head = nn.Linear(feature_dim * 2, 1)
        self.last_gate_entropy: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.einsum("blf,fhl->bfh", x, self.w1) + self.b1
        signals = torch.einsum("bfh,fh->bf", self.drop(torch.relu(z)), self.w2) + self.b2
        weights = torch.softmax(self.gate(x) / self.tau, dim=1)
        self.last_gate_entropy = -(weights * weights.clamp_min(1e-9).log()).sum(1).mean()
        return self.head(torch.cat([weights, weights * signals], dim=1)).squeeze(-1)


class MomentBackbone(nn.Module):
    """Optional MOMENT foundation-model adapter retained from rw0710."""
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float,
                 model_name: str, freeze_encoder: bool):
        super().__init__()
        try:
            from momentfm import MOMENTPipeline
        except ImportError as exc:
            raise ImportError("model=moment requires the optional 'momentfm' package") from exc
        self.encoder = MOMENTPipeline.from_pretrained(
            model_name, model_kwargs={"task_name": "embedding"}
        )
        self.encoder.init()
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        self.proj = nn.LazyLinear(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.encoder(x_enc=x.transpose(1, 2))
        embedding = output.embeddings if hasattr(output, "embeddings") else output
        z = self.dropout(torch.relu(self.proj(embedding.reshape(embedding.shape[0], -1))))
        return self.head(z).squeeze(-1)


class MoiraiBackbone(nn.Module):
    """Optional Moirai adapter; kept explicit because uni2ts is not a core dependency."""
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        try:
            import uni2ts  # noqa: F401
        except ImportError as exc:
            raise ImportError("model=moirai requires the optional 'uni2ts' package") from exc
        raise NotImplementedError(
            "Moirai's forecasting API requires a checkpoint-specific context/patch setup; "
            "install uni2ts and provide that adapter before selecting model=moirai."
        )


def build_model(name: str, seq_len: int, feature_dim: int, hidden_dim: int = 128,
                dropout: float = 0.1, resnet_blocks: int = 3,
                attention_layers: int = 2, attention_heads: int = 8,
                moe_tau: float = 1.0, moe_expert_hidden: int = 8,
                tlob_layers: int = 4, tlob_heads: int = 1,
                foundation_model_name: str = "",
                foundation_freeze_encoder: bool = True) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return MLPBackbone(seq_len, feature_dim, hidden_dim, dropout)
    if name == "cnn":
        return CNNBackbone(feature_dim, hidden_dim, dropout)
    if name == "resnet":
        return ResNetBackbone(feature_dim, hidden_dim, dropout, resnet_blocks)
    if name == "resnet_attn":
        return ResNetAttentionBackbone(feature_dim, hidden_dim, dropout, resnet_blocks,
                                       attention_layers, attention_heads)
    if name == "gru":
        return GRUBackbone(feature_dim, hidden_dim, dropout)
    if name == "transformer":
        return TransformerBackbone(feature_dim, hidden_dim, dropout,
                                   attention_layers, attention_heads)
    if name == "tlob":
        return TLOBBackbone(seq_len, feature_dim, hidden_dim, dropout, tlob_layers, tlob_heads)
    if name == "moe":
        return MoEBackbone(seq_len, feature_dim, hidden_dim, dropout, moe_tau, moe_expert_hidden)
    if name == "moment":
        return MomentBackbone(feature_dim, hidden_dim, dropout,
                              foundation_model_name or "AutonLab/MOMENT-1-large",
                              foundation_freeze_encoder)
    if name == "moirai":
        return MoiraiBackbone()
    raise ValueError(f"Unsupported model {name!r}; choose from {SUPPORTED_MODELS}")
