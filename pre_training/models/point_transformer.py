"""
Point-cloud encoder for mmBody radar points.

The original file was a one-off experiment that tried to import an external PointTransformerV3
implementation. For training in this repo, we provide a self-contained PyTorch encoder that
turns a variable-size point set into a fixed-dimensional representation.

Expected input:
  - points: FloatTensor of shape (B, P, F) where F includes xyz and other radar features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class PoseRegressorConfig:
    encoder: PointTransformerEncoderConfig
    label_dim: int = 45


class PoseRegressor(nn.Module):
    """
    Encoder + linear head for pose regression.
    Output shape: (B, label_dim) (default 45).
    """

    def __init__(self, cfg: PoseRegressorConfig):
        super().__init__()
        self.encoder = PointTransformerEncoder(cfg.encoder)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.encoder.out_dim),
            nn.Linear(cfg.encoder.out_dim, cfg.label_dim),
        )

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.encoder(points, mask=mask)
        return self.head(z)

@dataclass
class PointTransformerEncoderConfig:
    in_dim: int = 5          # mmBody radar points appear to have 5 values per point
    embed_dim: int = 128
    out_dim: int = 64
    num_layers: int = 2
    num_heads: int = 2
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    attn_dropout: float = 0.1
    pooling: str = "cls"    # mean | max | cls


class PointTransformerEncoder(nn.Module):
    """
    Lightweight point transformer:
      point MLP -> TransformerEncoder -> global pooling -> projection head
    """

    def __init__(self, cfg: PointTransformerEncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.point_embed = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.embed_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=int(cfg.embed_dim * cfg.mlp_ratio),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        # Learnable CLS token used when pooling == "cls"
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.proj = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.out_dim),
        )

    def forward(self, points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
          points: (B, P, F)
          mask: optional BoolTensor (B, P) where True means "valid point".
                If None, all points are considered valid.
        Returns:
          emb: (B, out_dim), L2-normalized.
        """
        if points.ndim != 3:
            raise ValueError(f"points must be (B,P,F), got {tuple(points.shape)}")

        x = self.point_embed(points)  # (B,P,D)

        # PyTorch Transformer uses src_key_padding_mask where True means "ignore"
        key_padding_mask = None
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()

        if self.cfg.pooling == "cls":
            # prepend CLS token
            cls = self.cls_token.expand(x.shape[0], -1, -1)  # (B,1,D)
            x = torch.cat([cls, x], dim=1)  # (B,1+P,D)
            if mask is None:
                # all points valid
                mask2 = torch.ones((x.shape[0], x.shape[1]), dtype=torch.bool, device=x.device)
            else:
                # CLS always valid
                mask2 = torch.cat(
                    [torch.ones((mask.shape[0], 1), dtype=torch.bool, device=mask.device), mask],
                    dim=1,
                )
            key_padding_mask = ~mask2
        else:
            if mask is not None:
                key_padding_mask = ~mask  # ignore padded / invalid

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B,P,D)

        if self.cfg.pooling == "mean":
            if mask is None:
                pooled = x.mean(dim=1)
            else:
                w = mask.float().unsqueeze(-1)  # (B,P,1)
                pooled = (x * w).sum(dim=1) / (w.sum(dim=1).clamp_min(1.0))
        elif self.cfg.pooling == "max":
            if mask is None:
                pooled = x.max(dim=1).values
            else:
                # mask invalids to -inf before max
                x2 = x.masked_fill((~mask).unsqueeze(-1), float("-inf"))
                pooled = x2.max(dim=1).values
        elif self.cfg.pooling == "cls":
            pooled = x[:, 0, :]  # CLS output
        else:
            raise ValueError(f"Unknown pooling={self.cfg.pooling}")

        z = self.proj(pooled)
        z = F.normalize(z, dim=-1)
        return z
