"""
Cross-attention decoder variant for mmWave pose regression.

Key difference vs `PoseRegressor` / `PoseRegressorPoseDec`:
  - We keep per-point token representations (B,P,D) from the transformer.
  - We decode joints using learnable joint queries + cross-attention over point tokens.

Output layout matches training/eval reshapes used elsewhere:
  pred.view(B, 3, J) assumes the flat vector is [x(0..J-1), y(0..J-1), z(0..J-1)].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .point_transformer import PointTransformerEncoderConfig


class PointTransformerTokenEncoder(nn.Module):
    """
    Same spirit as `PointTransformerEncoder`, but returns per-point tokens (B,P,D)
    (no pooling) so a downstream decoder can attend to them.
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

    def forward(self, points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
          points: (B,P,F)
          mask: optional BoolTensor (B,P) where True means "valid point".
        Returns:
          tokens: (B,P,D) where D=embed_dim
        """
        if points.ndim != 3:
            raise ValueError(f"points must be (B,P,F), got {tuple(points.shape)}")

        x = self.point_embed(points)  # (B,P,D)

        key_padding_mask = None
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            key_padding_mask = ~mask  # True means "ignore"

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B,P,D)
        return x


@dataclass
class PoseRegressorCrossDecConfig:
    encoder: PointTransformerEncoderConfig
    label_dim: int = 45  # default: 3*15

    # decoder
    dec_heads: int = 2
    dec_dropout: float = 0.1


class PoseRegressorCrossDec(nn.Module):
    """
    Token encoder + cross-attention joint decoder.
    Output: (B, label_dim) where label_dim must be divisible by 3.
    """

    def __init__(self, cfg: PoseRegressorCrossDecConfig):
        super().__init__()
        if cfg.label_dim % 3 != 0:
            raise ValueError(f"label_dim must be divisible by 3, got {cfg.label_dim}")
        self.cfg = cfg
        self.joints = cfg.label_dim // 3

        self.token_encoder = PointTransformerTokenEncoder(cfg.encoder)

        d = int(cfg.encoder.embed_dim)
        self.joint_queries = nn.Parameter(torch.zeros(1, self.joints, d))
        nn.init.trunc_normal_(self.joint_queries, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=int(cfg.dec_heads),
            dropout=float(cfg.dec_dropout),
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Dropout(float(cfg.dec_dropout)),
            nn.Linear(d * 2, d),
        )
        self.out_norm = nn.LayerNorm(d)

        # joint -> xyz (per joint)
        self.xyz_head = nn.Linear(d, 3)

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # memory tokens from points
        mem = self.token_encoder(points, mask=mask)  # (B,P,D)

        # cross-attn: Q=joint queries, K/V=point tokens
        q = self.joint_queries.expand(mem.shape[0], -1, -1)  # (B,J,D)

        key_padding_mask = None
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            key_padding_mask = ~mask  # (B,P) True means "ignore"

        attn_out, _ = self.cross_attn(q, mem, mem, key_padding_mask=key_padding_mask, need_weights=False)  # (B,J,D)
        x = q + attn_out
        x = x + self.ffn(x)
        x = self.out_norm(x)

        xyz = self.xyz_head(x)  # (B,J,3)
        # flatten as [x(0..J-1), y(0..J-1), z(0..J-1)]
        xyz = xyz.permute(0, 2, 1).contiguous().view(xyz.shape[0], 3 * self.joints)
        return xyz

