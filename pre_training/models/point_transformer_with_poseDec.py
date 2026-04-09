"""
Pose-decoder variant for mmWave pose regression.

Idea:
  - Use the same PointTransformerEncoder as `models/point_transformer.py`
  - Replace the single 45-D head with 3 separate heads (x/y/z), each predicting J joints.

Output layout matches the training/eval reshapes used elsewhere:
  pred.view(B, 3, J) assumes the flat vector is [x(0..J-1), y(0..J-1), z(0..J-1)].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .point_transformer import PointTransformerEncoder, PointTransformerEncoderConfig


@dataclass
class PoseRegressorPoseDecConfig:
    encoder: PointTransformerEncoderConfig
    label_dim: int = 45  # default: 3*15


class PoseRegressorPoseDec(nn.Module):
    """
    Encoder + (x,y,z) decoders for pose regression.
    Output shape: (B, label_dim) where label_dim must be divisible by 3.
    """

    def __init__(self, cfg: PoseRegressorPoseDecConfig):
        super().__init__()
        if cfg.label_dim % 3 != 0:
            raise ValueError(f"label_dim must be divisible by 3, got {cfg.label_dim}")
        self.cfg = cfg
        joints = cfg.label_dim // 3

        self.encoder = PointTransformerEncoder(cfg.encoder)

        # Separate "projectors" for each coordinate axis
        self.head_x = nn.Sequential(
            nn.LayerNorm(cfg.encoder.out_dim),
            nn.Linear(cfg.encoder.out_dim, joints),
        )
        self.head_y = nn.Sequential(
            nn.LayerNorm(cfg.encoder.out_dim),
            nn.Linear(cfg.encoder.out_dim, joints),
        )
        self.head_z = nn.Sequential(
            nn.LayerNorm(cfg.encoder.out_dim),
            nn.Linear(cfg.encoder.out_dim, joints),
        )

    def forward(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.encoder(points, mask=mask)  # (B, out_dim)
        x = self.head_x(z)  # (B, J)
        y = self.head_y(z)  # (B, J)
        zc = self.head_z(z)  # (B, J)
        return torch.cat([x, y, zc], dim=1)  # (B, 3J)

