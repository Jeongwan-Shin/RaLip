"""
VisionGuidedText: vision-augmented contrastive alignment.

Training:
  [Point Encoder → point_emb] + [CLIP Vision → img_emb]
       → Fusion Projector → fused_emb
       ↔ [CLIP Text → text_emb]  (contrastive loss)

  + auxiliary loss: point_emb ↔ text_emb  (ensures point encoder alone is useful)

Testing:
  Point Encoder → point_emb  ↔  text_emb   (R@1, R@5, R@10)
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pre_training"))
from models.point_transformer import PointTransformerEncoder, PointTransformerEncoderConfig


def load_pretrained_encoder(
    encoder: PointTransformerEncoder, checkpoint_path: str, device: str = "cpu",
) -> PointTransformerEncoder:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"]
    mapped = {k.replace("token_encoder.", "", 1): v
              for k, v in state_dict.items() if k.startswith("token_encoder.")}
    missing, _ = encoder.load_state_dict(mapped, strict=False)
    print(f"[encoder] loaded {len(mapped)} params, randomly init: {missing}")
    return encoder


def _contrastive_loss(emb_a, emb_b, logit_scale):
    logits = logit_scale * emb_a @ emb_b.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.T, labels)
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return (loss_ab + loss_ba) / 2.0, acc


@dataclass
class VisionGuidedTextConfig:
    in_dim: int = 5
    embed_dim: int = 256
    out_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    pooling: str = "cls"
    proj_dim: int = 512
    clip_model_name: str = "openai/clip-vit-base-patch32"
    freeze_clip: bool = True
    aux_weight: float = 0.5       # weight for auxiliary point-only loss
    init_logit_scale: float = np.log(1 / 0.07)


class VisionGuidedText(nn.Module):
    def __init__(self, cfg: VisionGuidedTextConfig):
        super().__init__()
        self.cfg = cfg

        # ---- point encoder ----
        enc_cfg = PointTransformerEncoderConfig(
            in_dim=cfg.in_dim, embed_dim=cfg.embed_dim, out_dim=cfg.out_dim,
            num_layers=cfg.num_layers, num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout,
            attn_dropout=cfg.dropout, pooling=cfg.pooling,
        )
        self.point_encoder = PointTransformerEncoder(enc_cfg)

        # ---- point projection (used standalone at test time) ----
        self.point_proj = nn.Sequential(
            nn.Linear(cfg.out_dim, cfg.proj_dim),
            nn.GELU(),
            nn.Linear(cfg.proj_dim, cfg.proj_dim),
        )

        # ---- fusion projector: combines point + vision ----
        self.fusion_proj = nn.Sequential(
            nn.Linear(cfg.proj_dim * 2, cfg.proj_dim),
            nn.GELU(),
            nn.Linear(cfg.proj_dim, cfg.proj_dim),
        )

        # ---- CLIP (frozen) ----
        from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

        self.processor = CLIPProcessor.from_pretrained(cfg.clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(
            cfg.clip_model_name, torch_dtype=torch.float32, use_safetensors=True,
        )
        if cfg.freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False
            self.clip_model.eval()

        # ---- temperature ----
        self.logit_scale = nn.Parameter(torch.tensor(float(cfg.init_logit_scale)))

    # ---- encoders ----

    def encode_points(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Point encoder + projection → (B, proj_dim) L2-normed."""
        z = self.point_encoder(points, mask=mask)
        z = self.point_proj(z)
        return F.normalize(z, dim=-1)

    def encode_images(self, images: List[Image.Image], device: torch.device) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            return F.normalize(self.clip_model.get_image_features(**inputs), dim=-1)

    def encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        tok = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=77, return_tensors="pt").to(device)
        with torch.no_grad():
            return F.normalize(self.clip_model.get_text_features(**tok), dim=-1)

    def fuse(self, point_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        """Fuse point + vision embeddings → (B, proj_dim) L2-normed."""
        combined = torch.cat([point_emb, image_emb], dim=-1)  # (B, 2*proj_dim)
        return F.normalize(self.fusion_proj(combined), dim=-1)

    # ---- training forward ----

    def forward(
        self,
        points: torch.Tensor,
        mask: torch.Tensor,
        images: List[Image.Image],
        texts: List[str],
    ) -> Dict[str, torch.Tensor]:
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        point_emb = self.encode_points(points, mask)       # (B, 512)
        image_emb = self.encode_images(images, points.device)  # (B, 512)
        text_emb = self.encode_text(texts, points.device)   # (B, 512)

        # Fusion: point + vision → fused
        fused_emb = self.fuse(point_emb, image_emb)         # (B, 512)

        # Main loss: fused ↔ text
        loss_fused, acc_fused = _contrastive_loss(fused_emb, text_emb, logit_scale)

        # Auxiliary loss: point-only ↔ text (ensures point encoder works standalone)
        loss_point, acc_point = _contrastive_loss(point_emb, text_emb, logit_scale)

        # Combined loss
        alpha = self.cfg.aux_weight
        loss = (1 - alpha) * loss_fused + alpha * loss_point

        return {
            "loss": loss,
            "loss_fused": loss_fused.detach(),
            "loss_point": loss_point.detach(),
            "acc_fused": acc_fused,
            "acc_point": acc_point,
            "logit_scale": logit_scale.detach(),
        }
