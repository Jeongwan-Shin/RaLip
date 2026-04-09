"""
RadarCLIP_Vision: contrastive alignment of radar point-cloud embeddings
with CLIP vision (image) embeddings.

Components
----------
1. PointTransformerEncoder  (pre-trained, fine-tuned)
2. CLIP vision encoder      (frozen)
3. Point projection head    (MLP -> shared dim)
4. Learnable temperature    (logit_scale)
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


# ---------------------------------------------------------------------------
# Load pre-trained encoder
# ---------------------------------------------------------------------------

def load_pretrained_encoder(
    encoder: PointTransformerEncoder,
    checkpoint_path: str,
    device: str = "cpu",
) -> PointTransformerEncoder:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"]
    mapped: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("token_encoder."):
            mapped[k.replace("token_encoder.", "", 1)] = v
    missing, unexpected = encoder.load_state_dict(mapped, strict=False)
    print(f"[encoder] loaded {len(mapped)} params from checkpoint")
    if missing:
        print(f"[encoder] randomly initialised: {missing}")
    return encoder


# ---------------------------------------------------------------------------
# RadarCLIP_Vision model
# ---------------------------------------------------------------------------

@dataclass
class RadarCLIPVisionConfig:
    # point encoder
    in_dim: int = 5
    embed_dim: int = 256
    out_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    pooling: str = "cls"

    # projection
    proj_dim: int = 512

    # CLIP
    clip_model_name: str = "openai/clip-vit-base-patch32"
    freeze_vision: bool = True

    # temperature
    init_logit_scale: float = np.log(1 / 0.07)


class RadarCLIPVision(nn.Module):
    def __init__(self, cfg: RadarCLIPVisionConfig):
        super().__init__()
        self.cfg = cfg

        # ---- point encoder ----
        enc_cfg = PointTransformerEncoderConfig(
            in_dim=cfg.in_dim,
            embed_dim=cfg.embed_dim,
            out_dim=cfg.out_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            attn_dropout=cfg.dropout,
            pooling=cfg.pooling,
        )
        self.point_encoder = PointTransformerEncoder(enc_cfg)

        # ---- point projection head ----
        self.point_proj = nn.Sequential(
            nn.Linear(cfg.out_dim, cfg.proj_dim),
            nn.GELU(),
            nn.Linear(cfg.proj_dim, cfg.proj_dim),
        )

        # ---- CLIP model (vision for training, text for eval — shared space) ----
        from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

        self.processor = CLIPProcessor.from_pretrained(cfg.clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(
            cfg.clip_model_name,
            torch_dtype=torch.float32,
            use_safetensors=True,
        )

        if cfg.freeze_vision:
            for p in self.clip_model.parameters():
                p.requires_grad = False
            self.clip_model.eval()

        # ---- learnable temperature ----
        self.logit_scale = nn.Parameter(
            torch.tensor(float(cfg.init_logit_scale))
        )

    # ---- forward helpers ----

    def encode_points(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.point_encoder(points, mask=mask)
        z = self.point_proj(z)
        return F.normalize(z, dim=-1)

    def encode_images(self, images: List[Image.Image], device: torch.device) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad() if self.cfg.freeze_vision else torch.enable_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        return F.normalize(image_features, dim=-1)

    def encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Encode text via CLIP text encoder (same shared space as vision)."""
        tok = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=77, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**tok)
        return F.normalize(text_features, dim=-1)

    # ---- contrastive loss ----

    def forward(
        self,
        points: torch.Tensor,
        mask: torch.Tensor,
        images: List[Image.Image],
    ) -> Dict[str, torch.Tensor]:
        point_embeds = self.encode_points(points, mask)
        image_embeds = self.encode_images(images, points.device)

        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        logits_p2i = logit_scale * point_embeds @ image_embeds.T
        logits_i2p = logits_p2i.T

        labels = torch.arange(logits_p2i.shape[0], device=logits_p2i.device)
        loss_p2i = F.cross_entropy(logits_p2i, labels)
        loss_i2p = F.cross_entropy(logits_i2p, labels)
        loss = (loss_p2i + loss_i2p) / 2.0

        with torch.no_grad():
            acc_p2i = (logits_p2i.argmax(dim=1) == labels).float().mean()
            acc_i2p = (logits_i2p.argmax(dim=1) == labels).float().mean()

        return {
            "loss": loss,
            "loss_p2i": loss_p2i.detach(),
            "loss_i2p": loss_i2p.detach(),
            "acc_p2i": acc_p2i,
            "acc_i2p": acc_i2p,
            "logit_scale": logit_scale.detach(),
        }
