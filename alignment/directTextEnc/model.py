"""
RadarCLIP: contrastive alignment of radar point-cloud embeddings
with CLIP text embeddings.

Components
----------
1. PointTransformerEncoder  (pre-trained, fine-tuned)
2. CLIP text encoder        (frozen)
3. Point projection head    (MLP -> shared dim)
4. Learnable temperature    (logit_scale)
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow importing from pre_training
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pre_training"))
from models.point_transformer import PointTransformerEncoder, PointTransformerEncoderConfig


# ---------------------------------------------------------------------------
# Load pre-trained encoder from PoseRegressorCrossDec checkpoint
# ---------------------------------------------------------------------------

def load_pretrained_encoder(
    encoder: PointTransformerEncoder,
    checkpoint_path: str,
    device: str = "cpu",
) -> PointTransformerEncoder:
    """
    Map weights from PoseRegressorCrossDec checkpoint
    (token_encoder.point_embed.*, token_encoder.encoder.*)
    into a PointTransformerEncoder (point_embed.*, encoder.*).

    cls_token and proj are randomly initialised (not present in the
    cross-dec checkpoint).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"]

    mapped: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k.startswith("token_encoder."):
            new_key = k.replace("token_encoder.", "", 1)
            mapped[new_key] = v

    missing, unexpected = encoder.load_state_dict(mapped, strict=False)
    print(f"[encoder] loaded {len(mapped)} params from checkpoint")
    if missing:
        print(f"[encoder] randomly initialised: {missing}")
    if unexpected:
        print(f"[encoder] ignored unexpected: {unexpected}")

    return encoder


# ---------------------------------------------------------------------------
# RadarCLIP model
# ---------------------------------------------------------------------------

@dataclass
class RadarCLIPConfig:
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
    proj_dim: int = 512          # shared embedding dim (matches CLIP)

    # CLIP
    clip_model_name: str = "openai/clip-vit-base-patch32"
    freeze_text: bool = True

    # temperature
    init_logit_scale: float = np.log(1 / 0.07)


class RadarCLIP(nn.Module):
    def __init__(self, cfg: RadarCLIPConfig):
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

        # ---- CLIP text encoder (frozen) ----
        from transformers import CLIPModel, CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(
            cfg.clip_model_name,
            torch_dtype=torch.float32,
            use_safetensors=True,
        )

        if cfg.freeze_text:
            for p in self.clip_model.parameters():
                p.requires_grad = False
            self.clip_model.eval()

        # ---- learnable temperature ----
        self.logit_scale = nn.Parameter(
            torch.tensor(float(cfg.init_logit_scale))
        )

    # ---- forward helpers ----

    def encode_points(self, points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """(B, P, 5), (B, P) -> (B, proj_dim) L2-normalised."""
        z = self.point_encoder(points, mask=mask)      # (B, out_dim), already L2-normed
        z = self.point_proj(z)                         # (B, proj_dim)
        return F.normalize(z, dim=-1)

    def encode_text(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """list[str] -> (B, proj_dim) L2-normalised."""
        tok = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=77, return_tensors="pt",
        ).to(device)

        with torch.no_grad() if self.cfg.freeze_text else torch.enable_grad():
            text_features = self.clip_model.get_text_features(**tok)  # (B, 512)

        return F.normalize(text_features, dim=-1)

    # ---- contrastive loss ----

    def forward(
        self,
        points: torch.Tensor,
        mask: torch.Tensor,
        texts: List[str],
    ) -> Dict[str, torch.Tensor]:
        point_embeds = self.encode_points(points, mask)          # (B, proj_dim)
        text_embeds = self.encode_text(texts, points.device)     # (B, 512)

        # scale
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        # cosine similarity matrix
        logits_p2t = logit_scale * point_embeds @ text_embeds.T  # (B, B)
        logits_t2p = logits_p2t.T

        labels = torch.arange(logits_p2t.shape[0], device=logits_p2t.device)
        loss_p2t = F.cross_entropy(logits_p2t, labels)
        loss_t2p = F.cross_entropy(logits_t2p, labels)
        loss = (loss_p2t + loss_t2p) / 2.0

        # accuracy (for logging)
        with torch.no_grad():
            acc_p2t = (logits_p2t.argmax(dim=1) == labels).float().mean()
            acc_t2p = (logits_t2p.argmax(dim=1) == labels).float().mean()

        return {
            "loss": loss,
            "loss_p2t": loss_p2t.detach(),
            "loss_t2p": loss_t2p.detach(),
            "acc_p2t": acc_p2t,
            "acc_t2p": acc_t2p,
            "logit_scale": logit_scale.detach(),
        }
