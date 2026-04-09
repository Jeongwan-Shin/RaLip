"""
ptVision (CLS): Point Cloud Tokenizer → CLIP Vision Encoder alignment.

Architecture:
  Path A: Point Cloud → Point Encoder (CLS pool) → point_proj → point_emb (512D)
  Path B: Point Cloud → PointCloudTokenizer (MLP) → CLIP Vision Transformer (frozen)
          → visual_projection → vision_emb (512D)

Training: contrastive(point_emb, vision_emb)
Testing:  point_emb vs text_emb → R@1, R@5, R@10
"""

from __future__ import annotations
import sys, os
from dataclasses import dataclass
from typing import Dict, List
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pre_training"))
from models.point_transformer import PointTransformerEncoder, PointTransformerEncoderConfig


def load_pretrained_encoder(encoder, checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    mapped = {k.replace("token_encoder.", "", 1): v
              for k, v in ckpt["model_state_dict"].items() if k.startswith("token_encoder.")}
    missing, _ = encoder.load_state_dict(mapped, strict=False)
    print(f"[encoder] loaded {len(mapped)} params, missing: {missing}")
    return encoder


class PointCloudTokenizer(nn.Module):
    """Learnable MLP that converts point cloud features into
    pseudo-visual tokens compatible with CLIP Vision Transformer."""

    def __init__(self, in_dim=5, hidden_dim=768, max_points=196):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_points + 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, points: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """(B, P, 5) → (B, 1+P, hidden_dim) with CLS prepended."""
        x = self.mlp(points)  # (B, P, 768)
        B, P, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+P, 768)
        x = x + self.pos_embed[:, :1 + P, :]
        return x


def _contrastive_loss(a, b, logit_scale):
    logits = logit_scale * a @ b.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    acc = (logits.argmax(1) == labels).float().mean()
    return loss, acc


@dataclass
class PtVisionConfig:
    # point encoder
    embed_dim: int = 256; out_dim: int = 128; num_layers: int = 4
    num_heads: int = 4; mlp_ratio: float = 2.0; dropout: float = 0.1
    pooling: str = "cls"
    proj_dim: int = 512; max_points: int = 196
    clip_model_name: str = "openai/clip-vit-base-patch32"
    init_logit_scale: float = np.log(1 / 0.07)


class PtVision(nn.Module):
    def __init__(self, cfg: PtVisionConfig):
        super().__init__()
        self.cfg = cfg

        # Point encoder (trainable)
        enc_cfg = PointTransformerEncoderConfig(
            in_dim=5, embed_dim=cfg.embed_dim, out_dim=cfg.out_dim,
            num_layers=cfg.num_layers, num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout,
            attn_dropout=cfg.dropout, pooling=cfg.pooling)
        self.point_encoder = PointTransformerEncoder(enc_cfg)
        self.point_proj = nn.Sequential(
            nn.Linear(cfg.out_dim, cfg.proj_dim), nn.GELU(),
            nn.Linear(cfg.proj_dim, cfg.proj_dim))

        # Point cloud tokenizer (trainable) → CLIP vision space
        from transformers import CLIPModel, CLIPTokenizer
        self.clip_model = CLIPModel.from_pretrained(
            cfg.clip_model_name, torch_dtype=torch.float32, use_safetensors=True)
        self.tokenizer_text = CLIPTokenizer.from_pretrained(cfg.clip_model_name)

        # Freeze CLIP entirely
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

        # Learnable tokenizer: point cloud → pseudo visual tokens (768D for ViT-B/32)
        vision_hidden = self.clip_model.vision_model.config.hidden_size  # 768
        self.pt_tokenizer = PointCloudTokenizer(
            in_dim=5, hidden_dim=vision_hidden, max_points=cfg.max_points)

        self.logit_scale = nn.Parameter(torch.tensor(float(cfg.init_logit_scale)))

    def encode_points(self, points, mask):
        """Point encoder path → (B, proj_dim) L2-normed."""
        z = self.point_encoder(points, mask=mask)
        return F.normalize(self.point_proj(z), dim=-1)

    def encode_via_vision(self, points, mask):
        """Point cloud → tokenizer → CLIP vision transformer → (B, 512) L2-normed."""
        tokens = self.pt_tokenizer(points, mask)  # (B, 1+P, 768)

        # Build attention mask: CLS always valid + point mask
        B, P = mask.shape
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)  # (B, 1+P)
        # CLIP encoder expects 4D attention mask
        attn_mask = full_mask[:, None, None, :].float()  # (B, 1, 1, 1+P)
        attn_mask = (1.0 - attn_mask) * torch.finfo(tokens.dtype).min

        with torch.no_grad():
            hidden = self.clip_model.vision_model.pre_layrnorm(tokens)
            encoder_out = self.clip_model.vision_model.encoder(
                inputs_embeds=hidden, attention_mask=attn_mask)
            pooled = encoder_out[0][:, 0, :]  # CLS token
            pooled = self.clip_model.vision_model.post_layernorm(pooled)
            vision_emb = self.clip_model.visual_projection(pooled)

        return F.normalize(vision_emb, dim=-1)

    def encode_text(self, texts, device):
        tok = self.tokenizer_text(texts, padding=True, truncation=True,
                                  max_length=77, return_tensors="pt").to(device)
        with torch.no_grad():
            return F.normalize(self.clip_model.get_text_features(**tok), dim=-1)

    def forward(self, points, mask):
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        point_emb = self.encode_points(points, mask)
        vision_emb = self.encode_via_vision(points, mask)
        loss, acc = _contrastive_loss(point_emb, vision_emb, logit_scale)
        return {"loss": loss, "acc": acc, "logit_scale": logit_scale.detach()}
