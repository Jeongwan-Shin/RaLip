"""
Token-level directTextEnc: cross-attention between point cloud tokens
and CLIP text tokens for contrastive learning.

Training:
  cross_attn(point_tokens, text_tokens) → fused_emb ↔ text_cls  (contrastive)
  + auxiliary: standalone_pool(point_tokens) → point_emb ↔ text_cls

Testing:
  standalone_pool(point_tokens) → point_emb  vs  text_cls → R@1, R@5, R@10
"""

from __future__ import annotations

import sys, os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pre_training"))
from models.point_transformer_with_crossDec import PointTransformerTokenEncoder
from models.point_transformer import PointTransformerEncoderConfig


def load_pretrained_token_encoder(encoder, checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    mapped = {k.replace("token_encoder.", "", 1): v
              for k, v in ckpt["model_state_dict"].items()
              if k.startswith("token_encoder.")}
    missing, _ = encoder.load_state_dict(mapped, strict=False)
    print(f"[token_encoder] loaded {len(mapped)} params, missing: {missing}")
    return encoder


class CrossModalAttention(nn.Module):
    """Q=point_tokens, K/V=other_modal_tokens → pool → embedding."""

    def __init__(self, point_dim, other_dim, proj_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.kv_proj = nn.Linear(other_dim, point_dim)
        self.cross_attn = nn.MultiheadAttention(
            point_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(point_dim)
        self.ffn = nn.Sequential(
            nn.Linear(point_dim, point_dim * 2), nn.GELU(),
            nn.Linear(point_dim * 2, point_dim))
        self.norm2 = nn.LayerNorm(point_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(point_dim, proj_dim), nn.GELU(),
            nn.Linear(proj_dim, proj_dim))

    def forward(self, point_tokens, other_tokens, point_mask=None):
        kv = self.kv_proj(other_tokens)
        key_padding_mask = ~point_mask if point_mask is not None else None
        attn_out, _ = self.cross_attn(
            point_tokens, kv, kv, need_weights=False)
        x = self.norm1(point_tokens + attn_out)
        x = self.norm2(x + self.ffn(x))
        # Masked mean pool
        if point_mask is not None:
            w = point_mask.float().unsqueeze(-1)
            pooled = (x * w).sum(1) / w.sum(1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        return F.normalize(self.out_proj(pooled), dim=-1)


class StandalonePool(nn.Module):
    """Mean pool + projection for test-time (no cross-attention)."""

    def __init__(self, in_dim, proj_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim), nn.GELU(),
            nn.Linear(proj_dim, proj_dim))

    def forward(self, tokens, mask=None):
        if mask is not None:
            w = mask.float().unsqueeze(-1)
            pooled = (tokens * w).sum(1) / w.sum(1).clamp(min=1)
        else:
            pooled = tokens.mean(dim=1)
        return F.normalize(self.proj(pooled), dim=-1)


def _contrastive_loss(a, b, logit_scale):
    logits = logit_scale * a @ b.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    acc = (logits.argmax(1) == labels).float().mean()
    return loss, acc


@dataclass
class TokenLevelDirectTextConfig:
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    mlp_ratio: float = 2.0
    dropout: float = 0.1
    proj_dim: int = 512
    cross_heads: int = 4
    clip_model_name: str = "openai/clip-vit-base-patch32"
    aux_weight: float = 0.5
    init_logit_scale: float = np.log(1 / 0.07)


class TokenLevelDirectText(nn.Module):
    def __init__(self, cfg: TokenLevelDirectTextConfig):
        super().__init__()
        self.cfg = cfg

        enc_cfg = PointTransformerEncoderConfig(
            in_dim=5, embed_dim=cfg.embed_dim, out_dim=cfg.embed_dim,
            num_layers=cfg.num_layers, num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout,
            attn_dropout=cfg.dropout, pooling="cls")
        self.token_encoder = PointTransformerTokenEncoder(enc_cfg)

        # Cross-attention: point(256) × text(512)
        self.cross_attn = CrossModalAttention(
            cfg.embed_dim, 512, cfg.proj_dim, cfg.cross_heads, cfg.dropout)

        # Standalone pool (for test time)
        self.standalone_pool = StandalonePool(cfg.embed_dim, cfg.proj_dim)

        # CLIP (frozen)
        from transformers import CLIPModel, CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(
            cfg.clip_model_name, torch_dtype=torch.float32, use_safetensors=True)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.clip_model.eval()

        self.logit_scale = nn.Parameter(torch.tensor(float(cfg.init_logit_scale)))

    def _get_point_tokens(self, points, mask):
        """(B,P,5), (B,P) → (B,P,embed_dim) per-point tokens."""
        return self.token_encoder(points, mask=mask)

    def _get_text_tokens_and_cls(self, texts, device):
        """list[str] → text_tokens (B,T,512), text_cls (B,512)."""
        tok = self.tokenizer(texts, padding=True, truncation=True,
                             max_length=77, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.clip_model.text_model(
                input_ids=tok.input_ids, attention_mask=tok.attention_mask)
            text_tokens = out.last_hidden_state
            text_cls = self.clip_model.text_projection(out.pooler_output)
        return text_tokens, F.normalize(text_cls, dim=-1)

    def encode_points_standalone(self, points, mask):
        """Test-time: point tokens → standalone pool → embedding."""
        tokens = self._get_point_tokens(points, mask)
        return self.standalone_pool(tokens, mask)

    def encode_text_cls(self, texts, device):
        _, text_cls = self._get_text_tokens_and_cls(texts, device)
        return text_cls

    def forward(self, points, mask, texts):
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        point_tokens = self._get_point_tokens(points, mask)
        text_tokens, text_cls = self._get_text_tokens_and_cls(texts, points.device)

        # Cross-attention path
        cross_emb = self.cross_attn(point_tokens, text_tokens, mask)
        loss_cross, acc_cross = _contrastive_loss(cross_emb, text_cls, logit_scale)

        # Standalone path (auxiliary)
        stand_emb = self.standalone_pool(point_tokens, mask)
        loss_stand, acc_stand = _contrastive_loss(stand_emb, text_cls, logit_scale)

        alpha = self.cfg.aux_weight
        loss = (1 - alpha) * loss_cross + alpha * loss_stand

        return {
            "loss": loss,
            "acc_cross": acc_cross, "acc_stand": acc_stand,
            "logit_scale": logit_scale.detach(),
        }
