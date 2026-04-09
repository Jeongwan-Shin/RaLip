"""
token_level_ptVision: Cross-attention between point tokens and
tokenized-vision tokens for contrastive learning.

Training:
  cross_attn(point_tokens, vision_tokens_from_tokenizer) → fused ↔ vision_cls  (main)
  standalone_pool(point_tokens) ↔ vision_cls                                    (aux)

Testing:
  standalone_pool(point_tokens) vs text_cls → R@K
"""

from __future__ import annotations
import sys, os
from dataclasses import dataclass
from typing import Dict, List
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pre_training"))
from models.point_transformer_with_crossDec import PointTransformerTokenEncoder
from models.point_transformer import PointTransformerEncoderConfig


def load_pretrained_token_encoder(encoder, checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    mapped = {k.replace("token_encoder.", "", 1): v
              for k, v in ckpt["model_state_dict"].items() if k.startswith("token_encoder.")}
    missing, _ = encoder.load_state_dict(mapped, strict=False)
    print(f"[token_encoder] loaded {len(mapped)} params, missing: {missing}")
    return encoder


class PointCloudTokenizer(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=768, max_points=196):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_points + 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, points, mask=None):
        x = self.mlp(points); B, P, _ = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :1+P, :]
        return x


class CrossModalAttention(nn.Module):
    def __init__(self, point_dim, other_dim, proj_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.kv_proj = nn.Linear(other_dim, point_dim)
        self.cross_attn = nn.MultiheadAttention(point_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(point_dim)
        self.ffn = nn.Sequential(nn.Linear(point_dim, point_dim*2), nn.GELU(), nn.Linear(point_dim*2, point_dim))
        self.norm2 = nn.LayerNorm(point_dim)
        self.out_proj = nn.Sequential(nn.Linear(point_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))

    def forward(self, point_tokens, other_tokens, point_mask=None):
        kv = self.kv_proj(other_tokens)
        attn_out, _ = self.cross_attn(point_tokens, kv, kv, need_weights=False)
        x = self.norm1(point_tokens + attn_out)
        x = self.norm2(x + self.ffn(x))
        if point_mask is not None:
            w = point_mask.float().unsqueeze(-1)
            pooled = (x * w).sum(1) / w.sum(1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        return F.normalize(self.out_proj(pooled), dim=-1)


class StandalonePool(nn.Module):
    def __init__(self, in_dim, proj_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))

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
class TokenLevelPtVisionConfig:
    embed_dim: int = 256; num_layers: int = 4; num_heads: int = 4
    mlp_ratio: float = 2.0; dropout: float = 0.1; proj_dim: int = 512
    cross_heads: int = 4; max_points: int = 196
    clip_model_name: str = "openai/clip-vit-base-patch32"
    aux_weight: float = 0.5; init_logit_scale: float = np.log(1 / 0.07)


class TokenLevelPtVision(nn.Module):
    def __init__(self, cfg: TokenLevelPtVisionConfig):
        super().__init__()
        self.cfg = cfg

        enc_cfg = PointTransformerEncoderConfig(
            in_dim=5, embed_dim=cfg.embed_dim, out_dim=cfg.embed_dim,
            num_layers=cfg.num_layers, num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout,
            attn_dropout=cfg.dropout, pooling="cls")
        self.token_encoder = PointTransformerTokenEncoder(enc_cfg)

        from transformers import CLIPModel, CLIPTokenizer
        self.clip_model = CLIPModel.from_pretrained(
            cfg.clip_model_name, torch_dtype=torch.float32, use_safetensors=True)
        self.tokenizer_text = CLIPTokenizer.from_pretrained(cfg.clip_model_name)
        for p in self.clip_model.parameters(): p.requires_grad = False
        self.clip_model.eval()

        vision_hidden = self.clip_model.vision_model.config.hidden_size  # 768
        self.pt_tokenizer = PointCloudTokenizer(in_dim=5, hidden_dim=vision_hidden, max_points=cfg.max_points)

        # Cross-attention: point(256) × vision(768)
        self.cross_attn = CrossModalAttention(cfg.embed_dim, vision_hidden, cfg.proj_dim, cfg.cross_heads, cfg.dropout)
        self.standalone_pool = StandalonePool(cfg.embed_dim, cfg.proj_dim)

        self.logit_scale = nn.Parameter(torch.tensor(float(cfg.init_logit_scale)))

    def _get_point_tokens(self, points, mask):
        return self.token_encoder(points, mask=mask)

    def _get_vision_tokens_and_cls(self, points, mask):
        """Point cloud → tokenizer → CLIP vision transformer → tokens + CLS."""
        tokens = self.pt_tokenizer(points, mask)  # (B, 1+P, 768)
        B, P = mask.shape
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)
        attn_mask = full_mask[:, None, None, :].float()
        attn_mask = (1.0 - attn_mask) * torch.finfo(tokens.dtype).min

        with torch.no_grad():
            hidden = self.clip_model.vision_model.pre_layrnorm(tokens)
            encoder_out = self.clip_model.vision_model.encoder(
                inputs_embeds=hidden, attention_mask=attn_mask)
            last_hidden = encoder_out[0]  # (B, 1+P, 768) all tokens
            pooled = self.clip_model.vision_model.post_layernorm(last_hidden[:, 0, :])
            vision_cls = F.normalize(self.clip_model.visual_projection(pooled), dim=-1)

        return last_hidden, vision_cls  # tokens for cross-attn, CLS for contrastive

    def encode_points_standalone(self, points, mask):
        return self.standalone_pool(self._get_point_tokens(points, mask), mask)

    def encode_text(self, texts, device):
        tok = self.tokenizer_text(texts, padding=True, truncation=True,
                                  max_length=77, return_tensors="pt").to(device)
        with torch.no_grad():
            return F.normalize(self.clip_model.get_text_features(**tok), dim=-1)

    def forward(self, points, mask):
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        point_tokens = self._get_point_tokens(points, mask)
        vision_tokens, vision_cls = self._get_vision_tokens_and_cls(points, mask)

        cross_emb = self.cross_attn(point_tokens, vision_tokens, mask)
        loss_cross, acc_cross = _contrastive_loss(cross_emb, vision_cls, logit_scale)

        stand_emb = self.standalone_pool(point_tokens, mask)
        loss_stand, acc_stand = _contrastive_loss(stand_emb, vision_cls, logit_scale)

        alpha = self.cfg.aux_weight
        loss = (1 - alpha) * loss_cross + alpha * loss_stand
        return {"loss": loss, "acc_cross": acc_cross, "acc_stand": acc_stand, "logit_scale": logit_scale.detach()}
