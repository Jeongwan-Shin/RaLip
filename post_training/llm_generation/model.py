"""
RadarLLM: Radar Point Cloud Encoder + Projector + LLM for text generation.

Architecture:
  PointTransformerTokenEncoder (frozen, from alignment checkpoint)
  → Linear Projector (trainable, 256 → LLM hidden dim)
  → LLM (frozen)

Pipeline:
  1. Encoder extracts per-point tokens from radar point cloud
  2. Mean pool → single radar embedding
  3. Projector maps radar embedding to LLM input space
  4. [radar_token] + [prompt_tokens] fed to LLM for generation
"""

from __future__ import annotations

import sys, os

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "pre_training"))

from pre_training.models.point_transformer_with_crossDec import PointTransformerTokenEncoder
from pre_training.models.point_transformer import PointTransformerEncoderConfig


# ---------------------------------------------------------------------------
# Encoder loader
# ---------------------------------------------------------------------------

def load_token_encoder_from_alignment(checkpoint_path, embed_dim=256, num_layers=4,
                                       num_heads=4, dropout=0.1, device="cpu"):
    """Load PointTransformerTokenEncoder weights from an alignment checkpoint."""
    enc_cfg = PointTransformerEncoderConfig(
        in_dim=5, embed_dim=embed_dim, out_dim=embed_dim,
        num_layers=num_layers, num_heads=num_heads,
        mlp_ratio=2.0, dropout=dropout, attn_dropout=dropout, pooling="cls")
    encoder = PointTransformerTokenEncoder(enc_cfg)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model_state_dict"]
    mapped = {k.replace("token_encoder.", "", 1): v
              for k, v in state.items() if k.startswith("token_encoder.")}
    missing, unexpected = encoder.load_state_dict(mapped, strict=False)
    print(f"[TokenEncoder] loaded {len(mapped)} params, missing={missing}")
    return encoder


# ---------------------------------------------------------------------------
# RadarLLM
# ---------------------------------------------------------------------------

class RadarLLM(nn.Module):
    """Radar encoder + projector + frozen LLM."""

    def __init__(self, encoder: PointTransformerTokenEncoder,
                 llm_model_name: str = "microsoft/Phi-3.5-mini-instruct",
                 encoder_dim: int = 256):
        super().__init__()

        # --- Frozen radar encoder ---
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        # --- Frozen LLM ---
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading LLM: {llm_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, torch_dtype=torch.float32,
            trust_remote_code=True, device_map="auto")
        for p in self.llm.parameters():
            p.requires_grad = False
        self.llm.eval()

        llm_hidden = self.llm.config.hidden_size

        # --- Trainable projector ---
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, llm_hidden),
            nn.GELU(),
            nn.Linear(llm_hidden, llm_hidden),
        )

    def get_radar_embeds(self, points, mask):
        """Point cloud → single projected radar token for LLM input.

        Args:
            points: (B, P, 5) radar point cloud
            mask:   (B, P) bool, True = valid point
        Returns:
            (B, 1, llm_hidden) radar token embedding
        """
        with torch.no_grad():
            tokens = self.encoder(points, mask=mask)  # (B, P, encoder_dim)
        if mask is not None:
            w = mask.float().unsqueeze(-1)
            pooled = (tokens * w).sum(1) / w.sum(1).clamp(min=1)
        else:
            pooled = tokens.mean(dim=1)
        radar_emb = self.projector(pooled)
        return radar_emb.unsqueeze(1)

    def forward(self, points, mask, prompt_ids, prompt_attn, label_ids):
        """Training forward: next-token prediction loss on label tokens only."""
        B = points.shape[0]
        device = points.device

        radar_embeds = self.get_radar_embeds(points, mask)

        with torch.no_grad():
            prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)
            label_embeds = self.llm.get_input_embeddings()(label_ids)

        inputs_embeds = torch.cat([radar_embeds, prompt_embeds, label_embeds], dim=1)

        radar_attn = torch.ones(B, 1, device=device, dtype=prompt_attn.dtype)
        label_attn = (label_ids != self.tokenizer.pad_token_id).long()
        attn_mask = torch.cat([radar_attn, prompt_attn, label_attn], dim=1)

        ignore = torch.full((B, 1 + prompt_ids.shape[1]), -100,
                            device=device, dtype=label_ids.dtype)
        labels = torch.cat([ignore, label_ids], dim=1)
        labels[labels == self.tokenizer.pad_token_id] = -100

        seq_len = inputs_embeds.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            position_ids=position_ids,
            labels=labels,
        )
        return outputs.loss

    @torch.no_grad()
    def generate(self, points, mask, prompt_text, max_new_tokens=64):
        """Greedy token-by-token generation with KV cache."""
        device = points.device
        B = points.shape[0]

        radar_embeds = self.get_radar_embeds(points, mask)

        tok = self.tokenizer(prompt_text, return_tensors="pt",
                             padding=True, truncation=True, max_length=128)
        prompt_ids = tok.input_ids.to(device)
        prompt_attn = tok.attention_mask.to(device)
        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)

        inputs_embeds = torch.cat([
            radar_embeds.expand(prompt_ids.shape[0], -1, -1),
            prompt_embeds
        ], dim=1)
        radar_attn = torch.ones(B, 1, device=device, dtype=prompt_attn.dtype)
        attn_mask = torch.cat([radar_attn, prompt_attn], dim=1)

        seq_len = inputs_embeds.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)

        generated_ids = []
        past_key_values = None
        cur_embeds = inputs_embeds
        cur_attn = attn_mask
        cur_pos = position_ids

        for step in range(max_new_tokens):
            outputs = self.llm(
                inputs_embeds=cur_embeds,
                attention_mask=cur_attn,
                position_ids=cur_pos,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            generated_ids.append(next_token)
            cur_embeds = self.llm.get_input_embeddings()(next_token.unsqueeze(-1))
            cur_attn = torch.cat([
                cur_attn, torch.ones(B, 1, device=device, dtype=cur_attn.dtype)
            ], dim=1)
            cur_pos = torch.tensor([[seq_len + step]], device=device).expand(B, -1)

        if generated_ids:
            token_ids = torch.stack(generated_ids, dim=1)
            return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return [""] * B
