"""
Train RadarCLIP_Vision: contrastive alignment of radar point-cloud encoder
with CLIP vision encoder.

Usage:
  python alignment/imageSpaceMap/train.py --help
  bash  alignment/imageSpaceMap/scripts/train.sh
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from functools import partial
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.dirname(__file__))

from dataset import (
    ActionEntry,
    RadarImageDataset,
    build_entries,
    collate_radar_image,
    split_entries,
)
from model import RadarCLIPVision, RadarCLIPVisionConfig, load_pretrained_encoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tqdm_maybe(it, **kwargs):
    try:
        from tqdm import tqdm
        return tqdm(it, file=sys.__stdout__, dynamic_ncols=True, leave=True, **kwargs)
    except Exception:
        return it


class Tee:
    def __init__(self, *writers):
        self._writers = writers

    def write(self, data):
        for w in self._writers:
            w.write(data)
        return len(data)

    def flush(self):
        for w in self._writers:
            w.flush()


def setup_log(log_dir: str, run_tag: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{run_tag}.log")
    f = open(path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = Tee(sys.__stderr__, f)
    print(f"[info] logging to: {path}")
    return path


# ---------------------------------------------------------------------------
# Evaluation — Radar-to-Image retrieval  (R@1, R@5, R@10)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: RadarCLIPVision,
    test_entries: List[ActionEntry],
    test_text_entries: List[Dict],
    dataset: RadarImageDataset,
    device: torch.device,
    max_points: int = 196,
    batch_size: int = 64,
    desc: str = "eval",
) -> Dict[str, float]:
    """
    Radar-to-Text retrieval evaluation (via CLIP shared space).

    Training aligns radar → CLIP image space.
    Since CLIP image & text share the same space, we evaluate
    radar → text retrieval at test time.

    1. Encode every test radar sample → point embeddings.
    2. Encode ALL test text descriptions (5 per entry) via CLIP text encoder.
    3. For each radar query, rank texts by cosine similarity.
    4. Hit if any text with the same action appears in top-K.
    5. Report R@1, R@5, R@10.
    """
    model.eval()

    # ---- 1. Encode radar point clouds ----
    all_point_embeds: List[torch.Tensor] = []
    all_query_actions: List[str] = []

    for i in tqdm_maybe(range(len(dataset)), desc=f"{desc} encode-radar", unit="sample"):
        entry = test_entries[i]
        pts_tensor, _ = dataset[i]

        n = min(pts_tensor.shape[0], max_points)
        pts = torch.zeros(1, max_points, 5)
        mask = torch.zeros(1, max_points, dtype=torch.bool)
        pts[0, :n] = pts_tensor[:n]
        mask[0, :n] = True

        pts = pts.to(device)
        mask = mask.to(device)
        emb = model.encode_points(pts, mask)
        all_point_embeds.append(emb.cpu())
        all_query_actions.append(entry.action)

    point_embeds = torch.cat(all_point_embeds, dim=0)

    # ---- 2. Build text gallery (5 sentences × N entries) ----
    all_texts: List[str] = []
    all_text_actions: List[str] = []
    for te in test_text_entries:
        action = te["annotation"]["action"]
        for sent in te["sentences"]:
            all_texts.append(sent)
            all_text_actions.append(action)

    all_text_embeds: List[torch.Tensor] = []
    for start in tqdm_maybe(range(0, len(all_texts), batch_size), desc=f"{desc} encode-text", unit="batch"):
        batch_texts = all_texts[start : start + batch_size]
        emb = model.encode_text(batch_texts, device)
        all_text_embeds.append(emb.cpu())

    text_embeds = torch.cat(all_text_embeds, dim=0)

    # ---- 3. Similarity & Recall@K ----
    sims = point_embeds @ text_embeds.T

    n_queries = sims.shape[0]
    recall = {1: 0, 5: 0, 10: 0}

    for i in range(n_queries):
        query_action = all_query_actions[i]
        ranked_indices = sims[i].argsort(descending=True)
        for k in recall:
            top_k_actions = [all_text_actions[idx] for idx in ranked_indices[:k]]
            if query_action in top_k_actions:
                recall[k] += 1

    return {
        "R@1": recall[1] / n_queries,
        "R@5": recall[5] / n_queries,
        "R@10": recall[10] / n_queries,
        "n_queries": n_queries,
        "n_gallery": len(all_texts),
    }


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RadarCLIP_Vision (contrastive radar-image alignment)")

    # data
    p.add_argument("--actions_json", type=str, required=True)
    p.add_argument("--actions_text_jsonl", type=str, required=True,
                   help="Path to mm_actions_text.jsonl (text descriptions for R2T eval)")
    # point cloud dirs
    p.add_argument("--mmbody_pc_dir", type=str, required=True)
    p.add_argument("--mmfi_pc_dir", type=str, required=True)
    p.add_argument("--mri_pc_dir", type=str, required=True)

    # image dirs
    p.add_argument("--mmbody_img_dir", type=str, required=True)
    p.add_argument("--mmfi_img_dir", type=str, required=True)
    p.add_argument("--mri_img_dir", type=str, required=True)

    p.add_argument("--feature_norm", type=str, default="per_sample_zscore")
    p.add_argument("--max_points", type=int, default=196)
    p.add_argument("--min_points", type=int, default=15)

    # pre-trained checkpoint
    p.add_argument("--pretrained_ckpt", type=str, default="")

    # model
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--out_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--pooling", type=str, default="cls")
    p.add_argument("--proj_dim", type=int, default=512)
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--freeze_vision", action="store_true", default=True)

    # training
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-6)

    # output
    p.add_argument("--ckpt_dir", type=str, default="alignment/imageSpaceMap/checkpoints")
    p.add_argument("--log_dir", type=str, default="alignment/imageSpaceMap/log")
    p.add_argument("--run_tag", type=str, default="")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    run_tag = args.run_tag or (
        f"radarclip_vision_ep{args.epochs}_bs{args.batch_size}"
        f"_lr{args.lr}_proj{args.proj_dim}"
        f"_D{args.embed_dim}_L{args.num_layers}_H{args.num_heads}"
    )
    setup_log(args.log_dir, run_tag)

    # ---- data ----
    print("[info] building entries ...")
    entries = build_entries(args.actions_json)
    train_entries, test_entries = split_entries(entries)
    print(f"[info] entries: total={len(entries)}  train={len(train_entries)}  test={len(test_entries)}")

    # Load text descriptions for R2T eval
    import json as _json
    test_actions_set = {e.action for e in test_entries}
    test_text_entries: List[Dict] = []
    with open(args.actions_text_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = _json.loads(line)
            if obj["annotation"]["action"] in test_actions_set:
                test_text_entries.append(obj)
    # Keep only 1 text entry per action (matching the test split)
    _seen_actions: set = set()
    _deduped: List[Dict] = []
    for te in test_text_entries:
        act = te["annotation"]["action"]
        if act not in _seen_actions:
            _seen_actions.add(act)
            _deduped.append(te)
    test_text_entries = _deduped
    print(f"[info] test text entries for R2T eval: {len(test_text_entries)} (5 sentences each)")

    train_ds = RadarImageDataset(
        train_entries,
        mmbody_pc_dir=args.mmbody_pc_dir,
        mmfi_pc_dir=args.mmfi_pc_dir,
        mri_pc_dir=args.mri_pc_dir,
        mmbody_img_dir=args.mmbody_img_dir,
        mmfi_img_dir=args.mmfi_img_dir,
        mri_img_dir=args.mri_img_dir,
        train=True,
        feature_norm=args.feature_norm,
    )
    test_ds = RadarImageDataset(
        test_entries,
        mmbody_pc_dir=args.mmbody_pc_dir,
        mmfi_pc_dir=args.mmfi_pc_dir,
        mri_pc_dir=args.mri_pc_dir,
        mmbody_img_dir=args.mmbody_img_dir,
        mmfi_img_dir=args.mmfi_img_dir,
        mri_img_dir=args.mri_img_dir,
        train=False,
        feature_norm=args.feature_norm,
    )

    collate = partial(collate_radar_image, max_points=args.max_points, min_points=args.min_points)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # ---- model ----
    device = torch.device(args.device)

    cfg = RadarCLIPVisionConfig(
        embed_dim=args.embed_dim,
        out_dim=args.out_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        pooling=args.pooling,
        proj_dim=args.proj_dim,
        clip_model_name=args.clip_model,
        freeze_vision=args.freeze_vision,
    )
    model = RadarCLIPVision(cfg).to(device)

    if args.pretrained_ckpt and os.path.isfile(args.pretrained_ckpt):
        load_pretrained_encoder(model.point_encoder, args.pretrained_ckpt, device=args.device)
        print(f"[info] loaded pre-trained encoder from {args.pretrained_ckpt}")
    else:
        print("[info] training point encoder from scratch (no --pretrained_ckpt)")

    params = [
        {"params": model.point_encoder.parameters(), "lr": args.lr},
        {"params": model.point_proj.parameters(), "lr": args.lr},
        {"params": [model.logit_scale], "lr": args.lr},
    ]
    optim = torch.optim.AdamW(params, weight_decay=args.weight_decay)

    # ---- scheduler ----
    scheduler = None
    if args.scheduler == "cosine":
        steps_per_epoch = len(train_loader)
        total_steps = max(1, args.epochs * steps_per_epoch)
        warmup_steps = max(0, args.warmup_epochs * steps_per_epoch)
        base_lr = args.lr
        min_lr = args.min_lr

        def lr_mult(step: int) -> float:
            s = float(max(0, min(step, total_steps)))
            if warmup_steps > 0 and s < warmup_steps:
                return s / float(warmup_steps)
            if total_steps <= warmup_steps:
                return 1.0
            t = (s - warmup_steps) / float(total_steps - warmup_steps)
            cos = 0.5 * (1.0 + math.cos(math.pi * t))
            min_mult = (min_lr / base_lr) if base_lr > 0 else 0.0
            return min_mult + (1.0 - min_mult) * cos

        scheduler = LambdaLR(optim, lr_lambda=lr_mult)

    n_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_total = sum(p.numel() for p in model.parameters())
    print(f"[info] model params: trainable={n_params_train:,}  total={n_params_total:,}")
    print(f"[info] device={device}  epochs={args.epochs}  batch_size={args.batch_size}")

    # ---- training loop ----
    best_r10 = 0.0

    for epoch in range(args.epochs):
        model.train()
        if cfg.freeze_vision:
            model.clip_model.eval()

        pbar = tqdm_maybe(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"train {epoch+1}/{args.epochs}",
            unit="batch",
        )

        for step, (pts, mask, images) in pbar:
            if pts.shape[0] == 0:
                continue

            pts = pts.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            out = model(pts, mask, images)
            loss = out["loss"]

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if scheduler is not None:
                scheduler.step()

            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    loss=f"{float(loss):.4f}",
                    acc=f"{float(out['acc_p2i']):.2f}",
                    temp=f"{float(out['logit_scale']):.1f}",
                    lr=f"{optim.param_groups[0]['lr']:.2e}",
                )

        # ---- eval: Radar-to-Text retrieval (via CLIP shared space) ----
        metrics = evaluate(
            model,
            test_entries=test_entries,
            test_text_entries=test_text_entries,
            dataset=test_ds,
            device=device,
            max_points=args.max_points,
            batch_size=args.eval_batch_size,
            desc=f"eval {epoch+1}/{args.epochs}",
        )
        print(
            f"\n[epoch {epoch+1}] R2T retrieval | "
            f"R@1={metrics['R@1']:.4f}  "
            f"R@5={metrics['R@5']:.4f}  "
            f"R@10={metrics['R@10']:.4f}  "
            f"(queries={metrics['n_queries']}  gallery={metrics['n_gallery']})"
        )

        # ---- save best (by R@10) ----
        if metrics["R@10"] > best_r10:
            best_r10 = metrics["R@10"]
            os.makedirs(args.ckpt_dir, exist_ok=True)
            best_path = os.path.join(args.ckpt_dir, f"{run_tag}_best.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "metrics": metrics,
                "args": vars(args),
                "config": vars(cfg),
            }, best_path)
            print(f"[info] new best R@10={best_r10:.4f}, saved: {best_path}")

    # ---- save final ----
    os.makedirs(args.ckpt_dir, exist_ok=True)
    final_path = os.path.join(args.ckpt_dir, f"{run_tag}_final.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "metrics": metrics,
        "args": vars(args),
        "config": vars(cfg),
    }, final_path)
    print(f"[info] saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
