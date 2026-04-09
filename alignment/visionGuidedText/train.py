"""
Train VisionGuidedText: vision-augmented contrastive alignment.

Training: (point + vision) fused ↔ text contrastive + point ↔ text auxiliary
Testing:  point-only → text retrieval (R@1, R@5, R@10)
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
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.dirname(__file__))

from dataset import (
    ActionEntry, RadarImageTextDataset,
    build_entries, collate_radar_image_text, split_entries,
)
from model import VisionGuidedText, VisionGuidedTextConfig, load_pretrained_encoder


def tqdm_maybe(it, **kwargs):
    try:
        from tqdm import tqdm
        return tqdm(it, file=sys.__stdout__, dynamic_ncols=True, leave=True, **kwargs)
    except Exception:
        return it


class Tee:
    def __init__(self, *w):
        self._w = w
    def write(self, d):
        for w in self._w: w.write(d)
        return len(d)
    def flush(self):
        for w in self._w: w.flush()


def setup_log(log_dir, run_tag):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{run_tag}.log")
    f = open(path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, f)
    sys.stderr = Tee(sys.__stderr__, f)
    print(f"[info] logging to: {path}")


# ---------------------------------------------------------------------------
# Eval: point-only → text retrieval  (R@1, R@5, R@10)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: VisionGuidedText,
    test_entries: List[ActionEntry],
    dataset: RadarImageTextDataset,
    device: torch.device,
    max_points: int = 196,
    batch_size: int = 64,
    desc: str = "eval",
) -> Dict[str, float]:
    model.eval()

    # 1. Encode radar (point-only, no vision)
    all_point_embeds, all_query_actions = [], []
    for i in tqdm_maybe(range(len(dataset)), desc=f"{desc} encode-radar", unit="sample"):
        entry = test_entries[i]
        pts_tensor, _, _ = dataset[i]
        n = min(pts_tensor.shape[0], max_points)
        pts = torch.zeros(1, max_points, 5)
        mask = torch.zeros(1, max_points, dtype=torch.bool)
        pts[0, :n] = pts_tensor[:n]
        mask[0, :n] = True
        emb = model.encode_points(pts.to(device), mask.to(device))
        all_point_embeds.append(emb.cpu())
        all_query_actions.append(entry.action)

    point_embeds = torch.cat(all_point_embeds, dim=0)

    # 2. Text gallery (5 sentences per test entry)
    all_texts, all_text_actions = [], []
    for entry in test_entries:
        for sent in entry.sentences:
            all_texts.append(sent)
            all_text_actions.append(entry.action)

    all_text_embeds = []
    for start in tqdm_maybe(range(0, len(all_texts), batch_size), desc=f"{desc} encode-text", unit="batch"):
        emb = model.encode_text(all_texts[start:start + batch_size], device)
        all_text_embeds.append(emb.cpu())
    text_embeds = torch.cat(all_text_embeds, dim=0)

    # 3. R@K
    sims = point_embeds @ text_embeds.T
    n_queries = sims.shape[0]
    recall = {1: 0, 5: 0, 10: 0}
    for i in range(n_queries):
        ranked = sims[i].argsort(descending=True)
        for k in recall:
            if all_query_actions[i] in [all_text_actions[idx] for idx in ranked[:k]]:
                recall[k] += 1

    return {
        "R@1": recall[1] / n_queries, "R@5": recall[5] / n_queries,
        "R@10": recall[10] / n_queries,
        "n_queries": n_queries, "n_gallery": len(all_texts),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Train VisionGuidedText")
    p.add_argument("--actions_json", type=str, required=True)
    p.add_argument("--actions_text_jsonl", type=str, required=True)
    p.add_argument("--mmbody_pc_dir", type=str, required=True)
    p.add_argument("--mmfi_pc_dir", type=str, required=True)
    p.add_argument("--mri_pc_dir", type=str, required=True)
    p.add_argument("--mmbody_img_dir", type=str, required=True)
    p.add_argument("--mmfi_img_dir", type=str, required=True)
    p.add_argument("--mri_img_dir", type=str, required=True)
    p.add_argument("--feature_norm", type=str, default="per_sample_zscore")
    p.add_argument("--max_points", type=int, default=196)
    p.add_argument("--min_points", type=int, default=15)
    p.add_argument("--pretrained_ckpt", type=str, default="")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--out_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--pooling", type=str, default="cls")
    p.add_argument("--proj_dim", type=int, default=512)
    p.add_argument("--aux_weight", type=float, default=0.5,
                   help="Weight for auxiliary point-only loss (0=fused only, 1=point only)")
    p.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"])
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--ckpt_dir", type=str, default="alignment/visionGuidedText/checkpoints")
    p.add_argument("--log_dir", type=str, default="alignment/visionGuidedText/log")
    p.add_argument("--run_tag", type=str, default="")
    return p.parse_args()


def main():
    args = parse_args()
    run_tag = args.run_tag or (
        f"vgt_ep{args.epochs}_bs{args.batch_size}_lr{args.lr}"
        f"_aux{args.aux_weight}_proj{args.proj_dim}"
        f"_D{args.embed_dim}_L{args.num_layers}_H{args.num_heads}"
    )
    setup_log(args.log_dir, run_tag)

    # ---- data ----
    entries = build_entries(args.actions_json, args.actions_text_jsonl)
    train_entries, test_entries = split_entries(entries)
    print(f"[info] entries: total={len(entries)}  train={len(train_entries)}  test={len(test_entries)}")

    ds_kwargs = dict(
        mmbody_pc_dir=args.mmbody_pc_dir, mmfi_pc_dir=args.mmfi_pc_dir, mri_pc_dir=args.mri_pc_dir,
        mmbody_img_dir=args.mmbody_img_dir, mmfi_img_dir=args.mmfi_img_dir, mri_img_dir=args.mri_img_dir,
        feature_norm=args.feature_norm,
    )
    train_ds = RadarImageTextDataset(train_entries, **ds_kwargs, train=True)
    test_ds = RadarImageTextDataset(test_entries, **ds_kwargs, train=False)

    collate = partial(collate_radar_image_text, max_points=args.max_points, min_points=args.min_points)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # ---- model ----
    device = torch.device(args.device)
    cfg = VisionGuidedTextConfig(
        embed_dim=args.embed_dim, out_dim=args.out_dim,
        num_layers=args.num_layers, num_heads=args.num_heads,
        pooling=args.pooling, proj_dim=args.proj_dim,
        clip_model_name=args.clip_model,
        aux_weight=args.aux_weight,
    )
    model = VisionGuidedText(cfg).to(device)

    if args.pretrained_ckpt and os.path.isfile(args.pretrained_ckpt):
        load_pretrained_encoder(model.point_encoder, args.pretrained_ckpt, device=args.device)

    params = [
        {"params": model.point_encoder.parameters(), "lr": args.lr},
        {"params": model.point_proj.parameters(), "lr": args.lr},
        {"params": model.fusion_proj.parameters(), "lr": args.lr},
        {"params": [model.logit_scale], "lr": args.lr},
    ]
    optim = torch.optim.AdamW(params, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == "cosine":
        spe = len(train_loader)
        total = max(1, args.epochs * spe)
        wu = max(0, args.warmup_epochs * spe)
        base, mn = args.lr, args.min_lr

        def lr_mult(step):
            s = float(max(0, min(step, total)))
            if wu > 0 and s < wu:
                return s / float(wu)
            if total <= wu:
                return 1.0
            t = (s - wu) / float(total - wu)
            cos = 0.5 * (1.0 + math.cos(math.pi * t))
            return (mn / base) + (1.0 - mn / base) * cos

        scheduler = LambdaLR(optim, lr_lambda=lr_mult)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[info] params: trainable={n_train:,}  total={n_total:,}")
    print(f"[info] aux_weight={cfg.aux_weight}  device={device}  epochs={args.epochs}")

    # ---- train ----
    best_r10 = 0.0

    for epoch in range(args.epochs):
        model.train()
        model.clip_model.eval()

        pbar = tqdm_maybe(enumerate(train_loader), total=len(train_loader),
                          desc=f"train {epoch+1}/{args.epochs}", unit="batch")

        for step, (pts, mask, images, texts) in pbar:
            if pts.shape[0] == 0:
                continue
            pts = pts.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            out = model(pts, mask, images, texts)
            loss = out["loss"]

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if scheduler:
                scheduler.step()

            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(
                    loss=f"{float(loss):.4f}",
                    fused=f"{float(out['acc_fused']):.2f}",
                    point=f"{float(out['acc_point']):.2f}",
                    lr=f"{optim.param_groups[0]['lr']:.2e}",
                )

        # ---- eval: point-only → text ----
        metrics = evaluate(model, test_entries, test_ds, device,
                           max_points=args.max_points, batch_size=args.eval_batch_size,
                           desc=f"eval {epoch+1}/{args.epochs}")
        print(
            f"\n[epoch {epoch+1}] point→text | "
            f"R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}  "
            f"(queries={metrics['n_queries']}  gallery={metrics['n_gallery']})"
        )

        if metrics["R@10"] > best_r10:
            best_r10 = metrics["R@10"]
            os.makedirs(args.ckpt_dir, exist_ok=True)
            best_path = os.path.join(args.ckpt_dir, f"{run_tag}_best.pt")
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                        "metrics": metrics, "args": vars(args), "config": vars(cfg)}, best_path)
            print(f"[info] new best R@10={best_r10:.4f}, saved: {best_path}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    final_path = os.path.join(args.ckpt_dir, f"{run_tag}_final.pt")
    torch.save({"epoch": args.epochs, "model_state_dict": model.state_dict(),
                "metrics": metrics, "args": vars(args), "config": vars(cfg)}, final_path)
    print(f"[info] saved final: {final_path}")


if __name__ == "__main__":
    main()
