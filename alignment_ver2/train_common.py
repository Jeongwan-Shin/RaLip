"""
Common training loop for all 8 alignment_ver2 methods.

Each method provides:
  - model construction
  - encode_points_fn(pts, mask) -> emb
  - encode_text_fn(texts, device) -> emb
  - train_step(model, pts, mask, texts, images=None) -> loss_dict
"""

from __future__ import annotations
import argparse, math, os, sys
from functools import partial
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

# Add parent dir for common imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from common import (
    TEXT_TYPES, ActionEntry, MultiTextRadarDataset,
    build_entries, split_entries, collate_multitext,
    build_negative_mask, masked_contrastive_loss, select_text_for_batch,
    evaluate_all, setup_log, tqdm_maybe,
)


def add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--actions_text_jsonl", required=True)
    p.add_argument("--mmbody_pc_dir", required=True)
    p.add_argument("--mmfi_pc_dir", required=True)
    p.add_argument("--mri_pc_dir", required=True)
    p.add_argument("--mmbody_img_dir", default="")
    p.add_argument("--mmfi_img_dir", default="")
    p.add_argument("--mri_img_dir", default="")
    p.add_argument("--pretrained_ckpt", default="")
    p.add_argument("--feature_norm", default="per_sample_zscore")
    p.add_argument("--max_points", type=int, default=196)
    p.add_argument("--min_points", type=int, default=15)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--out_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--proj_dim", type=int, default=512)
    p.add_argument("--cross_heads", type=int, default=4)
    p.add_argument("--aux_weight", type=float, default=0.5)
    p.add_argument("--text_types", type=str, default="all",
                   help="CSV of text types to use: all | sentences,QA,ActionRec,QA-LimbFocus")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--scheduler", default="cosine")
    p.add_argument("--warmup_epochs", type=int, default=50)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--ckpt_dir", default="")
    p.add_argument("--log_dir", default="")
    p.add_argument("--run_tag", default="")
    return p


def run_training(
    args,
    model: torch.nn.Module,
    train_params: List[Dict],
    encode_points_fn: Callable,
    encode_text_fn: Callable,
    train_step_fn: Callable,
    *,
    need_image: bool = False,
    freeze_clip_attr: str = "clip_model",
):
    tag = args.run_tag
    setup_log(args.log_dir, tag)

    # --- text types ---
    if args.text_types.lower() == "all":
        use_text_types = TEXT_TYPES
    else:
        use_text_types = [t.strip() for t in args.text_types.split(",")]
    print(f"[info] text_types: {use_text_types}")

    # --- data ---
    all_entries = build_entries(args.actions_text_jsonl)
    train_entries, test_entries = split_entries(all_entries)
    print(f"[info] entries: total={len(all_entries)} train={len(train_entries)} test={len(test_entries)}")

    ds_kw = dict(mmbody_pc_dir=args.mmbody_pc_dir, mmfi_pc_dir=args.mmfi_pc_dir,
                 mri_pc_dir=args.mri_pc_dir, feature_norm=args.feature_norm)
    if need_image:
        ds_kw.update(mmbody_img_dir=args.mmbody_img_dir, mmfi_img_dir=args.mmfi_img_dir,
                     mri_img_dir=args.mri_img_dir, return_image=True)

    train_ds = MultiTextRadarDataset(train_entries, **ds_kw, train=True)
    test_ds = MultiTextRadarDataset(test_entries, **ds_kw, train=False)

    collate = partial(collate_multitext, max_points=args.max_points,
                      min_points=args.min_points, return_image=need_image)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              pin_memory=True, persistent_workers=(args.num_workers > 0))

    # --- optimizer ---
    device = torch.device(args.device)
    model = model.to(device)
    optim = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == "cosine":
        spe = len(train_loader); total = max(1, args.epochs * spe)
        wu = max(0, args.warmup_epochs * spe)
        base, mn = args.lr, args.min_lr

        def lr_m(s):
            s = float(max(0, min(s, total)))
            if wu > 0 and s < wu: return s / float(wu)
            if total <= wu: return 1.0
            t = (s - wu) / float(total - wu)
            return (mn / base) + (1 - mn / base) * 0.5 * (1 + math.cos(math.pi * t))

        scheduler = LambdaLR(optim, lr_m)

    nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[info] trainable={nt:,}  device={device}  epochs={args.epochs}  bs={args.batch_size}")

    # --- train loop ---
    best_r10 = 0.0
    for epoch in range(args.epochs):
        model.train()
        if hasattr(model, freeze_clip_attr):
            getattr(model, freeze_clip_attr).eval()

        pbar = tqdm_maybe(enumerate(train_loader), total=len(train_loader),
                          desc=f"train {epoch+1}/{args.epochs}", unit="b")

        for step, batch in pbar:
            if need_image:
                pts, mask, images, indices = batch
            else:
                pts, mask, indices = batch
                images = None
            if pts.shape[0] == 0: continue
            pts = pts.to(device); mask = mask.to(device)

            # Random text type
            text_type = random.choice(use_text_types)
            texts = select_text_for_batch(train_entries, indices, text_type, train=True)
            neg_mask = build_negative_mask(train_entries, indices, text_type)

            loss_dict = train_step_fn(model, pts, mask, texts, neg_mask,
                                      images=images, text_type=text_type)
            loss = loss_dict["loss"]

            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
            if scheduler: scheduler.step()

            if hasattr(pbar, "set_postfix"):
                pf = {k: f"{float(v):.4f}" for k, v in loss_dict.items() if k != "loss"}
                pf["loss"] = f"{float(loss):.4f}"
                pf["type"] = text_type[:4]
                pbar.set_postfix(**pf)

        # --- eval ---
        metrics = evaluate_all(model, test_entries, all_entries, test_ds, device,
                               encode_points_fn, encode_text_fn,
                               max_points=args.max_points, batch_size=args.eval_batch_size,
                               desc=f"eval {epoch+1}/{args.epochs}")
        print(
            f"\n[epoch {epoch+1}] "
            f"R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}  "
            f"QA={metrics['QA_acc']:.4f}  AR={metrics['AR_acc']:.4f}  Limb={metrics['Limb_acc']:.4f}"
        )

        if metrics["R@10"] > best_r10:
            best_r10 = metrics["R@10"]; os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(), "metrics": metrics},
                       os.path.join(args.ckpt_dir, f"{tag}_best.pt"))
            print(f"[info] new best R@10={best_r10:.4f}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.save({"epoch": args.epochs, "model_state_dict": model.state_dict(), "metrics": metrics},
               os.path.join(args.ckpt_dir, f"{tag}_final.pt"))
    print(f"[info] saved final")


import random
