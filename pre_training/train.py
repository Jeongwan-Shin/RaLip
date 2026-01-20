import argparse
import os
import sys
import hashlib
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import WeightedRandomSampler

from models.point_transformer import PoseRegressor, PoseRegressorConfig, PointTransformerEncoderConfig
from models.point_transformer_with_poseDec import PoseRegressorPoseDec, PoseRegressorPoseDecConfig
from models.point_transformer_with_crossDec import PoseRegressorCrossDec, PoseRegressorCrossDecConfig
from utils.dataloader import (MARS_Dataset, mRI_Dataset, mmBody_Dataset, mmBody_train_test_path, mmFI_Dataset, train_test_cross_split,)
from utils.train_utils import (EvalMetrics, collate_pad_mask, default_run_tag, evaluate_pose_regressor, require_dir, selected_datasets, setup_log_file, tqdm_maybe, save_checkpoint,)
from utils.train_utils import root_relative_pose
from utils.dataset_config import ( MMFI_CROSS_SUBJECT_TRAIN, MMFI_CROSS_SUBJECT_TEST, MRI_CROSS_SUBJECT_TRAIN, MRI_CROSS_SUBJECT_TEST,)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PointTransformerEncoder on mmWave pose datasets (variable-length points).")
    p.add_argument("--mars_dir", type=str, default=None, help="MARS root containing train/ and test/")
    p.add_argument("--mri_dir", type=str, default=None, help="mRI root containing subject folders with .npz")
    p.add_argument("--mmfi_dir", type=str, default=None, help="mmFI root containing subject folders with .npz")
    p.add_argument("--mmbody_dir", type=str, default=None, help="mmBody root containing train/ and test/")
    p.add_argument("--datasets", type=str, default="MARS,mRI,mmFI,mmBody", help="CSV: MARS,mRI,mmFI,mmBody (or 'all')")
    p.add_argument("--transform", action="store_true", help="Use dataset transform=True for train split")
    p.add_argument("--strict", action="store_true", help="Fail fast if selected dataset path is missing/invalid")

    # training
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_points", type=int, default=196, help="Pad/truncate all point clouds to this length")
    p.add_argument("--min_points", type=int, default=15, help="Drop samples with fewer than this many valid points (train+eval).")
    p.add_argument("--label_dim", type=int, default=45, help="Regression target dimension (default: 3*15)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for evaluation (default: 1)")
    p.add_argument(
        "--eval_compare_filter",
        action="store_true",
        help="If set, evaluation prints both unfiltered(min_points=1) and filtered(min_points=--min_points) metrics.",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=0, help="Debug: stop after N steps per epoch (0=full epoch)")
    p.add_argument(
        "--sampling",
        type=str,
        default="proportional",
        choices=["proportional", "balanced"],
        help="How to sample across datasets when using ConcatDataset. proportional=by dataset size (default). balanced=equalize datasets via WeightedRandomSampler.",
    )
    p.add_argument(
        "--dataset_weights",
        type=str,
        default="",
        help="Optional CSV weights matching selected dataset order (e.g. '1,1,1,1'). Only used when --sampling=balanced.",
    )
    p.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "cosine"],
        help="LR scheduler. cosine = linear warmup then cosine decay over total training steps.",
    )
    p.add_argument("--warmup_epochs", type=int, default=0, help="(cosine) warmup epochs (linear ramp to base lr)")
    p.add_argument("--min_lr", type=float, default=1e-6, help="(cosine) minimum LR at the end of schedule")
    p.add_argument("--ckpt_dir", type=str, default="/workspace/mmWave_pose_estimation/checkpoints", help="Where to save checkpoints")
    p.add_argument("--log_dir", type=str, default="/workspace/mmWave_pose_estimation/log", help="Where to save training logs")
    p.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional run tag for log/checkpoint filenames. If empty, a tag is derived from key args.",
    )
    # model capacity
    p.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "cls"], help="Encoder pooling method")
    p.add_argument("--embed_dim", type=int, default=128, help="Transformer embed dim")
    p.add_argument("--out_dim", type=int, default=64, help="Encoder output dim (after projection)")
    p.add_argument("--num_layers", type=int, default=2, help="Transformer encoder layers")
    p.add_argument("--num_heads", type=int, default=2, help="Transformer attention heads")
    p.add_argument("--dropout", type=float, default=0.1, help="Transformer dropout")
    p.add_argument("--mlp_ratio", type=float, default=2.0, help="Transformer FFN expansion ratio")
    p.add_argument("--dec_heads", type=int, default=2, help="(crossDec) cross-attention heads")
    p.add_argument("--dec_dropout", type=float, default=0.1, help="(crossDec) decoder dropout")
    p.add_argument(
        "--decoding",
        type=str,
        default="original",
        choices=["original", "poseDec", "crossDec"],
        help="Model head/decoding type: original (single head), poseDec (separate x/y/z heads), crossDec (cross-attn decoder).",
    )
    p.add_argument(
        "--label_mode",
        type=str,
        default="absolute",
        choices=["absolute", "root_relative"],
        help="Label mode for training/eval loss: absolute vs root_relative (subtract pelvis joint 0).",
    )
    p.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "l1", "smoothl1"],
        help="Training loss function (robust losses help with label outliers).",
    )
    p.add_argument(
        "--feature_norm",
        type=str,
        default="none",
        choices=["none", "per_sample_zscore"],
        help="Optional feature normalization applied in dataset __getitem__.",
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()
    selected = selected_datasets(args.datasets)

    run_tag = str(args.run_tag).strip() or default_run_tag(args)
    setup_log_file(str(args.log_dir), run_tag)

    datasets = {}

    if "MARS" in selected and require_dir(args.mars_dir, "mars_dir", args.strict):
        train_path = os.path.join(args.mars_dir, "train")
        test_path = os.path.join(args.mars_dir, "test")
        if not (os.path.isdir(train_path) and os.path.isdir(test_path)):
            if args.strict:
                raise FileNotFoundError(f"MARS expects '{args.mars_dir}/train' and '{args.mars_dir}/test'")
        else:
            mars_train_dataset = MARS_Dataset(train_path, transform=args.transform, train=True, feature_norm=args.feature_norm)
            mars_test_dataset = MARS_Dataset(test_path, transform=False, train=False, feature_norm=args.feature_norm)
            datasets["MARS"] = (mars_train_dataset, mars_test_dataset)

    if "mRI" in selected and require_dir(args.mri_dir, "mri_dir", args.strict):
        mri_train_files, mri_test_files = train_test_cross_split(args.mri_dir, MRI_CROSS_SUBJECT_TRAIN, MRI_CROSS_SUBJECT_TEST)
        mri_train_dataset = mRI_Dataset(mri_train_files, transform=args.transform, train=True, feature_norm=args.feature_norm)
        mri_test_dataset = mRI_Dataset(mri_test_files, transform=False, train=False, feature_norm=args.feature_norm)
        datasets["mRI"] = (mri_train_dataset, mri_test_dataset)

    if "mmFI" in selected and require_dir(args.mmfi_dir, "mmfi_dir", args.strict):
        mmfi_train_files, mmfi_test_files = train_test_cross_split(args.mmfi_dir, MMFI_CROSS_SUBJECT_TRAIN, MMFI_CROSS_SUBJECT_TEST)
        mmfi_train_dataset = mmFI_Dataset(mmfi_train_files, transform=args.transform, train=True, feature_norm=args.feature_norm)
        mmfi_test_dataset = mmFI_Dataset(mmfi_test_files, transform=False, train=False, feature_norm=args.feature_norm)
        datasets["mmFI"] = (mmfi_train_dataset, mmfi_test_dataset)

    if "mmBody" in selected and require_dir(args.mmbody_dir, "mmbody_dir", args.strict):
        mmbody_train_files, mmbody_test_files = mmBody_train_test_path(args.mmbody_dir)
        mmbody_train_dataset = mmBody_Dataset(mmbody_train_files, transform=args.transform, train=True, feature_norm=args.feature_norm)
        mmbody_test_dataset = mmBody_Dataset(mmbody_test_files, transform=False, train=False, feature_norm=args.feature_norm)
        datasets["mmBody"] = (mmbody_train_dataset, mmbody_test_dataset)

    if not datasets:
        print("No datasets loaded. Provide paths or use --strict to see errors.")
        return

    # Build concat datasets
    dataset_names = list(datasets.keys())
    train_splits = [datasets[name][0] for name in dataset_names]
    test_splits = [datasets[name][1] for name in dataset_names]
    train_ds = ConcatDataset(train_splits) if len(train_splits) > 1 else train_splits[0]
    test_ds = ConcatDataset(test_splits) if len(test_splits) > 1 else test_splits[0]

    collate = lambda b: collate_pad_mask(b, max_points=args.max_points, label_dim=args.label_dim, min_points=args.min_points)
    # NOTE: sortish (length pre-scan + length-aware batch sampling) removed to reduce startup IO/CPU.
    # This makes the first run much lighter, at the cost of more padding.
    train_sampler = None
    if args.sampling == "balanced" and isinstance(train_ds, ConcatDataset):
        # Default: inverse-frequency so each dataset contributes equally in expectation.
        # weights are per-sample; total size = sum(len(ds_i))
        per_dataset_weights: List[float]
        if str(args.dataset_weights).strip():
            parts = [p.strip() for p in str(args.dataset_weights).split(",") if p.strip()]
            if len(parts) != len(train_splits):
                raise ValueError(
                    f"--dataset_weights must have {len(train_splits)} values (matching selected dataset order {dataset_names}), got {len(parts)}"
                )
            per_dataset_weights = [float(p) for p in parts]
        else:
            per_dataset_weights = [1.0 for _ in train_splits]

        weights = []
        for w_ds, ds_i in zip(per_dataset_weights, train_splits):
            n = len(ds_i)
            if n <= 0:
                continue
            # Each dataset gets total mass proportional to w_ds
            weights.extend([float(w_ds) / float(n)] * n)
        if len(weights) != len(train_ds):
            raise RuntimeError("Internal error: weights length mismatch for ConcatDataset.")
        train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(args.device.startswith("cuda")),
        persistent_workers=(args.num_workers > 0),
    )

    # Reuse eval DataLoaders across epochs to avoid respawning workers / repeated IO setup.
    eval_loaders: Dict[str, DataLoader] = {}
    for name, (_, test_split) in datasets.items():
        eval_loaders[name] = DataLoader(
            test_split,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=(args.device.startswith("cuda")),
            persistent_workers=(args.num_workers > 0),
        )

    device = torch.device(args.device)
    encoder_cfg = PointTransformerEncoderConfig(
        in_dim=5,
        embed_dim=int(args.embed_dim),
        out_dim=int(args.out_dim),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        mlp_ratio=float(args.mlp_ratio),
        dropout=float(args.dropout),
        attn_dropout=float(args.dropout),
        pooling=args.pooling,
    )
    print(
        "[info] PointTransformerEncoderConfig: "
        f"in_dim={encoder_cfg.in_dim} embed_dim={encoder_cfg.embed_dim} out_dim={encoder_cfg.out_dim} "
        f"num_layers={encoder_cfg.num_layers} num_heads={encoder_cfg.num_heads} mlp_ratio={encoder_cfg.mlp_ratio} "
        f"dropout={encoder_cfg.dropout} attn_dropout={encoder_cfg.attn_dropout} pooling={encoder_cfg.pooling}"
    )

    if args.decoding == "original":
        model = PoseRegressor(PoseRegressorConfig(encoder=encoder_cfg, label_dim=args.label_dim)).to(device)
    elif args.decoding == "poseDec":
        model = PoseRegressorPoseDec(PoseRegressorPoseDecConfig(encoder=encoder_cfg, label_dim=args.label_dim)).to(device)
    elif args.decoding == "crossDec":
        model = PoseRegressorCrossDec(
            PoseRegressorCrossDecConfig(
                encoder=encoder_cfg,
                label_dim=args.label_dim,
                dec_heads=int(args.dec_heads),
                dec_dropout=float(args.dec_dropout),
            )
        ).to(device)
    else:
        raise ValueError(f"Unknown --decoding={args.decoding}")
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.loss == "mse":
        criterion = nn.MSELoss()
    elif args.loss == "l1":
        criterion = nn.L1Loss()
    elif args.loss == "smoothl1":
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown --loss={args.loss}")

    # Scheduler (step-based)
    scheduler = None
    if args.scheduler == "cosine":
        steps_per_epoch = int(args.max_steps) if int(args.max_steps) > 0 else int(len(train_loader))
        total_steps = max(1, int(args.epochs) * steps_per_epoch)
        warmup_steps = max(0, int(args.warmup_epochs) * steps_per_epoch)
        base_lr = float(args.lr)
        min_lr = float(args.min_lr)
        if min_lr < 0:
            raise ValueError("--min_lr must be >= 0")
        if min_lr > base_lr:
            raise ValueError("--min_lr must be <= --lr")

        def lr_mult(step: int) -> float:
            # step is 0-based, called after scheduler.step(); clamp to [0, total_steps]
            s = float(max(0, min(step, total_steps)))
            if warmup_steps > 0 and s < warmup_steps:
                return s / float(warmup_steps)
            # cosine decay from 1 -> min_lr/base_lr
            if total_steps <= warmup_steps:
                return 1.0
            t = (s - warmup_steps) / float(total_steps - warmup_steps)  # 0..1
            cos = 0.5 * (1.0 + np.cos(np.pi * t))
            min_mult = (min_lr / base_lr) if base_lr > 0 else 0.0
            return min_mult + (1.0 - min_mult) * cos

        scheduler = LambdaLR(optim, lr_lambda=lambda step: lr_mult(step))
    elif args.scheduler == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown --scheduler={args.scheduler}")

    print(f"[info] device={device} max_points={args.max_points} label_dim={args.label_dim} train_batches={len(train_loader)}")

    for epoch in range(args.epochs):
        model.train()

        pbar = tqdm_maybe(enumerate(train_loader), total=len(train_loader), desc=f"train epoch {epoch+1}/{args.epochs}", unit="batch")
        for step, (pts, mask, y) in pbar:
            if int(pts.shape[0]) == 0:
                continue
            pts = pts.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            pred = model(pts, mask)
            y_used = y
            if args.label_mode == "root_relative":
                y_used = root_relative_pose(y_used, label_dim=args.label_dim)
            loss = criterion(pred, y_used)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if scheduler is not None:
                scheduler.step()

            if hasattr(pbar, "set_postfix"):
                lr_now = float(optim.param_groups[0]["lr"])
                pbar.set_postfix(loss=float(loss.detach().cpu()), lr=lr_now)

            if args.max_steps and (step + 1) >= args.max_steps:
                break

        # Per-dataset evaluation (test split) each epoch
        metrics_by_dataset: Dict[str, EvalMetrics] = {}
        metrics_by_dataset_unfiltered: Dict[str, EvalMetrics] = {}
        for name, (_, test_split) in datasets.items():
            metrics_by_dataset[name] = evaluate_pose_regressor(
                model,
                test_split,
                loader=eval_loaders[name],
                device=device,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                max_points=args.max_points,
                label_dim=args.label_dim,
                label_mode=args.label_mode,
                min_points=args.min_points,
                desc=f"eval {name}/test (epoch {epoch+1})",
            )
            if args.eval_compare_filter:
                metrics_by_dataset_unfiltered[name] = evaluate_pose_regressor(
                    model,
                    test_split,
                    device=device,
                    batch_size=args.eval_batch_size,
                    num_workers=args.num_workers,
                    max_points=args.max_points,
                    label_dim=args.label_dim,
                    label_mode=args.label_mode,
                    min_points=1,
                    desc=f"eval {name}/test unfiltered (epoch {epoch+1})",
                )

        print(f"\n[epoch {epoch+1}] per-dataset metrics (test):")
        for name, m in metrics_by_dataset.items():
            if args.eval_compare_filter and name in metrics_by_dataset_unfiltered:
                u = metrics_by_dataset_unfiltered[name]
                print(
                    f"- {name:6s} | "
                    f"mpjpe2={m.mpjpe2:.6f} (filtered n={m.n_samples})  "
                    f"mpjpe2_unf={u.mpjpe2:.6f} (unf n={u.n_samples})"
                )
            else:
                print(
                    f"- {name:6s} | mpjpe2={m.mpjpe2:.6f}  mpjpe={m.mpjpe:.6f}  mse={m.mse:.6f}  mae={m.mae:.6f}  n={m.n_samples}"
                )

    # Save ONE final checkpoint
    ckpt_dir = str(args.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    final_path = os.path.join(ckpt_dir, f"{run_tag}.pt")
    save_checkpoint(
        final_path,
        model=model,
        optim=optim,
        epoch=int(args.epochs),
        metrics_by_dataset=metrics_by_dataset,
        args=args,
    )
    print(f"[info] saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()