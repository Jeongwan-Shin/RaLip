import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def selected_datasets(value: str):
    value = value.strip()
    if value.lower() in {"all", "*"}:
        return ["MARS", "mRI", "mmFI", "mmBody"]
    return [v.strip() for v in value.split(",") if v.strip()]


def require_dir(path: str, name: str, strict: bool) -> bool:
    if path is None:
        if strict:
            raise ValueError(f"--{name} is required (path is missing).")
        return False
    if not os.path.isdir(path):
        if strict:
            raise FileNotFoundError(f"--{name}='{path}' does not exist or is not a directory.")
        return False
    return True


def tqdm_maybe(it, **kwargs):
    try:
        from tqdm import tqdm  # type: ignore

        # Important: when stdout is tee'd into a log file, tqdm's frequent redraws create huge logs.
        # Route tqdm output to the real console only (not the log file).
        return tqdm(it, file=sys.__stdout__, disable=False, dynamic_ncols=True, leave=True, **kwargs)
    except Exception:
        return it


def collate_pad_mask(
    batch,
    *,
    max_points: int,
    label_dim: int,
    min_points: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points_list, labels_list = zip(*batch)
    bsz = len(points_list)
    feat_dim = int(points_list[0].shape[-1])

    # Filter out tiny point clouds (after dataset-side zero-padding removal)
    min_points = int(min_points)
    if min_points > 1:
        kept = [(p, y) for (p, y) in zip(points_list, labels_list) if int(p.shape[0]) >= min_points]
        if kept:
            points_list, labels_list = zip(*kept)
            bsz = len(points_list)
        else:
            # Return an empty batch (caller should skip)
            pts = torch.zeros((0, max_points, feat_dim), dtype=torch.float32)
            mask = torch.zeros((0, max_points), dtype=torch.bool)
            labels = torch.zeros((0, label_dim), dtype=torch.float32)
            return pts, mask, labels

    pts = torch.zeros((bsz, max_points, feat_dim), dtype=torch.float32)
    mask = torch.zeros((bsz, max_points), dtype=torch.bool)
    labels = torch.zeros((bsz, label_dim), dtype=torch.float32)

    for i, (p, y) in enumerate(zip(points_list, labels_list)):
        if p.ndim != 2:
            raise ValueError(f"points must be (N,F), got {tuple(p.shape)}")
        if p.shape[1] != feat_dim:
            raise ValueError("inconsistent feature dims in batch")

        if p.shape[0] > max_points:
            p = p[:max_points]
        n = int(p.shape[0])
        pts[i, :n] = p
        mask[i, :n] = True

        y = y.reshape(-1).to(torch.float32)
        if int(y.numel()) != label_dim:
            raise ValueError(f"label_dim mismatch: expected {label_dim}, got {int(y.numel())}")
        labels[i] = y

    return pts, mask, labels


def root_relative_pose(y: torch.Tensor, *, label_dim: int) -> torch.Tensor:
    """
    Root-align pose by subtracting pelvis (joint 0) from all joints.
    Assumes label layout is compatible with view(B,3,J).
    """
    if y.ndim != 2:
        raise ValueError(f"y must be (B,label_dim), got {tuple(y.shape)}")
    if label_dim % 3 != 0:
        return y
    joints = label_dim // 3
    y3 = y.view(-1, 3, joints)
    root = y3[:, :, 0].unsqueeze(2)  # (B,3,1)
    y3 = y3 - root
    return y3.reshape(-1, label_dim)


@dataclass
class EvalMetrics:
    mse: float
    mae: float
    mpjpe: float
    mpjpe2: float
    n_samples: int


@torch.no_grad()
def evaluate_pose_regressor(
    model: nn.Module,
    ds,
    *,
    loader: Optional[DataLoader] = None,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_points: int,
    label_dim: int,
    label_mode: str = "absolute",
    min_points: int = 1,
    desc: str,
) -> EvalMetrics:
    model.eval()
    if loader is None:
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda b: collate_pad_mask(b, max_points=max_points, label_dim=label_dim, min_points=min_points),
            pin_memory=device.type == "cuda",
            persistent_workers=(num_workers > 0),
        )

    se_sum = 0.0
    ae_sum = 0.0
    mpjpe_sum = 0.0
    mpjpe2_sum = 0.0
    n_elem = 0
    n_joint = 0
    n_joint2 = 0
    n_samp = 0

    joints = None
    if label_dim % 3 == 0:
        joints = label_dim // 3

    pbar = tqdm_maybe(enumerate(loader), total=len(loader), desc=desc, unit="batch")
    for _, (pts, mask, y) in pbar:
        if int(pts.shape[0]) == 0:
            continue
        pts = pts.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(pts, mask)

        y_used = y
        if label_mode == "root_relative":
            y_used = root_relative_pose(y_used, label_dim=label_dim)

        diff = pred - y_used
        se_sum += float((diff * diff).sum().detach().cpu())
        ae_sum += float(diff.abs().sum().detach().cpu())
        n_elem += int(y_used.numel())
        n_samp += int(y_used.shape[0])

        if joints is not None:
            pred3 = pred.view(-1, 3, joints)
            y3 = y_used.view(-1, 3, joints)
            per_joint = torch.linalg.norm(pred3 - y3, ord=2, dim=1)  # (B,J)
            mpjpe_sum += float(per_joint.sum().detach().cpu())
            n_joint += int(per_joint.numel())

            # MPJPE2: root-aligned (pelvis=joint 0), mirroring Train_mmWave_Encoder's compute_mpjpe()
            pred_root = pred3[:, :, 0].unsqueeze(2)  # (B,3,1)
            y_root = y3[:, :, 0].unsqueeze(2)  # (B,3,1)
            pred_aligned = pred3 - pred_root
            y_aligned = y3 - y_root
            per_joint2 = torch.linalg.norm(pred_aligned - y_aligned, ord=2, dim=1)  # (B,J)
            mpjpe2_sum += float(per_joint2.sum().detach().cpu())
            n_joint2 += int(per_joint2.numel())

    mse = (se_sum / n_elem) if n_elem else float("nan")
    mae = (ae_sum / n_elem) if n_elem else float("nan")
    mpjpe = (mpjpe_sum / n_joint) if n_joint else float("nan")
    mpjpe2 = (mpjpe2_sum / n_joint2) if n_joint2 else float("nan")
    return EvalMetrics(mse=mse, mae=mae, mpjpe=mpjpe, mpjpe2=mpjpe2, n_samples=n_samp)


def save_checkpoint(
    path: str,
    *,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    epoch: int,
    metrics_by_dataset: Dict[str, EvalMetrics],
    args: argparse.Namespace,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "metrics_by_dataset": {k: vars(v) for k, v in metrics_by_dataset.items()},
            "args": vars(args),
        },
        path,
    )


def format_sci(x: float) -> str:
    """Format like 1e-5 (no leading zero in exponent)."""
    if x == 0 or not math.isfinite(x):
        return str(x)
    sign = "-" if x < 0 else ""
    ax = abs(x)
    exp = int(math.floor(math.log10(ax)))
    mant = ax / (10**exp)
    # round mantissa a bit for stable strings
    mant_s = f"{mant:.6g}".rstrip("0").rstrip(".")
    if mant_s == "1":
        return f"{sign}1e{exp}"
    return f"{sign}{mant_s}e{exp}"


def default_run_tag(args: argparse.Namespace) -> str:
    # Match requested style: epochs50_batch_size_32_lr_1e-5_pooling_cls
    lr_s = format_sci(float(args.lr))
    return f"epochs{int(args.epochs)}_batch_size_{int(args.batch_size)}_lr_{lr_s}_pooling_{str(args.pooling)}"


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


def setup_log_file(log_dir: str, run_tag: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{run_tag}.log")
    f = open(path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, f)  # type: ignore[assignment]
    sys.stderr = Tee(sys.__stderr__, f)  # type: ignore[assignment]
    print(f"[info] logging to: {path}")
    return path

