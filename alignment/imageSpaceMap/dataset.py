"""
Radar-Image pair dataset for contrastive learning (image-space alignment).

Each sample pairs a single radar point-cloud frame with its corresponding
RGB image for the same frame.

Image paths:
  mmBody : {img_dir}/{train|test}/{segment}_high/frame_{i:06d}.jpg
  mmFI   : {img_dir}/{Env}/{Subject}/{Action}/rgb/frame{i+1:03d}.png   (1-indexed)
  mRI    : {img_dir}/subject{N}_{date}/subject{N}_color0/frame_{i:06d}.jpg
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# mRI subject number → folder name mapping  (subject1 → subject1_0427, etc.)
# ---------------------------------------------------------------------------

MRI_SUBJECT_FOLDERS: Dict[int, str] = {
    1: "subject1_0427",   2: "subject2_0428",
    3: "subject3_0501",   4: "subject4_0501",
    5: "subject5_0502",   6: "subject6_0503",
    7: "subject7_0503",   8: "subject8_0503",
    9: "subject9_0504",  10: "subject10_0504",
   11: "subject11_0505", 12: "subject12_0505",
   13: "subject13_0505", 14: "subject14_0505",
   15: "subject15_0508", 16: "subject16_0508",
   17: "subject17_0510", 18: "subject18_0511",
   19: "subject19_0513", 20: "subject20_0513",
}


# ---------------------------------------------------------------------------
# Feature pre-processing (same as directTextEnc/dataset.py)
# ---------------------------------------------------------------------------

def remove_zero_padded_points(pc: np.ndarray) -> np.ndarray:
    return pc[~np.all(pc == 0, axis=1)]


def apply_feature_norm(points: np.ndarray, mode: str) -> np.ndarray:
    if mode == "per_sample_zscore":
        xyz = points[:, :3]
        mu = xyz.mean(axis=0, keepdims=True)
        std = np.maximum(xyz.std(axis=0, keepdims=True), 1e-6)
        points = points.copy()
        points[:, :3] = (xyz - mu) / std
    return points


# ---------------------------------------------------------------------------
# Entry data-class
# ---------------------------------------------------------------------------

@dataclass
class ActionEntry:
    key: str
    source: str          # mmbody | mmfi | mri
    mode: str            # train | test (dataset split)
    segment: str
    frame_start: int
    frame_end: int
    action: str
    body_part: Dict[str, Any]


# ---------------------------------------------------------------------------
# Build entries from mm_actions.json
# ---------------------------------------------------------------------------

def _action_key(item: Dict[str, Any]) -> str:
    info = item.get("info") or {}
    frames = item.get("frames") or {}
    segment = info.get("segment", "unknown_segment")
    start = frames.get("start", "na")
    end = frames.get("end", "na")
    action = item.get("action", "unknown_action")
    return f"{segment}:{start}-{end}:{action}"


def build_entries(actions_json_path: str) -> List[ActionEntry]:
    with open(actions_json_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    entries: List[ActionEntry] = []
    for item in raw_items:
        info = item.get("info", {})
        frames = item.get("frames", {})
        entries.append(ActionEntry(
            key=_action_key(item),
            source=info.get("from", "unknown"),
            mode=info.get("mode", "train"),
            segment=info.get("segment", ""),
            frame_start=int(frames.get("start", 0)),
            frame_end=int(frames.get("end", 0)),
            action=item.get("action", ""),
            body_part=item.get("body_part", {}),
        ))
    return entries


def split_entries(
    entries: List[ActionEntry],
    seed: int = 42,
) -> Tuple[List[ActionEntry], List[ActionEntry]]:
    """Pick exactly 1 entry per unique action for test, rest go to train."""
    rng = random.Random(seed)
    by_action: Dict[str, List[ActionEntry]] = {}
    for e in entries:
        by_action.setdefault(e.action, []).append(e)

    train, test = [], []
    for action, group in by_action.items():
        chosen = rng.choice(group)
        test.append(chosen)
        for e in group:
            if e is not chosen:
                train.append(e)
    return train, test


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

MMFI_FRAMES_PER_ACTION = 297


class RadarImageDataset(Dataset):
    """
    Each __getitem__ returns (point_cloud: Tensor[N,5], image: PIL.Image).

    Training: random frame from action segment.
    Eval: middle frame (deterministic).
    """

    def __init__(
        self,
        entries: List[ActionEntry],
        # point cloud dirs
        mmbody_pc_dir: str,
        mmfi_pc_dir: str,
        mri_pc_dir: str,
        # image dirs
        mmbody_img_dir: str,
        mmfi_img_dir: str,
        mri_img_dir: str,
        *,
        train: bool = True,
        feature_norm: str = "per_sample_zscore",
    ):
        self.entries = entries
        self.mmbody_pc_dir = mmbody_pc_dir
        self.mmfi_pc_dir = mmfi_pc_dir
        self.mri_pc_dir = mri_pc_dir
        self.mmbody_img_dir = mmbody_img_dir
        self.mmfi_img_dir = mmfi_img_dir
        self.mri_img_dir = mri_img_dir
        self.train = train
        self.feature_norm = feature_norm

        # Pre-build mmBody point-cloud index (same as directTextEnc)
        self._mmbody_pc_index: Dict[str, str] = {}
        for split in ("train", "test"):
            root = os.path.join(mmbody_pc_dir, split)
            if not os.path.isdir(root):
                continue
            for dirpath, _, fnames in os.walk(root):
                for fn in fnames:
                    if fn.endswith(".npz"):
                        full = os.path.join(dirpath, fn)
                        rel = os.path.relpath(full, os.path.join(mmbody_pc_dir, split))
                        self._mmbody_pc_index[rel] = full

    def __len__(self) -> int:
        return len(self.entries)

    # ---- point cloud resolution ----

    def _resolve_pc(self, entry: ActionEntry, frame_idx: int) -> Optional[str]:
        if entry.source == "mmbody":
            fname = f"frame_{frame_idx}.npz"
            key = os.path.join(entry.segment, fname)
            if key in self._mmbody_pc_index:
                return self._mmbody_pc_index[key]
            # test: "furnished_sequence_0" → "furnished/sequence_0"
            sep = "_sequence_"
            idx = entry.segment.rfind(sep)
            if idx >= 0:
                env = entry.segment[:idx]
                seq = "sequence_" + entry.segment[idx + len(sep):]
                key2 = os.path.join(env, seq, fname)
                if key2 in self._mmbody_pc_index:
                    return self._mmbody_pc_index[key2]
            return None

        elif entry.source == "mmfi":
            parts = entry.segment.split("_")
            subject = parts[1]
            action_num = int(parts[2][1:])
            global_idx = (action_num - 1) * MMFI_FRAMES_PER_ACTION + frame_idx
            path = os.path.join(self.mmfi_pc_dir, subject, f"{global_idx}.npz")
            return path if os.path.isfile(path) else None

        elif entry.source == "mri":
            subj_str = entry.segment.split("_")[0]
            subj_num = int(subj_str.replace("subject", ""))
            path = os.path.join(self.mri_pc_dir, f"{subj_num:02d}", f"{frame_idx}.npz")
            return path if os.path.isfile(path) else None

        return None

    # ---- image resolution ----

    def _resolve_img(self, entry: ActionEntry, frame_idx: int) -> Optional[str]:
        if entry.source == "mmbody":
            # {img_dir}/{train|test}/{segment}_high/frame_{i:06d}.jpg
            if entry.mode == "train":
                folder = f"{entry.segment}_high"
            else:
                folder = f"{entry.segment}_high"
            path = os.path.join(
                self.mmbody_img_dir, entry.mode, folder,
                f"frame_{frame_idx:06d}.jpg",
            )
            return path if os.path.isfile(path) else None

        elif entry.source == "mmfi":
            # {img_dir}/{Env}/{Subject}/{Action}/rgb/frame{i+1:03d}.png  (1-indexed)
            parts = entry.segment.split("_")
            env = parts[0]       # "E01"
            subject = parts[1]   # "S01"
            action = parts[2]    # "A02"
            img_idx = frame_idx + 1  # 1-indexed
            path = os.path.join(
                self.mmfi_img_dir, env, subject, action, "rgb",
                f"frame{img_idx:03d}.png",
            )
            return path if os.path.isfile(path) else None

        elif entry.source == "mri":
            # {img_dir}/subject{N}_{date}/subject{N}_color0/frame_{i:06d}.jpg
            subj_str = entry.segment.split("_")[0]
            subj_num = int(subj_str.replace("subject", ""))
            subj_folder = MRI_SUBJECT_FOLDERS.get(subj_num)
            if subj_folder is None:
                return None
            path = os.path.join(
                self.mri_img_dir, subj_folder,
                f"subject{subj_num}_color0",
                f"frame_{frame_idx:06d}.jpg",
            )
            return path if os.path.isfile(path) else None

        return None

    # ---- feature loading ----

    def _load_points(self, file_path: str, source: str) -> np.ndarray:
        data = np.load(file_path, allow_pickle=True, mmap_mode=None)
        if source == "mmbody":
            feature = data["feature"].copy()[:, :5]
        elif source == "mmfi":
            feature = data["feature"].copy()
            feature = feature[:, [1, 0, 2, 3, 4]]
        elif source == "mri":
            feature = data["feature"].copy().reshape(-1, 5)
        else:
            raise ValueError(f"Unknown source: {source}")
        feature = remove_zero_padded_points(feature)
        if feature.shape[0] == 0:
            feature = np.zeros((1, 5), dtype=np.float32)
        feature = apply_feature_norm(feature, self.feature_norm)
        return feature.astype(np.float32)

    # ---- __getitem__ ----

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Image.Image]:
        entry = self.entries[idx]

        # Select frame
        if self.train:
            frame_idx = random.randint(entry.frame_start, entry.frame_end)
        else:
            frame_idx = (entry.frame_start + entry.frame_end) // 2

        # Try to find a valid frame (both pc and image exist)
        pc_path = self._resolve_pc(entry, frame_idx)
        img_path = self._resolve_img(entry, frame_idx)

        if pc_path is None or img_path is None:
            # Search backwards for a valid frame
            for fi in range(entry.frame_end, entry.frame_start - 1, -1):
                pc_path = self._resolve_pc(entry, fi)
                img_path = self._resolve_img(entry, fi)
                if pc_path is not None and img_path is not None:
                    break

        if pc_path is None or img_path is None:
            # Fallback: dummy data
            dummy_pts = torch.zeros((1, 5), dtype=torch.float32)
            dummy_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
            return dummy_pts, dummy_img

        points = self._load_points(pc_path, entry.source)
        image = Image.open(img_path).convert("RGB")

        return torch.from_numpy(points), image


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_radar_image(
    batch: List[Tuple[torch.Tensor, Image.Image]],
    max_points: int = 196,
    min_points: int = 15,
) -> Tuple[torch.Tensor, torch.Tensor, List[Image.Image]]:
    """
    Returns:
      pts:    (B, max_points, 5)
      mask:   (B, max_points) bool
      images: list[PIL.Image] of length B
    """
    points_list, images_list = zip(*batch)

    kept = [
        (p, img) for p, img in zip(points_list, images_list)
        if p.shape[0] >= min_points
    ]
    if not kept:
        pts = torch.zeros((0, max_points, 5), dtype=torch.float32)
        mask = torch.zeros((0, max_points), dtype=torch.bool)
        return pts, mask, []

    points_list, images_list = zip(*kept)
    bsz = len(points_list)

    pts = torch.zeros((bsz, max_points, 5), dtype=torch.float32)
    mask = torch.zeros((bsz, max_points), dtype=torch.bool)

    for i, p in enumerate(points_list):
        n = min(int(p.shape[0]), max_points)
        pts[i, :n] = p[:n]
        mask[i, :n] = True

    return pts, mask, list(images_list)
