"""
Shared components for alignment_ver2.

Multi-text-type contrastive learning with negative masking:
  - sentences:     standard contrastive (all negatives valid)
  - QA:            mask negatives with same action
  - ActionRec:     mask negatives with same action
  - QA-LimbFocus:  mask negatives with overlapping active body parts

Evaluation:
  - R@K (sentences gallery, same as ver1)
  - QA accuracy (action classification)
  - ActionRec accuracy (action classification)
  - QA-LimbFocus accuracy (body part detection)
"""

from __future__ import annotations

import json, os, random, sys, math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEXT_TYPES = ["sentences", "QA", "ActionRec", "QA-LimbFocus"]
MMFI_FRAMES_PER_ACTION = 297
MRI_SUBJECT_FOLDERS = {
    1: "subject1_0427", 2: "subject2_0428", 3: "subject3_0501",
    4: "subject4_0501", 5: "subject5_0502", 6: "subject6_0503",
    7: "subject7_0503", 8: "subject8_0503", 9: "subject9_0504",
    10: "subject10_0504", 11: "subject11_0505", 12: "subject12_0505",
    13: "subject13_0505", 14: "subject14_0505", 15: "subject15_0508",
    16: "subject16_0508", 17: "subject17_0510", 18: "subject18_0511",
    19: "subject19_0513", 20: "subject20_0513",
}
BODY_PARTS = ["left arm", "right arm", "left leg", "right leg"]


# ---------------------------------------------------------------------------
# Point cloud utils
# ---------------------------------------------------------------------------

def remove_zero_padded_points(pc):
    return pc[~np.all(pc == 0, axis=1)]

def apply_feature_norm(points, mode):
    if mode == "per_sample_zscore":
        xyz = points[:, :3]
        mu = xyz.mean(axis=0, keepdims=True)
        std = np.maximum(xyz.std(axis=0, keepdims=True), 1e-6)
        points = points.copy()
        points[:, :3] = (xyz - mu) / std
    return points


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

@dataclass
class ActionEntry:
    key: str; source: str; mode: str; segment: str
    frame_start: int; frame_end: int
    action: str; body_part: Dict[str, bool]
    sentences: List[str]; qa: str; action_rec: str; qa_limb_focus: List[str]


def build_entries(path: str) -> List[ActionEntry]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            info = obj.get("info", {})
            fr = info.get("frame", "0-0").split("-")
            entries.append(ActionEntry(
                key=obj["key"], source=info.get("from", ""), mode=info.get("mode", "train"),
                segment=info.get("segment", ""),
                frame_start=int(fr[0]), frame_end=int(fr[1]) if len(fr) > 1 else int(fr[0]),
                action=obj["annotation"]["action"], body_part=obj["annotation"]["body_part"],
                sentences=obj["sentences"], qa=obj["QA"],
                action_rec=obj["ActionRec"], qa_limb_focus=obj["QA-LimbFocus"],
            ))
    return entries


def split_entries(entries, seed=42):
    rng = random.Random(seed)
    by_action = {}
    for e in entries:
        by_action.setdefault(e.action, []).append(e)
    train, test = [], []
    for group in by_action.values():
        chosen = rng.choice(group)
        test.append(chosen)
        for e in group:
            if e is not chosen: train.append(e)
    return train, test


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiTextRadarDataset(Dataset):
    def __init__(self, entries, mmbody_pc_dir, mmfi_pc_dir, mri_pc_dir,
                 *, train=True, feature_norm="per_sample_zscore",
                 mmbody_img_dir=None, mmfi_img_dir=None, mri_img_dir=None,
                 return_image=False):
        self.entries = entries
        self.mmbody_pc_dir = mmbody_pc_dir
        self.mmfi_pc_dir = mmfi_pc_dir
        self.mri_pc_dir = mri_pc_dir
        self.mmbody_img_dir = mmbody_img_dir
        self.mmfi_img_dir = mmfi_img_dir
        self.mri_img_dir = mri_img_dir
        self.train = train
        self.feature_norm = feature_norm
        self.return_image = return_image
        self._mmbody_pc_index = {}
        for split in ("train", "test"):
            root = os.path.join(mmbody_pc_dir, split)
            if not os.path.isdir(root): continue
            for dp, _, fns in os.walk(root):
                for fn in fns:
                    if fn.endswith(".npz"):
                        full = os.path.join(dp, fn)
                        self._mmbody_pc_index[os.path.relpath(full, os.path.join(mmbody_pc_dir, split))] = full

    def __len__(self): return len(self.entries)

    def _resolve_pc(self, e, fi):
        if e.source == "mmbody":
            fn = f"frame_{fi}.npz"
            k = os.path.join(e.segment, fn)
            if k in self._mmbody_pc_index: return self._mmbody_pc_index[k]
            sep = "_sequence_"; idx = e.segment.rfind(sep)
            if idx >= 0:
                k2 = os.path.join(e.segment[:idx], "sequence_" + e.segment[idx+len(sep):], fn)
                if k2 in self._mmbody_pc_index: return self._mmbody_pc_index[k2]
            return None
        elif e.source == "mmfi":
            p = e.segment.split("_"); gi = (int(p[2][1:]) - 1) * MMFI_FRAMES_PER_ACTION + fi
            path = os.path.join(self.mmfi_pc_dir, p[1], f"{gi}.npz")
            return path if os.path.isfile(path) else None
        elif e.source == "mri":
            sn = int(e.segment.split("_")[0].replace("subject", ""))
            path = os.path.join(self.mri_pc_dir, f"{sn:02d}", f"{fi}.npz")
            return path if os.path.isfile(path) else None
        return None

    def _resolve_img(self, e, fi):
        if not self.return_image: return None
        if e.source == "mmbody":
            path = os.path.join(self.mmbody_img_dir, e.mode, f"{e.segment}_high", f"frame_{fi:06d}.jpg")
            return path if os.path.isfile(path) else None
        elif e.source == "mmfi":
            p = e.segment.split("_")
            path = os.path.join(self.mmfi_img_dir, p[0], p[1], p[2], "rgb", f"frame{fi+1:03d}.png")
            return path if os.path.isfile(path) else None
        elif e.source == "mri":
            sn = int(e.segment.split("_")[0].replace("subject", ""))
            sf = MRI_SUBJECT_FOLDERS.get(sn)
            if not sf: return None
            path = os.path.join(self.mri_img_dir, sf, f"subject{sn}_color0", f"frame_{fi:06d}.jpg")
            return path if os.path.isfile(path) else None
        return None

    def _load_points(self, path, src):
        d = np.load(path, allow_pickle=True, mmap_mode=None)
        if src == "mmbody": feat = d["feature"].copy()[:, :5]
        elif src == "mmfi": feat = d["feature"].copy(); feat = feat[:, [1,0,2,3,4]]
        elif src == "mri": feat = d["feature"].copy().reshape(-1, 5)
        else: raise ValueError(src)
        feat = remove_zero_padded_points(feat)
        if feat.shape[0] == 0: feat = np.zeros((1, 5), dtype=np.float32)
        return apply_feature_norm(feat, self.feature_norm).astype(np.float32)

    def __getitem__(self, idx):
        e = self.entries[idx]
        fi = random.randint(e.frame_start, e.frame_end) if self.train else (e.frame_start + e.frame_end) // 2
        pc = self._resolve_pc(e, fi); img = self._resolve_img(e, fi) if self.return_image else None
        if pc is None or (self.return_image and img is None):
            for fi2 in range(e.frame_end, e.frame_start - 1, -1):
                pc = self._resolve_pc(e, fi2)
                if self.return_image: img = self._resolve_img(e, fi2)
                if pc and (not self.return_image or img): break
        pts = torch.from_numpy(self._load_points(pc, e.source)) if pc else torch.zeros((1, 5))
        if self.return_image:
            pil = Image.open(img).convert("RGB") if img else Image.new("RGB", (224, 224))
            return pts, pil, idx
        return pts, idx


def collate_multitext(batch, max_points=196, min_points=15, return_image=False):
    if return_image:
        pts_l, imgs_l, idxs = zip(*batch)
    else:
        pts_l, idxs = zip(*batch); imgs_l = None
    kept = [i for i, p in enumerate(pts_l) if p.shape[0] >= min_points]
    if not kept:
        z = torch.zeros((0, max_points, 5)); m = torch.zeros((0, max_points), dtype=torch.bool)
        return (z, m, [], []) if return_image else (z, m, [])
    bsz = len(kept)
    pts = torch.zeros((bsz, max_points, 5)); mask = torch.zeros((bsz, max_points), dtype=torch.bool)
    oi, oimgs = [], []
    for i, ki in enumerate(kept):
        p = pts_l[ki]; n = min(p.shape[0], max_points)
        pts[i, :n] = p[:n]; mask[i, :n] = True
        oi.append(idxs[ki])
        if return_image: oimgs.append(imgs_l[ki])
    return (pts, mask, oimgs, oi) if return_image else (pts, mask, oi)


# ---------------------------------------------------------------------------
# Masked Contrastive Loss
# ---------------------------------------------------------------------------

def build_negative_mask(entries, indices, text_type):
    B = len(indices)
    if text_type == "sentences":
        return None
    if text_type in ("QA", "ActionRec"):
        actions = [entries[i].action for i in indices]
        mask = torch.ones(B, B, dtype=torch.bool)
        for i in range(B):
            for j in range(B):
                if i != j and actions[i] == actions[j]: mask[i, j] = False
        return mask
    if text_type == "QA-LimbFocus":
        bps = []
        for i in indices:
            active = {p for p, v in entries[i].body_part.items() if v}
            bps.append(active if active else {"none"})
        mask = torch.ones(B, B, dtype=torch.bool)
        for i in range(B):
            for j in range(B):
                if i != j and bps[i] & bps[j]: mask[i, j] = False
        return mask
    return None


def masked_contrastive_loss(emb_a, emb_b, logit_scale, neg_mask=None):
    logits = logit_scale * emb_a @ emb_b.T; B = logits.shape[0]
    labels = torch.arange(B, device=logits.device)
    if neg_mask is not None:
        neg_mask = neg_mask.to(logits.device)
        valid = neg_mask | torch.eye(B, dtype=torch.bool, device=logits.device)
        logits_m = logits.masked_fill(~valid, float("-inf"))
    else:
        logits_m = logits
    loss = (F.cross_entropy(logits_m, labels) + F.cross_entropy(logits_m.T, labels)) / 2
    acc = (logits_m.argmax(1) == labels).float().mean()
    return loss, acc


def select_text_for_batch(entries, indices, text_type, train=True):
    texts = []
    for i in indices:
        e = entries[i]
        if text_type == "sentences":
            texts.append(random.choice(e.sentences) if train else e.sentences[0])
        elif text_type == "QA": texts.append(e.qa)
        elif text_type == "ActionRec": texts.append(e.action_rec)
        elif text_type == "QA-LimbFocus":
            texts.append(random.choice(e.qa_limb_focus) if train else e.qa_limb_focus[0])
    return texts


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_all(model, test_entries, all_entries, dataset, device,
                 encode_points_fn, encode_text_fn,
                 max_points=196, batch_size=64, desc="eval"):
    model.eval()

    # --- Encode all test point clouds ---
    all_pe, all_actions = [], []
    for i in tqdm_maybe(range(len(dataset)), desc=f"{desc} enc-radar", unit="s"):
        e = test_entries[i]; pts_t = dataset[i][0]
        n = min(pts_t.shape[0], max_points)
        pts = torch.zeros(1, max_points, 5); mask = torch.zeros(1, max_points, dtype=torch.bool)
        pts[0, :n] = pts_t[:n]; mask[0, :n] = True
        all_pe.append(encode_points_fn(pts.to(device), mask.to(device)).cpu())
        all_actions.append(e.action)
    point_embeds = torch.cat(all_pe, 0)
    nq = point_embeds.shape[0]

    # --- 1. R@K (sentences gallery, 5 per entry) ---
    st, sa = [], []
    for e in test_entries:
        for s in e.sentences: st.append(s); sa.append(e.action)
    se = _enc_batch(st, encode_text_fn, device, batch_size)
    sims = point_embeds @ se.T
    recall = {1: 0, 5: 0, 10: 0}
    for i in range(nq):
        ranked = sims[i].argsort(descending=True)
        for k in recall:
            if all_actions[i] in [sa[idx] for idx in ranked[:k]]: recall[k] += 1

    # --- 2. QA R@K ---
    unique_actions = sorted(set(e.action for e in all_entries))
    qa_cands = [f"Which action is represented by this point cloud? {a}" for a in unique_actions]
    qa_emb = _enc_batch(qa_cands, encode_text_fn, device, batch_size)
    sims_qa = point_embeds @ qa_emb.T
    qa_recall = {1: 0, 5: 0, 10: 0}
    for i in range(nq):
        ranked = sims_qa[i].argsort(descending=True)
        for k in qa_recall:
            if all_actions[i] in [unique_actions[idx] for idx in ranked[:k]]:
                qa_recall[k] += 1

    # --- 3. ActionRec R@K ---
    ar_cands = [f"The action being performed is {a}" for a in unique_actions]
    ar_emb = _enc_batch(ar_cands, encode_text_fn, device, batch_size)
    sims_ar = point_embeds @ ar_emb.T
    ar_recall = {1: 0, 5: 0, 10: 0}
    for i in range(nq):
        ranked = sims_ar[i].argsort(descending=True)
        for k in ar_recall:
            if all_actions[i] in [unique_actions[idx] for idx in ranked[:k]]:
                ar_recall[k] += 1

    # --- 4. QA-LimbFocus accuracy ---
    limb_cands = [f"Which single body part is primarily moving in this action? {bp}" for bp in BODY_PARTS]
    limb_cands.append("Which single body part is primarily moving in this action? none")
    limb_labels = BODY_PARTS + ["none"]
    limb_emb = _enc_batch(limb_cands, encode_text_fn, device, batch_size)
    sims_limb = point_embeds @ limb_emb.T  # (N, 5)
    limb_ok, limb_total = 0, 0
    for i in range(nq):
        e = test_entries[i]
        active = {p for p, v in e.body_part.items() if v}
        if not active: active = {"none"}
        med = sims_limb[i].median().item()
        for j, bp in enumerate(limb_labels):
            gt = bp in active
            pred = sims_limb[i, j].item() > med
            if pred == gt: limb_ok += 1
            limb_total += 1

    return {
        "R@1": recall[1] / nq, "R@5": recall[5] / nq, "R@10": recall[10] / nq,
        "QA@1": qa_recall[1] / nq, "QA@5": qa_recall[5] / nq, "QA@10": qa_recall[10] / nq,
        "AR@1": ar_recall[1] / nq, "AR@5": ar_recall[5] / nq, "AR@10": ar_recall[10] / nq,
        "Limb_acc": limb_ok / limb_total if limb_total else 0.0,
        "n_queries": nq,
    }


def _enc_batch(texts, fn, device, bs):
    e = []
    for i in range(0, len(texts), bs):
        e.append(fn(texts[i:i+bs], device).cpu())
    return torch.cat(e, 0)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class Tee:
    def __init__(self, *w): self._w = w
    def write(self, d):
        for w in self._w: w.write(d)
        return len(d)
    def flush(self):
        for w in self._w: w.flush()

def setup_log(d, t):
    os.makedirs(d, exist_ok=True)
    f = open(os.path.join(d, f"{t}.log"), "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, f); sys.stderr = Tee(sys.__stderr__, f)

def tqdm_maybe(it, **kw):
    try:
        from tqdm import tqdm; return tqdm(it, file=sys.__stdout__, dynamic_ncols=True, leave=True, **kw)
    except: return it
