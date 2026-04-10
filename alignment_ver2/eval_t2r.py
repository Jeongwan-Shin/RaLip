"""
Text-to-Radar retrieval evaluation.

For each test entry, sample 20 frames from the action segment → 20 point clouds.
Query = sentence text, Gallery = all point clouds from all test entries.
Hit if any point cloud from the same action appears in top-K.
"""

import os, sys, json, torch, argparse, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    build_entries, split_entries, MultiTextRadarDataset,
    remove_zero_padded_points, apply_feature_norm, tqdm_maybe,
    MMFI_FRAMES_PER_ACTION, MRI_SUBJECT_FOLDERS,
)

METHODS = [
    ("directTextEnc", "cls", False),
    ("token_level_directTextEnc", "token", False),
    ("ptVision", "cls", False),
    ("token_level_ptVision", "token", False),
    ("imageSpaceMap", "cls", False),
    ("visionGuidedText", "cls", True),
    ("token_level_imageSpaceMap", "token", True),
    ("token_level_visionGuidedText", "token", True),
]


def _import_from(filepath, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"_t2r_{name}", filepath)
    mod = importlib.util.module_from_spec(spec)
    mod.__name__ = f"_t2r_{name}"
    sys.modules[mod.__name__] = mod
    spec.loader.exec_module(mod)
    return mod


def load_model(method, base_dir, ckpt_path, device):
    model_path = os.path.join(base_dir, method, "model.py")
    mod = _import_from(model_path, method.replace("/", "_"))

    if method == "directTextEnc":
        cfg = mod.RadarCLIPConfig(embed_dim=256, out_dim=128, num_layers=4, num_heads=4, proj_dim=512, pooling="cls")
        model = mod.RadarCLIP(cfg)
        enc_pt, enc_txt = model.encode_points, model.encode_text
    elif method == "token_level_directTextEnc":
        cfg = mod.TokenLevelDirectTextConfig(embed_dim=256, num_layers=4, num_heads=4, proj_dim=512, cross_heads=4, aux_weight=0.5)
        model = mod.TokenLevelDirectText(cfg)
        enc_pt, enc_txt = model.encode_points_standalone, model.encode_text_cls
    elif method == "ptVision":
        cfg = mod.PtVisionConfig(embed_dim=256, out_dim=128, num_layers=4, num_heads=4, proj_dim=512, max_points=196)
        model = mod.PtVision(cfg)
        enc_pt, enc_txt = model.encode_points, model.encode_text
    elif method == "token_level_ptVision":
        cfg = mod.TokenLevelPtVisionConfig(embed_dim=256, num_layers=4, num_heads=4, proj_dim=512, cross_heads=4, aux_weight=0.5, max_points=196)
        model = mod.TokenLevelPtVision(cfg)
        enc_pt, enc_txt = model.encode_points_standalone, model.encode_text
    elif method == "imageSpaceMap":
        cfg = mod.RadarCLIPVisionConfig(embed_dim=256, out_dim=128, num_layers=4, num_heads=4, proj_dim=512, pooling="cls")
        model = mod.RadarCLIPVision(cfg)
        enc_pt, enc_txt = model.encode_points, model.encode_text
    elif method == "visionGuidedText":
        cfg = mod.VisionGuidedTextConfig(embed_dim=256, out_dim=128, num_layers=4, num_heads=4, proj_dim=512, aux_weight=0.5)
        model = mod.VisionGuidedText(cfg)
        enc_pt, enc_txt = model.encode_points, model.encode_text
    elif method == "token_level_imageSpaceMap":
        cfg = mod.TokenLevelImageSpaceConfig(embed_dim=256, num_layers=4, num_heads=4, proj_dim=512, cross_heads=4, aux_weight=0.5)
        model = mod.TokenLevelImageSpace(cfg)
        enc_pt, enc_txt = model.encode_points_standalone, model.encode_text_cls
    elif method == "token_level_visionGuidedText":
        cfg = mod.TokenLevelVGTConfig(embed_dim=256, num_layers=4, num_heads=4, proj_dim=512, cross_heads=4, aux_weight=0.5)
        model = mod.TokenLevelVGT(cfg)
        enc_pt, enc_txt = model.encode_points_standalone, model.encode_text_cls
    else:
        raise ValueError(method)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, enc_pt, enc_txt


class PointCloudLoader:
    """Load point clouds from disk for test entries."""

    def __init__(self, mmbody_pc_dir, mmfi_pc_dir, mri_pc_dir, feature_norm="per_sample_zscore"):
        self.mmbody_pc_dir = mmbody_pc_dir
        self.mmfi_pc_dir = mmfi_pc_dir
        self.mri_pc_dir = mri_pc_dir
        self.feature_norm = feature_norm
        # mmBody index
        self._mmbody_idx = {}
        for split in ("train", "test"):
            root = os.path.join(mmbody_pc_dir, split)
            if not os.path.isdir(root): continue
            for dp, _, fns in os.walk(root):
                for fn in fns:
                    if fn.endswith(".npz"):
                        full = os.path.join(dp, fn)
                        self._mmbody_idx[os.path.relpath(full, os.path.join(mmbody_pc_dir, split))] = full

    def resolve(self, entry, frame_idx):
        if entry.source == "mmbody":
            fn = f"frame_{frame_idx}.npz"
            k = os.path.join(entry.segment, fn)
            if k in self._mmbody_idx: return self._mmbody_idx[k]
            sep = "_sequence_"; idx = entry.segment.rfind(sep)
            if idx >= 0:
                k2 = os.path.join(entry.segment[:idx], "sequence_" + entry.segment[idx+len(sep):], fn)
                if k2 in self._mmbody_idx: return self._mmbody_idx[k2]
            return None
        elif entry.source == "mmfi":
            p = entry.segment.split("_")
            gi = (int(p[2][1:]) - 1) * MMFI_FRAMES_PER_ACTION + frame_idx
            path = os.path.join(self.mmfi_pc_dir, p[1], f"{gi}.npz")
            return path if os.path.isfile(path) else None
        elif entry.source == "mri":
            sn = int(entry.segment.split("_")[0].replace("subject", ""))
            path = os.path.join(self.mri_pc_dir, f"{sn:02d}", f"{frame_idx}.npz")
            return path if os.path.isfile(path) else None
        return None

    def load(self, path, source):
        d = np.load(path, allow_pickle=True, mmap_mode=None)
        if source == "mmbody": feat = d["feature"].copy()[:, :5]
        elif source == "mmfi": feat = d["feature"].copy(); feat = feat[:, [1,0,2,3,4]]
        elif source == "mri": feat = d["feature"].copy().reshape(-1, 5)
        else: raise ValueError(source)
        feat = remove_zero_padded_points(feat)
        if feat.shape[0] == 0: feat = np.zeros((1, 5), dtype=np.float32)
        return apply_feature_norm(feat, self.feature_norm).astype(np.float32)

    def sample_frames(self, entry, n=20):
        """Sample up to n valid frames from entry's range."""
        candidates = list(range(entry.frame_start, entry.frame_end + 1))
        random.shuffle(candidates)
        frames = []
        for fi in candidates:
            path = self.resolve(entry, fi)
            if path is not None:
                frames.append((fi, path))
            if len(frames) >= n:
                break
        return frames


@torch.no_grad()
def evaluate_t2r(model, test_entries, device, encode_points_fn, encode_text_fn,
                 pc_loader, max_points=196, n_frames=20, desc="eval"):
    """
    Text-to-Radar retrieval.
    Query = sentences (5 per entry, each tested separately)
    Gallery = 20 point clouds per entry = 110 * 20 = 2200 point clouds
    Hit if any point cloud from the same action appears in top-K.
    """
    model.eval()

    # 1. Build point cloud gallery: 20 frames per entry
    all_pc_embeds = []
    all_pc_actions = []

    for entry in tqdm_maybe(test_entries, desc=f"{desc} build-gallery", unit="entry"):
        frames = pc_loader.sample_frames(entry, n=n_frames)
        for fi, path in frames:
            pts_np = pc_loader.load(path, entry.source)
            pts_t = torch.from_numpy(pts_np)
            n = min(pts_t.shape[0], max_points)
            if n < 15: continue  # skip tiny
            pts = torch.zeros(1, max_points, 5)
            mask = torch.zeros(1, max_points, dtype=torch.bool)
            pts[0, :n] = pts_t[:n]; mask[0, :n] = True
            emb = encode_points_fn(pts.to(device), mask.to(device))
            all_pc_embeds.append(emb.cpu())
            all_pc_actions.append(entry.action)

    pc_embeds = torch.cat(all_pc_embeds, 0)  # (N_gallery, D)
    n_gallery = pc_embeds.shape[0]

    # 2. Build text queries: first sentence per entry
    all_texts = []
    all_text_actions = []
    for entry in test_entries:
        all_texts.append(entry.sentences[0])
        all_text_actions.append(entry.action)

    # Encode texts
    all_text_embeds = []
    for i in range(0, len(all_texts), 64):
        emb = encode_text_fn(all_texts[i:i+64], device)
        all_text_embeds.append(emb.cpu())
    text_embeds = torch.cat(all_text_embeds, 0)  # (N_queries, D)

    # 3. Text → Radar similarity
    sims = text_embeds @ pc_embeds.T  # (N_queries, N_gallery)
    nq = sims.shape[0]

    recall = {1: 0, 5: 0, 10: 0}
    for i in range(nq):
        query_action = all_text_actions[i]
        ranked = sims[i].argsort(descending=True)
        for k in recall:
            top_k_actions = [all_pc_actions[idx] for idx in ranked[:k]]
            if query_action in top_k_actions:
                recall[k] += 1

    return {
        "T2R@1": recall[1] / nq,
        "T2R@5": recall[5] / nq,
        "T2R@10": recall[10] / nq,
        "n_queries": nq,
        "n_gallery": n_gallery,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_frames", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    device = torch.device(args.device)
    random.seed(args.seed)

    base = os.path.dirname(os.path.abspath(__file__))
    text_path = os.path.join(base, "..", "data_generation", "mm_actions_text.jsonl")

    all_entries = build_entries(text_path)
    _, test_entries = split_entries(all_entries)

    pc_loader = PointCloudLoader(
        mmbody_pc_dir="/workspace/DATA/mmBody_pointcloud",
        mmfi_pc_dir="/workspace/DATA/split_mmFI",
        mri_pc_dir="/workspace/DATA/split_mRI",
    )

    results = []
    for method, mtype, _ in METHODS:
        for ver, ckpt_sub in [("v1", "ver1"), ("v2", "")]:
            ckpt_dir = os.path.join(base, method, "checkpoints", ckpt_sub) if ckpt_sub else os.path.join(base, method, "checkpoints")
            best_pt = None
            if os.path.isdir(ckpt_dir):
                for f in os.listdir(ckpt_dir):
                    if f.endswith("_best.pt"):
                        best_pt = os.path.join(ckpt_dir, f); break
            if best_pt is None:
                print(f"SKIP {method} {ver}"); continue

            print(f"\n=== {method} ({ver}) ===")
            model, enc_pt, enc_txt = load_model(method, base, best_pt, device)
            random.seed(args.seed)  # reset seed for consistent gallery
            metrics = evaluate_t2r(model, test_entries, device, enc_pt, enc_txt,
                                   pc_loader, n_frames=args.n_frames, desc=f"{method}/{ver}")
            print(f"  T2R@1={metrics['T2R@1']:.4f}  T2R@5={metrics['T2R@5']:.4f}  T2R@10={metrics['T2R@10']:.4f}  "
                  f"(queries={metrics['n_queries']}  gallery={metrics['n_gallery']})")
            results.append({"method": method, "type": mtype, "ver": ver, **metrics})
            del model; torch.cuda.empty_cache()

    print("\n" + "="*100)
    print(f"{'#':>2} {'Method':>30} {'Pool':>5} {'Ver':>3} {'T2R@1':>7} {'T2R@5':>7} {'T2R@10':>7} {'gallery':>7}")
    print("-"*100)
    for i, r in enumerate(results, 1):
        print(f"{i:>2} {r['method']:>30} {r['type']:>5} {r['ver']:>3} "
              f"{r['T2R@1']:>7.4f} {r['T2R@5']:>7.4f} {r['T2R@10']:>7.4f} {r['n_gallery']:>7}")


if __name__ == "__main__":
    main()
