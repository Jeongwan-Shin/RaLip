"""Evaluate all 16 saved models (8 methods × ver1/ver2) with updated action names."""

import os, sys, json, torch, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_entries, split_entries, MultiTextRadarDataset, evaluate_all, tqdm_maybe

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
    spec = importlib.util.spec_from_file_location(f"_eval_{name}", filepath)
    mod = importlib.util.module_from_spec(spec)
    # Set __name__ to avoid dataclass issues
    mod.__name__ = f"_eval_{name}"
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    base = os.path.dirname(os.path.abspath(__file__))
    text_path = os.path.join(base, "..", "data_generation", "mm_actions_text.jsonl")

    all_entries = build_entries(text_path)
    _, test_entries = split_entries(all_entries)

    test_ds = MultiTextRadarDataset(
        test_entries,
        mmbody_pc_dir="/home/jwshin/2025/JH_folder/dataset/mmBody_pointcloud",
        mmfi_pc_dir="/home/jwshin/2025/JH_folder/dataset/split_mmFI",
        mri_pc_dir="/home/jwshin/2025/JH_folder/dataset/split_mRI",
        train=False,
    )

    results = []
    for method, mtype, _ in METHODS:
        for ver, ckpt_sub in [("v1", "ver1"), ("v2", "")]:
            ckpt_dir = os.path.join(base, method, "checkpoints", ckpt_sub) if ckpt_sub else os.path.join(base, method, "checkpoints")
            # Find best checkpoint
            best_pt = None
            if os.path.isdir(ckpt_dir):
                for f in os.listdir(ckpt_dir):
                    if f.endswith("_best.pt"):
                        best_pt = os.path.join(ckpt_dir, f)
                        break
            if best_pt is None:
                print(f"SKIP {method} {ver}: no checkpoint found in {ckpt_dir}")
                continue

            print(f"\n=== {method} ({ver}) ===")
            model, enc_pt, enc_txt = load_model(method, base, best_pt, device)
            metrics = evaluate_all(model, test_entries, all_entries, test_ds, device,
                                   enc_pt, enc_txt, max_points=196, batch_size=64,
                                   desc=f"{method}/{ver}")
            print(f"  R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}  "
                  f"QA@1={metrics['QA@1']:.4f}  QA@5={metrics['QA@5']:.4f}  QA@10={metrics['QA@10']:.4f}  "
                  f"AR@1={metrics['AR@1']:.4f}  AR@5={metrics['AR@5']:.4f}  AR@10={metrics['AR@10']:.4f}  "
                  f"Limb={metrics['Limb_acc']:.4f}")
            results.append({"method": method, "type": mtype, "ver": ver, **metrics})

            # Free GPU
            del model; torch.cuda.empty_cache()

    # Print summary table
    print("\n" + "="*160)
    print(f"{'#':>2} {'Method':>30} {'Pool':>5} {'Ver':>3} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'QA@1':>6} {'QA@5':>6} {'QA@10':>6} {'AR@1':>6} {'AR@5':>6} {'AR@10':>6} {'Limb':>6}")
    print("-"*160)
    for i, r in enumerate(results, 1):
        print(f"{i:>2} {r['method']:>30} {r['type']:>5} {r['ver']:>3} "
              f"{r['R@1']:>6.4f} {r['R@5']:>6.4f} {r['R@10']:>6.4f} "
              f"{r['QA@1']:>6.4f} {r['QA@5']:>6.4f} {r['QA@10']:>6.4f} "
              f"{r['AR@1']:>6.4f} {r['AR@5']:>6.4f} {r['AR@10']:>6.4f} "
              f"{r['Limb_acc']:>6.4f}")


if __name__ == "__main__":
    main()
