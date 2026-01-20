import argparse
import os
import sys
from typing import Dict, Tuple

import torch

from models.point_transformer import PoseRegressor, PoseRegressorConfig, PointTransformerEncoderConfig
from models.point_transformer_with_poseDec import PoseRegressorPoseDec, PoseRegressorPoseDecConfig
from models.point_transformer_with_crossDec import PoseRegressorCrossDec, PoseRegressorCrossDecConfig
from utils.dataloader import (
    MARS_Dataset,
    mRI_Dataset,
    mmFI_Dataset,
    mmBody_Dataset,
    mmBody_train_test_path,
    train_test_cross_split,
)
from utils.dataset_config import (
    MMFI_CROSS_SUBJECT_TEST,
    MMFI_CROSS_SUBJECT_TRAIN,
    MRI_CROSS_SUBJECT_TEST,
    MRI_CROSS_SUBJECT_TRAIN,
)
from utils.train_utils import EvalMetrics, evaluate_pose_regressor, require_dir, selected_datasets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate all checkpoints in a folder (batch_size=1) and print test metrics.")
    p.add_argument("--ckpt_dir", type=str, default="/workspace/mmWave_pose_estimation/checkpoints")
    p.add_argument("--glob", type=str, default="*.pt", help="Checkpoint glob inside ckpt_dir (default: *.pt)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)

    # dataset paths (if not provided, try to read from checkpoint's saved args)
    p.add_argument("--mars_dir", type=str, default=None)
    p.add_argument("--mri_dir", type=str, default=None)
    p.add_argument("--mmfi_dir", type=str, default=None)
    p.add_argument("--mmbody_dir", type=str, default=None)
    p.add_argument("--datasets", type=str, default="MARS,mRI,mmFI,mmBody", help="CSV or 'all'")
    p.add_argument("--strict", action="store_true", help="Fail if any selected dataset path is missing/invalid")
    p.add_argument(
        "--compare_filter",
        action="store_true",
        help="If set, print both unfiltered(min_points=1) and filtered(min_points from checkpoint/args) metrics.",
    )
    return p.parse_args()


def _build_model_from_args(args_dict: dict, device: torch.device) -> torch.nn.Module:
    decoding = str(args_dict.get("decoding", "original"))
    pooling = str(args_dict.get("pooling", "mean"))
    label_dim = int(args_dict.get("label_dim", 45))

    encoder_cfg = PointTransformerEncoderConfig(
        in_dim=int(args_dict.get("in_dim", 5)),
        embed_dim=int(args_dict.get("embed_dim", 128)),
        out_dim=int(args_dict.get("out_dim", 64)),
        num_layers=int(args_dict.get("num_layers", 2)),
        num_heads=int(args_dict.get("num_heads", 2)),
        mlp_ratio=float(args_dict.get("mlp_ratio", 2.0)),
        dropout=float(args_dict.get("dropout", 0.1)),
        attn_dropout=float(args_dict.get("attn_dropout", args_dict.get("dropout", 0.1))),
        pooling=pooling,
    )

    if decoding == "original":
        model = PoseRegressor(PoseRegressorConfig(encoder=encoder_cfg, label_dim=label_dim)).to(device)
    elif decoding == "poseDec":
        model = PoseRegressorPoseDec(PoseRegressorPoseDecConfig(encoder=encoder_cfg, label_dim=label_dim)).to(device)
    elif decoding == "crossDec":
        model = PoseRegressorCrossDec(
            PoseRegressorCrossDecConfig(
                encoder=encoder_cfg,
                label_dim=label_dim,
                dec_heads=int(args_dict.get("dec_heads", 2)),
                dec_dropout=float(args_dict.get("dec_dropout", 0.1)),
            )
        ).to(device)
    else:
        raise ValueError(f"Unknown decoding={decoding} in checkpoint args")

    return model


def _load_test_datasets(
    selected: list,
    *,
    mars_dir: str | None,
    mri_dir: str | None,
    mmfi_dir: str | None,
    mmbody_dir: str | None,
    strict: bool,
    feature_norm: str,
) -> Dict[str, Tuple[object, object]]:
    datasets: Dict[str, Tuple[object, object]] = {}

    if "MARS" in selected and require_dir(mars_dir, "mars_dir", strict):
        train_path = os.path.join(mars_dir, "train")
        test_path = os.path.join(mars_dir, "test")
        if not (os.path.isdir(train_path) and os.path.isdir(test_path)):
            if strict:
                raise FileNotFoundError(f"MARS expects '{mars_dir}/train' and '{mars_dir}/test'")
        else:
            datasets["MARS"] = (
                MARS_Dataset(train_path, transform=False, train=True, feature_norm=feature_norm),
                MARS_Dataset(test_path, transform=False, train=False, feature_norm=feature_norm),
            )

    if "mRI" in selected and require_dir(mri_dir, "mri_dir", strict):
        mri_train_files, mri_test_files = train_test_cross_split(mri_dir, MRI_CROSS_SUBJECT_TRAIN, MRI_CROSS_SUBJECT_TEST)
        datasets["mRI"] = (
            mRI_Dataset(mri_train_files, transform=False, train=True, feature_norm=feature_norm),
            mRI_Dataset(mri_test_files, transform=False, train=False, feature_norm=feature_norm),
        )

    if "mmFI" in selected and require_dir(mmfi_dir, "mmfi_dir", strict):
        mmfi_train_files, mmfi_test_files = train_test_cross_split(mmfi_dir, MMFI_CROSS_SUBJECT_TRAIN, MMFI_CROSS_SUBJECT_TEST)
        datasets["mmFI"] = (
            mmFI_Dataset(mmfi_train_files, transform=False, train=True, feature_norm=feature_norm),
            mmFI_Dataset(mmfi_test_files, transform=False, train=False, feature_norm=feature_norm),
        )

    if "mmBody" in selected and require_dir(mmbody_dir, "mmbody_dir", strict):
        mmbody_train_files, mmbody_test_files = mmBody_train_test_path(mmbody_dir)
        datasets["mmBody"] = (
            mmBody_Dataset(mmbody_train_files, transform=False, train=True, feature_norm=feature_norm),
            mmBody_Dataset(mmbody_test_files, transform=False, train=False, feature_norm=feature_norm),
        )

    return datasets


def _evaluate_checkpoint(ckpt_path: str, cli_args: argparse.Namespace) -> None:
    device = torch.device(cli_args.device)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    args_dict = dict(ckpt.get("args", {}))
    epoch = ckpt.get("epoch", None)

    # In case the checkpoint predates these options, default to current behavior.
    label_mode = str(args_dict.get("label_mode", "absolute"))
    feature_norm = str(args_dict.get("feature_norm", "none"))
    max_points = int(args_dict.get("max_points", 196))
    label_dim = int(args_dict.get("label_dim", 45))
    min_points = int(args_dict.get("min_points", 15))

    mars_dir = cli_args.mars_dir or args_dict.get("mars_dir")
    mri_dir = cli_args.mri_dir or args_dict.get("mri_dir")
    mmfi_dir = cli_args.mmfi_dir or args_dict.get("mmfi_dir")
    mmbody_dir = cli_args.mmbody_dir or args_dict.get("mmbody_dir")

    selected = selected_datasets(cli_args.datasets)
    datasets = _load_test_datasets(
        selected,
        mars_dir=mars_dir,
        mri_dir=mri_dir,
        mmfi_dir=mmfi_dir,
        mmbody_dir=mmbody_dir,
        strict=bool(cli_args.strict),
        feature_norm=feature_norm,
    )
    if not datasets:
        print("[warn] No datasets loaded for evaluation (check dataset paths).")
        return

    model = _build_model_from_args(args_dict, device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    print("\n" + "=" * 100)
    print(f"[ckpt] {os.path.basename(ckpt_path)}")
    print(f"  epoch={epoch} decoding={args_dict.get('decoding','?')} pooling={args_dict.get('pooling','?')}")
    print(f"  label_mode={label_mode} feature_norm={feature_norm} min_points={min_points} max_points={max_points} label_dim={label_dim}")

    for name, (_, test_split) in datasets.items():
        m: EvalMetrics = evaluate_pose_regressor(
            model,
            test_split,
            device=device,
            batch_size=1,  # requested
            num_workers=int(cli_args.num_workers),
            max_points=max_points,
            label_dim=label_dim,
            label_mode=label_mode,
            min_points=min_points,
            desc=f"eval {name}/test",
        )
        if cli_args.compare_filter:
            u: EvalMetrics = evaluate_pose_regressor(
                model,
                test_split,
                device=device,
                batch_size=1,
                num_workers=int(cli_args.num_workers),
                max_points=max_points,
                label_dim=label_dim,
                label_mode=label_mode,
                min_points=1,
                desc=f"eval {name}/test unfiltered",
            )
            print(
                f"- {name:6s} | "
                f"mpjpe2={m.mpjpe2:.6f} (filtered n={m.n_samples})  "
                f"mpjpe2_unf={u.mpjpe2:.6f} (unf n={u.n_samples})"
            )
        else:
            print(f"- {name:6s} | mpjpe2={m.mpjpe2:.6f}  mpjpe={m.mpjpe:.6f}  mse={m.mse:.6f}  mae={m.mae:.6f}  n={m.n_samples}")


def main() -> None:
    args = parse_args()
    ckpt_dir = str(args.ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"--ckpt_dir='{ckpt_dir}' does not exist")

    # basic glob without importing glob (keeps dependencies minimal)
    pat = str(args.glob)
    if pat.startswith("/"):
        raise ValueError("--glob must be a filename pattern, not an absolute path")
    ckpts = sorted([os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
    if pat not in ("*.pt", ""):
        # very small subset of glob: support '*SUBSTR*' patterns
        s = pat.replace("*", "")
        ckpts = [p for p in ckpts if s in os.path.basename(p)]

    if not ckpts:
        print(f"No checkpoints found in {ckpt_dir} (pattern={args.glob})")
        return

    for ckpt_path in ckpts:
        try:
            _evaluate_checkpoint(ckpt_path, args)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print("\n" + "=" * 100)
            print(f"[ckpt] {os.path.basename(ckpt_path)}  -> ERROR")
            print(f"  {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

