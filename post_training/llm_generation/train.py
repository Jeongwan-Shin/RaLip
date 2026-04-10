"""
Train the projector that bridges radar encoder → LLM.
Evaluates on test set after each epoch.

Usage:
  CUDA_VISIBLE_DEVICES=2 python post_training/llm_generation/train.py --epochs 30
"""

from __future__ import annotations

import argparse, json, os, sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from alignment_ver2.common import (
    build_entries, split_entries, MultiTextRadarDataset, collate_multitext,
)
from post_training.llm_generation.model import RadarLLM, load_token_encoder_from_alignment
from post_training.llm_generation.utils import (
    load_jsonl_index, get_training_text, get_all_actions,
    build_qa_prompt, TASK_PROMPTS, LIMB_CANDIDATES,
)
from post_training.llm_generation.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Train RadarLLM projector")
    parser.add_argument("--actions_jsonl", default="data_generation/mm_actions_text.jsonl")
    parser.add_argument("--actions_gen_jsonl", default="data_generation/mm_actions_text_gen.jsonl")
    parser.add_argument("--alignment_ckpt",
                        default="alignment_ver2/token_level_directTextEnc/checkpoints/ver1/v1_tl_direct_ep100_bs64_best.pt")
    parser.add_argument("--mmbody_pc_dir", default="/workspace/DATA/mmBody_pointcloud")
    parser.add_argument("--mmfi_pc_dir", default="/workspace/DATA/split_mmFI")
    parser.add_argument("--mri_pc_dir", default="/workspace/DATA/split_mRI")
    parser.add_argument("--llm_model", default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_points", type=int, default=196)
    parser.add_argument("--min_points", type=int, default=15)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--ckpt_dir", default="post_training/llm_generation/checkpoints")
    parser.add_argument("--output_dir", default="post_training/llm_generation/output")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load data ----
    print("Loading entries...")
    entries = build_entries(args.actions_jsonl)
    jsonl_index = load_jsonl_index(args.actions_gen_jsonl)
    all_actions = get_all_actions(args.actions_jsonl)
    qa_prompt = build_qa_prompt(all_actions)
    train_entries, test_entries = split_entries(entries)
    print(f"Train: {len(train_entries)}, Test: {len(test_entries)}, Actions: {len(all_actions)}")

    train_ds = MultiTextRadarDataset(
        train_entries, args.mmbody_pc_dir, args.mmfi_pc_dir, args.mri_pc_dir, train=True)
    test_ds = MultiTextRadarDataset(
        test_entries, args.mmbody_pc_dir, args.mmfi_pc_dir, args.mri_pc_dir, train=False)

    def collate_fn(batch):
        return collate_multitext(batch, max_points=args.max_points, min_points=args.min_points)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=0, collate_fn=collate_fn)

    # ---- Build model ----
    print("Loading encoder...")
    encoder = load_token_encoder_from_alignment(args.alignment_ckpt, device=args.device)
    encoder = encoder.to(args.device)

    print("Building RadarLLM...")
    model = RadarLLM(encoder, llm_model_name=args.llm_model, encoder_dim=256)
    model.projector = model.projector.to(args.device)

    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=args.lr, weight_decay=1e-4)

    # ---- Epoch log ----
    epoch_log = []
    best_f1 = -1

    for epoch in range(args.epochs):
        # ---- Train ----
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        model.projector.train()
        total_loss, n_steps = 0, 0

        for batch_idx, (pts, mask, orig_indices) in enumerate(train_loader):
            if pts.shape[0] == 0:
                continue

            pts = pts.to(args.device)
            mask = mask.to(args.device)

            prompts, targets = [], []
            for oi in orig_indices:
                entry = train_entries[oi]
                jsonl_obj = jsonl_index.get(entry.key)
                if jsonl_obj is None:
                    prompts.append(TASK_PROMPTS["ActionRec"])
                    targets.append(entry.action_rec)
                else:
                    p, t = get_training_text(entry, jsonl_obj, qa_prompt)
                    prompts.append(p)
                    targets.append(t)

            prompt_tok = model.tokenizer(prompts, return_tensors="pt",
                                         padding=True, truncation=True, max_length=256)
            target_tok = model.tokenizer(targets, return_tensors="pt",
                                         padding=True, truncation=True, max_length=128)

            prompt_ids = prompt_tok.input_ids.to(args.device)
            prompt_attn = prompt_tok.attention_mask.to(args.device)
            label_ids = target_tok.input_ids.to(args.device)

            loss = model(pts, mask, prompt_ids, prompt_attn, label_ids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1

            if batch_idx % 100 == 0:
                avg = total_loss / n_steps
                print(f"  step {batch_idx}/{len(train_loader)}  loss={loss.item():.4f}  avg={avg:.4f}")

        avg_loss = total_loss / max(n_steps, 1)
        print(f"Train loss: {avg_loss:.4f}")

        # ---- Evaluate ----
        print(f"Evaluating epoch {epoch+1}...")
        metrics = evaluate_model(
            model, test_entries, test_loader, jsonl_index,
            qa_prompt, args.device, args.max_new_tokens,
        )
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = avg_loss
        epoch_log.append(metrics)

        # Print metrics
        print(f"  [Epoch {epoch+1}] train_loss={avg_loss:.4f}")
        for k, v in metrics.items():
            if k in ("epoch", "train_loss", "n_samples"):
                continue
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

        # Save best
        avg_f1 = (metrics.get("sentences_token_F1", 0) +
                  metrics.get("ActionRec_token_F1", 0) +
                  metrics.get("QA_cls_accuracy", 0) +
                  metrics.get("LimbFocus_cls_accuracy", 0)) / 4
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save(model.projector.state_dict(),
                       os.path.join(args.ckpt_dir, "projector_best.pt"))
            print(f"  ** New best (avg_score={avg_f1:.4f}) saved **")

        # Save latest
        torch.save(model.projector.state_dict(),
                   os.path.join(args.ckpt_dir, "projector_latest.pt"))

    # ---- Save epoch log ----
    log_path = os.path.join(args.output_dir, "epoch_log.json")
    with open(log_path, "w") as f:
        json.dump(epoch_log, f, indent=2)
    print(f"\nEpoch log saved to {log_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Ep':>3} | {'Loss':>7} | {'Sent_F1':>7} | {'AR_F1':>7} | {'QA_Acc':>7} | {'Limb_Acc':>8}")
    print(f"{'-'*3}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}")
    for m in epoch_log:
        print(f"{m['epoch']:3d} | {m['train_loss']:7.4f} | "
              f"{m.get('sentences_token_F1',0):7.4f} | "
              f"{m.get('ActionRec_token_F1',0):7.4f} | "
              f"{m.get('QA_cls_accuracy',0):7.4f} | "
              f"{m.get('LimbFocus_cls_accuracy',0):8.4f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
