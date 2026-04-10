"""
Evaluate RadarLLM: generate text + classification for test set.

Tasks:
  - sentences:    Free generation → token F1 against GT
  - ActionRec:    Free generation → token F1 against GT
  - QA:           Classification → pick one from all actions
  - QA-LimbFocus: Classification → pick one from [left arm, right arm, left leg, right leg, none]

Usage:
  CUDA_VISIBLE_DEVICES=2 python post_training/llm_generation/evaluate.py \
    --projector_ckpt post_training/llm_generation/checkpoints/projector_best.pt
"""

from __future__ import annotations

import argparse, json, os, sys
from typing import List

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from alignment_ver2.common import (
    build_entries, split_entries, MultiTextRadarDataset, collate_multitext,
)
from post_training.llm_generation.model import RadarLLM, load_token_encoder_from_alignment
from post_training.llm_generation.utils import (
    load_jsonl_index, get_ground_truths, get_all_actions,
    build_qa_prompt, TASK_PROMPTS, LIMB_CANDIDATES,
)


# ---------------------------------------------------------------------------
# Token matching metrics (for generation tasks: sentences, ActionRec)
# ---------------------------------------------------------------------------

def token_f1(pred: str, ref: str) -> float:
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def token_precision(pred: str, ref: str) -> float:
    pred_tokens = pred.lower().split()
    ref_tokens = set(ref.lower().split())
    if not pred_tokens:
        return 0.0
    return sum(1 for t in pred_tokens if t in ref_tokens) / len(pred_tokens)


def token_recall(pred: str, ref: str) -> float:
    ref_tokens = ref.lower().split()
    pred_tokens = set(pred.lower().split())
    if not ref_tokens:
        return 0.0
    return sum(1 for t in ref_tokens if t in pred_tokens) / len(ref_tokens)


def best_token_scores(pred: str, refs) -> dict:
    if isinstance(refs, str):
        refs = [refs]
    best_f1, best_p, best_r = 0, 0, 0
    for ref in refs:
        f1 = token_f1(pred, ref)
        if f1 > best_f1:
            best_f1 = f1
            best_p = token_precision(pred, ref)
            best_r = token_recall(pred, ref)
    return {"f1": best_f1, "precision": best_p, "recall": best_r}


# ---------------------------------------------------------------------------
# Classification: find best matching candidate in generated text
# ---------------------------------------------------------------------------

def classify_from_generation(gen_text: str, candidates: List[str]) -> str:
    """Pick the candidate that best matches the generated text."""
    gen_lower = gen_text.lower().strip()
    # First: exact match
    for c in candidates:
        if c.lower() == gen_lower:
            return c
    # Second: candidate appears as substring
    best_c, best_pos = candidates[0], len(gen_lower) + 1
    for c in candidates:
        pos = gen_lower.find(c.lower())
        if pos != -1 and pos < best_pos:
            best_pos = pos
            best_c = c
    if best_pos <= len(gen_lower):
        return best_c
    # Third: token overlap
    gen_tokens = set(gen_lower.split())
    best_c, best_overlap = candidates[0], 0
    for c in candidates:
        c_tokens = set(c.lower().split())
        overlap = len(gen_tokens & c_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_c = c
    return best_c


# ---------------------------------------------------------------------------
# Evaluate model (called from train.py per epoch)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, test_entries, test_loader, jsonl_index,
                   qa_prompt, device, max_new_tokens=64):
    """Run evaluation and return metrics dict."""
    model.projector.eval()

    all_actions = sorted(set(e.action for e in test_entries))
    # Use full action list for classification
    all_actions_full = list(set(e.action for e in test_entries))

    results = []
    for pts, mask, orig_indices in test_loader:
        if pts.shape[0] == 0:
            continue
        pts = pts.to(device)
        mask = mask.to(device)
        oi = orig_indices[0]
        entry = test_entries[oi]
        jsonl_obj = jsonl_index.get(entry.key)
        if jsonl_obj is None:
            continue

        gt = get_ground_truths(entry, jsonl_obj)
        generated = {}

        # sentences: free generation
        generated["sentences"] = model.generate(
            pts, mask, [TASK_PROMPTS["sentences"]], max_new_tokens=max_new_tokens)[0]

        # ActionRec: free generation
        generated["ActionRec"] = model.generate(
            pts, mask, [TASK_PROMPTS["ActionRec"]], max_new_tokens=max_new_tokens)[0]

        # QA: classification (generate then match to action list)
        qa_gen = model.generate(
            pts, mask, [qa_prompt], max_new_tokens=max_new_tokens)[0]
        generated["QA_raw"] = qa_gen
        generated["QA_predicted"] = classify_from_generation(qa_gen, all_actions_full)

        # QA-LimbFocus: classification
        limb_gen = model.generate(
            pts, mask, [TASK_PROMPTS["QA-LimbFocus"]], max_new_tokens=max_new_tokens)[0]
        generated["LimbFocus_raw"] = limb_gen
        generated["LimbFocus_predicted"] = classify_from_generation(limb_gen, LIMB_CANDIDATES)

        results.append({
            "key": entry.key,
            "action": entry.action,
            "source": entry.source,
            "ground_truth": gt,
            "generated": generated,
        })

    return compute_metrics(results)


def compute_metrics(results):
    """Compute all metrics."""
    n = len(results)
    if n == 0:
        return {"error": "no results"}

    metrics = {}

    # --- sentences: token F1 ---
    sent_scores = []
    for r in results:
        sent_scores.append(best_token_scores(
            r["generated"]["sentences"], r["ground_truth"]["sentences"]))
    metrics["sentences_token_F1"] = sum(s["f1"] for s in sent_scores) / n
    metrics["sentences_token_precision"] = sum(s["precision"] for s in sent_scores) / n
    metrics["sentences_token_recall"] = sum(s["recall"] for s in sent_scores) / n

    # --- ActionRec: token F1 ---
    ar_scores = []
    for r in results:
        ar_scores.append(best_token_scores(
            r["generated"]["ActionRec"], r["ground_truth"]["ActionRec"]))
    metrics["ActionRec_token_F1"] = sum(s["f1"] for s in ar_scores) / n
    metrics["ActionRec_token_precision"] = sum(s["precision"] for s in ar_scores) / n
    metrics["ActionRec_token_recall"] = sum(s["recall"] for s in ar_scores) / n

    # --- QA: classification accuracy ---
    qa_correct = sum(
        1 for r in results
        if r["generated"]["QA_predicted"].lower() == r["action"].lower()
    )
    metrics["QA_cls_accuracy"] = qa_correct / n

    # --- QA-LimbFocus: classification accuracy ---
    limb_correct, limb_total = 0, 0
    for r in results:
        pred = r["generated"]["LimbFocus_predicted"].lower()
        gt_parts = r["ground_truth"]["QA-LimbFocus"]
        # Check if prediction matches any GT part
        for part in gt_parts:
            limb_total += 1
            if pred == part.lower():
                limb_correct += 1
    metrics["LimbFocus_cls_accuracy"] = limb_correct / max(limb_total, 1)

    metrics["n_samples"] = n
    return metrics


# ---------------------------------------------------------------------------
# Standalone evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate RadarLLM")
    parser.add_argument("--actions_jsonl", default="data_generation/mm_actions_text.jsonl")
    parser.add_argument("--actions_gen_jsonl", default="data_generation/mm_actions_text_gen.jsonl")
    parser.add_argument("--alignment_ckpt",
                        default="alignment_ver2/token_level_directTextEnc/checkpoints/ver1/v1_tl_direct_ep100_bs64_best.pt")
    parser.add_argument("--projector_ckpt",
                        default="post_training/llm_generation/checkpoints/projector_best.pt")
    parser.add_argument("--mmbody_pc_dir", default="/workspace/DATA/mmBody_pointcloud")
    parser.add_argument("--mmfi_pc_dir", default="/workspace/DATA/split_mmFI")
    parser.add_argument("--mri_pc_dir", default="/workspace/DATA/split_mRI")
    parser.add_argument("--llm_model", default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--max_points", type=int, default=196)
    parser.add_argument("--min_points", type=int, default=15)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--output_dir", default="post_training/llm_generation/output")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading entries...")
    entries = build_entries(args.actions_jsonl)
    jsonl_index = load_jsonl_index(args.actions_gen_jsonl)
    all_actions = get_all_actions(args.actions_jsonl)
    qa_prompt = build_qa_prompt(all_actions)
    _, test_entries = split_entries(entries)
    print(f"Test: {len(test_entries)}, Actions: {len(all_actions)}")

    test_ds = MultiTextRadarDataset(
        test_entries, args.mmbody_pc_dir, args.mmfi_pc_dir, args.mri_pc_dir, train=False)

    def collate_fn(batch):
        return collate_multitext(batch, max_points=args.max_points, min_points=args.min_points)

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=0, collate_fn=collate_fn)

    print("Loading encoder...")
    encoder = load_token_encoder_from_alignment(args.alignment_ckpt, device=args.device)
    encoder = encoder.to(args.device)

    print("Building RadarLLM...")
    model = RadarLLM(encoder, llm_model_name=args.llm_model, encoder_dim=256)
    model.projector = model.projector.to(args.device)

    if os.path.isfile(args.projector_ckpt):
        proj_state = torch.load(args.projector_ckpt, map_location=args.device)
        model.projector.load_state_dict(proj_state)
        print(f"Loaded projector from {args.projector_ckpt}")

    metrics = evaluate_model(
        model, test_entries, test_loader, jsonl_index,
        qa_prompt, args.device, args.max_new_tokens,
    )

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
