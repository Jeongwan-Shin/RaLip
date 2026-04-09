"""Token-level directTextEnc training."""

from __future__ import annotations
import argparse, math, os, sys
from functools import partial
from typing import Dict, List
import numpy as np, torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.dirname(__file__))
from model import TokenLevelDirectText, TokenLevelDirectTextConfig, load_pretrained_token_encoder

# Import dataset from CLS directTextEnc
_direct_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "directTextEnc"))
sys.path.insert(0, _direct_dir)
import dataset as _ds_mod
sys.path.pop(0)
RadarTextDataset = _ds_mod.RadarTextDataset
build_entries = _ds_mod.build_entries
split_entries = _ds_mod.split_entries
collate_radar_text = _ds_mod.collate_radar_text


def tqdm_maybe(it, **kw):
    try:
        from tqdm import tqdm; return tqdm(it, file=sys.__stdout__, dynamic_ncols=True, leave=True, **kw)
    except: return it

class Tee:
    def __init__(self, *w): self._w = w
    def write(self, d):
        for w in self._w: w.write(d)
        return len(d)
    def flush(self):
        for w in self._w: w.flush()

def setup_log(d, t):
    os.makedirs(d, exist_ok=True); f = open(os.path.join(d, f"{t}.log"), "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, f); sys.stderr = Tee(sys.__stderr__, f)


@torch.no_grad()
def evaluate(model, test_entries, dataset, device, max_points=196, batch_size=64, desc="eval"):
    model.eval()
    all_pe, all_qa = [], []
    for i in tqdm_maybe(range(len(dataset)), desc=f"{desc} encode-radar", unit="s"):
        e = test_entries[i]; pts_t, _ = dataset[i]
        n = min(pts_t.shape[0], max_points)
        pts = torch.zeros(1, max_points, 5); mask = torch.zeros(1, max_points, dtype=torch.bool)
        pts[0,:n] = pts_t[:n]; mask[0,:n] = True
        emb = model.encode_points_standalone(pts.to(device), mask.to(device))
        all_pe.append(emb.cpu()); all_qa.append(e.action)
    point_embeds = torch.cat(all_pe, 0)

    all_t, all_ta = [], []
    for e in test_entries:
        for s in e.sentences: all_t.append(s); all_ta.append(e.action)
    all_te = []
    for i in tqdm_maybe(range(0, len(all_t), batch_size), desc=f"{desc} encode-text", unit="b"):
        emb = model.encode_text_cls(all_t[i:i+batch_size], device); all_te.append(emb.cpu())
    text_embeds = torch.cat(all_te, 0)

    sims = point_embeds @ text_embeds.T; nq = sims.shape[0]
    recall = {1:0, 5:0, 10:0}
    for i in range(nq):
        ranked = sims[i].argsort(descending=True)
        for k in recall:
            if all_qa[i] in [all_ta[idx] for idx in ranked[:k]]: recall[k] += 1
    return {"R@1": recall[1]/nq, "R@5": recall[5]/nq, "R@10": recall[10]/nq, "n_queries": nq, "n_gallery": len(all_t)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--actions_json", required=True); p.add_argument("--actions_text_jsonl", required=True)
    p.add_argument("--mmbody_dir", required=True); p.add_argument("--mmfi_dir", required=True); p.add_argument("--mri_dir", required=True)
    p.add_argument("--pretrained_ckpt", default=""); p.add_argument("--feature_norm", default="per_sample_zscore")
    p.add_argument("--max_points", type=int, default=196); p.add_argument("--min_points", type=int, default=15)
    p.add_argument("--embed_dim", type=int, default=256); p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4); p.add_argument("--proj_dim", type=int, default=512)
    p.add_argument("--cross_heads", type=int, default=4); p.add_argument("--aux_weight", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=50); p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=64); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4); p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--scheduler", default="cosine", choices=["none","cosine"])
    p.add_argument("--warmup_epochs", type=int, default=5); p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--ckpt_dir", default="alignment/token_level_directTextEnc/checkpoints")
    p.add_argument("--log_dir", default="alignment/token_level_directTextEnc/log")
    p.add_argument("--run_tag", default="")
    return p.parse_args()


def main():
    args = parse_args()
    tag = args.run_tag or f"tl_direct_ep{args.epochs}_bs{args.batch_size}_lr{args.lr}_aux{args.aux_weight}"
    setup_log(args.log_dir, tag)

    entries = build_entries(args.actions_json, args.actions_text_jsonl)
    train_e, test_e = split_entries(entries)
    print(f"[info] entries: total={len(entries)} train={len(train_e)} test={len(test_e)}")

    dk = dict(mmbody_dir=args.mmbody_dir, mmfi_dir=args.mmfi_dir, mri_dir=args.mri_dir, feature_norm=args.feature_norm)
    train_ds = RadarTextDataset(train_e, **dk, train=True)
    test_ds = RadarTextDataset(test_e, **dk, train=False)
    collate = partial(collate_radar_text, max_points=args.max_points, min_points=args.min_points)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate, pin_memory=True, persistent_workers=(args.num_workers>0))

    device = torch.device(args.device)
    cfg = TokenLevelDirectTextConfig(
        embed_dim=args.embed_dim, num_layers=args.num_layers, num_heads=args.num_heads,
        proj_dim=args.proj_dim, cross_heads=args.cross_heads, aux_weight=args.aux_weight)
    model = TokenLevelDirectText(cfg).to(device)

    if args.pretrained_ckpt and os.path.isfile(args.pretrained_ckpt):
        load_pretrained_token_encoder(model.token_encoder, args.pretrained_ckpt, args.device)

    params = [
        {"params": model.token_encoder.parameters()}, {"params": model.cross_attn.parameters()},
        {"params": model.standalone_pool.parameters()}, {"params": [model.logit_scale]},
    ]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == "cosine":
        spe = len(train_loader); total = max(1, args.epochs*spe); wu = max(0, args.warmup_epochs*spe)
        base, mn = args.lr, args.min_lr
        def lr_m(s):
            s = float(max(0, min(s, total)))
            if wu>0 and s<wu: return s/float(wu)
            if total<=wu: return 1.0
            t = (s-wu)/float(total-wu); return (mn/base)+(1-mn/base)*0.5*(1+math.cos(math.pi*t))
        scheduler = LambdaLR(optim, lr_m)

    nt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[info] trainable={nt:,}  device={device}  epochs={args.epochs}  aux={cfg.aux_weight}")

    best_r10 = 0.0
    for epoch in range(args.epochs):
        model.train(); model.clip_model.eval()
        pbar = tqdm_maybe(enumerate(train_loader), total=len(train_loader), desc=f"train {epoch+1}/{args.epochs}", unit="b")
        for step, (pts, mask, texts) in pbar:
            if pts.shape[0]==0: continue
            pts, mask = pts.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            out = model(pts, mask, texts); loss = out["loss"]
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
            if scheduler: scheduler.step()
            if hasattr(pbar,"set_postfix"):
                pbar.set_postfix(loss=f"{float(loss):.4f}", cross=f"{float(out['acc_cross']):.2f}",
                                 stand=f"{float(out['acc_stand']):.2f}", lr=f"{optim.param_groups[0]['lr']:.2e}")

        metrics = evaluate(model, test_e, test_ds, device, args.max_points, args.eval_batch_size, f"eval {epoch+1}/{args.epochs}")
        print(f"\n[epoch {epoch+1}] point→text | R@1={metrics['R@1']:.4f}  R@5={metrics['R@5']:.4f}  R@10={metrics['R@10']:.4f}")

        if metrics["R@10"] > best_r10:
            best_r10 = metrics["R@10"]; os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save({"epoch":epoch+1,"model_state_dict":model.state_dict(),"metrics":metrics,"config":vars(cfg)},
                       os.path.join(args.ckpt_dir, f"{tag}_best.pt"))
            print(f"[info] new best R@10={best_r10:.4f}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.save({"epoch":args.epochs,"model_state_dict":model.state_dict(),"metrics":metrics,"config":vars(cfg)},
               os.path.join(args.ckpt_dir, f"{tag}_final.pt"))
    print(f"[info] saved final: {tag}_final.pt")

if __name__ == "__main__": main()
