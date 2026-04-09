import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model import RadarCLIP, RadarCLIPConfig, load_pretrained_encoder
from train_common import add_common_args, run_training
from common import masked_contrastive_loss
import torch

def train_step(model, pts, mask, texts, neg_mask, images=None, text_type=None):
    ls = model.logit_scale.exp().clamp(max=100.0)
    pe = model.encode_points(pts, mask)
    te = model.encode_text(texts, pts.device)
    loss, acc = masked_contrastive_loss(pe, te, ls, neg_mask)
    return {"loss": loss, "acc": acc.detach()}

if __name__ == "__main__":
    p = add_common_args(argparse.ArgumentParser())
    args = p.parse_args()
    if not args.run_tag: args.run_tag = f"v2_direct_ep{args.epochs}_bs{args.batch_size}"
    if not args.ckpt_dir: args.ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    if not args.log_dir: args.log_dir = os.path.join(os.path.dirname(__file__), "log")
    cfg = RadarCLIPConfig(embed_dim=args.embed_dim, out_dim=args.out_dim, num_layers=args.num_layers,
                          num_heads=args.num_heads, proj_dim=args.proj_dim, pooling="cls")
    model = RadarCLIP(cfg)
    if args.pretrained_ckpt and os.path.isfile(args.pretrained_ckpt):
        load_pretrained_encoder(model.point_encoder, args.pretrained_ckpt)
    params = [{"params": model.point_encoder.parameters()}, {"params": model.point_proj.parameters()},
              {"params": [model.logit_scale]}]
    run_training(args, model, params, model.encode_points, model.encode_text, train_step)
