import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model import TokenLevelImageSpace, TokenLevelImageSpaceConfig, load_pretrained_token_encoder
from train_common import add_common_args, run_training
from common import masked_contrastive_loss

def train_step(model, pts, mask, texts, neg_mask, images=None, text_type=None):
    ls = model.logit_scale.exp().clamp(max=100.0)
    pt = model._get_point_tokens(pts, mask)
    it, ic = model._get_image_tokens_and_cls(images, pts.device)
    cross = model.cross_attn(pt, it, mask)
    stand = model.standalone_pool(pt, mask)
    lc, _ = masked_contrastive_loss(cross, ic, ls)
    ls2, acc = masked_contrastive_loss(stand, ic, ls)
    a = model.cfg.aux_weight
    return {"loss": (1-a)*lc + a*ls2, "acc": acc.detach()}

if __name__ == "__main__":
    p = add_common_args(argparse.ArgumentParser())
    args = p.parse_args()
    if not args.run_tag: args.run_tag = f"v2_tl_imgspace_ep{args.epochs}_bs{args.batch_size}"
    if not args.ckpt_dir: args.ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    if not args.log_dir: args.log_dir = os.path.join(os.path.dirname(__file__), "log")
    cfg = TokenLevelImageSpaceConfig(embed_dim=args.embed_dim, num_layers=args.num_layers,
                                     num_heads=args.num_heads, proj_dim=args.proj_dim,
                                     cross_heads=args.cross_heads, aux_weight=args.aux_weight)
    model = TokenLevelImageSpace(cfg)
    if args.pretrained_ckpt and os.path.isfile(args.pretrained_ckpt):
        load_pretrained_token_encoder(model.token_encoder, args.pretrained_ckpt)
    params = [{"params": model.token_encoder.parameters()}, {"params": model.cross_attn.parameters()},
              {"params": model.standalone_pool.parameters()}, {"params": [model.logit_scale]}]
    run_training(args, model, params, model.encode_points_standalone, model.encode_text_cls, train_step, need_image=True)
