#!/usr/bin/env bash
set -euo pipefail
SD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PD="$(cd "$SD/.." && pwd)"; ROOT="$(cd "$PD/../.." && pwd)"
python "$PD/train.py" \
  --actions_json "$ROOT/data_generation/mm_actions.json" \
  --actions_text_jsonl "$ROOT/data_generation/mm_actions_text.jsonl" \
  --mmbody_dir "/home/jwshin/2025/JH_folder/dataset/mmBody_pointcloud" \
  --mmfi_dir "/home/jwshin/2025/JH_folder/dataset/split_mmFI" \
  --mri_dir "/home/jwshin/2025/JH_folder/dataset/split_mRI" \
  --pretrained_ckpt "$ROOT/pre_training/checkpoints/epochs100_batch_size_128_lr_1e-4_pooling_cls_decoding_crossDec_label_root_relative_loss_l1_feat_per_sample_zscore_sampling_balanced_minpts15_sched_cosine_wu5_minlr1e-6_D256_L4_H4_OD128_DH4_HuPR.pt" \
  --embed_dim 256 --num_layers 4 --num_heads 4 --proj_dim 512 --cross_heads 4 --aux_weight 0.5 \
  --epochs 50 --batch_size 64 --lr 1e-4 --weight_decay 1e-4 --scheduler cosine --warmup_epochs 5 --min_lr 1e-6 \
  --max_points 196 --min_points 15 --feature_norm per_sample_zscore --num_workers 4 \
  --ckpt_dir "$PD/checkpoints" --log_dir "$PD/log" \
  --run_tag "tl_direct_ep50_bs64_lr1e-4_aux0.5"
