#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../.." && pwd)"

# ---- data paths ----
ACTIONS_JSON="${REPO_ROOT}/data_generation/mm_actions.json"
ACTIONS_TEXT="${REPO_ROOT}/data_generation/mm_actions_text.jsonl"
MMBODY_DIR="/home/jwshin/2025/JH_folder/dataset/mmBody_pointcloud"
MMFI_DIR="/home/jwshin/2025/JH_folder/dataset/split_mmFI"
MRI_DIR="/home/jwshin/2025/JH_folder/dataset/split_mRI"

# ---- pre-trained checkpoint ----
PRETRAINED_CKPT="${REPO_ROOT}/pre_training/checkpoints/epochs100_batch_size_128_lr_1e-4_pooling_cls_decoding_crossDec_label_root_relative_loss_l1_feat_per_sample_zscore_sampling_balanced_minpts15_sched_cosine_wu5_minlr1e-6_D256_L4_H4_OD128_DH4_HuPR.pt"

# ---- training config ----
EPOCHS=50
BATCH_SIZE=64
LR="1e-4"
PROJ_DIM=512
EMBED_DIM=256
OUT_DIM=128
NUM_LAYERS=4
NUM_HEADS=4
POOLING="cls"
SCHEDULER="cosine"
WARMUP_EPOCHS=5
MIN_LR="1e-6"

RUN_TAG="radarclip_ep${EPOCHS}_bs${BATCH_SIZE}_lr${LR}_proj${PROJ_DIM}_D${EMBED_DIM}_L${NUM_LAYERS}_H${NUM_HEADS}"

python "${PROJECT_DIR}/train.py" \
  --actions_json "${ACTIONS_JSON}" \
  --actions_text_jsonl "${ACTIONS_TEXT}" \
  --mmbody_dir "${MMBODY_DIR}" \
  --mmfi_dir "${MMFI_DIR}" \
  --mri_dir "${MRI_DIR}" \
  --pretrained_ckpt "${PRETRAINED_CKPT}" \
  --embed_dim "${EMBED_DIM}" \
  --out_dim "${OUT_DIM}" \
  --num_layers "${NUM_LAYERS}" \
  --num_heads "${NUM_HEADS}" \
  --pooling "${POOLING}" \
  --proj_dim "${PROJ_DIM}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay 1e-4 \
  --scheduler "${SCHEDULER}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --min_lr "${MIN_LR}" \
  --max_points 196 \
  --min_points 15 \
  --feature_norm "per_sample_zscore" \
  --num_workers 4 \
  --freeze_text \
  --run_tag "${RUN_TAG}" \
  --ckpt_dir "${PROJECT_DIR}/checkpoints" \
  --log_dir "${PROJECT_DIR}/log"
