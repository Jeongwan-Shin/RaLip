#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/../.." && pwd)"

ACTIONS_JSON="${REPO_ROOT}/data_generation/mm_actions.json"
ACTIONS_TEXT="${REPO_ROOT}/data_generation/mm_actions_text.jsonl"

MMBODY_PC_DIR="/workspace/DATA/mmBody_pointcloud"
MMFI_PC_DIR="/workspace/DATA/split_mmFI"
MRI_PC_DIR="/workspace/DATA/split_mRI"

MMBODY_IMG_DIR="/home/jwshin/Train_mmWave_Encoder/datasets/mmBody/img"
MMFI_IMG_DIR="/home/jwshin/Train_mmWave_Encoder/datasets/mmFI/img"
MRI_IMG_DIR="/home/jwshin/Train_mmWave_Encoder/datasets/mRI"

PRETRAINED_CKPT="${REPO_ROOT}/pre_training/checkpoints/epochs100_batch_size_128_lr_1e-4_pooling_cls_decoding_crossDec_label_root_relative_loss_l1_feat_per_sample_zscore_sampling_balanced_minpts15_sched_cosine_wu5_minlr1e-6_D256_L4_H4_OD128_DH4_HuPR.pt"

EPOCHS=50
BATCH_SIZE=64
LR="1e-4"
AUX_WEIGHT=0.5
PROJ_DIM=512
EMBED_DIM=256
OUT_DIM=128
NUM_LAYERS=4
NUM_HEADS=4

RUN_TAG="vgt_ep${EPOCHS}_bs${BATCH_SIZE}_lr${LR}_aux${AUX_WEIGHT}_proj${PROJ_DIM}_D${EMBED_DIM}_L${NUM_LAYERS}_H${NUM_HEADS}"

python "${PROJECT_DIR}/train.py" \
  --actions_json "${ACTIONS_JSON}" \
  --actions_text_jsonl "${ACTIONS_TEXT}" \
  --mmbody_pc_dir "${MMBODY_PC_DIR}" \
  --mmfi_pc_dir "${MMFI_PC_DIR}" \
  --mri_pc_dir "${MRI_PC_DIR}" \
  --mmbody_img_dir "${MMBODY_IMG_DIR}" \
  --mmfi_img_dir "${MMFI_IMG_DIR}" \
  --mri_img_dir "${MRI_IMG_DIR}" \
  --pretrained_ckpt "${PRETRAINED_CKPT}" \
  --embed_dim "${EMBED_DIM}" \
  --out_dim "${OUT_DIM}" \
  --num_layers "${NUM_LAYERS}" \
  --num_heads "${NUM_HEADS}" \
  --pooling cls \
  --proj_dim "${PROJ_DIM}" \
  --aux_weight "${AUX_WEIGHT}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay 1e-4 \
  --scheduler cosine \
  --warmup_epochs 5 \
  --min_lr 1e-6 \
  --max_points 196 \
  --min_points 15 \
  --feature_norm per_sample_zscore \
  --num_workers 4 \
  --run_tag "${RUN_TAG}" \
  --ckpt_dir "${PROJECT_DIR}/checkpoints" \
  --log_dir "${PROJECT_DIR}/log"
