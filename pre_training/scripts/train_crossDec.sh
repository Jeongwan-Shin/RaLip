#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Keep args + tag in sync
EPOCHS=100
BATCH_SIZE=128
LR="1e-4"
POOLING="cls"
DECODING="crossDec"
LABEL_MODE="root_relative"
LOSS="l1"
FEATURE_NORM="per_sample_zscore"
# Dataset sampling / filtering
SAMPLING="balanced"
DATASET_WEIGHTS=""
MIN_POINTS=15
# Scheduler (warmup + cosine)
SCHEDULER="cosine"
WARMUP_EPOCHS=5
MIN_LR="1e-6"
# Try a stronger model; tune these if you want faster/slower.
EMBED_DIM=256
OUT_DIM=128
NUM_LAYERS=4
NUM_HEADS=4
DEC_HEADS=4
RUN_TAG="epochs${EPOCHS}_batch_size_${BATCH_SIZE}_lr_${LR}_pooling_${POOLING}_decoding_${DECODING}_label_${LABEL_MODE}_loss_${LOSS}_feat_${FEATURE_NORM}_sampling_${SAMPLING}_minpts${MIN_POINTS}_sched_${SCHEDULER}_wu${WARMUP_EPOCHS}_minlr${MIN_LR}_D${EMBED_DIM}_L${NUM_LAYERS}_H${NUM_HEADS}_OD${OUT_DIM}_DH${DEC_HEADS}"

python "${PROJECT_DIR}/train.py" \
  --mars_dir "/workspace/2025/JH_folder/dataset/split_MARS" \
  --mri_dir "/workspace/2025/JH_folder/dataset/split_mRI" \
  --mmfi_dir "/workspace/2025/JH_folder/dataset/split_mmFI" \
  --mmbody_dir "/workspace/2025/JH_folder/dataset/mmBody_pointcloud" \
  --datasets "all" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --pooling "${POOLING}" \
  --decoding "${DECODING}" \
  --embed_dim "${EMBED_DIM}" \
  --out_dim "${OUT_DIM}" \
  --num_layers "${NUM_LAYERS}" \
  --num_heads "${NUM_HEADS}" \
  --dec_heads "${DEC_HEADS}" \
  --label_mode "${LABEL_MODE}" \
  --loss "${LOSS}" \
  --feature_norm "${FEATURE_NORM}" \
  --sampling "${SAMPLING}" \
  --dataset_weights "${DATASET_WEIGHTS}" \
  --min_points "${MIN_POINTS}" \
  --scheduler "${SCHEDULER}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --min_lr "${MIN_LR}" \
  --weight_decay 1e-4 \
  --max_points 196 \
  --num_workers 4 \
  --run_tag "${RUN_TAG}" \
  --log_dir "${PROJECT_DIR}/log" \
  --ckpt_dir "${PROJECT_DIR}/checkpoints"

