#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Keep args + tag in sync
EPOCHS=100
BATCH_SIZE=128
LR="1e-4"
POOLING="cls"
DECODING="poseDec"
LABEL_MODE="root_relative"
LOSS="l1"
FEATURE_NORM="per_sample_zscore"
# Dataset sampling / filtering
SAMPLING="balanced"
DATASET_WEIGHTS=""
MIN_POINTS=15
RUN_TAG="epochs${EPOCHS}_batch_size_${BATCH_SIZE}_lr_${LR}_pooling_${POOLING}_decoding_${DECODING}_label_${LABEL_MODE}_loss_${LOSS}_feat_${FEATURE_NORM}_sampling_${SAMPLING}_minpts${MIN_POINTS}"

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
  --label_mode "${LABEL_MODE}" \
  --loss "${LOSS}" \
  --feature_norm "${FEATURE_NORM}" \
  --sampling "${SAMPLING}" \
  --dataset_weights "${DATASET_WEIGHTS}" \
  --min_points "${MIN_POINTS}" \
  --weight_decay 1e-4 \
  --max_points 196 \
  --num_workers 4 \
  --run_tag "${RUN_TAG}" \
  --log_dir "${PROJECT_DIR}/log" \
  --ckpt_dir "${PROJECT_DIR}/checkpoints"
