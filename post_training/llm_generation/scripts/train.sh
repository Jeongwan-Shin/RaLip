#!/bin/bash
# Train RadarLLM projector (30 epochs, eval each epoch)
cd /workspace/RaLip
export CUDA_VISIBLE_DEVICES=2

python post_training/llm_generation/train.py \
  --actions_jsonl "data_generation/mm_actions_text.jsonl" \
  --actions_gen_jsonl "data_generation/mm_actions_text_gen.jsonl" \
  --alignment_ckpt "alignment_ver2/token_level_directTextEnc/checkpoints/ver1/v1_tl_direct_ep100_bs64_best.pt" \
  --mmbody_pc_dir "/workspace/DATA/mmBody_pointcloud" \
  --mmfi_pc_dir "/workspace/DATA/split_mmFI" \
  --mri_pc_dir "/workspace/DATA/split_mRI" \
  --llm_model "microsoft/Phi-3.5-mini-instruct" \
  --epochs 30 \
  --batch_size 4 \
  --lr 1e-4 \
  --max_new_tokens 64 \
  --ckpt_dir "post_training/llm_generation/checkpoints" \
  --output_dir "post_training/llm_generation/output"
