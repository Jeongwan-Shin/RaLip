#!/bin/bash
# Evaluate RadarLLM on test set
cd /workspace/RaLip
export CUDA_VISIBLE_DEVICES=2

python post_training/llm_generation/evaluate.py \
  --actions_jsonl "data_generation/mm_actions_text.jsonl" \
  --actions_gen_jsonl "data_generation/mm_actions_text_gen.jsonl" \
  --alignment_ckpt "alignment_ver2/token_level_directTextEnc/checkpoints/ver1/v1_tl_direct_ep100_bs64_best.pt" \
  --projector_ckpt "post_training/llm_generation/checkpoints/projector.pt" \
  --mmbody_pc_dir "/workspace/DATA/mmBody_pointcloud" \
  --mmfi_pc_dir "/workspace/DATA/split_mmFI" \
  --mri_pc_dir "/workspace/DATA/split_mRI" \
  --llm_model "microsoft/Phi-3.5-mini-instruct" \
  --max_new_tokens 64 \
  --output_dir "post_training/llm_generation/output"
