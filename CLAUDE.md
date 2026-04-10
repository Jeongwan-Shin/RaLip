# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**RaLIP** (Radar to Language–Image space via Pretraining) is a research framework for training mmWave radar encoders to understand human pose/action by aligning radar point clouds with language-image embedding spaces (CLIP-like). Python 3.10+, PyTorch-based.

## 4-Stage Pipeline

1. **Data Generation** (`data_generation/`): Uses OpenAI GPT-4.1 to generate 5 descriptive sentences per radar action annotation → JSONL radar-text pairs
2. **Pre-Training** (`pre_training/`): **Most mature stage.** Trains a Point Transformer encoder on pose regression across multiple radar datasets
3. **Transfer Learning** (`alignment/`): Placeholder — will align radar embeddings to CLIP-like space
4. **Post-Training** (`post_training/`): Placeholder — will fine-tune with radar-text pairs

## Commands

### Pre-training
```bash
# Show all training arguments
python pre_training/train.py --help

# Train with default config (original decoder, L1 loss, root-relative labels)
bash pre_training/scripts/train.sh

# Train with separate x/y/z decoder heads
bash pre_training/scripts/train_poseDec.sh

# Train with cross-attention decoder
bash pre_training/scripts/train_crossDec.sh
```

### Evaluation
```bash
# Evaluate a single checkpoint
python pre_training/test.py --checkpoint pre_training/checkpoints/<name>.pt

# Batch evaluate with glob pattern
python pre_training/test.py --glob "pre_training/checkpoints/*.pt"
```

### Data Generation
```bash
export OPENAI_API_KEY="YOUR_KEY"
python data_generation/Generate_ActText.py \
  --input data_generation/mm_actions.json \
  --output data_generation/mm_actions_text.jsonl \
  --model gpt-4.1
```

## Architecture

### Core Model (`pre_training/models/point_transformer.py`)
- **PointTransformerEncoder**: Point embedding MLP → TransformerEncoder layers → global pooling (mean/max/cls) → projection head. Input: variable-length point clouds padded to max_points=196. Output: fixed embedding (default 64D).
- **PoseRegressor**: Encoder + single linear head (45D output = 15 joints × 3 coords)
- **PoseRegressorPoseDec**: Separate x/y/z decoder heads
- **PoseRegressorCrossDec**: Cross-attention decoder variant

### Datasets (`pre_training/utils/dataloader.py`)
Supports 5 radar datasets with different skeleton formats, each mapped to a unified 15-joint representation:
- **MARS**, **mRI**, **mmFI**, **mmBody**, **HuPR**
- Dataset-specific joint mappings and train/test splits defined in `pre_training/utils/dataset_config.py`

### Training (`pre_training/train.py`)
- Multi-dataset training via ConcatDataset with balanced/proportional sampling
- Cosine LR scheduler with warmup
- Per-dataset evaluation each epoch
- Configurable loss (MSE, L1, SmoothL1) and label modes (absolute, root_relative)
- Metrics: MPJPE, MPJPE2, MSE, MAE

## Dependencies (no requirements.txt — install manually)

```bash
pip install torch numpy tqdm openai
```

## Notes

- Training scripts in `pre_training/scripts/` contain hardcoded dataset paths that need updating for your environment
- No formal test suite, linting, or CI — this is a research codebase
- Checkpoints saved to `pre_training/checkpoints/`, logs to `pre_training/log/`
