# Pre-Training

Pre-train a Point Transformer encoder on mmWave radar pose regression across multiple datasets.

## Architecture

- **PointTransformerEncoder**: Point embedding MLP → TransformerEncoder → global pooling (mean/max/cls) → projection head
- **PoseRegressorCrossDec**: Token encoder + cross-attention joint decoder (best performing variant)

## Supported Datasets

| Dataset | Points/Frame | Joints | Source |
|---|---|---|---|
| MARS | 64 (8x8) | 19 → 15 | Radar |
| mRI | 196 (14x14) | 17 → 15 | Radar |
| mmFI | Variable | 17 → 15 | Radar |
| mmBody | Variable | 15 | Radar |
| HuPR | Variable | 15 (2D) | Radar |

All datasets are mapped to a unified 15-joint skeleton representation.

## Training

```bash
# Default training
bash scripts/train.sh

# Cross-attention decoder variant
bash scripts/train_crossDec.sh
```

## Evaluation

```bash
python test.py --checkpoint checkpoints/<name>.pt
python test.py --glob "checkpoints/*.pt"
```

## Key Files

| File | Description |
|---|---|
| `train.py` | Training script with multi-dataset support |
| `test.py` | Evaluation script |
| `models/point_transformer.py` | PointTransformerEncoder, PoseRegressor |
| `models/point_transformer_with_crossDec.py` | Cross-attention decoder variant |
| `utils/dataloader.py` | Dataset classes for all 5 radar datasets |
| `utils/dataset_config.py` | Cross-subject train/test splits |
