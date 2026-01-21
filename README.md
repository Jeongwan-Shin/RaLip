<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="asset/Icon.png">
    <img alt="Icon" src="Icon.png" width="150" style="max-width: 100%;">
  </picture>
</p>

# RaLIP: Aligning mmWave Radar to the Language–Image Space via Pretraining for Human Understanding
<p align="center">
  <strong>Fully Open Framework for Radar Encoder Training</strong>
</p>

## NEWS
- 2026-01-20: Released

## Contents
- [Quick Start For Training](#quick-start-guide)

## Quick Start For Training

### 1. Stage 1. Data Generation

**Goal**: Generate *Radar–Text* pairs by converting action annotations into 5 short English sentences per action.

- **Script**: `Train_mmWave_Encoder/preprocessing/radar_text_pair_generation.py`
- **Input**: `Train_mmWave_Encoder/datasets/mm_actions.json`
- **Output**: JSONL file (append-only, resumable) with 5 sentences per action segment.

**Run**:

```bash
export OPENAI_API_KEY="YOUR_KEY"
python /workspace/Train_mmWave_Encoder/preprocessing/radar_text_pair_generation.py \
  --input  /workspace/Train_mmWave_Encoder/datasets/mm_actions.json \
  --output /workspace/Train_mmWave_Encoder/datasets/mm_actions_text3.jsonl \
  --model  gpt-4.1
```

### 2. Stage 2. Pre-Training

**Goal**: Pre-train a radar/point-cloud encoder to better understand point-cloud structure (representation learning on point clouds).

- **What it does**: Learns point-cloud features so the encoder captures human/radar geometry patterns before alignment.
- **Where**: `RaLip/pre_training/` (see `RaLip/pre_training/train.py`)

**Run (example)**:

```bash
python /workspace/RaLip/pre_training/train.py --help
```

### 3. Stage 3. Transfer Learning

**Goal**: Align point-cloud inputs into the **Language–Image space** (e.g., CLIP-like embedding space) via transfer learning.

- **What it does**: Fine-tunes the radar encoder so radar embeddings match the language–image embedding space.
- **Typical setup**: radar encoder + projection head, trained with alignment loss using paired radar/text (and/or radar/image) supervision.

### 4. Stage 4. Post-Training

**Goal**: Post-train using **Radar–Text pairs only**, to further improve language alignment after transfer learning.

- **What it does**: Continues training (or re-training) focusing only on radar–text supervision generated in Stage 1.
