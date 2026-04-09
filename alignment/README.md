# Alignment (ver1)

Contrastive learning to align radar point cloud encoder with CLIP text/vision embeddings.
Training uses **sentences only** for text. Evaluation uses R@K retrieval.

## 8 Methods

### CLS Pooling

| Method | Train Alignment | Test |
|---|---|---|
| `directTextEnc` | Radar CLS ↔ CLIP Text CLS | Radar → Text R@K |
| `imageSpaceMap` | Radar CLS ↔ CLIP Vision CLS (real images) | Radar → Text R@K |
| `visionGuidedText` | (Radar+Vision fused) ↔ Text + aux: Radar ↔ Text | Radar → Text R@K |
| `ptVision` | Radar CLS ↔ (Tokenizer → CLIP Vision TF) | Radar → Text R@K |

### Token-Level Cross-Attention

| Method | Train Alignment | Test |
|---|---|---|
| `token_level_directTextEnc` | CrossAttn(Radar tokens, Text tokens) ↔ Text CLS | standalone pool → Text R@K |
| `token_level_imageSpaceMap` | CrossAttn(Radar tokens, Image tokens) ↔ Image CLS | standalone pool → Text R@K |
| `token_level_visionGuidedText` | CrossAttn(Radar tokens, Image tokens) ↔ Text CLS | standalone pool → Text R@K |
| `token_level_ptVision` | CrossAttn(Radar tokens, Tokenized Vision tokens) ↔ Vision CLS | standalone pool → Text R@K |

## Best Results (bs64, ep100)

| Method | R@1 | R@5 | R@10 |
|---|---|---|---|
| tl_directTextEnc | **0.2000** | **0.3091** | 0.3364 |
| tl_visionGuidedText | 0.1818 | 0.2455 | **0.3455** |
| directTextEnc | 0.1545 | 0.2273 | 0.3182 |
| visionGuidedText | 0.1545 | 0.2273 | 0.3000 |

## Key Findings

1. **Radar ↔ Text direct alignment** is most effective
2. **Token-level cross-attention** consistently outperforms CLS pooling
3. **Vision-guided** methods are comparable but do not improve over direct text alignment
4. **Image-space / ptVision** indirect alignment is ineffective
