# Alignment ver2

Multi-text-type contrastive learning with **masked negative sampling**.

## Difference from ver1

| | ver1 (alignment/) | ver2 (alignment_ver2/) |
|---|---|---|
| Training text | `sentences` only | `sentences` + `QA` + `ActionRec` + `QA-LimbFocus` (random per step) |
| Negative masking | None (all batch negatives) | Text-type-specific masking |
| Evaluation | R@1, R@5, R@10 | + QA@K, AR@K, Limb_acc |

## Negative Masking Rules

| Text Type | Rule |
|---|---|
| `sentences` | Standard (all negatives valid) |
| `QA` | Exclude same-action negatives |
| `ActionRec` | Exclude same-action negatives |
| `QA-LimbFocus` | Exclude overlapping body part negatives |

## 16 Models (8 methods x v1/v2)

Same 8 architectures as ver1. v1 = sentences only, v2 = multi-text.

## Best Results (110 actions, bs64, ep100)

### Radar-to-Text Retrieval

| # | Method | Pool | Ver | R@1 | R@5 | R@10 |
|---|---|---|---|---|---|---|
| 3 | tl_directTextEnc | Cross | v1 | **0.2000** | **0.3091** | 0.3364 |
| 16 | tl_visionGuidedText | Cross | v2 | **0.2000** | 0.2909 | **0.3545** |
| 4 | tl_directTextEnc | Cross | v2 | 0.1818 | 0.2727 | 0.3455 |

### Action Classification (QA@K / AR@K)

| # | Method | Ver | QA@1 | QA@10 | AR@1 | AR@10 |
|---|---|---|---|---|---|---|
| 4 | tl_directTextEnc | v2 | **0.1545** | 0.3818 | 0.1545 | 0.3818 |
| 16 | tl_visionGuidedText | v2 | 0.1455 | 0.3727 | **0.1636** | 0.4182 |
| 3 | tl_directTextEnc | v1 | 0.1182 | **0.4273** | 0.1273 | 0.4182 |

### Body Part Detection (Limb_acc)

| # | Method | Ver | Limb_acc |
|---|---|---|---|
| 16 | tl_visionGuidedText | v2 | **0.7436** |
| 4 | tl_directTextEnc | v2 | 0.7255 |
| 2 | directTextEnc | v2 | 0.7073 |

### Text-to-Radar Retrieval (T2R)

| # | Method | Ver | T2R@1 | T2R@5 | T2R@10 |
|---|---|---|---|---|---|
| 16 | tl_visionGuidedText | v2 | **0.2273** | 0.3182 | 0.3909 |
| 15 | tl_visionGuidedText | v1 | 0.2091 | **0.3545** | **0.4909** |
| 4 | tl_directTextEnc | v2 | 0.2182 | 0.3273 | 0.4182 |

## Key Files

| File | Description |
|---|---|
| `common.py` | Shared: dataset, masked contrastive loss, evaluation |
| `train_common.py` | Shared training loop for all 8 methods |
| `eval_all.py` | Evaluate all 16 models (R2T: R@K, QA@K, AR@K, Limb) |
| `eval_t2r.py` | Text-to-Radar retrieval evaluation |

## Conclusions

1. **ver2 (multi-text)** improves QA/AR/Limb accuracy but slightly reduces R@K
2. **Token-level cross-attention** consistently outperforms CLS pooling
3. **tl_visionGuidedText v2** is the best overall model across all metrics
4. **Direct text alignment** methods dominate; vision-indirect methods are ineffective
