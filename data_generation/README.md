# Data Generation

Generate Radar-Text pairs by converting action annotations into descriptive text using GPT-4.1.

## Files

| File | Description |
|---|---|
| `Generate_ActText.py` | GPT-4.1을 사용하여 action annotation → 5개 영어 문장 생성 |
| `mm_actions.json` | 원본 action annotation (3개 데이터셋: mmBody, mmFI, mRI) |
| `mm_actions2.json` | action 이름 통일 버전 (119 → 110 unique actions) |
| `mm_actions_text.jsonl` | 생성된 Radar-Text 쌍 (1,978 entries) |
| `mm_actions_text_train.jsonl` | Train split (1,859 entries) |
| `mm_actions_text_test.jsonl` | Test split (119 entries, action당 1개) |

## JSONL Format

```json
{
  "key": "sequence_0:0-100:T-Pose",
  "annotation": {"action": "T-Pose", "body_part": {"left arm": false, ...}},
  "sentences": ["The person is holding a T-Pose with both arms extended.", ...],
  "info": {"from": "mmbody", "mode": "train", "segment": "sequence_0", "frame": "0-100"},
  "ActionRec": "The action being performed is T-Pose",
  "QA": "Which action is represented by this point cloud? T-Pose",
  "QA-LimbFocus": ["Which single body part is primarily moving in this action? none"]
}
```

## Text Types

| Type | Description | Example |
|---|---|---|
| `sentences` | 5개 자연어 설명 | "The person is holding a T-Pose..." |
| `ActionRec` | Action recognition 문장 | "The action being performed is T-Pose" |
| `QA` | Action QA 문장 | "Which action is represented by this point cloud? T-Pose" |
| `QA-LimbFocus` | Body part QA 문장 | "Which single body part is primarily moving? left arm" |

## Usage

```bash
export OPENAI_API_KEY="YOUR_KEY"
python Generate_ActText.py \
  --input mm_actions.json \
  --output mm_actions_text.jsonl \
  --model gpt-4.1
```
