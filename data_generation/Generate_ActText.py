"""
Generate simple English text sentences (5) for each action annotation in datasets/mm_actions.json
using OpenAI GPT-4.1, and save results as JSONL (resumable).

Usage:
  export OPENAI_API_KEY="..."
  python preprocessing/radar_text_pair_generation.py \
    --input /workspace/Train_mmWave_Encoder/datasets/mm_actions.json \
    --output /workspace/Train_mmWave_Encoder/datasets/mm_actions_text.jsonl \
    --model gpt-4.1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Set
from tqdm import tqdm

PROMPT_TEMPLATE = """You will generate **5 simple English sentences** describing the given human action annotation.

Requirements:
- The action name (field `action`) must appear in every sentence **verbatim** (exact spelling/case/hyphens/spaces). Example: if action is "T-Pose", each sentence must include the exact substring "T-Pose".
- This dataset describes exactly **one person**. Use singular phrasing such as "The person ..." in every sentence. Do NOT use plural pronouns like "they/them/their/theirs".
- The 5 sentences must describe the same action but use **slightly different wording/phrasing** (no near-duplicates).
- Prefer short present progressive / present tense sentences (e.g., "is holding...", "is doing...", "is standing in...").
- Do NOT add numbering or bullet points; return plain sentences only.
- Avoid extra guesses (location, emotion, speed). You may lightly mention limbs only if consistent with the provided fields.

Example (desired style):
Input annotation:
{{"action":"T-Pose","body_part":{{"left arm":false,"right arm":false,"left leg":false,"right leg":false}},"movement":false}}
Output sentences:
[
  "The person is holding a T-Pose with both arms extended.",
  "The person is standing in a T-Pose with arms stretched out to the sides.",
  "The person is doing a T-Pose by raising both arms to shoulder height.",
  "The person is maintaining a T-Pose posture with arms spread wide.",
  "The person is posing in a T-Pose with both arms held straight out."
]

Now generate 5 sentences in the same style for this annotation:
{annotation_json}
"""


def _load_actions(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    return data


def _action_key(item: Dict[str, Any]) -> str:
    info = item.get("info") or {}
    frames = item.get("frames") or {}
    segment = info.get("segment", "unknown_segment")
    start = frames.get("start", "na")
    end = frames.get("end", "na")
    action = item.get("action", "unknown_action")
    return f"{segment}:{start}-{end}:{action}"


def _read_done_keys(output_path: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                k = obj.get("key")
                if isinstance(k, str):
                    done.add(k)
            except Exception:
                # ignore malformed lines
                continue
    return done


def _build_prompt(item: Dict[str, Any]) -> str:
    # Provide the key fields to the model (avoid stripping away useful supervision).
    annotation = {
        "action": item.get("action"),
        "body_part": item.get("body_part"),
        "movement": item.get("movement"),
    }
    return PROMPT_TEMPLATE.format(
        action=item.get("action"),
        annotation_json=json.dumps(annotation, ensure_ascii=False),
    )


def _call_gpt_41(prompt: str, model: str, temperature: float, max_retries: int) -> List[str]:
    """
    Returns a list of exactly 5 sentences.
    Requires: `pip install openai` and OPENAI_API_KEY env var.
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency `openai`. Install with: pip install openai"
        ) from e

    client = OpenAI()

    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You write short English sentences describing a human action from an annotation. "
                            "Return ONLY valid JSON."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            prompt
                            + "\n\nReturn JSON in this exact shape:\n"
                            + '{"sentences":["...","...","...","...","..."]}'
                        ),
                    },
                ],
            )
            content = resp.choices[0].message.content or ""
            obj = json.loads(content)
            sentences = obj.get("sentences")
            if not isinstance(sentences, list) or len(sentences) != 5 or not all(
                isinstance(s, str) and s.strip() for s in sentences
            ):
                raise ValueError(f"Model returned unexpected JSON: {obj}")
            return [s.strip() for s in sentences]
        except Exception as e:
            last_err = e
            # exponential backoff with jitter
            sleep_s = min(30.0, (2**attempt) + random.random())
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed after {max_retries} retries") from last_err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/workspace/RaLip/data_generation/mm_actions.json", help="Path to mm_actions.json (JSON array).")
    ap.add_argument("--output", default="/workspace/RaLip/data_generation/mm_actions_text.jsonl", help="Output JSONL path (append-only, resumable).")
    ap.add_argument("--model", default="gpt-4.1", help="OpenAI model name (default: gpt-4.1).")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="If > 0, process only the first N new items.")
    args = ap.parse_args()

    # Don't hardcode API keys in code. Prefer:
    #   export OPENAI_API_KEY="..."
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY env var is required.")

    items = _load_actions(args.input)
    done = _read_done_keys(args.output)

    processed_new = 0
    with open(args.output, "a", encoding="utf-8") as out_f:
        for item in tqdm(items):
            key = _action_key(item)
            if key in done:
                continue

            prompt = _build_prompt(item)
            sentences = _call_gpt_41(prompt=prompt, model=args.model, temperature=args.temperature, max_retries=args.max_retries)

            record = {
                "key": key,
                "annotation": {
                    "action": item.get("action"),
                    "body_part": item.get("body_part"),
                },
                "sentences": sentences,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            done.add(key)  # prevent duplicates within the same run, too

            processed_new += 1
            if args.limit > 0 and processed_new >= args.limit:
                break


if __name__ == "__main__":
    main()
