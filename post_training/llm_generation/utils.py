"""
Utility functions for LLM generation pipeline.
- JSONL loading/indexing
- Task prompts (with candidate lists for classification tasks)
- Ground truth extraction
- Training text selection
"""

from __future__ import annotations

import json, random
from typing import Dict, List


# ---------------------------------------------------------------------------
# Limb candidates (fixed)
# ---------------------------------------------------------------------------

LIMB_CANDIDATES = ["left arm", "right arm", "left leg", "right leg", "none"]


# ---------------------------------------------------------------------------
# Task prompts
# ---------------------------------------------------------------------------

def build_qa_prompt(action_list: List[str]) -> str:
    """Build QA prompt with all action candidates listed."""
    candidates = ", ".join(action_list)
    return (
        "You are given a mmWave radar point cloud representing a human action. "
        "Each point consists of 3D spatial coordinates (x, y, z) and two radar-specific features. "
        "Question: Which action is represented by this point cloud? "
        f"Choose exactly one from the following actions: [{candidates}]. "
        "Answer with only the action name, nothing else."
    )


TASK_PROMPTS = {
    "sentences": (
        "You are given a mmWave radar point cloud that captures a person performing a physical action. "
        "Each point consists of 3D spatial coordinates (x, y, z) and two radar-specific features. "
        "Analyze the spatial distribution and motion pattern of the point cloud, "
        "then describe what human action or pose the person is performing in one detailed sentence. "
        "Focus on which body parts are moving and how they are positioned."
    ),
    "ActionRec": (
        "You are given a mmWave radar point cloud that captures a person performing a physical action. "
        "Each point consists of 3D spatial coordinates (x, y, z) and two radar-specific features. "
        "Based on the spatial structure and motion pattern of the point cloud, "
        "identify the specific action being performed. "
        "Respond in the format: 'The action being performed is <action name>'"
    ),
    # QA prompt is built dynamically with build_qa_prompt()
    "QA": None,
    "QA-LimbFocus": (
        "You are given a mmWave radar point cloud representing a human action. "
        "Each point consists of 3D spatial coordinates (x, y, z) and two radar-specific features. "
        "Question: Which single body part is primarily moving in this action? "
        "Choose exactly one from: [left arm, right arm, left leg, right leg, none]. "
        "Answer with only the body part name, nothing else."
    ),
}


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def load_jsonl_index(path: str) -> Dict[str, dict]:
    """Load JSONL file and return dict indexed by 'key' field."""
    index = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            index[obj["key"]] = obj
    return index


def get_all_actions(path: str) -> List[str]:
    """Extract sorted list of unique action names from JSONL."""
    actions = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            actions.add(obj["annotation"]["action"])
    return sorted(actions)


# ---------------------------------------------------------------------------
# Ground truth extraction
# ---------------------------------------------------------------------------

def get_ground_truths(entry, jsonl_entry: dict) -> Dict[str, any]:
    """Extract ground truth for each task from a JSONL entry."""
    gt = {}
    gt["sentences"] = jsonl_entry["sentences"]
    gt["ActionRec"] = jsonl_entry["ActionRec"]

    qa = jsonl_entry["QA"]
    gt["QA"] = qa["A"] if isinstance(qa, dict) else qa

    limb = jsonl_entry["QA-LimbFocus"]
    if isinstance(limb, list):
        gt["QA-LimbFocus"] = [
            item["A"] if isinstance(item, dict) else item
            for item in limb
        ]
    else:
        gt["QA-LimbFocus"] = [limb]

    return gt


# ---------------------------------------------------------------------------
# Training text selection
# ---------------------------------------------------------------------------

def get_training_text(entry, jsonl_entry: dict, qa_prompt: str):
    """Randomly select a (prompt, target) pair for training."""
    task = random.choice(["sentences", "ActionRec", "QA", "QA-LimbFocus"])

    if task == "QA":
        prompt = qa_prompt
    else:
        prompt = TASK_PROMPTS[task]

    if task == "sentences":
        target = random.choice(jsonl_entry["sentences"])
    elif task == "ActionRec":
        target = jsonl_entry["ActionRec"]
    elif task == "QA":
        qa = jsonl_entry["QA"]
        target = qa["A"] if isinstance(qa, dict) else qa
    elif task == "QA-LimbFocus":
        items = jsonl_entry["QA-LimbFocus"]
        item = random.choice(items)
        target = item["A"] if isinstance(item, dict) else item

    return prompt, target
