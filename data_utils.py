"""Loading helpers shared by train / search / evaluate."""
from __future__ import annotations

import json
import zlib

import numpy as np
import torch

from config import LABELS_JSONL, MODULES_JSON


def split_of(mod_id: str) -> str:
    h = zlib.crc32(mod_id.encode()) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def load_modules() -> dict:
    return json.loads(MODULES_JSON.read_text())


def load_labels():
    rows = []
    with open(LABELS_JSONL) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def module_tensors(mods: dict, ids):
    F = torch.tensor(np.stack([mods[i]["feats"] for i in ids]), dtype=torch.float32)
    facts = torch.tensor(np.stack([mods[i]["facts"] for i in ids]), dtype=torch.bool)
    return F, facts
