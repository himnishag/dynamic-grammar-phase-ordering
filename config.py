"""Central configuration and LLVM tool discovery."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IR_DIR = DATA_DIR / "ir"
ART_DIR = ROOT / "artifacts"
RES_DIR = ROOT / "results"
for _d in (DATA_DIR, IR_DIR, ART_DIR, RES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 20                    # max pass-sequence length (NOP padded)
MIN_INSTS, MAX_INSTS = 40, 2500  # module size filter (IR instructions)
MODULES_JSON = DATA_DIR / "modules.json"
LABELS_JSONL = DATA_DIR / "labels.jsonl"
CKPT = ART_DIR / "surrogate.pt"
PASSES_OK = ART_DIR / "passes_ok.json"


def _find(name: str):
    d = os.environ.get("LLVM_BIN")
    if d:
        for ext in ("", ".exe"):
            p = Path(d) / (name + ext)
            if p.exists():
                return str(p)
    p = shutil.which(name)
    if p:
        return p
    for v in range(25, 13, -1):  # e.g. opt-18 on Ubuntu
        p = shutil.which(f"{name}-{v}")
        if p:
            return p
    return None


TOOLS = {n: _find(n) for n in ("opt", "clang", "lli", "llc", "llvm-size")}
IS_LLVM_AVAILABLE = bool(TOOLS.get("opt"))
DEMO_MODE = os.environ.get("DEMO_MODE", "").lower() == "true" or not IS_LLVM_AVAILABLE


def require(name: str) -> str:
    p = TOOLS.get(name)
    if not p:
        if DEMO_MODE:
            return f"simulated_{name}"
        raise RuntimeError(
            f"LLVM tool '{name}' not found. Install LLVM (see README) or set the "
            f"LLVM_BIN environment variable to the folder that contains {name}."
        )
    return p

