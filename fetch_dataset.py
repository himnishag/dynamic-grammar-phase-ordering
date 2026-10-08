"""Build the IR corpus in data/ir/.

  python fetch_dataset.py --source synthetic --n 400     # works offline, recommended first run
  python fetch_dataset.py --source c-dir --path C:\\code\\AnghaBench_subset --n 800
  python fetch_dataset.py --source compile --n 800        # real-world IR from HuggingFace ComPile

Every module is stored as *unoptimised* IR (what a compiler would hand to the optimiser).
"""
from __future__ import annotations

import argparse
import hashlib
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import IR_DIR, MAX_INSTS, MIN_INSTS
from gen_synthetic_c import generate
from llvm_tools import WORKERS, clang_to_ll, count_insts, opt_text


def from_synthetic(n, seed):
    def one(i):
        out = IR_DIR / f"syn_{seed + i:05d}.ll"
        if out.exists():
            return True
        with tempfile.TemporaryDirectory() as td:
            c = Path(td) / "p.c"
            c.write_text(generate(seed + i))
            return clang_to_ll(c, out)

    with ThreadPoolExecutor(WORKERS) as ex:
        ok = sum(ex.map(one, range(n)))
    print(f"synthetic: {ok}/{n} programs compiled")


def from_c_dir(path, n):
    files = sorted(Path(path).rglob("*.c"))[: n * 3]

    def one(f):
        h = hashlib.sha1(str(f).encode()).hexdigest()[:10]
        out = IR_DIR / f"c_{h}.ll"
        if out.exists():
            return True
        return clang_to_ll(f, out)

    kept = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for ok in ex.map(one, files):
            kept += bool(ok)
            if kept >= n:
                break
    print(f"c-dir: {kept} modules compiled from {len(files)} files")


def from_compile(n, lang="c", min_bytes=1500, max_bytes=60000):
    """Stream bitcode from the ComPile dataset (https://huggingface.co/datasets/llvm-ml/ComPile)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("pip install datasets   (needed only for --source compile)")
    try:
        ds = load_dataset("llvm-ml/ComPile", lang, split="train", streaming=True)
    except Exception as e:  # config name differs between dataset revisions
        print("config", lang, "not accepted (", type(e).__name__, ") -> loading default and filtering")
        ds = load_dataset("llvm-ml/ComPile", split="train", streaming=True)
    seen, kept = set(), 0
    for row in ds:
        if row.get("language", lang) != lang:
            continue
        bc = row["content"]
        if not (min_bytes <= len(bc) <= max_bytes):
            continue
        h = hashlib.sha1(bc).hexdigest()[:12]
        if h in seen:
            continue
        seen.add(h)
        out = IR_DIR / f"cp_{h}.bc"
        out.write_bytes(bc)
        ok, txt = opt_text(out, "strip-debug")
        if not ok:
            out.unlink(missing_ok=True)
            continue
        if not (MIN_INSTS <= count_insts(txt) <= MAX_INSTS):
            out.unlink(missing_ok=True)
            continue
        (IR_DIR / f"cp_{h}.ll").write_text(txt)
        out.unlink(missing_ok=True)
        kept += 1
        if kept % 50 == 0:
            print("kept", kept)
        if kept >= n:
            break
    print(f"compile: kept {kept} modules")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "c-dir", "compile"], default="synthetic")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--path", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.source == "synthetic":
        from_synthetic(a.n, a.seed)
    elif a.source == "c-dir":
        if not a.path:
            raise SystemExit("--path required")
        from_c_dir(a.path, a.n)
    else:
        from_compile(a.n)
    print("IR files in", IR_DIR, ":", len(list(IR_DIR.glob('*.ll'))) + len(list(IR_DIR.glob('*.bc'))))
