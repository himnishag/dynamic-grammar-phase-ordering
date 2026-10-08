"""Stage 1: features + baselines per module.  Stage 2: label sampled pass sequences with real `opt`.

  python generate_labels.py --seqs-per-module 40

Writes data/modules.json and data/labels.jsonl  (one line = module, 20-token sequence, #instructions).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from config import IR_DIR, LABELS_JSONL, MAX_INSTS, MIN_INSTS, MODULES_JSON, SEQ_LEN
from data_utils import split_of
from ir_features import featurize
from llvm_tools import WORKERS, Evaluator, opt_text
from pass_grammar import REFERENCE_SEQS, Grammar, NOP, get_space, repair, sample


def prep(path):
    try:
        ok, txt = opt_text(path, "verify")
        if not ok:
            return None
        feats, facts, n = featurize(txt)
        if not (MIN_INSTS <= n <= MAX_INSTS):
            return None
        ev = Evaluator(path, workers=1)
        base = ev.baselines()
        if any(math.isinf(v) for v in base.values()) or base["Oz"] > 0.97 * ev.init:
            return None  # broken or already optimised (e.g. optnone) -> useless for phase ordering
        return dict(path=str(path), init=ev.init, feats=feats.tolist(), facts=facts.tolist(),
                    **{k.lower(): v for k, v in base.items()})
    except Exception as e:  # noqa
        print("skip", path, type(e).__name__, e)
        return None


def make_sequences(n, facts, space, L, rng, gen):
    gd, gs, gn = Grammar(space, "dynamic"), Grammar(space, "static"), Grammar(space, "none")
    nd, ns, nn_ = int(n * 0.5), int(n * 0.15), int(n * 0.10)
    nm = n - nd - ns - nn_
    seqs = []
    for g, k in ((gd, nd), (gs, ns), (gn, nn_)):
        if k:
            seqs += sample(g, facts, k, L, gen).tolist()
    refs = [[space.idx[x] for x in r if x in space.idx][:L] for r in REFERENCE_SEQS]
    for _ in range(nm):
        base = list(rng.choice(refs))
        base += [0] * (L - len(base))
        for i in range(L):
            if rng.random() < rng.choice([0.1, 0.2, 0.35]):
                base[i] = rng.randrange(1, space.V)
        if rng.random() < 0.25:  # shorter pipelines
            cut = rng.randrange(3, L)
            base[cut:] = [0] * (L - cut)
        seqs.append(repair(gd, facts, base, gen=gen))
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs-per-module", type=int, default=40)
    ap.add_argument("--max-modules", type=int, default=100000)
    ap.add_argument("--tail", default="", help='pipeline appended after every sequence, e.g. "default<Oz>"')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    files = sorted(IR_DIR.glob("*.ll")) + sorted(IR_DIR.glob("*.bc"))
    files = files[: a.max_modules * 2]
    t0 = time.time()
    if MODULES_JSON.exists() and a.resume:
        mods = json.loads(MODULES_JSON.read_text())
    else:
        print(f"stage 1: features + baselines for {len(files)} files ...")
        mods = {}
        with ThreadPoolExecutor(WORKERS) as ex:
            for f, r in zip(files, ex.map(prep, files)):
                if r is not None:
                    r["split"] = split_of(f.stem)
                    mods[f.stem] = r
                    if len(mods) >= a.max_modules:
                        break
        MODULES_JSON.write_text(json.dumps(mods))
    from collections import Counter
    print("usable modules:", len(mods), dict(Counter(m["split"] for m in mods.values())),
          f"({time.time() - t0:.0f}s)")

    done = set()
    if a.resume and LABELS_JSONL.exists():
        for line in open(LABELS_JSONL):
            done.add(json.loads(line)["m"])
    elif LABELS_JSONL.exists():
        LABELS_JSONL.unlink()

    space = get_space()
    gen = torch.Generator().manual_seed(a.seed)
    rng = random.Random(a.seed)
    print(f"stage 2: labelling {a.seqs_per_module} sequences per module (V={space.V}, L={SEQ_LEN}) ...")
    with open(LABELS_JSONL, "a") as out:
        for k, (mid, m) in enumerate(mods.items()):
            if mid in done:
                continue
            facts = torch.tensor(m["facts"], dtype=torch.bool)
            seqs = make_sequences(a.seqs_per_module, facts, space, SEQ_LEN, rng, gen)
            ev = Evaluator(m["path"], tail=a.tail)
            sizes = ev.sizes([space.names(s) for s in seqs])
            for s, z in zip(seqs, sizes):
                out.write(json.dumps({"m": mid, "seq": space.names(s),
                                      "size": -1 if math.isinf(z) else int(z)}) + "\n")
            out.flush()
            if (k + 1) % 20 == 0:
                print(f"  {k + 1}/{len(mods)} modules  ({time.time() - t0:.0f}s)")
    print(f"done in {time.time() - t0:.0f}s -> {LABELS_JSONL}")


if __name__ == "__main__":
    main()
