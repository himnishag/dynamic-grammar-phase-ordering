"""Empirical study of the grammar itself (no learning involved).

For none / static / dynamic grammar: sample random pipelines and measure, with the real compiler,
  * invalid-pipeline rate (opt or the IR verifier rejects it)
  * wasted-pass rate  (a pass that leaves the IR byte-identical)
  * geomean size/Oz of the samples and of the best-of-n per module.

  python grammar_ablation.py --modules 12 --seqs 12
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor

import torch

from config import RES_DIR, SEQ_LEN
from data_utils import load_modules
from llvm_tools import WORKERS, count_insts, opt_text
from pass_grammar import Grammar, build_pipeline, get_space, sample


def analyse(path, space, seq):
    """Returns dict(valid, wasted, passes, size). Prefix-by-prefix to find no-op passes."""
    names = [space.tokens[i] for i in seq if i != 0]
    prev_hash, wasted, size = None, 0, None
    ok, txt = opt_text(path, "verify")
    prev_hash = hashlib.md5(txt.encode()).hexdigest()
    for k in range(1, len(names) + 1):
        ok, txt = opt_text(path, build_pipeline(names[:k], space))
        if not ok:
            return dict(valid=False, wasted=wasted, passes=len(names), size=math.inf)
        h = hashlib.md5(txt.encode()).hexdigest()
        wasted += h == prev_hash
        prev_hash = h
    if names:
        size = count_insts(txt)
    else:
        size = count_insts(opt_text(path, "verify")[1])
    return dict(valid=True, wasted=wasted, passes=len(names), size=size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", type=int, default=12)
    ap.add_argument("--seqs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    mods = load_modules()
    ids = sorted(m for m, v in mods.items() if v["split"] == "test")[: a.modules] or sorted(mods)[: a.modules]
    space = get_space()
    out = {}
    for mode in ("none", "static", "dynamic"):
        gen = torch.Generator().manual_seed(a.seed)
        g = Grammar(space, mode)
        jobs = []
        for mid in ids:
            facts = torch.tensor(mods[mid]["facts"], dtype=torch.bool)
            for s in sample(g, facts, a.seqs, SEQ_LEN, gen).tolist():
                jobs.append((mid, s))
        with ThreadPoolExecutor(WORKERS) as ex:
            res = list(ex.map(lambda j: analyse(mods[j[0]]["path"], space, j[1]), jobs))
        valid = [r for r in res if r["valid"]]
        passes = sum(r["passes"] for r in valid)
        ratios = [r["size"] / mods[j[0]]["oz"] for r, j in zip(res, jobs) if r["valid"]]
        best = {}
        for r, j in zip(res, jobs):
            if r["valid"]:
                best[j[0]] = min(best.get(j[0], math.inf), r["size"] / mods[j[0]]["oz"])
        gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")  # noqa
        out[mode] = dict(invalid_rate=1 - len(valid) / len(res),
                         wasted_pass_rate=sum(r["wasted"] for r in valid) / max(passes, 1),
                         mean_effective_len=passes / max(len(valid), 1),
                         gm_sample_size_vs_oz=gm(ratios), gm_best_of_n_vs_oz=gm(list(best.values())), n=len(res))
        print(mode, json.dumps(out[mode]))
    (RES_DIR / "grammar_ablation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
