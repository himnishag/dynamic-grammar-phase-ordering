"""Optimise ONE module:  python optimize.py my_module.ll [--rounds 5 --per-round 10]

Input : LLVM IR (.ll / .bc)  (or a .c file - compiled with clang to unoptimised IR first)
Output: pass pipeline string, sizes (init / ours / -Oz / -O3), correctness check, optimized IR file
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from config import CKPT
from diff_search import DiffSearcher, closed_loop
from ir_features import featurize
from llvm_tools import Evaluator, clang_to_ll, equivalent, opt_text, verify_pipeline
from pass_grammar import get_space
from surrogate import load_ensemble


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--per-round", type=int, default=10)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--B", type=int, default=48)
    ap.add_argument("--kappa", type=float, default=0.5)
    ap.add_argument("--grammar", default="dynamic", choices=["dynamic", "static", "none"])
    ap.add_argument("--tail", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    path = Path(a.input)
    if path.suffix == ".c":
        ll = Path(tempfile.mkdtemp()) / (path.stem + ".ll")
        if not clang_to_ll(path, ll):
            raise SystemExit("clang failed on the C file")
        path = ll
    ok, text = opt_text(path, "verify")
    if not ok:
        raise SystemExit("cannot parse IR:\n" + text)
    feats, facts, n = featurize(text)
    ens, extra = load_ensemble(CKPT, "cpu")
    space = get_space()
    searcher = DiffSearcher(ens, space, a.grammar, prior=extra.get("prior"))
    ev = Evaluator(path, a.tail)
    base = ev.baselines()
    res = closed_loop(searcher, ev, torch.tensor(feats), torch.tensor(facts), a.rounds, a.per_round,
                      a.steps, a.B, a.kappa, verbose=True)
    pipe = ev.pipe(res["best_seq"] or [])
    ok, opt_ir = verify_pipeline(path, pipe)
    eq = equivalent(path, opt_ir) if ok else None
    out_ll = Path(a.out) if a.out else Path(path.stem + ".optimized.ll")
    if ok:
        out_ll.write_text(opt_ir)
    report = dict(input=str(a.input), pipeline=pipe, initial_size=ev.init, our_size=res["best_size"],
                  oz_size=base["Oz"], o3_size=base["O3"],
                  reduction_vs_init_pct=round(100 * (1 - res["best_size"] / ev.init), 2),
                  ratio_vs_oz=round(res["best_size"] / base["Oz"], 4),
                  compiler_calls=res["calls"], verified_each_pass=ok, output_identical=eq,
                  optimized_ir=str(out_ll) if ok else None)
    print(json.dumps(report, indent=2))
    Path(path.stem + ".report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
