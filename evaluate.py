"""Benchmark every method on the held-out test modules with the SAME real-compiler budget.

  python evaluate.py --n-test 20 --rounds 5 --per-round 10        # budget = 50 compiler calls
  python evaluate.py --n-test 20 --ablate                         # + static/none grammar variants

Output: results/results.csv (one row per module x method) and results/summary.json
Size = number of IR instructions after the pipeline (lower is better).
ratio_oz < 1  means smaller than clang's -Oz pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time

import torch

from baselines import genetic, hill_climb, random_search
from config import CKPT, RES_DIR
from data_utils import load_modules
from diff_search import DiffSearcher, closed_loop, open_loop
from llvm_tools import Evaluator, equivalent, verify_pipeline
from pass_grammar import get_space
from surrogate import load_ensemble


def geomean(xs):
    xs = [x for x in xs if x > 0 and not math.isinf(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--per-round", type=int, default=10)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--B", type=int, default=48)
    ap.add_argument("--kappa", type=float, default=0.5)
    ap.add_argument("--ablate", action="store_true", help="also run diff search with static / no grammar")
    ap.add_argument("--tail", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    budget = a.rounds * a.per_round

    mods = load_modules()
    ids = sorted(m for m, v in mods.items() if v["split"] == "test")
    if not ids:
        print("no test modules, using val split")
        ids = sorted(m for m, v in mods.items() if v["split"] == "val")
    ids = ids[: a.n_test]
    ens, extra = load_ensemble(CKPT, a.device)
    space = get_space(a.device)
    assert ens.tokens == space.tokens, "checkpoint was trained with a different pass vocabulary -> retrain"
    searchers = {m: DiffSearcher(ens, space, m, prior=extra.get("prior"), device=a.device)
                 for m in (["dynamic"] + (["static", "none"] if a.ablate else []))}

    rows = []
    t_all = time.time()
    for k, mid in enumerate(ids):
        m = mods[mid]
        feats = torch.tensor(m["feats"], dtype=torch.float32)
        facts = torch.tensor(m["facts"], dtype=torch.bool)
        init, oz, o3 = m["init"], m["oz"], m["o3"]
        res = {}

        def timed(name, fn):
            t0 = time.time()
            ev = Evaluator(m["path"], a.tail)
            r = fn(ev)
            r["time"] = time.time() - t0
            res[name] = (r, ev)

        timed("random-dynamic", lambda ev: random_search(ev, space, facts, budget, "dynamic", a.seed))
        timed("random-nogrammar", lambda ev: random_search(ev, space, facts, budget, "none", a.seed))
        timed("hillclimb", lambda ev: hill_climb(ev, space, facts, budget, "dynamic", a.seed))
        timed("genetic", lambda ev: genetic(ev, space, facts, budget, "dynamic", a.seed))
        timed("diff-openloop", lambda ev: open_loop(searchers["dynamic"], ev, feats, facts, budget,
                                                    a.steps, a.B, a.kappa, a.seed))
        timed("diff-closedloop", lambda ev: closed_loop(searchers["dynamic"], ev, feats, facts, a.rounds,
                                                        a.per_round, a.steps, a.B, a.kappa, a.seed))
        if a.ablate:
            for g in ("static", "none"):
                timed(f"diff-closedloop-{g}", lambda ev, g=g: closed_loop(
                    searchers[g], ev, feats, facts, a.rounds, a.per_round, a.steps, a.B, a.kappa, a.seed))
        # zero-shot = best-predicted single candidate (1 compiler call equivalent)
        r_open = res["diff-openloop"][0]
        res["diff-zeroshot"] = ({"best_size": r_open["zeroshot_size"], "best_seq": r_open["zeroshot_seq"],
                                 "calls": 1, "time": r_open["time"]}, res["diff-openloop"][1])

        for name, (r, ev) in res.items():
            names = r["best_seq"] or []
            pipe = ev.pipe(names)
            ok, txt = verify_pipeline(m["path"], pipe)
            eq = equivalent(m["path"], txt) if ok else None
            rows.append(dict(module=mid, method=name, init=init, oz=oz, o3=o3, size=r["best_size"],
                             ratio_oz=r["best_size"] / oz, ratio_o3=r["best_size"] / o3,
                             calls=r["calls"], verified=ok, equivalent=eq, time=round(r["time"], 2),
                             pipeline=pipe, history=json.dumps(r.get("history", []))))
        print(f"[{k + 1}/{len(ids)}] {mid}: init={init} Oz={oz} O3={o3} | " +
              " ".join(f"{n.replace('diff-', 'D-')}={r['best_size']}" for n, (r, _) in res.items()) +
              f"  ({time.time() - t_all:.0f}s)")
        with open(RES_DIR / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    summary = {}
    methods = list(dict.fromkeys(r["method"] for r in rows))
    print(f"\n{'method':26s} {'GM size/Oz':>10s} {'GM size/O3':>10s} {'GM red.vs init':>14s} "
          f"{'win/tie/loss vs Oz':>19s} {'calls':>6s} {'verified':>8s} {'time(s)':>8s}")
    for mth in methods + ["-Oz", "-O3"]:
        if mth == "-Oz":
            sz = [r["oz"] for r in rows if r["method"] == methods[0]]
            init_ = [r["init"] for r in rows if r["method"] == methods[0]]
            entry = dict(gm_vs_oz=1.0, gm_vs_o3=geomean([s / r["o3"] for s, r in zip(sz, [r for r in rows if r["method"] == methods[0]])]),
                         red=1 - geomean([s / i for s, i in zip(sz, init_)]), wtl="-", calls=0, ver=1.0, t=0.0)
        elif mth == "-O3":
            sub = [r for r in rows if r["method"] == methods[0]]
            entry = dict(gm_vs_oz=geomean([r["o3"] / r["oz"] for r in sub]), gm_vs_o3=1.0,
                         red=1 - geomean([r["o3"] / r["init"] for r in sub]), wtl="-", calls=0, ver=1.0, t=0.0)
        else:
            sub = [r for r in rows if r["method"] == mth]
            w = sum(r["size"] < r["oz"] for r in sub)
            t = sum(r["size"] == r["oz"] for r in sub)
            entry = dict(gm_vs_oz=geomean([r["ratio_oz"] for r in sub]), gm_vs_o3=geomean([r["ratio_o3"] for r in sub]),
                         red=1 - geomean([r["size"] / r["init"] for r in sub]),
                         wtl=f"{w}/{t}/{len(sub) - w - t}", calls=sum(r["calls"] for r in sub) / len(sub),
                         ver=sum(bool(r["verified"]) for r in sub) / len(sub),
                         t=sum(r["time"] for r in sub) / len(sub))
        summary[mth] = entry
        print(f"{mth:26s} {entry['gm_vs_oz']:10.4f} {entry['gm_vs_o3']:10.4f} {entry['red'] * 100:13.1f}% "
              f"{entry['wtl']:>19s} {entry['calls']:6.0f} {entry['ver'] * 100:7.0f}% {entry['t']:8.1f}")
    bad = [r for r in rows if r["equivalent"] is False]
    print(f"\nprograms whose output changed after optimisation: {len(bad)} (of "
          f"{sum(r['equivalent'] is not None for r in rows)} checked)")
    (RES_DIR / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
