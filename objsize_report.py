"""Real code size check: object-file bytes (llc + llvm-size) for the pipelines found by evaluate.py.

  python objsize_report.py          # reads results/results.csv, writes results/objsize.json
"""
from __future__ import annotations

import csv
import json
import math

from config import RES_DIR
from data_utils import load_modules
from llvm_tools import TOOLS, obj_text_size, opt_text


def gm(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def main():
    if not (TOOLS["llc"] and TOOLS["llvm-size"]):
        raise SystemExit("llc / llvm-size not found")
    mods = load_modules()
    rows = list(csv.DictReader(open(RES_DIR / "results.csv")))
    base = {}
    for mid in {r["module"] for r in rows}:
        p = mods[mid]["path"]
        base[mid] = {k: obj_text_size(opt_text(p, pipe)[1]) for k, pipe in
                     (("Oz", "default<Oz>"), ("O3", "default<O3>"))}
    out = {}
    for m in dict.fromkeys(r["method"] for r in rows):
        ratios_oz, ratios_o3 = [], []
        for r in rows:
            if r["method"] != m:
                continue
            ok, txt = opt_text(mods[r["module"]]["path"], r["pipeline"])
            s = obj_text_size(txt) if ok else None
            b = base[r["module"]]
            if s and b["Oz"] and b["O3"]:
                ratios_oz.append(s / b["Oz"])
                ratios_o3.append(s / b["O3"])
        out[m] = dict(gm_obj_vs_oz=gm(ratios_oz), gm_obj_vs_o3=gm(ratios_o3), n=len(ratios_oz))
        print(f"{m:26s} obj-size/Oz {out[m]['gm_obj_vs_oz']:.4f}  obj-size/O3 {out[m]['gm_obj_vs_o3']:.4f}  (n={len(ratios_oz)})")
    out["-O3"] = dict(gm_obj_vs_oz=gm([b["O3"] / b["Oz"] for b in base.values() if b["Oz"] and b["O3"]]), gm_obj_vs_o3=1.0)
    print(f"{'-O3':26s} obj-size/Oz {out['-O3']['gm_obj_vs_oz']:.4f}")
    (RES_DIR / "objsize.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
