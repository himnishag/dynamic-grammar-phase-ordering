"""Make the figures for the report from results/*.csv|json   ->  results/*.png"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import RES_DIR


def main():
    # 1. method comparison
    sp = RES_DIR / "summary.json"
    if sp.exists():
        s = json.loads(sp.read_text())
        names = [k for k in s if k not in ("-Oz",)]
        vals = [s[k]["gm_vs_oz"] for k in names]
        order = np.argsort(vals)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.barh([names[i] for i in order], [vals[i] for i in order], color="#3b6ea5")
        ax.axvline(1.0, color="k", ls="--", lw=1)
        ax.set_xlabel("geomean code size / -Oz   (lower is better)")
        ax.set_xlim(min(0.8, min(vals) - 0.02), max(vals) + 0.05)
        plt.tight_layout()
        plt.savefig(RES_DIR / "fig_methods.png", dpi=160)
        plt.close()
    # 2. per-module scatter + convergence
    cp = RES_DIR / "results.csv"
    if cp.exists():
        import csv
        rows = list(csv.DictReader(open(cp)))
        d = {r["module"]: float(r["oz"]) for r in rows}
        for m in ("diff-closedloop", "hillclimb"):
            xs = [(float(r["oz"]), float(r["size"])) for r in rows if r["method"] == m]
            if xs:
                fig, ax = plt.subplots(figsize=(4.8, 4.8))
                ax.scatter(*zip(*xs), s=18)
                lim = [0, max(max(x) for x in xs) * 1.05]
                ax.plot(lim, lim, "k--", lw=1)
                ax.set_xlabel("-Oz size (IR instr.)")
                ax.set_ylabel(f"{m} size")
                plt.tight_layout()
                plt.savefig(RES_DIR / f"fig_scatter_{m}.png", dpi=160)
                plt.close()
        hist = [json.loads(r["history"]) for r in rows if r["method"] == "diff-closedloop" and r["history"] != "[]"]
        if hist:
            n = min(len(h) for h in hist)
            oz = [float(r["oz"]) for r in rows if r["method"] == "diff-closedloop" and r["history"] != "[]"]
            curve = np.exp(np.mean([[np.log(h[i] / o) for i in range(n)] for h, o in zip(hist, oz)], 0))
            fig, ax = plt.subplots(figsize=(5, 3.6))
            ax.plot(range(1, n + 1), curve, "o-")
            ax.axhline(1, color="k", ls="--", lw=1)
            ax.set_xlabel("closed-loop round")
            ax.set_ylabel("geomean best size / -Oz")
            plt.tight_layout()
            plt.savefig(RES_DIR / "fig_convergence.png", dpi=160)
            plt.close()
    # 3. surrogate quality
    pp = RES_DIR / "surrogate_test_preds.npz"
    if pp.exists():
        z = np.load(pp)
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        ax.scatter(z["y"], z["pred"], s=3, alpha=0.3)
        lim = [min(z["y"].min(), z["pred"].min()), max(z["y"].max(), z["pred"].max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("true log(size/init)")
        ax.set_ylabel("predicted")
        plt.tight_layout()
        plt.savefig(RES_DIR / "fig_surrogate.png", dpi=160)
        plt.close()
    # 4. grammar ablation
    gp = RES_DIR / "grammar_ablation.json"
    if gp.exists():
        g = json.loads(gp.read_text())
        modes = list(g)
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
        for ax, key, title in zip(axs, ["invalid_rate", "wasted_pass_rate", "gm_best_of_n_vs_oz"],
                                  ["invalid pipelines", "wasted passes (no IR change)", "best-of-n size / -Oz"]):
            ax.bar(modes, [g[m][key] for m in modes], color=["#999", "#6a9", "#3b6ea5"])
            ax.set_title(title)
        plt.tight_layout()
        plt.savefig(RES_DIR / "fig_grammar.png", dpi=160)
        plt.close()
    print("figures written to", RES_DIR)


if __name__ == "__main__":
    main()
