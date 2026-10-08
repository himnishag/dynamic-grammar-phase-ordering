"""Train the surrogate ensemble on data/labels.jsonl.

  python train.py --epochs 40 --models 3
Reports MAE (log-ratio), per-module Spearman rho, and regret@1 on the val and test modules.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from config import CKPT, RES_DIR, SEQ_LEN
from data_utils import load_labels, load_modules
from pass_grammar import get_space
from surrogate import Ensemble, Surrogate, save_ensemble


def build(mods, rows, space):
    ids = sorted(mods)
    mid = {m: i for i, m in enumerate(ids)}
    keep = [r for r in rows if r["size"] > 0 and r["m"] in mid and len(r["seq"]) == SEQ_LEN]
    mod_of = np.array([mid[r["m"]] for r in keep])
    seq = torch.tensor([space.ids(r["seq"]) for r in keep], dtype=torch.long)
    init = np.array([mods[r["m"]]["init"] for r in keep], dtype=np.float32)
    y = torch.tensor(np.log(np.array([r["size"] for r in keep], dtype=np.float32) / init).clip(-3, 1.5))
    feats = torch.tensor(np.stack([mods[i]["feats"] for i in ids]), dtype=torch.float32)
    split = np.array([mods[i]["split"] for i in ids])
    return ids, mod_of, seq, y, feats, split


@torch.no_grad()
def predict(ens, feats, mod_of, seq, V, dev, bs=2048):
    out = []
    for i in range(0, len(seq), bs):
        x = F.one_hot(seq[i:i + bs], V).float().to(dev)
        f = feats[torch.as_tensor(mod_of[i:i + bs])].to(dev)
        m, s = ens(f, x)
        out.append(torch.stack([m, s]).cpu())
    o = torch.cat(out, 1)
    return o[0].numpy(), o[1].numpy()


def metrics(pred, y, mod_of):
    y = y.numpy()
    mae = float(np.abs(pred - y).mean())
    rhos, reg1, reg5 = [], [], []
    for m in np.unique(mod_of):
        ix = np.where(mod_of == m)[0]
        if len(ix) < 6:
            continue
        r = spearmanr(pred[ix], y[ix]).correlation
        if not np.isnan(r):
            rhos.append(r)
        best = y[ix].min()
        reg1.append(np.exp(y[ix][np.argmin(pred[ix])] - best) - 1)
        reg5.append(np.exp(y[ix][np.argsort(pred[ix])[:5]].min() - best) - 1)
    return dict(mae=mae, spearman=float(np.mean(rhos)), regret_top1=float(np.mean(reg1)),
                regret_top5=float(np.mean(reg5)), n=int(len(y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--models", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--M", type=int, default=16, help="modules per batch")
    ap.add_argument("--S", type=int, default=8, help="sequences per module per batch")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--rank-weight", type=float, default=0.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if (a.device == "auto" and torch.cuda.is_available()) else ("cpu" if a.device == "auto" else a.device)
    print("device:", dev)

    space = get_space()
    mods, rows = load_modules(), load_labels()
    ids, mod_of, seq, y, feats, split = build(mods, rows, space)
    V, D = space.V, feats.shape[1]
    tr_mods = np.where(split == "train")[0]
    mu = feats[tr_mods].mean(0)
    sd = feats[tr_mods].std(0).clamp_min(1e-3)
    fn = ((feats - mu) / sd)
    by_mod = {m: np.where(mod_of == m)[0] for m in tr_mods if (mod_of == m).sum() > 0}
    print(f"samples: {len(seq)}, modules train/val/test = "
          f"{(split == 'train').sum()}/{(split == 'val').sum()}/{(split == 'test').sum()}, V={V}")
    rng = np.random.default_rng(a.seed)

    vmask = np.isin(mod_of, np.where(split == "val")[0])
    members = []
    for k in range(a.models):
        torch.manual_seed(a.seed + k)
        model = Surrogate(D, V, SEQ_LEN, H=a.hidden).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
        mlist = [m for m in by_mod if rng.random() < 0.85] or list(by_mod)  # bootstrap modules
        steps = max(1, int(sum(len(by_mod[m]) for m in mlist) / (a.M * a.S)))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps)
        best, best_state = -1e9, None
        t0 = time.time()
        for ep in range(a.epochs):
            model.train()
            tot = 0.0
            for _ in range(steps):
                chosen = rng.choice(mlist, size=min(a.M, len(mlist)), replace=False)
                idx = np.concatenate([rng.choice(by_mod[m], a.S, replace=len(by_mod[m]) < a.S) for m in chosen])
                x = F.one_hot(seq[idx], V).float().to(dev)
                f = fn[torch.as_tensor(mod_of[idx])].to(dev)
                t = y[idx].to(dev)
                p = model(f, x)
                reg = F.smooth_l1_loss(p, t, beta=0.1)
                pr, yr = p.view(len(chosen), a.S), t.view(len(chosen), a.S)
                dp, dy = pr[:, :, None] - pr[:, None, :], yr[:, :, None] - yr[:, None, :]
                w = (dy.abs() > 0.02).float()
                rank = (w * F.softplus(-torch.sign(dy) * dp * 8.0)).sum() / w.sum().clamp_min(1)
                loss = reg + a.rank_weight * rank
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += loss.item()
            if (ep + 1) % 5 == 0 or ep == a.epochs - 1:
                model.eval()
                tmp = Ensemble([model], mu, sd, space.tokens, SEQ_LEN).to(dev)
                pv, _ = predict(tmp, feats, mod_of[vmask], seq[vmask], V, dev)
                mt = metrics(pv, y[vmask], mod_of[vmask])
                score = mt["spearman"] - mt["mae"]
                if score > best:
                    best, best_state = score, {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
                print(f"  model {k} ep {ep + 1:3d} loss {tot / steps:.4f}  val MAE {mt['mae']:.4f} "
                      f"rho {mt['spearman']:.3f} regret@1 {mt['regret_top1']:.3f}  ({time.time() - t0:.0f}s)")
        model.load_state_dict(best_state)
        members.append(model.eval())

    ens = Ensemble(members, mu, sd, space.tokens, SEQ_LEN).to(dev).eval()
    # prior over tokens per position, from the best 20% of sequences of every training module
    prior = torch.zeros(SEQ_LEN, V)
    for m, ix in by_mod.items():
        cut = np.quantile(y[ix].numpy(), 0.2)
        for j in ix[y[ix].numpy() <= cut]:
            prior[torch.arange(SEQ_LEN), seq[j]] += 1
    prior = (prior + 0.5) / (prior.sum(-1, keepdim=True) + 0.5 * V)
    res = {}
    for name in ("val", "test"):
        mk = np.isin(mod_of, np.where(split == name)[0])
        if mk.sum() == 0:
            continue
        pm, ps = predict(ens, feats, mod_of[mk], seq[mk], V, dev)
        res[name] = metrics(pm, y[mk], mod_of[mk])
        if name == "test":
            np.savez(RES_DIR / "surrogate_test_preds.npz", pred=pm, std=ps, y=y[mk].numpy(), mod=mod_of[mk])
        print(name, json.dumps(res[name]))
    (RES_DIR / "surrogate_metrics.json").write_text(json.dumps(res, indent=1))
    save_ensemble(CKPT, ens.cpu(), extra={"prior": prior, "metrics": res})
    print("saved", CKPT)


if __name__ == "__main__":
    main()
