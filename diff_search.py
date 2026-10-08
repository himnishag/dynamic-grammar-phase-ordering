"""Grammar-constrained, differentiable search over pass sequences  (the core contribution).

For one module:
  * theta[B,L,V] are free logits of B parallel restarts.
  * At step t the *dynamic grammar* (tracked on the hard prefix) masks illegal passes, a
    straight-through Gumbel-softmax samples a one-hot pass, and the grammar state advances.
  * The relaxed sequence goes through the differentiable surrogate; gradients move theta.
  * Every hard sample is a valid discrete sequence, so all of them become candidates.
  * Closed loop: the best candidates are compiled for real; the observed residuals
    (true - predicted) become a differentiable kernel correction for the next round.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from config import SEQ_LEN
from pass_grammar import Grammar, NOP, PassSpace

NEG = -1e9


def _gumbel(like):
    u = torch.rand_like(like).clamp(1e-9, 1 - 1e-7)
    return -torch.log(-torch.log(u))


class DiffSearcher:
    def __init__(self, ens, space: PassSpace, mode="dynamic", L=SEQ_LEN, prior=None, device="cpu"):
        self.ens, self.space, self.L, self.dev = ens.to(device).eval(), space.to(device), L, device
        self.grammar = Grammar(space, mode)
        self.mode = mode
        self.prior = None if prior is None else prior.to(device)

    # ------------------------------------------------------------------ objective pieces
    def _correction(self, Y, obs, beta=8.0, lam=0.3):
        """Differentiable kernel regression of observed residuals on (soft) sequences."""
        if obs is None or len(obs[1]) == 0:
            return 0.0
        oy, r = obs  # oy [M,L,V] one-hot, r [M]
        sim = torch.einsum("blv,mlv->bm", Y, oy) / Y.shape[1]
        k = torch.exp(-beta * (1.0 - sim))
        return (k * r[None]).sum(1) / (lam + k.sum(1))

    @torch.no_grad()
    def base_predict(self, feats, seqs):
        x = F.one_hot(torch.as_tensor(seqs, device=self.dev), self.space.V).float()
        f = feats.to(self.dev)[None].expand(len(seqs), -1)
        m, s = self.ens(f, x)
        return m.cpu(), s.cpu()

    # ------------------------------------------------------------------ search
    def search(self, feats, facts, obs=None, steps=80, B=48, lr=0.25, tau=(1.5, 0.4),
               kappa=0.5, use_prior=True, seed=0):
        """Returns list of (objective, tuple(token ids)) sorted ascending (lower = smaller code)."""
        torch.manual_seed(seed)
        V, L, dev = self.space.V, self.L, self.dev
        theta = 0.3 * torch.randn(B, L, V, device=dev)
        if use_prior and self.prior is not None:
            theta = theta + 0.7 * torch.log(self.prior)[None]
        theta.requires_grad_(True)
        opt = torch.optim.Adam([theta], lr=lr)
        f = feats.to(dev)[None].expand(B, -1)
        facts_b = facts.to(dev)[None].expand(B, -1)
        cand: dict = {}
        for s in range(steps):
            t_cur = tau[0] * (tau[1] / tau[0]) ** (s / max(steps - 1, 1))
            st = self.grammar.init_state(facts_b)
            Ys, toks = [], []
            for t in range(L):
                logits = theta[:, t, :].masked_fill(~self.grammar.mask(st), NEG)
                g = _gumbel(logits)
                soft = F.softmax((logits + g) / t_cur, -1)
                idx = soft.argmax(-1)
                y = F.one_hot(idx, V).float() + soft - soft.detach()  # straight-through
                st = self.grammar.step(st, idx)
                Ys.append(y)
                toks.append(idx)
            Y = torch.stack(Ys, 1)
            mean, std = self.ens(f, Y)
            obj = mean + self._correction(Y, obs) + kappa * std
            opt.zero_grad()
            obj.mean().backward()
            opt.step()
            T = torch.stack(toks, 1).cpu().tolist()
            for row, o in zip(T, obj.detach().cpu().tolist()):
                key = tuple(row)
                if key not in cand or o < cand[key]:
                    cand[key] = o
        # deterministic decode of the learned logits (argmax under the grammar)
        with torch.no_grad():
            st = self.grammar.init_state(facts_b)
            toks = []
            for t in range(L):
                idx = theta[:, t, :].masked_fill(~self.grammar.mask(st), NEG).argmax(-1)
                st = self.grammar.step(st, idx)
                toks.append(idx)
            T = torch.stack(toks, 1)
            Y = F.one_hot(T, V).float()
            mean, std = self.ens(f, Y)
            obj = mean + self._correction(Y, obs) + kappa * std
            for row, o in zip(T.cpu().tolist(), obj.cpu().tolist()):
                key = tuple(row)
                cand[key] = min(cand.get(key, 1e9), o)
        return sorted(((o, k) for k, o in cand.items()), key=lambda x: x[0])


def effective(seq_ids, space):
    """Drop NOPs -> list of pass names (what actually reaches the compiler)."""
    return [space.tokens[i] for i in seq_ids if i != 0]


def closed_loop(searcher: DiffSearcher, ev, feats, facts, rounds=5, per_round=10, steps=80, B=48,
                kappa=0.5, seed=0, verbose=False):
    """Active-learning loop. Total real compiler calls <= rounds * per_round (+0)."""
    space = searcher.space
    obs_y, obs_r = [], []
    best = (math.inf, None)
    seen: set = set()
    init = ev.init
    history = []
    for r in range(rounds):
        obs = None
        if obs_y:
            obs = (torch.stack(obs_y).to(searcher.dev), torch.tensor(obs_r, device=searcher.dev))
        cands = searcher.search(feats, facts, obs=obs, steps=steps, B=B, kappa=kappa, seed=seed + r)
        picked = []
        for o, key in cands:
            eff = tuple(effective(key, space))
            if eff in seen:
                continue
            seen.add(eff)
            picked.append(key)
            if len(picked) >= per_round:
                break
        if not picked:
            break
        sizes = ev.sizes([effective(k, space) for k in picked])
        base_mean, _ = searcher.base_predict(feats, [list(k) for k in picked])
        for k, z, bm in zip(picked, sizes, base_mean.tolist()):
            if math.isinf(z):
                continue
            y_true = math.log(z / init)
            obs_y.append(F.one_hot(torch.tensor(k), space.V).float())
            obs_r.append(y_true - bm)
            if z < best[0]:
                best = (z, effective(k, space))
        history.append(best[0])
        if verbose:
            print(f"   round {r + 1}: best={best[0]}  (init={init})")
    return dict(best_size=best[0], best_seq=best[1], calls=ev.calls, history=history)


def open_loop(searcher: DiffSearcher, ev, feats, facts, budget=50, steps=80, B=48, kappa=0.5, seed=0):
    """Single search, then compile the top-`budget` candidates (no feedback)."""
    space = searcher.space
    cands = searcher.search(feats, facts, steps=steps, B=B, kappa=kappa, seed=seed)
    seen, picked = set(), []
    for o, key in cands:
        eff = tuple(effective(key, space))
        if eff in seen:
            continue
        seen.add(eff)
        picked.append(eff)
        if len(picked) >= budget:
            break
    sizes = ev.sizes([list(p) for p in picked])
    i = min(range(len(sizes)), key=lambda j: sizes[j])
    return dict(best_size=sizes[i], best_seq=list(picked[i]), calls=ev.calls,
                zeroshot_size=sizes[0], zeroshot_seq=list(picked[0]))
