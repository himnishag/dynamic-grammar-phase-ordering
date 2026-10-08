"""Search baselines that get the SAME real-compiler budget as the differentiable method."""
from __future__ import annotations

import math
import random

import torch

from config import SEQ_LEN
from diff_search import effective
from pass_grammar import Grammar, PassSpace, repair, sample


def _best(ev, space, seqs):
    names = [effective(s, space) for s in seqs]
    sizes = ev.sizes(names)
    i = min(range(len(sizes)), key=lambda j: sizes[j])
    return sizes[i], names[i]


def random_search(ev, space: PassSpace, facts, budget, mode="dynamic", seed=0, L=SEQ_LEN):
    gen = torch.Generator().manual_seed(seed)
    seqs = sample(Grammar(space, mode), facts, budget, L, gen).tolist()
    s, n = _best(ev, space, seqs)
    return dict(best_size=s, best_seq=n, calls=ev.calls)


def hill_climb(ev, space, facts, budget, mode="dynamic", seed=0, L=SEQ_LEN, batch=5):
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    g = Grammar(space, mode)
    cur = sample(g, facts, 1, L, gen)[0].tolist()
    cur_size = ev.sizes([effective(cur, space)])[0]
    best_names = effective(cur, space)
    tries = 0
    while ev.calls < budget and tries < budget * 10:
        tries += 1
        nbrs = []
        for _ in range(min(batch, budget - ev.calls)):
            s = list(cur)
            op = rng.random()
            i, j = rng.randrange(L), rng.randrange(L)
            if op < 0.5:
                s[i] = rng.randrange(1, space.V)
            elif op < 0.8:
                s[i], s[j] = s[j], s[i]
            else:
                s.insert(i, rng.randrange(1, space.V))
                s.pop()
            nbrs.append(repair(g, facts, s, gen=gen))
        sizes = ev.sizes([effective(n, space) for n in nbrs])
        k = min(range(len(sizes)), key=lambda x: sizes[x])
        if sizes[k] <= cur_size:
            cur, cur_size = nbrs[k], sizes[k]
            best_names = effective(cur, space)
    return dict(best_size=cur_size, best_seq=best_names, calls=ev.calls)


def genetic(ev, space, facts, budget, mode="dynamic", seed=0, L=SEQ_LEN, pop=10):
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    g = Grammar(space, mode)
    P = sample(g, facts, pop, L, gen).tolist()
    best = (math.inf, None)
    while True:
        sizes = ev.sizes([effective(p, space) for p in P])
        for p, z in zip(P, sizes):
            if z < best[0]:
                best = (z, effective(p, space))
        if ev.calls + pop > budget:
            break
        order = sorted(range(len(P)), key=lambda i: sizes[i])
        elite = [P[i] for i in order[: max(2, pop // 3)]]
        new = list(elite)
        while len(new) < pop:
            a, b = rng.sample(elite, 2) if len(elite) > 1 else (elite[0], elite[0])
            cut = rng.randrange(1, L)
            child = a[:cut] + b[cut:]
            for i in range(L):
                if rng.random() < 0.08:
                    child[i] = rng.randrange(1, space.V)
            new.append(repair(g, facts, child, gen=gen))
        P = new
    return dict(best_size=best[0], best_seq=best[1], calls=ev.calls)
