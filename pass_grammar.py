"""Pass vocabulary + dynamic (state-dependent) grammar.

Three grammar modes (used for the ablation):
  none     : every token allowed everywhere (plain unconstrained search space).
  static   : structure + facts of the *input IR only* (no loops -> no loop passes,
             no calls -> no inline, ...), no immediate repeats, NOP is absorbing.
  dynamic  : static + an abstract *IR state* that evolves with every pass that is
             chosen (promoted allocas, canonical loop form, "already ran and nothing
             big changed since" bits).  The mask at step t depends on the prefix.

All state updates are tensor ops, so thousands of prefixes are tracked in parallel
inside the differentiable search loop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import torch

from config import PASSES_OK

NOP = "<nop>"
FACTS = ["ALLOCA", "LOOPS", "CALLS", "MULTI", "CANON"]
FI = {f: i for i, f in enumerate(FACTS)}
CONST_FACTS = ("LOOPS", "CALLS", "MULTI")  # never change during a pipeline (static view)

# name, level, need, forbid, sets, clears, big
#  level : module | cgscc | function | loop
#  need  : facts that must be true      forbid: facts that must be false
#  sets/clears : fact updates after running the pass
#  big   : structural pass -> re-enables every pass that was blocked by the "done" bit
_SPEC = [
    ("mem2reg", "function", ["ALLOCA"], [], [], ["ALLOCA"], True),
    ("sroa", "function", ["ALLOCA"], [], [], ["ALLOCA"], True),
    ("instcombine", "function", [], [], [], [], True),
    ("aggressive-instcombine", "function", [], ["ALLOCA"], [], [], False),
    ("instsimplify", "function", [], [], [], [], False),
    ("simplifycfg", "function", [], [], [], ["CANON"], True),
    ("early-cse", "function", [], [], [], [], False),
    ("gvn", "function", [], ["ALLOCA"], [], [], True),
    ("dse", "function", [], [], [], [], False),
    ("dce", "function", [], [], [], [], False),
    ("adce", "function", [], [], [], [], False),
    ("bdce", "function", [], ["ALLOCA"], [], [], False),
    ("sccp", "function", [], ["ALLOCA"], [], [], True),
    ("jump-threading", "function", [], ["ALLOCA"], [], ["CANON"], True),
    ("correlated-propagation", "function", [], ["ALLOCA"], [], [], False),
    ("reassociate", "function", [], [], [], [], False),
    ("memcpyopt", "function", [], [], [], [], False),
    ("tailcallelim", "function", ["CALLS"], [], [], [], False),
    ("sink", "function", [], ["ALLOCA"], [], [], False),
    ("mldst-motion", "function", [], ["ALLOCA"], [], [], False),
    ("div-rem-pairs", "function", [], [], [], [], False),
    ("float2int", "function", [], [], [], [], False),
    ("constraint-elimination", "function", [], ["ALLOCA"], [], [], False),
    ("loop-simplify", "function", ["LOOPS"], ["CANON"], ["CANON"], [], False),
    ("loop-unroll", "function", ["LOOPS", "CANON"], ["ALLOCA"], [], ["CANON"], True),
    ("loop-load-elim", "function", ["LOOPS", "CANON"], ["ALLOCA"], [], [], False),
    ("loop-rotate", "loop", ["LOOPS", "CANON"], ["ALLOCA"], [], [], True),
    ("licm", "loop", ["LOOPS", "CANON"], ["ALLOCA"], [], [], True),
    ("indvars", "loop", ["LOOPS", "CANON"], ["ALLOCA"], [], [], True),
    ("loop-deletion", "loop", ["LOOPS", "CANON"], ["ALLOCA"], [], [], False),
    ("loop-idiom", "loop", ["LOOPS", "CANON"], ["ALLOCA"], [], [], False),
    ("loop-instsimplify", "loop", ["LOOPS", "CANON"], ["ALLOCA"], [], [], False),
    ("loop-reduce", "loop", ["LOOPS", "CANON"], ["ALLOCA"], [], [], False),
    ("loop-unroll-full", "loop", ["LOOPS", "CANON"], ["ALLOCA"], [], ["CANON"], True),
    ("loop-simplifycfg", "loop", ["LOOPS", "CANON"], ["ALLOCA"], [], [], False),
    ("inline", "module", ["CALLS"], ["ALLOCA"], [], ["CANON"], True),
    ("globaldce", "module", [], [], [], [], False),
    ("globalopt", "module", [], ["ALLOCA"], [], [], True),
    ("ipsccp", "module", ["MULTI"], ["ALLOCA"], [], ["CANON"], True),
    ("deadargelim", "module", ["MULTI"], [], [], [], False),
    ("mergefunc", "module", ["MULTI"], [], [], [], False),
    ("function-attrs", "cgscc", ["CALLS"], [], [], [], False),
    ("argpromotion", "cgscc", ["MULTI", "CALLS"], ["ALLOCA"], [], [], True),
]

# Hand written reference orders (used only to seed the label distribution)
REFERENCE_SEQS = [
    ["sroa", "early-cse", "simplifycfg", "instcombine", "ipsccp", "globalopt", "inline",
     "instcombine", "simplifycfg", "loop-simplify", "loop-rotate", "licm", "indvars",
     "loop-deletion", "loop-unroll-full", "gvn", "sccp", "adce", "simplifycfg", "globaldce"],
    ["mem2reg", "instcombine", "simplifycfg", "inline", "sroa", "early-cse", "jump-threading",
     "correlated-propagation", "instcombine", "simplifycfg", "reassociate", "loop-simplify",
     "licm", "indvars", "loop-deletion", "gvn", "dse", "adce", "simplifycfg", "globaldce"],
]


@dataclass
class Spec:
    name: str
    level: str
    need: list
    forbid: list
    sets: list
    clears: list
    big: bool


class PassSpace:
    def __init__(self, names=None, device="cpu"):
        specs = [Spec(*s) for s in _SPEC]
        if names is not None:
            keep = set(names)
            specs = [s for s in specs if s.name in keep]
        self.specs = specs
        self.tokens = [NOP] + [s.name for s in specs]
        self.idx = {t: i for i, t in enumerate(self.tokens)}
        self.level = {s.name: s.level for s in specs}
        self.emit: dict = {}  # token -> string actually passed to opt (e.g. instcombine<no-verify-fixpoint>)
        self.V = len(self.tokens)
        K = len(FACTS)
        V = self.V

        def fm(key):
            t = torch.zeros(V, K, dtype=torch.bool)
            for s in specs:
                for f in getattr(s, key):
                    t[self.idx[s.name], FI[f]] = True
            return t

        need, forbid, sets, clears = fm("need"), fm("forbid"), fm("sets"), fm("clears")
        const_mask = torch.tensor([f in CONST_FACTS for f in FACTS])
        clr_done = torch.zeros(V, V, dtype=torch.bool)
        for s in specs:
            if s.big:
                clr_done[self.idx[s.name], 1:] = True
        only_nop = torch.zeros(V, dtype=torch.bool)
        only_nop[0] = True
        self._t = dict(need=need, forbid=forbid, sets=sets, clears=clears,
                       need_static=need & const_mask[None, :], clr_done=clr_done,
                       only_nop=only_nop, ar=torch.arange(V))
        self.to(device)

    def to(self, device):
        self.device = device
        for k, v in self._t.items():
            setattr(self, k, v.to(device))
        return self

    def ids(self, names):
        return [self.idx[n] for n in names]

    def names(self, ids):
        return [self.tokens[int(i)] for i in ids]


_SPACE_CACHE: dict = {}


def get_space(device="cpu") -> PassSpace:
    key = str(device)
    if key not in _SPACE_CACHE:
        names, emit = None, {}
        if PASSES_OK.exists():
            cfg = json.loads(PASSES_OK.read_text())
            names, emit = cfg["ok"], cfg.get("emit", {})
        sp = PassSpace(names, device)
        sp.emit = emit
        _SPACE_CACHE[key] = sp
    return _SPACE_CACHE[key]


@dataclass
class State:
    facts: torch.Tensor       # [B,K] evolving facts
    init_facts: torch.Tensor  # [B,K] facts of the input module
    done: torch.Tensor        # [B,V] pass already ran and nothing big happened since
    prev: torch.Tensor        # [B]   previous token
    ended: torch.Tensor       # [B]   NOP seen (absorbing)


class Grammar:
    def __init__(self, space: PassSpace, mode: str = "dynamic"):
        assert mode in ("none", "static", "dynamic")
        self.space, self.mode = space, mode

    def init_state(self, facts: torch.Tensor) -> State:
        sp = self.space
        facts = facts.to(sp.device).bool()
        B = facts.shape[0]
        return State(facts.clone(), facts.clone(),
                     torch.zeros(B, sp.V, dtype=torch.bool, device=sp.device),
                     torch.full((B,), -1, dtype=torch.long, device=sp.device),
                     torch.zeros(B, dtype=torch.bool, device=sp.device))

    def mask(self, st: State) -> torch.Tensor:
        sp = self.space
        B = st.prev.shape[0]
        if self.mode == "none":
            return torch.ones(B, sp.V, dtype=torch.bool, device=sp.device)
        if self.mode == "static":
            ok = ~((sp.need_static[None] & ~st.init_facts[:, None, :]).any(-1))
        else:
            f = st.facts
            ok = (~((sp.need[None] & ~f[:, None, :]).any(-1))
                  & ~((sp.forbid[None] & f[:, None, :]).any(-1))
                  & ~st.done)
        ok = ok & (sp.ar[None] != st.prev[:, None])
        ok[:, 0] = True
        return torch.where(st.ended[:, None], sp.only_nop[None], ok)

    def step(self, st: State, tok: torch.Tensor) -> State:
        sp = self.space
        facts, done, ended = st.facts, st.done, st.ended
        if self.mode == "dynamic":
            facts = (facts | sp.sets[tok]) & ~sp.clears[tok]
            done = st.done & ~sp.clr_done[tok]
            done[torch.arange(tok.shape[0], device=sp.device), tok] = True
        if self.mode != "none":
            ended = st.ended | (tok == 0)
        return State(facts, st.init_facts, done, tok, ended)


def _weights(m: torch.Tensor, nop_weight: float) -> torch.Tensor:
    w = m.float().clone()
    others = m[:, 1:].any(-1)
    w[:, 0] = torch.where(others, torch.full_like(w[:, 0], nop_weight), torch.ones_like(w[:, 0])) * m[:, 0].float()
    return w


def sample(grammar: Grammar, facts: torch.Tensor, B: int, L: int,
           gen: torch.Generator | None = None, nop_weight: float = 0.02) -> torch.Tensor:
    """Uniformly sample B grammar-valid sequences of length L -> LongTensor [B,L]."""
    if facts.dim() == 1:
        facts = facts[None].expand(B, -1)
    st = grammar.init_state(facts)
    out = []
    for _ in range(L):
        w = _weights(grammar.mask(st), nop_weight).cpu()
        tok = torch.multinomial(w, 1, generator=gen).squeeze(1).to(grammar.space.device)
        st = grammar.step(st, tok)
        out.append(tok)
    return torch.stack(out, 1)


def is_valid(grammar: Grammar, facts: torch.Tensor, seq) -> bool:
    st = grammar.init_state(facts[None])
    for t in seq:
        t = int(t)
        if not bool(grammar.mask(st)[0, t]):
            return False
        st = grammar.step(st, torch.tensor([t], device=grammar.space.device))
    return True


def repair(grammar: Grammar, facts: torch.Tensor, seq, gen=None, nop_weight: float = 0.02) -> list:
    """Replay a sequence; replace every token that is illegal at its position."""
    st = grammar.init_state(facts[None])
    out = []
    for t in seq:
        m = grammar.mask(st)
        if not bool(m[0, int(t)]):
            w = _weights(m, nop_weight).cpu()
            t = int(torch.multinomial(w, 1, generator=gen).item())
        t = int(t)
        out.append(t)
        st = grammar.step(st, torch.tensor([t], device=grammar.space.device))
    return out


def build_pipeline(names, space: PassSpace | None = None) -> str:
    """Flat list of pass names -> LLVM new-PM pipeline string with correct nesting."""
    space = space or get_space()
    items: list = []
    state = {"f": None, "l": None}

    def flush_loop():
        if state["l"]:
            if state["f"] is None:
                state["f"] = []
            state["f"].append("loop-mssa(" + ",".join(state["l"]) + ")")
            state["l"] = None

    def flush_fn():
        flush_loop()
        if state["f"]:
            items.append("function(" + ",".join(state["f"]) + ")")
            state["f"] = None

    for n in names:
        if n == NOP:
            continue
        lvl = space.level[n]
        e = space.emit.get(n, n)
        if lvl == "module":
            flush_fn()
            items.append(e)
        elif lvl == "cgscc":
            flush_fn()
            items.append(f"cgscc({e})")
        elif lvl == "function":
            flush_loop()
            if state["f"] is None:
                state["f"] = []
            state["f"].append(e)
        else:  # loop
            if state["l"] is None:
                state["l"] = []
            state["l"].append(e)
    flush_fn()
    return ",".join(items)
