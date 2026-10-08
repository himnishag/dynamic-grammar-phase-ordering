"""Autophase-style features + initial grammar facts, computed from textual LLVM IR."""
from __future__ import annotations

import math
import re

import numpy as np

OPCODES = ["alloca", "load", "store", "getelementptr", "add", "sub", "mul", "sdiv", "udiv", "srem",
           "urem", "shl", "lshr", "ashr", "and", "or", "xor", "icmp", "fcmp", "fadd", "fsub", "fmul",
           "fdiv", "frem", "fneg", "zext", "sext", "trunc", "bitcast", "ptrtoint", "inttoptr",
           "sitofp", "uitofp", "fptosi", "fptoui", "fpext", "fptrunc", "select", "phi", "br_cond",
           "br_uncond", "switch", "ret", "call_direct", "call_defined", "call_intrinsic",
           "call_indirect", "invoke", "unreachable", "extractvalue", "insertvalue", "extractelement",
           "insertelement", "shufflevector", "atomic", "freeze", "other"]
_OPSET = set(OPCODES)
STRUCT = ["n_def", "n_decl", "n_blocks", "n_back", "n_glob", "n_insts", "max_fn",
          "insts_per_block", "back_per_block", "load_frac", "store_frac", "call_frac",
          "phi_frac", "alloca_frac", "condbr_frac"]
FEATURE_NAMES = OPCODES + STRUCT
DIM = len(FEATURE_NAMES)

_DEF = re.compile(r'^define[^@]*@("[^"]+"|[\w.$\-]+)\(', re.M)
_LABEL = re.compile(r'^("[^"]+"|[^\s;:]+):')
_TGT = re.compile(r'label %("[^"]+"|[^\s,\]\)]+)')
_ASSIGN = re.compile(r'^(?:%"[^"]*"|%[^\s=]+)\s*=\s*(.*)$')
_CALLEE = re.compile(r'@("[^"]+"|[\w.$\-]+)\s*\(')


def scan(text: str) -> dict:
    defined = {m.strip('"') for m in _DEF.findall(text)}
    cnt = {o: 0 for o in OPCODES}
    n_decl = sum(1 for l in text.splitlines() if l.startswith("declare "))
    n_glob = sum(1 for l in text.splitlines() if l.startswith("@") and "=" in l)
    n_blocks = n_back = n_insts = max_fn = 0
    in_fn = in_sw = False
    fn_insts = 0
    names: list = []
    cur = 0
    seen_inst = False
    edges: list = []  # (src_block_idx, target_name)
    for line in text.splitlines():
        if not in_fn:
            if line.startswith("define "):
                in_fn, in_sw = True, False
                fn_insts, names, cur, seen_inst, edges = 0, [None], 0, False, []
            continue
        if line == "}":
            in_fn = False
            n_blocks += len(names)
            pos = {n: i for i, n in enumerate(names) if n is not None}
            for src, tgt in edges:
                if tgt in pos and pos[tgt] <= src:
                    n_back += 1
            max_fn = max(max_fn, fn_insts)
            continue
        if line and line[0] not in " \t":
            m = _LABEL.match(line)
            if m:
                nm = m.group(1).strip('"')
                if not seen_inst and names[0] is None:
                    names[0] = nm
                else:
                    names.append(nm)
                    cur = len(names) - 1
            continue
        s = line.strip()
        if not s or s[0] == ";":
            continue
        if in_sw:
            for t in _TGT.findall(s):
                edges.append((cur, t.strip('"')))
            if s == "]":
                in_sw = False
            continue
        seen_inst = True
        m = _ASSIGN.match(s)
        body = m.group(1) if m else s
        toks = body.split()
        while toks and toks[0] in ("tail", "musttail", "notail"):
            toks = toks[1:]
        if not toks:
            continue
        op = toks[0]
        if op in ("call", "invoke"):
            cm = _CALLEE.search(body)
            if "@llvm.dbg." in body or "@llvm.lifetime." in body:
                continue
            if op == "invoke":
                cnt["invoke"] += 1
            elif cm is None:
                cnt["call_indirect"] += 1
            else:
                nm = cm.group(1).strip('"')
                if nm.startswith("llvm."):
                    cnt["call_intrinsic"] += 1
                elif nm in defined:
                    cnt["call_defined"] += 1
                else:
                    cnt["call_direct"] += 1
            for t in _TGT.findall(s):
                edges.append((cur, t.strip('"')))
        elif op == "br":
            if toks[1] == "i1":
                cnt["br_cond"] += 1
            else:
                cnt["br_uncond"] += 1
            for t in _TGT.findall(s):
                edges.append((cur, t.strip('"')))
        elif op == "switch":
            cnt["switch"] += 1
            for t in _TGT.findall(s):
                edges.append((cur, t.strip('"')))
            if s.endswith("["):
                in_sw = True
        elif op in ("atomicrmw", "cmpxchg", "fence"):
            cnt["atomic"] += 1
        elif op in _OPSET:
            cnt[op] += 1
        else:
            cnt["other"] += 1
        n_insts += 1
        fn_insts += 1
    return dict(cnt=cnt, n_def=len(defined), n_decl=n_decl, n_blocks=n_blocks, n_back=n_back,
                n_glob=n_glob, n_insts=n_insts, max_fn=max_fn)


def featurize(text: str):
    """Returns (features float32 [DIM], facts bool [5] = ALLOCA,LOOPS,CALLS,MULTI,CANON, n_insts)."""
    d = scan(text)
    c = d["cnt"]
    n = max(d["n_insts"], 1)
    nb = max(d["n_blocks"], 1)
    vec = [math.log1p(c[o]) for o in OPCODES]
    vec += [math.log1p(d["n_def"]), math.log1p(d["n_decl"]), math.log1p(d["n_blocks"]),
            math.log1p(d["n_back"]), math.log1p(d["n_glob"]), math.log1p(d["n_insts"]),
            math.log1p(d["max_fn"]), d["n_insts"] / nb, d["n_back"] / nb,
            c["load"] / n, c["store"] / n, (c["call_direct"] + c["call_defined"]) / n,
            c["phi"] / n, c["alloca"] / n, c["br_cond"] / n]
    facts = [c["alloca"] > 0, d["n_back"] > 0, c["call_defined"] > 0, d["n_def"] > 1, False]
    return np.asarray(vec, dtype=np.float32), np.asarray(facts, dtype=bool), d["n_insts"]
