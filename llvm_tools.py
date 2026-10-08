"""Thin wrapper around LLVM's `opt` / `clang` / `lli` + a cached, parallel evaluator."""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import TOOLS, require
from pass_grammar import build_pipeline

WORKERS = max(1, min(8, (os.cpu_count() or 2)))
BASELINES = {"Oz": "default<Oz>", "O3": "default<O3>", "O2": "default<O2>", "Os": "default<Os>"}


def run(cmd, timeout=120, stdin=None):
    try:
        p = subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, b"", b"timeout"


def count_insts(text: str) -> int:
    """Number of IR instructions inside function bodies (dbg/lifetime markers excluded)."""
    n = 0
    in_fn = in_sw = False
    for line in text.splitlines():
        if not in_fn:
            if line.startswith("define "):
                in_fn = True
            continue
        if line == "}":
            in_fn = False
            continue
        if not line or line[0] not in " \t":
            continue  # block label
        s = line.strip()
        if not s or s[0] == ";":
            continue
        if in_sw:
            if s == "]":
                in_sw = False
            continue
        if "@llvm.dbg." in s or "@llvm.lifetime." in s:
            continue
        n += 1
        if s.startswith("switch ") and s.endswith("["):
            in_sw = True
    return n


def opt_text(path, pipeline: str, verify_each: bool = False, timeout: int = 120):
    cmd = [require("opt"), f"-passes={pipeline or 'verify'}", "-S", "-o", "-"]
    if verify_each:
        cmd.append("-verify-each")
    cmd.append(str(path))
    rc, out, err = run(cmd, timeout)
    if rc != 0:
        return False, err.decode("utf-8", "replace")[:2000]
    return True, out.decode("utf-8", "replace")


class Evaluator:
    """Real compiler feedback for one module. Counts unique `opt` invocations (the budget)."""

    def __init__(self, path, tail: str = "", workers: int = WORKERS):
        self.path, self.tail, self.workers = str(path), tail, workers
        self.cache: dict = {}
        self.calls = 0
        ok, txt = opt_text(self.path, "verify")
        if not ok:
            raise RuntimeError(f"cannot parse {path}: {txt[:200]}")
        self.init = count_insts(txt)

    def pipe(self, names) -> str:
        return ",".join(x for x in (build_pipeline(names), self.tail) if x)

    def _one(self, pipe: str) -> float:
        ok, txt = opt_text(self.path, pipe)
        return count_insts(txt) if ok else math.inf

    def sizes(self, seqs):
        pipes = [self.pipe(s) for s in seqs]
        todo = sorted({p for p in pipes if p not in self.cache})
        if todo:
            with ThreadPoolExecutor(self.workers) as ex:
                res = list(ex.map(self._one, todo))
            for p, r in zip(todo, res):
                self.cache[p] = r
            self.calls += len(todo)
        return [self.cache[p] for p in pipes]

    def baselines(self) -> dict:
        names = list(BASELINES)
        with ThreadPoolExecutor(self.workers) as ex:
            res = list(ex.map(self._one, [BASELINES[n] for n in names]))
        return dict(zip(names, res))


# ----------------------------------------------------------------- correctness checks
def verify_pipeline(path, pipeline: str):
    """Run the pipeline with the IR verifier after *every* pass. Returns (ok, text_or_err)."""
    ok, txt = opt_text(path, f"{pipeline},verify" if pipeline else "verify", verify_each=True)
    return ok, txt


def run_ir(ir_path: str, timeout: int = 30):
    """Execute a module with lli. Returns (returncode, stdout) or None when lli is missing."""
    lli = TOOLS.get("lli")
    if not lli:
        return None
    rc, out, _ = run([lli, ir_path], timeout)
    return rc, out


def has_main(text: str) -> bool:
    return "define " in text and "@main(" in text


def equivalent(orig_path, optimized_text: str):
    """Compare program output of original vs optimized IR (only if the module has a main)."""
    orig = Path(orig_path).read_text(errors="replace") if str(orig_path).endswith(".ll") else ""
    if not has_main(orig) or not TOOLS.get("lli"):
        return None  # not applicable
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "opt.ll"
        p.write_text(optimized_text)
        a = run_ir(str(orig_path))
        b = run_ir(str(p))
    if a is None or b is None or a[0] == 124 or b[0] == 124:
        return None
    return a == b


def clang_to_ll(c_path, out_ll, extra=()):
    """C -> *unoptimised* IR (no optnone/noinline so every pass can act on it)."""
    cmd = [require("clang"), "-O1", "-Xclang", "-disable-llvm-passes", "-S", "-emit-llvm", "-w",
           *extra, str(c_path), "-o", str(out_ll)]
    rc, _, err = run(cmd, 120)
    return rc == 0


def obj_text_size(ir_text: str, level: str = "2"):
    """Real code size: bytes of non-writable sections (text+rodata) of the object file from llc.
    Returns None when llc / llvm-size are unavailable or codegen fails."""
    llc, sz = TOOLS.get("llc"), TOOLS.get("llvm-size")
    if not (llc and sz):
        return None
    with tempfile.TemporaryDirectory() as td:
        ll, o = Path(td) / "m.ll", Path(td) / "m.o"
        ll.write_text(ir_text)
        rc, _, _ = run([llc, f"-O{level}", "-filetype=obj", str(ll), "-o", str(o)], 120)
        if rc != 0:
            return None
        rc, out, _ = run([sz, "--format=berkeley", str(o)], 30)
        lines = out.decode(errors="replace").strip().splitlines()
        try:
            return int(lines[1].split()[0])
        except (IndexError, ValueError):
            return None
