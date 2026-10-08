"""Check every pass in the vocabulary against *your* LLVM version; keep only the ones that work.

Writes artifacts/passes_ok.json.  Run once after installing LLVM / after changing LLVM version.
"""
from __future__ import annotations

import json
import sys

from config import IR_DIR, PASSES_OK, TOOLS
from llvm_tools import opt_text, verify_pipeline
from pass_grammar import _SPEC, PassSpace, Spec, build_pipeline


def main():
    if not TOOLS["opt"]:
        raise SystemExit("opt not found - see README (LLVM install)")
    files = sorted(IR_DIR.glob("*.ll")) + sorted(IR_DIR.glob("*.bc"))
    if not files:
        raise SystemExit("no IR in data/ir - run fetch_dataset.py first")
    # use a few modules that contain loops/calls so loop & ipo passes really run
    test = files[:8]
    full = PassSpace(None)
    ok, bad, emit = [], [], {}
    # some LLVM versions need parameters on a pass to avoid spurious fatal errors
    ALTERNATIVES = {"instcombine": ["instcombine<no-verify-fixpoint>", "instcombine<max-iterations=1000>"]}
    for s in map(lambda t: Spec(*t), _SPEC):
        pre = ["sroa"] if "sroa" in full.idx else []
        if s.level == "loop" or s.name in ("loop-unroll", "loop-load-elim"):
            pre += ["loop-simplify"]
        names = pre + [s.name]
        works = False
        msg = ["?"]
        for cand in [s.name] + ALTERNATIVES.get(s.name, []):
            full.emit = {} if cand == s.name else {s.name: cand}
            pipe = build_pipeline(names, full)
            results = [verify_pipeline(f, pipe) for f in test]
            if all(r[0] for r in results):
                works = True
                if cand != s.name:
                    emit[s.name] = cand
                    print(f"  note: '{s.name}' emitted as '{cand}' for this LLVM version")
                break
            msg = next(r[1] for r in results if not r[0]).strip().splitlines() or ["?"]
        if works:
            ok.append(s.name)
        else:
            bad.append((s.name, pipe, msg[0]))
    full.emit = emit
    PASSES_OK.write_text(json.dumps({"ok": ok, "bad": [b[0] for b in bad], "emit": emit}, indent=1))
    print(f"{len(ok)} passes OK, {len(bad)} rejected by this LLVM")
    for n, p, m in bad:
        print("  rejected:", n, "|", p, "|", m[:120])
    # sanity: whole vocabulary in one pipeline
    sp = PassSpace(ok)
    sp.emit = emit
    allp = build_pipeline(ok, sp)
    r = opt_text(test[0], allp, verify_each=True)
    print("full-vocabulary pipeline:", "OK" if r[0] else "FAILED -> " + r[1][:200])


if __name__ == "__main__":
    sys.exit(main())
