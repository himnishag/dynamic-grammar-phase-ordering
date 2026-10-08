import warnings
warnings.filterwarnings("ignore")

import sys
import tempfile
from pathlib import Path

from config import DEMO_MODE, IS_LLVM_AVAILABLE, TOOLS
from llvm_tools import clang_to_ll, count_insts, opt_text, run


def main():
    print("python", sys.version.split()[0])
    try:
        import torch
        print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
    except ImportError:
        print("torch MISSING -> pip install torch")
    for n, p in TOOLS.items():
        ver = ""
        if p:
            rc, out, _ = run([p, "--version"], 20)
            ver = (out.decode(errors="replace").splitlines() or [""])[0]
        print(f"{n:10s} {'OK  ' + p if p else 'MISSING'}  {ver}")
    
    if DEMO_MODE or not IS_LLVM_AVAILABLE:
        print("\n[!] LLVM binary 'opt' not found in PATH.")
        print("[!] Entering DEMO / SIMULATION MODE (using pre-calculated results & surrogate inference).")
        print("[!] For real LLVM compilation, install LLVM or run inside WSL2.")
        return 0
        
    with tempfile.TemporaryDirectory() as td:
        c = Path(td) / "t.c"
        c.write_text("int f(int n){int s=0;for(int i=0;i<n;i++)s+=i*i;return s;}\n")
        ll = Path(td) / "t.ll"
        if TOOLS["clang"] and clang_to_ll(c, ll):
            ok, t0 = opt_text(ll, "verify")
            ok2, t1 = opt_text(ll, "default<Oz>")
            ok3, t2 = opt_text(ll, "function(sroa,instcombine,simplifycfg)")
            print("smoke test:", "OK" if ok and ok2 and ok3 else "FAILED",
                  f"| insts init={count_insts(t0)} Oz={count_insts(t1)} custom={count_insts(t2)}")
        else:
            print("clang missing or failed: fallback demo mode enabled")
    return 0


if __name__ == "__main__":
    sys.exit(main())

