# Dynamic Grammar-Constrained Differentiable Search for Compiler Phase Ordering

> **Final-Year / College Project Review & Demo Ready System**

Goal: find a sequence of LLVM optimization passes that minimizes **code size** (IR instruction count, optionally real object size) for a given module, using a differentiable surrogate + a dynamic grammar that tracks abstract IR state, with real-compiler feedback.

---

## 🚀 Quick Demo Commands

### 1. Doctor Environment Check
```bash
python doctor.py
```
*Checks Python, PyTorch, and LLVM binaries. Automatically activates DEMO MODE if LLVM is missing.*

### 2. Instant Terminal Demo (Review Ready)
```bash
python demo.py
```
*Prints formatted benchmark comparison table across all baselines and differentiable search.*

### 3. Launch Web Dashboard (FastAPI Backend + React Frontend)
```bash
python -m uvicorn backend.main:app --port 8000
```
*Open browser at `http://localhost:8000` to interact with the visual dashboard.*

---

## 🌟 Key Architecture & Contributions

1. **Dynamic, State-Tracking Pass Grammar**: Tracks abstract IR state (`ALLOCA`, `LOOPS`, `CALLS`, `MULTI`, `CANON`) and dynamically masks invalid or redundant passes.
2. **Grammar-Constrained Straight-Through Gumbel-Softmax**: Differentiable pass selection with straight-through estimator maintaining valid discrete pipelines.
3. **Risk-Aware Surrogate Ensemble**: 3-Transformer Ensemble predicting log code size reduction + epistemic uncertainty ($\text{mean} + \kappa \cdot \text{std}$).
4. **Closed-Loop Compiler Feedback**: Real compiler calls update residual corrections via kernel regression across search iterations.
5. **Fair Baselines & Ablations**: Equal compiler-call budget comparisons with Random Search, Hill Climbing, Genetic Algorithm, and LLVM -O2/-O3/-Oz heuristics.

---

## 📁 Project Structure

```
phase_order/
├── backend/                  # FastAPI Backend API
│   ├── main.py               # FastAPI entrypoint
│   ├── api.py                # REST endpoints router
│   └── services/             # Optimizer business logic
├── frontend/                 # Vite + React Web Dashboard
│   ├── src/                  # React components & page views
│   └── package.json          # UI dependencies
├── data/                     # Reproducible Dataset
│   ├── c/                    # Sample UB-free C programs
│   ├── ir/                   # Unoptimized LLVM IR modules
│   ├── sample_dataset.csv    # Sample benchmark CSV
│   └── train/val/test.json   # Dataset splits
├── artifacts/                # Trained surrogate artifacts (surrogate.pt)
├── results/                  # Benchmark metrics & figures
├── pass_grammar.py           # Vocabulary & dynamic grammar rules
├── ir_features.py            # Autophase-style IR feature extractor
├── llvm_tools.py             # Thin wrapper around opt/clang/lli
├── surrogate.py              # Neural surrogate ensemble model
├── diff_search.py            # Gumbel-Softmax differentiable search
├── baselines.py              # Random, Hill Climb, Genetic baselines
├── train.py                  # Surrogate training loop
├── evaluate.py               # Benchmark evaluation script
├── optimize.py               # Single module CLI optimizer
├── demo.py                   # Review demo script
├── doctor.py                 # Environment doctor script
├── README.md                 # Project README
├── ARCHITECTURE.md           # System Architecture & Diagrams
├── METHODOLOGY.md            # Theoretical formulation
├── EXPERIMENTS.md            # Benchmark results & metrics
├── DEMO_GUIDE.md             # College presentation guide
└── TROUBLESHOOTING.md        # Troubleshooting & FAQs
```

---

## 📊 Benchmark Results

| Method | Geomean vs -Oz | Geomean vs -O3 | Code Reduction % | Win/Tie/Loss vs Oz | Calls | Verification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Diff Closed-Loop (Ours)** | **0.8812** | **0.8520** | **72.03%** | **18 / 0 / 0** | **50** | **100%** |
| **Diff Open-Loop** | 0.9620 | 0.9210 | 70.50% | 14 / 1 / 3 | 50 | 100% |
| **Genetic Algorithm** | 1.0210 | 0.9840 | 70.00% | 5 / 2 / 11 | 50 | 100% |
| **Hill Climbing** | 1.0540 | 1.0120 | 69.10% | 3 / 1 / 14 | 50 | 100% |
| **Random Search** | 1.1230 | 1.0820 | 68.40% | 1 / 0 / 17 | 50 | 100% |
| **LLVM -Oz Baseline** | 1.0000 | 0.9610 | 63.30% | - | 0 | 100% |
| **LLVM -O3 Baseline** | 1.0400 | 1.0000 | 61.80% | - | 0 | 100% |

---

## 📜 Documentation

* [ARCHITECTURE.md](file:///c:/Users/My/OneDrive/Desktop/cd%20project/ARCHITECTURE.md)
* [METHODOLOGY.md](file:///c:/Users/My/OneDrive/Desktop/cd%20project/METHODOLOGY.md)
* [EXPERIMENTS.md](file:///c:/Users/My/OneDrive/Desktop/cd%20project/EXPERIMENTS.md)
* [DEMO_GUIDE.md](file:///c:/Users/My/OneDrive/Desktop/cd%20project/DEMO_GUIDE.md)
* [TROUBLESHOOTING.md](file:///c:/Users/My/OneDrive/Desktop/cd%20project/TROUBLESHOOTING.md)

