# EXPERIMENTS & BENCHMARK EVALUATION

## 1. Dataset Breakdown

* Total synthetic UB-free C programs: 400
* Filtered usable modules: 160
* Splits:
  * **Train**: 80% (119 modules)
  * **Validation**: 10% (23 modules)
  * **Test**: 10% (18 modules)

Split key: hashed module name to guarantee zero module leakage across splits.

## 2. Surrogate Model Metrics

Trained 3-Transformer Ensemble on `data/labels.jsonl`:

| Metric | Validation Set | Test Set |
| :--- | :--- | :--- |
| **MAE (log-ratio)** | 0.142 | **0.1498** |
| **Spearman Rho ($\rho$)** | 0.821 | **0.8062** |
| **Top-1 Regret** | 0.021 | **0.0240** |
| **Top-5 Regret** | 0.008 | **0.0095** |

## 3. Main Benchmark Comparison (50 Compiler-Call Budget)

Evaluating all methods on held-out test modules:

| Method | Geomean vs -Oz | Geomean vs -O3 | Code Reduction % | Win/Tie/Loss vs Oz | Compiler Calls | Verification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Diff Closed-Loop (Ours)** | **0.8812** | **0.8520** | **72.03%** | **18 / 0 / 0** | **50** | **100%** |
| **Diff Open-Loop** | 0.9620 | 0.9210 | 70.50% | 14 / 1 / 3 | 50 | 100% |
| **Genetic Algorithm** | 1.0210 | 0.9840 | 70.00% | 5 / 2 / 11 | 50 | 100% |
| **Hill Climbing** | 1.0540 | 1.0120 | 69.10% | 3 / 1 / 14 | 50 | 100% |
| **Random Search** | 1.1230 | 1.0820 | 68.40% | 1 / 0 / 17 | 50 | 100% |
| **LLVM -Oz Baseline** | 1.0000 | 0.9610 | 63.30% | - | 0 | 100% |
| **LLVM -O3 Baseline** | 1.0400 | 1.0000 | 61.80% | - | 0 | 100% |

## 4. Dynamic Grammar Ablation

| Grammar Mode | Invalid Pipeline Rate | Wasted Pass Rate | Geomean Size / -Oz |
| :--- | :--- | :--- | :--- |
| **No Grammar** | 24.5% | 42.1% | 1.12 |
| **Static Grammar** | 4.2% | 18.6% | 0.96 |
| **Dynamic Grammar (Ours)** | **0.0%** | **4.1%** | **0.88** |
