# COLLEGE PROJECT DEMO & REVIEW GUIDE

This guide details how to demonstrate the project during a college project review.

## 1. Fast Start Commands (Terminal)

### Step A: System Health & Environment Check
```bash
python doctor.py
```
*Expected Output:* Shows Python, PyTorch, and LLVM tool status. If LLVM is not in PATH, cleanly displays `[!] Entering DEMO / SIMULATION MODE`.

### Step B: Run the Instant CLI Demo
```bash
python demo.py
```
*Expected Output:* Prints the benchmark table comparing LLVM -O2, -O3, -Oz, Random, Hill Climb, Genetic, Diff Open Loop, and Diff Closed Loop on sample modules.

### Step C: Launch Web Dashboard
```bash
python -m uvicorn backend.main:app --port 8000
```
Open browser at: `http://localhost:8000`

---

## 2. Review Presentation Flow (5 Minutes)

1. **Open Dashboard Overview**: Highlight the high-level cards (72.03% code size reduction, 0.8062 Spearman correlation, 0% invalid pipelines).
2. **Show DEMO MODE Badge**: Point out that the application transparently indicates whether real LLVM binaries or surrogate simulations are running.
3. **Run Interactive Optimization**: Navigate to **Optimize Module**, enter or use default IR, click **Optimize Module**, and show the discovered pipeline.
4. **Show Pipeline View**: Demonstrate how passes update the abstract IR facts step-by-step.
5. **Open Grammar Explorer**: Explain how dynamic boolean masking blocks invalid transformations (e.g. preventing `mem2reg` twice).
6. **Open Review Mode Page**: Walk through the "2-Minute Pitch" and click through the 10 Professor Q&As.
