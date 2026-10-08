# SYSTEM ARCHITECTURE

## Overview

The **Dynamic Grammar-Constrained Differentiable Search Framework** automatically discovers high-performing LLVM optimization pass pipelines to minimize code size.

```
+-----------------------------------------------------------------------------------+
|                                  INPUT STAGE                                      |
|                                                                                   |
|    +------------------------+             +----------------------------------+    |
|    | C Source Code (.c)     |             | Raw LLVM IR Module (.ll / .bc)   |    |
|    +-----------+------------+             +----------------+-----------------+    |
|                | (clang -O1 -disable-llvm-passes)          |                      |
|                +--------------------+----------------------+                      |
|                                     |                                             |
+-------------------------------------v---------------------------------------------+
                                      |
+-------------------------------------v---------------------------------------------+
|                             FEATURE & STATE EXTRACTION                            |
|                                                                                   |
|                    +----------------------------------+                           |
|                    | ir_features.py (20 IR metrics)   |                           |
|                    +----------------+-----------------+                           |
|                                     |                                             |
+-------------------------------------v---------------------------------------------+
                                      |
+-------------------------------------v---------------------------------------------+
|                          DYNAMIC GRAMMAR & SEARCH MASK                            |
|                                                                                   |
|          +-------------------------------------------------------+                |
|          | Abstract IR Facts: [ALLOCA, LOOPS, CALLS, MULTI, CANON|                |
|          +--------------------------+----------------------------+                |
|                                     |                                             |
|          +--------------------------v----------------------------+                |
|          | pass_grammar.py (Mask illegal & redundant passes)     |                |
|          +--------------------------+----------------------------+                |
|                                     |                                             |
+-------------------------------------v---------------------------------------------+
                                      |
+-------------------------------------v---------------------------------------------+
|                        DIFFERENTIABLE PASS SELECTION                              |
|                                                                                   |
|          +-------------------------------------------------------+                |
|          | Straight-Through Gumbel-Softmax Sampling              |                |
|          +--------------------------+----------------------------+                |
|                                     |                                             |
|          +--------------------------v----------------------------+                |
|          | Soft One-Hot Sequence Tensor [B, L, V]                 |                |
|          +--------------------------+----------------------------+                |
|                                     |                                             |
+-------------------------------------v---------------------------------------------+
                                      |
+-------------------------------------v---------------------------------------------+
|                         SURROGATE ENSEMBLE EVALUATION                             |
|                                                                                   |
|    +------------------------+   +-----------------------+   +------------------+  |
|    | Transformer Model 1    |   | Transformer Model 2   |   | Transformer Model 3 |
|    +-----------+------------+   +-----------+-----------+   +--------+---------+  |
|                |                        |                            |            |
|                +------------------------+----------------------------+            |
|                                         |                                         |
|                 +-----------------------v-----------------------+                 |
|                 | Mean Prediction + Epistemic Uncertainty (std) |                 |
|                 +-----------------------+-----------------------+                 |
|                                         |                                         |
+-----------------------------------------v-----------------------------------------+
                                          |
+-----------------------------------------v-----------------------------------------+
|                        CLOSED-LOOP COMPILER FEEDBACK                              |
|                                                                                   |
|          +-------------------------------------------------------+                |
|          | Run Top Candidates with LLVM 'opt' (or Simulation)    |                |
|          +--------------------------+----------------------------+                |
|                                     |                                             |
|          +--------------------------v----------------------------+                |
|          | Compute Residual: e_i = True_Size - Predicted_Size    |                |
|          +--------------------------+----------------------------+                |
|                                     |                                             |
|          +--------------------------v----------------------------+                |
|          | Kernel Regression Feedback -> Next Search Iteration    |                |
|          +-------------------------------------------------------+                |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

## Baseline Branch Comparison

```
+-------------------+      +-------------------+      +-------------------+
| LLVM Heuristics   |      | Random Search     |      | Genetic Algorithm |
| (-O2 / -O3 / -Oz) |      | (Dynamic Grammar) |      | (Elitism + Cross) |
+---------+---------+      +---------+---------+      +---------+---------+
          |                          |                          |
          +--------------------------+--------------------------+
                                     |
                          +----------v----------+
                          | Evaluator Benchmark |
                          +---------------------+
```

## System Components

1. **`pass_grammar.py`**: Encapsulates pass vocabulary (V=43 tokens) and rule matrices.
2. **`ir_features.py`**: Extracts instruction distributions, loop depth, call site frequency, and alloca densities into 20 normalized feature vectors.
3. **`surrogate.py`**: Implements 3 Transformer Encoders with FiLM conditioning.
4. **`diff_search.py`**: Straight-Through Gumbel-Softmax optimization and closed-loop residual feedback.
5. **`backend/`**: FastAPI REST API providing server endpoints for dashboard UI.
6. **`frontend/`**: Vite + React modern web dashboard interface.
