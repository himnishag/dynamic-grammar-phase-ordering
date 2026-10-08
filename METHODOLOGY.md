# METHODOLOGY

## 1. Dynamic State-Tracking Pass Grammar

Standard phase ordering search spaces suffer from two main issues:
1. **Illegal Pipelines**: Running loop transformations when no loops exist.
2. **Redundant Passes**: Re-running `mem2reg` immediately after allocas have already been promoted to registers.

Our **Dynamic Grammar** maintains state vectors across sequence steps $t=1 \dots L$:

$$\text{State}_t = (\text{Facts}_t, \text{Done}_t, \text{Prev}_t, \text{Ended}_t)$$

Where facts track binary abstract IR properties:
* `ALLOCA`: Unpromoted stack allocations exist.
* `LOOPS`: Loop structures exist in the function.
* `CALLS`: Function calls are present.
* `MULTI`: Multiple functions exist in the module.
* `CANON`: Loops are in canonical form.

At step $t$, illegal passes are masked out by setting their logits to $-\infty$ ($10^{-9}$):

$$\text{Logits}_t[v] = \begin{cases} \theta_t[v] & \text{if } \text{Valid}(v \mid \text{State}_t) \\ -10^9 & \text{otherwise} \end{cases}$$

## 2. Grammar-Constrained Straight-Through Gumbel-Softmax

To optimize categorical pass choices with gradients, we apply Gumbel-Softmax relaxation:

$$y_{t,v} = \frac{\exp((\text{Logits}_{t,v} + g_{t,v}) / \tau)}{\sum_j \exp((\text{Logits}_{t,j} + g_{t,j}) / \tau)}$$

Where $g \sim \text{Gumbel}(0,1)$ and $\tau$ is a temperature parameter annealed from $\tau_{start} = 1.5$ to $\tau_{end} = 0.4$.

During forward propagation, hard one-hot vectors are used ($y_{hard} = \text{one\_hot}(\arg\max y)$). In backward propagation, soft gradients flow via straight-through estimation:

$$y_{ST} = y_{hard} + y - \text{detach}(y)$$

## 3. Risk-Aware Surrogate Ensemble Objective

The surrogate ensemble contains $K=3$ neural networks. The optimization objective minimizes expected code size plus an uncertainty penalty (epistemic risk):

$$\mathcal{J}(\theta) = \mu(f, Y) + \kappa \cdot \sigma(f, Y) + \text{ResidualCorrection}(Y)$$

Where $\mu$ is predicted log size reduction, $\sigma$ is standard deviation across ensemble members, and $\kappa = 0.5$.

## 4. Closed-Loop Compiler Feedback

Surrogate models exhibit distribution drift when evaluating unseen pass combinations. To fix this, after each search iteration:
1. Top candidates are compiled with real LLVM `opt`.
2. Actual instruction counts $z_{true}$ are recorded.
3. Residuals $r_i = \log(z_{true}/init) - \mu_i$ are computed.
4. Kernel regression weights candidate predictions in subsequent iterations.
