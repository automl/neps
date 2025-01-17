# Multi-Fidelity Optimizers

## What is Multi-Fidelity Optimization?

Multi-Fidelity optimization leverages the idea of running an AutoML problem on a small scale, which is cheaper and faster, and then using this information to train full-scale models. The _low-fidelity_ runs could be on a smaller dataset, a smaller model, or for shorter training times. MF-algorithms then infer which configurations are likely to perform well on the full problem, before investing larger compute amounts.

### Advantages of Multi-Fidelity

- **Heuristic**: MF-algorithms can use the information from the low-fidelity runs to guide the search in the high-fidelity runs.
- **Exploration**: By using low-fidelity runs, the optimizer can explore more of the search space.

### Disadvantages of Multi-Fidelity

- **More compute**: Running multiple iterations on different fidelities is generally more compute-intensive.
- **Variance**: The performance of a configuration on a low-fidelity run might not correlate well with its performance on a high-fidelity run. This can result in misguided conclusions.

## Optimizers using Multi-Fidelity

### 1 `Successive Halving`

`Successive Halving` (SH) is a simple but effective Multi-Fidelity optimizer. It starts with a large number of configurations and evaluates them on a low-fidelity. The best-performing $1/\eta$ configurations are then promoted to the next fidelity, where they are evaluated again. This process is repeated until only a few configurations remain, evaluated on the highest fidelity.
The process allows for broad exploration in the beginning and focus on the most promising configurations towards the end.

!!! tip Note:

    - For the same total compute, SH outperforms uninformed search algorithms.
    - It highly depends on the correlation between lower and higher fidelities. If the correlation is low, SH underperforms.

### 2 `HyperBand`

Detailed explanation of `hyperband`:

Link to Sucessive Halving for its explanation.
Write how its called as subroutine and the advantages.

Explain parallelization mode

### 3 `ASHA`

Detailed explanation of `asha`:

Link to Sucessive Halving for its explanation.
Explain the problems of SH for parallelization and how rungs are used to maximize compute use and quick promotion

### 4 `Mobster`

Detailed explanation of `mobster`:

Link to BO-explanation (extern), to explain BO.

Explain the problem when parallelizing BO and how Mobster fantasizes outcomes via joint GP
Explain the difference between using promotion and stopping and advantages of each (good defaults vs conservative)
-> To consider when using Mobster in Neps.

### 5 `IfBO`

Detailed explanation of `IfBO`:

Link to FT, PFNs and BO-explanation (extern), to explain BO, In-Context-Learning and Freeze-Thaw.

Explain the combination of the a PFN as surrogate for Freeze-Thaw-BO.

Write about what to consider when using IfBO  in Neps
___

For optimizers using both Priors and Multi-Fidelity, please refer [here](multifidelity_prior.md).
