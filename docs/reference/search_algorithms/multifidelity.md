# Multi-Fidelity Optimizers

## What is Multi-Fidelity Optimization?

Multi-Fidelity optimization leverages the idea of running an AutoML problem on a small scale, which is cheaper and faster, and then using this information to train full-scale models. The _low-fidelity_ runs could be on a smaller dataset, a smaller model, or for shorter training times. MF-algorithms then infer which configurations are likely to perform well on the full problem, before investing larger compute amounts.

!!! tip "Advantages of Multi-Fidelity"

    - **Parallelization**: MF-algorithms can use the information from many parallel low-fidelity runs to guide the search in the few high-fidelity runs.
    - **Exploration**: By using low-fidelity runs, the optimizer can explore more of the search space.

!!! warning "Disadvantages of Multi-Fidelity"

    - **More compute**: Running multiple iterations on different fidelities is generally more compute-intensive.
    - **Variance**: The performance of a configuration on a low-fidelity run might not correlate well with its performance on a high-fidelity run. This can result in misguided decisions.

## Optimizers using Multi-Fidelity

### 1 `Successive Halfing`

`Successive Halfing` (SH, see [Paper](https://proceedings.mlr.press/v51/jamieson16.pdf)) is a simple but effective Multi-Fidelity algorithm. It starts with a large number of random configurations and evaluates them on a low-fidelity. The best-performing $1/\eta$ configurations are then promoted to the next fidelity, where they are evaluated again. This process is repeated until only a few configurations remain, evaluated on the highest fidelity.
The process allows for broad exploration in the beginning and focus on the most promising configurations towards the end.

!!! example "Practical Tips"

    - For the same total compute, SH outperforms uninformed search algorithms.
    - It highly depends on the correlation between lower and higher fidelities. If the correlation is low, SH underperforms.

#### _Asynchronous_ Successive Halving

`Asynchronous Successive Halving` (ASHA, see [Paper](https://arxiv.org/pdf/1810.05934)) is an asynchronous version of SH that maximizes parallel evaluations. Instead of waiting for all $n$ configurations to finish on one fidelity, ASHA promotes the best configuration to the next fidelity as soon as there are enough evaluations to make a decision ($1/\eta*n\geq 1$). This allows for quicker promotions and earlier high fidelity-results. When there are no promotable configurations, ASHA spawns new configurations at the lowest fidelity, so it always utilizes the available compute and increases exploration compared to SH.

### 2 `HyperBand`

`HyperBand` (HB, see [Paper](https://arxiv.org/pdf/1603.06560)) is an extension of [Successive Halfing](../search_algorithms/multifidelity.md#1-successive-halfing) that employs multiple Successive Halfing-runs in parallel. Each of these runs has a different resource budget and different number of configurations. This makes HyperBand more flexible and parallelizable than SH.

!!! example "Practical Tips"

    - HyperBand is a good choice when you have a limited budget and want to parallelize your search.
    - It is more efficient than SH when the correlation between lower and higher fidelities is low.

### 3 `BOHB`

`BOHB` (see [Paper](https://arxiv.org/pdf/1807.01774)) is a combination of [Bayesian Optimization (BO)](../search_algorithms/bayesian_optimization.md) and [HyperBand](../search_algorithms/multifidelity.md#2-hyperband). Contrary to HyperBand, which uses random configurations, BOHB uses BO to choose the next configurations for HyperBand. This way, it can leverage the advantages of both algorithms: the flexibility and parallelization of HyperBand for low budgets and the efficiency of BO for higher budgets.

!!! example "Practical Tips"

    - BOHB is more efficient than both HyperBand and BO on their own.
    - The effects of BO only start showing after some evaluations, as it needs those to build its model.

!!! info
    ``BOHB`` is chosen as the default optimizer in NePS when there is no [Prior](../search_algorithms/prior.md), only Multi-Fidelity information available.

### 4 `Mobster`

Detailed explanation of `mobster`:

Link to BO-explanation.

Explain the problem when parallelizing BO and how Mobster fantasizes outcomes via joint GP
Explain the difference between using promotion and stopping and advantages of each (good defaults vs conservative)
-> To consider when using Mobster in Neps.

### 5 `IfBO`

Detailed explanation of `IfBO`:

Link to FT, PFNs and BO-explanation, to explain BO, In-Context-Learning and Freeze-Thaw.

Explain the combination of a PFN as surrogate for Freeze-Thaw-BO.

Write about what to consider when using IfBO  in Neps
___

For optimizers using both Priors and Multi-Fidelity, please refer [here](multifidelity_prior.md).
