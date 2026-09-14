# Multi-Objective Optimization

Multi-Objective Optimization (MOO) addresses the challenge of optimizing multiple, often competing objectives simultaneously. Unlike single-objective optimization where there is one clear optimum, multi-objective problems have a set of trade-off solutions known as the **Pareto Front**. This is particularly relevant in deep learning, where practitioners often need to optimize for multiple goals such as validation accuracy, inference latency, training time, and fairness.

## What is Multi-Objective Optimization?

In multi-objective optimization, we seek to minimize (or maximize) a vector-valued objective function:

$$\min_{\boldsymbol{\lambda} \in \mathcal{\Lambda}} \mathbf{f}(\boldsymbol{\lambda}) = \min_{\boldsymbol{\lambda} \in \mathcal{\Lambda}} \begin{pmatrix} f_1(\boldsymbol{\lambda}) \\ f_2(\boldsymbol{\lambda}) \\ \vdots \\ f_n(\boldsymbol{\lambda}) \end{pmatrix}$$

Since optimizing all objectives simultaneously is generally impossible, the goal is to find the **Pareto Set**, a set of non-dominated solutions where improving one objective necessarily worsens another.

### Pareto Optimality

A configuration $\boldsymbol{\lambda}_2$ **Pareto dominates** $\boldsymbol{\lambda}_1$ (denoted $\boldsymbol{\lambda}_2 \prec \boldsymbol{\lambda}_1$) if:
- It is at least as good as $\boldsymbol{\lambda}_1$ in all objectives
- It is strictly better in at least one objective

$$\boldsymbol{\lambda}_2 \prec \boldsymbol{\lambda}_1 \iff \begin{cases} \forall i: f_i(\boldsymbol{\lambda}_2) \leq f_i(\boldsymbol{\lambda}_1) \\ \exists k: f_k(\boldsymbol{\lambda}_2) < f_k(\boldsymbol{\lambda}_1) \end{cases}$$

A configuration is **Pareto Optimal** if no other configuration dominates it. The set of all Pareto Optimal configurations is the **Pareto Set**, and the set of their objective values is the **Pareto Front**.

!!! tip "Advantages of Multi-Objective Optimization"

    - **Real-world relevance**: Captures the practical reality that practitioners optimize for multiple goals simultaneously
    - **Informed trade-offs**: Identifies the Pareto Front to help practitioners choose configurations that match their priorities
    - **Resource awareness**: Can explicitly optimize for computational costs, inference latency, and other practical constraints

!!! warning "Challenges with Multi-Objective Optimization"

    - **Computational cost**: Exploring the entire Pareto Front is typically more expensive than finding a single optimum
    - **Scale with objectives**: Difficulty increases with the number of objectives
    - **Prior integration**: Traditional MOO algorithms lack robust mechanisms to incorporate expert knowledge for multiple objectives

___

## PriMO: Prior-Informed Multi-Objective Optimizer

`PriMO` (see [paper](https://arxiv.org/html/2511.08371v1)) is the first hyperparameter optimization algorithm that effectively integrates **multi-objective expert priors** while exploiting **cheap approximations** of expensive objectives. It combines the strengths of multi-fidelity optimization with prior-guided Bayesian optimization, achieving state-of-the-art performance in both multi-objective and single-objective settings.

### Key Components

#### 1. Multi-Objective Expert Priors

PriMO extends single-objective priors to the multi-objective setting using a **factorized prior approach**. For each objective $f_i$, a prior $\pi_{f_i}(\boldsymbol{\lambda})$ represents a probability distribution over the location of that objective's optimum:

$$\pi_{f_i}(\boldsymbol{\lambda}) = \mathbb{P}(f_i(\boldsymbol{\lambda}) = \min_{\boldsymbol{\lambda}' \in \mathcal{\Lambda}} f_i(\boldsymbol{\lambda}'))$$

The compound prior is then:

$$\Pi_\mathbf{f}(\boldsymbol{\lambda}) = \{\pi_{f_i}(\boldsymbol{\lambda})\}_{i=1}^n$$

This allows experts to specify beliefs about optimal regions for each objective independently, which is more intuitive than specifying priors over Pareto fronts directly.

#### 2. Initial Design Strategy: Leveraging Cheap Approximations

PriMO uses a multi-fidelity algorithm (`MOASHA` - Multi-Objective Asynchronous Successive Halving) during an initial design phase to generate strong seed points at full fidelity. This exploits cheap low-fidelity evaluations to quickly identify promising regions before transitioning to expensive full-fidelity optimization.

The initial design terminates once a budget threshold is reached, after which only full-fidelity evaluations are inserted into the dataset $\mathcal{D}$ for use in the Bayesian optimization phase. This approach provides a "warm start" to the BO phase while consuming resources efficiently.

#### 3. Prior-Augmented Bayesian Optimization with ε-Greedy Strategy

PriMO's core innovation is an **ε-greedy prior-weighted acquisition function** that balances prior exploitation with exploration. At each iteration, the acquisition function is:

$$\alpha_{\epsilon\pi}(\boldsymbol{\lambda}, \mathcal{D}) = \begin{cases} 
\alpha(\boldsymbol{\lambda}, \mathcal{D}), & \text{with probability } \epsilon \\
\alpha(\boldsymbol{\lambda}, \mathcal{D}) \cdot \pi_{f_j}(\boldsymbol{\lambda})^{\gamma}, & \text{with probability } 1-\epsilon, \quad j \sim \mathcal{U}(1, \ldots, n)
\end{cases}$$

where:
- $\alpha(\boldsymbol{\lambda}, \mathcal{D})$ is the base acquisition function (qLogNoisyEI)
- One prior is randomly selected at each iteration
- The decay factor is: $\gamma = \exp\left(-\frac{n_{\text{BO}}^2}{n_d}\right)$
- $n_{\text{BO}}$ is the number of BO samples
- $n_d$ is the dimensionality of the search space

**Key insight**: Unlike previous methods that depend on priors too long, PriMO's quadratic decay ensures rapid reduction of prior influence, allowing quick recovery from misleading priors while still exploiting good priors early.

### Algorithm Advantages

!!! tip "Why Use PriMO?"

    - **Robust to prior quality**: The ε-greedy strategy with quadratic decay allows PriMO to leverage good priors for early speedup while recovering from bad priors
    - **Compute-efficient**: Up to 10x speedups observed on DL benchmarks when good priors are available
    - **Multi-fidelity support**: Leverages cheap proxy tasks to accelerate initial seed discovery
    - **True multi-objective**: Unlike scalarization baselines, PriMO explores the full Pareto Front through random weight resampling
    - **No learning curve**: Works effectively even with modest budgets typical in deep learning (10-20 evaluations)

### When to Use PriMO

PriMO should be your choice when:

- ✅ You have **multiple objectives** to optimize (e.g., accuracy and latency)
- ✅ You have **expert knowledge** about good hyperparameter regions for each objective
- ✅ You **can leverage cheap approximations** (lower fidelity runs, smaller datasets, fewer epochs)
- ✅ You have **limited compute budget** (typical practical deep learning scenarios)

PriMO may not be ideal when:

- ❌ You only have a single objective (use [`PiBO`](../search_algorithms/prior.md#1-pibo) instead)
- ❌ You have no access to multi-fidelity information
- ❌ You cannot express expert knowledge as priors for individual objectives

### Practical Considerations

!!! warning "Caution with Priors"

    - **Misleading priors**: Even if priors are bad, PriMO recovers after ~10 BO samples due to its decay schedule. However, multiple runs with different seeds may be needed for reliable coverage
    - **Prior generation**: Good priors should reflect expert intuition or be derived from similar tasks. Use the highest-fidelity data available when constructing priors
    - **Objective scaling**: Ensure objectives are on comparable scales for effective scalarization, or normalize them appropriately

See the algorithm's implementation details in the [API][neps.optimizers.algorithms.primo].

!!! info
    `PriMO` is the recommended choice in NePS when you have **both Multi-Fidelity and Multi-Objective** settings with **expert priors** available.

___

### Single-Objective Special Case

PriMO can be specialized to single-objective optimization by:
- Replacing the multi-fidelity `MOASHA` component with `ASHA` in the initial design
- Using the single objective's prior throughout (instead of randomly selecting among priors)
- Maintaining the ε-greedy prior-weighted acquisition function

### Comparison to Related Work

| Aspect | MOASHA | BO+RW | ParEGO | πBO+RW | MO-Priorband | PriMO |
| :- | :-: | :-: | :-: | :-: | :-: | :-: |
| Multi-Objective | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Multi-Fidelity | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Multi-Objective Priors | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Robust to Bad Priors | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| Sample Efficiency | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |

