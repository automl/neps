# Algorithms

Algorithms are the search strategies determining what configurations to evaluate next. In NePS, we provide a variety of pre-implemented algorithms and offer the possibility to implement custom algorithms. This chapter gives an overview of the different algorithms available in NePS and practical tips for their usage.

We distinguish between algorithms that use different types of information and strategies to guide the search process:

✅ = supported/necessary, ❌ = not supported, ✔️* = optional, click for details, ✖️\* ignorable, click for details

| Algorithm         | [Multi-Fidelity](../search_algorithms/multifidelity.md) | [Priors](../search_algorithms/prior.md) | Model-based | [NePS-ready](../neps_spaces.md#3-constructing-architecture-spaces) | Multi-Objective |
| :- | :------------: | :----: | :---------: | :-----------------: | :---------------: |
| `Grid Search`|[️️✖️*][neps.optimizers.algorithms.grid_search]|❌|❌|✅|❌|
| `Random Search`|[️️✖️*][neps.optimizers.algorithms.random_search]|[✔️*][neps.optimizers.algorithms.random_search]|❌|✅|❌|
| `Complex Random Search`|[️️✖️*][neps.optimizers.algorithms.complex_random_search]|[✔️*][neps.optimizers.algorithms.complex_random_search]|❌|✅|❌|
| [`Bayesian Optimization`](../search_algorithms/bayesian_optimization.md)|[️️✖️*][neps.optimizers.algorithms.bayesian_optimization]|❌|✅|❌|❌|
| [`Successive Halving`](../search_algorithms/multifidelity.md#1-successive-halfing)|✅|[✔️*][neps.optimizers.algorithms.successive_halving]|❌|✅|❌|
| [`ASHA`](../search_algorithms/multifidelity.md#asynchronous-successive-halving)|✅|[✔️*][neps.optimizers.algorithms.asha]|❌|✅|❌|
| [`Hyperband`](../search_algorithms/multifidelity.md#2-hyperband)|✅|[✔️*][neps.optimizers.algorithms.hyperband]|❌|✅|❌|
| [`Asynch HB`](../search_algorithms/multifidelity.md)|✅|[✔️*][neps.optimizers.algorithms.async_hb]|❌|✅|❌|
| [`IfBO`](../search_algorithms/multifidelity.md#3-in-context-freeze-thaw-bayesian-optimization)|✅|[✔️*][neps.optimizers.algorithms.ifbo]|✅|❌|❌|
| [`PiBO`](../search_algorithms/prior.md#1-pibo)|[️️✖️*][neps.optimizers.algorithms.pibo]|✅|✅|❌|❌|
| [`PriorBand`](../search_algorithms/multifidelity_prior.md#1-priorband)|✅|✅|✅|✅|❌|
| `PriMO`|❌|✅|❌|❌|✅|

## What is Multi-Fidelity Optimization?

Multi-Fidelity (MF) optimization leverages the idea of running an AutoML problem on a small scale, which is cheaper and faster, and then using this information to train full-scale models. The _low-fidelity_ runs could be on a smaller dataset, a smaller model, or for shorter training times. MF-algorithms then infer which configurations are likely to perform well on the full problem, before investing larger compute amounts.

!!! tip "Advantages of Multi-Fidelity"

    - **Parallelization**: MF-algorithms can use the information from many parallel low-fidelity runs to guide the search in the few high-fidelity runs.
    - **Exploration**: By using low-fidelity runs, the optimizer can explore more of the search space.

!!! warning "Disadvantages of Multi-Fidelity"

    - **Variance**: The performance of a configuration on a low-fidelity run might not correlate well with its performance on a high-fidelity run. This can result in misguided decisions.

We present a collection of MF-algorithms [here](./multifidelity.md) and algorithms that combine MF with priors [here](./multifidelity_prior.md).

## What are Priors?

Priors are used when there exists some information about the search space, that can be used to guide the optimization process. This information could come from expert domain knowledge or previous experiments. A Prior is provided in the form of a distribution over one dimension of the search space, with a `mean` (the suspected optimum) and a `confidence level`, or `variance`. We discuss how Priors can be included in your NePS-search space [here](../../reference/neps_spaces.md#1-constructing-hyperparameter-spaces).

!!! tip "Advantages of using Priors"

    - **Less compute**: By providing a Prior, the optimizer can focus on the most promising regions of the search space, potentially saving a lot of compute.
    - **More exploitation**: By focusing on these regions, the optimizer might find a better final solution.

!!! warning "Disadvantages of using Priors"

    - **Less exploration**: By focusing on these regions, the optimizer _might_ miss out on other regions that could potentially be better.
    - **Bad priors**: If the Prior is not a good representation of the search space, the optimizer might deliver suboptimal results, compared to a search without Priors. The optimizers we provide in NePS are specifically designed to handle bad priors, but they still slow down the search process.

We present a collection of algorithms that use Priors [here](./prior.md) and algorithms that combine priors with Multi-Fidelity [here](./multifidelity_prior.md).
