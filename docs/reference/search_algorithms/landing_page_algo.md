# Algorithms

Algorithms are the search strategies determining what configurations to evaluate next. In NePS, we provide a variety of pre-implemented algorithms and offer the possibility to implement custom algorithms.This chapter gives an overview of the different algorithms available in NePS and practical tips for their usage. We distinguish between algorithms that use different types of information and strategies to guide the search process.

| Algorithm         | [Multi-Fidelity](./multifidelity.md) | [Priors](./prior.md) | Model-based | Asynchronous |
| :- | :------------: | :----: | :---------: | :-: |
| `Grid Search`||||✅|
| `Random Search`||||✅|
| [`Successive Halving`](./multifidelity.md#1-successive-halfing)|✅||||
| [`ASHA`](./multifidelity.md#asynchronous-successive-halving)|✅|||✅|
| [`Hyperband`](./multifidelity.md#2-hyperband)|✅||||
| [`Asynch HB`](./multifidelity.md)|✅|||✅|
| [`IfBO`](./multifidelity.md#5-in-context-freeze-thaw-bayesian-optimization)|✅||✅||
| [`PiBO`](./prior.md#1-pibo)||✅|✅||
| [`PriorBand`](./multifidelity_prior.md#1-priorband)|✅|✅|✅||

## What is Multi-Fidelity Optimization?

Multi-Fidelity (MF) optimization leverages the idea of running an AutoML problem on a small scale, which is cheaper and faster, and then using this information to train full-scale models. The _low-fidelity_ runs could be on a smaller dataset, a smaller model, or for shorter training times. MF-algorithms then infer which configurations are likely to perform well on the full problem, before investing larger compute amounts.

!!! tip "Advantages of Multi-Fidelity"

    - **Parallelization**: MF-algorithms can use the information from many parallel low-fidelity runs to guide the search in the few high-fidelity runs.
    - **Exploration**: By using low-fidelity runs, the optimizer can explore more of the search space.

!!! warning "Disadvantages of Multi-Fidelity"

    - **More compute**: Running multiple iterations on different fidelities is generally more compute-intensive.
    - **Variance**: The performance of a configuration on a low-fidelity run might not correlate well with its performance on a high-fidelity run. This can result in misguided decisions.

## What are Priors?

Priors are used when there exists some information about the search space, that can be used to guide the optimization process. This information could come from expert domain knowledge or previous experiments. A Prior is provided in the form of a distribution over one dimension of the search space, with a `mean` (the suspected optimum) and a `confidence level`, or `variance`. We discuss how Priors can be included in your NePS-search space [here](../../reference/pipeline_space.md#using-your-knowledge-providing-a-prior).

!!! tip "Advantages of using Priors"

    - **Less compute**: By providing a Prior, the optimizer can focus on the most promising regions of the search space, potentially saving a lot of compute.
    - **More exploitation**: By focusing on these regions, the optimizer might find a better final solution.

!!! warning "Disadvantages of using Priors"

    - **Less exploration**: By focusing on these regions, the optimizer _might_ miss out on other regions that could potentially be better.
    - **Bad priors**: If the Prior is not a good representation of the search space, the optimizer might deliver suboptimal results, compared to a search without Priors.
