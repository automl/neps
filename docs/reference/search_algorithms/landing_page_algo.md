# Algorithms

Algorithms are the search strategies determining what configurations to evaluate next. In Extrapolaw, we provide a few pre-implemented algorithms for scaling studies and offer the possibility to implement custom algorithms. This chapter gives an overview of the algorithms available in Extrapolaw.

✅ = supported/necessary, ❌ = not supported, ✔️* = optional, click for details, ✖️\* ignorable, click for details

| Algorithm         | [Fidelity](#fidelities) | [Priors](#priors) | Model-based | [Extrapolaw-ready](../extrapolaw_spaces.md#3-constructing-architecture-spaces) | Scaling-aware |
| :- | :------------: | :----: | :---------: | :-----------------: | :---------------: |
| `Grid Search`|[️️✖️*][extrapolaw.optimizers.algorithms.grid_search]|❌|❌|✅|❌|
| `Random Search`|[️️✖️*][extrapolaw.optimizers.algorithms.random_search]|[✔️*][extrapolaw.optimizers.algorithms.random_search]|❌|✅|❌|
| `Complex Random Search`|[️️✖️*][extrapolaw.optimizers.algorithms.complex_random_search]|❌|❌|✅|❌|
| [`Bayesian Optimization`](../search_algorithms/bayesian_optimization.md)|[️️✖️*][extrapolaw.optimizers.algorithms.bayesian_optimization]|❌|✅|❌|[✔️*][extrapolaw.optimizers.algorithms.bayesian_optimization]|
| [`CARBS`](../optimizers.md#31-carbs)|❌|[✔️*][extrapolaw.optimizers.algorithms.carbs]|✅|❌|✅|
| [`Chinchilla`](../optimizers.md#32-chinchilla-guided-scaling)|❌|❌|❌|❌|✅|

Scaling-aware optimizers use the [`cost_estimator`](../extrapolaw_acquire.md#scaling-studies-estimating-the-cost-of-a-configuration) of the run, see [scaling optimizers](../optimizers.md#3-scaling-optimizers).

## Fidelities

A fidelity parameter, e.g. the number of epochs, can be declared with `IntegerFidelity` or `FloatFidelity`. None of the optimizers uses it to evaluate cheaper versions of a configuration. Grid search, random search and bayesian optimization take `ignore_fidelity`: `"highest_fidelity"` always sets it to its maximum, and `True` treats it like any other parameter.

## Priors

A prior is a value of a parameter that is expected to work well, with a `prior_confidence`. We discuss how priors can be included in your search space [here](../../reference/extrapolaw_spaces.md#1-constructing-hyperparameter-spaces). [`random_search`][extrapolaw.optimizers.algorithms.random_search] samples around them with `use_priors=True`, and [`carbs`][extrapolaw.optimizers.algorithms.carbs] starts its search from them. The other optimizers do not use them.
