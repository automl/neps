# Optimizer Configuration

## 1 What optimizer works best for my problem?

The best optimizers utilizes all information available in the search space to guide the optimization process. Besides a fully black-box search, there are two sources of information an optimizer can draw from: using small scale proxies ([Multi-Fidelity](#11-multi-fidelity-mf)) and intuition ([Priors](#12-priors)).

### 1.1 Multi-Fidelity (MF)

Multi-Fidelity uses small scale version of the problem, which run cheaper and faster. This could mean training models for a shorter time, using only a subset of the training data, or a smaller model entirely. From these *low fidelity* runs, MF-algorithms can infer which configurations are likely to perform well on the full problem.

It is defined using the `is_fidelity` parameter in the `pipeline_space` definition.

```python
pipeline_space = {
    "epoch": neps.Integer(lower=1, upper=100, is_fidelity=True),
    # epoch will be available as fidelity to the optimizer
}
```

For a more detailed explanation of Multi-Fidelity and a list of NePS-optimizers using MF please refer [here](../reference/search_algorithms/multifidelity.md).

### 1.2 Priors

Optimization with Priors is used, when there already exists an intuition for what region or specific value of a hyperparameter _could_ work well. By providing this intuition as Prior (knowledge) to the optimizer, it can prioritize these most promising regions of the search space, potentially saving a lot of compute.

It is defined using the `prior` parameter in the `pipeline_space` definition.

```python
pipeline_space = {
    "alpha": neps.Float(lower=0.1, upper=1.0, prior=0.4, prior_confidence="high"),
    # alpha will have a prior pointing towards 0.4 with high confidence
}
```

For a more detailed explanation of Priors and a list of NePS-optimizers using Priors please refer [here](../reference/search_algorithms/prior.md).

## 2 NePS Optimizer Selection

### 2.1 Automatic Optimizer Selection

NePS provides a multitude of optimizers from the literature, the [algorithms](../reference/search_algorithms/landing_page_algo.md) section goes into detail on each of them. This chapter focusses on how to select them when using NePS.

✅ = supported/necessary, ❌ = not supported, ✔️* = optional, click for details, ✖️\* ignorable, click for details

| Algorithm         | [Multi-Fidelity](../reference/search_algorithms/multifidelity.md) | [Priors](../reference/search_algorithms/prior.md) | Model-based | [NePS-ready](../reference/neps_spaces.md#3-architectures) |
| :- | :------------: | :----: | :---------: | :-----------------: |
| `Grid Search`|[️️✖️*][neps.optimizers.algorithms.grid_search]|❌|❌|❌|
| `Random Search`|[️️✖️*][neps.optimizers.algorithms.random_search]|[✔️*][neps.optimizers.algorithms.random_search]|❌|✅|
| `Complex Random Search`|[️️✖️*][neps.optimizers.algorithms.complex_random_search]|[✔️*][neps.optimizers.algorithms.complex_random_search]|❌|✅|
| [`Bayesian Optimization`](../reference/search_algorithms/bayesian_optimization.md)|[️️✖️*][neps.optimizers.algorithms.bayesian_optimization]|❌|✅|❌|
| [`Successive Halving`](../reference/search_algorithms/multifidelity.md#1-successive-halfing)|✅|[✔️*][neps.optimizers.algorithms.successive_halving]|❌|❌|
| [`ASHA`](../reference/search_algorithms/multifidelity.md#asynchronous-successive-halving)|✅|[✔️*][neps.optimizers.algorithms.asha]|❌|❌|
| [`Hyperband`](../reference/search_algorithms/multifidelity.md#2-hyperband)|✅|[✔️*][neps.optimizers.algorithms.hyperband]|❌|❌|
| [`Asynch HB`](../reference/search_algorithms/multifidelity.md)|✅|[✔️*][neps.optimizers.algorithms.async_hb]|❌|❌|
| [`IfBO`](../reference/search_algorithms/multifidelity.md#3-in-context-freeze-thaw-bayesian-optimization)|✅|[✔️*][neps.optimizers.algorithms.ifbo]|✅|❌|
| [`PiBO`](../reference/search_algorithms/prior.md#1-pibo)|[️️✖️*][neps.optimizers.algorithms.pibo]|✅|✅|❌|
| [`PriorBand`](../reference/search_algorithms/multifidelity_prior.md#1-priorband)|✅|✅|✅|✅|

If you prefer not to specify a particular optimizer for your AutoML task, you can simply pass `"auto"` or `None`
for the neps optimizer. This provides a hassle-free way to get started quickly, as NePS will automatically choose the best optimizer based on the characteristics of your search
space:

- If it has fidelity: [`hyperband`](../reference/search_algorithms/multifidelity.md#2-hyperband)
- If it has both fidelity and a prior: [`priorband`](../reference/search_algorithms/multifidelity_prior.md#1-priorband)
- If it has a prior: [`pibo`](../reference/search_algorithms/prior.md#1-pibo)
- If it has neither: [`bayesian_optimization`](../reference/search_algorithms/bayesian_optimization.md)

For example, running the following format, without specifying a optimizer will choose an optimizer depending on
the `pipeline_space` passed.

```python
neps.run(
    evaluate_pipeline=run_function,
    pipeline_space=pipeline_space,
    root_directory="results/",
    max_evaluations_total=25,
    # no optimizer specified
)
```

### 2.2 Choosing one of NePS Optimizers

We have also prepared some optimizers with specific hyperparameters that we believe can generalize well to most AutoML tasks and use cases. The available optimizers are imported via the `neps.algorithms` module.
You can use either the optimizer name or the optimizer class itself as the optimizer argument.

```python
neps.run(
    evaluate_pipeline=run_function,
    pipeline_space=pipeline_space,
    root_directory="results/",
    max_evaluations_total=25,
    # optimizer specified, along with an argument
    optimizer=neps.algorithms.bayesian_optimization, # or as string: "bayesian_optimization"
)
```

For a list of available optimizers, please refer [here](./search_algorithms/landing_page_algo.md).

### 2.3 Hyperparameter Overrides

For users who want more control over the optimizer's hyperparameters, you can input a dictionary with your parameter choices together with the optimizer name.

```python
neps.run(
    evaluate_pipeline=run_function,
    pipeline_space=pipeline_space,
    root_directory="results/",
    max_evaluations_total=25,
    optimizer=("bayesian_optimization", {"initial_design_size": 5})
)
```

## 3 Custom Optimizers

To design entirely new optimizers, you can define them as class with a `__call__` method outside of NePS and pass them to the `neps.run()` function:

```python
@dataclass
class MyOptimizer:
    space: SearchSpace
    sampler: Sampler
    encoder: ConfigEncoder

    def __call__(
        self,
        trials: Mapping[str, Trial],
        budget_info: BudgetInfo | None,
        n: int | None,
    ) -> SampledConfig | list[SampledConfig]:
        # Your custom sampling logic here
        ...
```

The class is then passed to the `neps.run()` function just like the built-in optimizers and can be configured the same way, using a dictionary:

```python
neps.run(
    evaluate_pipeline=run_function,
    pipeline_space=pipeline_space,
    root_directory="results/",
    max_evaluations_total=25,
    optimizer=MyOptimizer,
)
```

For more details on how to define a custom optimizer see the [Optimizer Interface][neps.optimizers.optimizer.AskFunction].
