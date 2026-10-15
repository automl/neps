# Optimizer Configuration

## 1 What optimizer works best for my problem?

The best optimizers utilizes all information available in the search space to guide the optimization process. Besides a fully black-box search, there are two sources of information an optimizer can draw from: using small scale proxies ([Multi-Fidelity](#11-multi-fidelity-mf)) and intuition ([Priors](#12-priors)).

### 1.1 Multi-Fidelity (MF)

Multi-Fidelity uses small scale version of the problem, which run cheaper and faster. This could mean training models for a shorter time, using only a subset of the training data, or a smaller model entirely. From these *low fidelity* runs, MF-algorithms can infer which configurations are likely to perform well on the full problem.

It is defined using the `is_fidelity` parameter in the `pipeline_space` definition.

```python
pipeline_space = {
    "epoch": autoscan.Integer(lower=1, upper=100, is_fidelity=True),
    # epoch will be available as fidelity to the optimizer
}
```

For a more detailed explanation of Multi-Fidelity and a list of AutoScAn-optimizers using MF please refer [here](../reference/search_algorithms/multifidelity.md).

### 1.2 Priors

Optimization with Priors is used, when there already exists an intuition for what region or specific value of a hyperparameter _could_ work well. By providing this intuition as Prior (knowledge) to the optimizer, it can prioritize these most promising regions of the search space, potentially saving a lot of compute.

It is defined using the `prior` parameter in the `pipeline_space` definition.

```python
pipeline_space = {
    "alpha": autoscan.Float(lower=0.1, upper=1.0, prior=0.4, prior_confidence="high"),
    # alpha will have a prior pointing towards 0.4 with high confidence
}
```

For a more detailed explanation of Priors and a list of AutoScAn-optimizers using Priors please refer [here](../reference/search_algorithms/prior.md).

## 2 AutoScAn Optimizer Selection

### 2.1 Automatic Optimizer Selection

AutoScAn provides a multitude of optimizers from the literature, the [algorithms](../reference/search_algorithms/landing_page_algo.md) section goes into detail on each of them. This chapter focusses on how to select them when using AutoScAn.

✅ = supported/necessary, ❌ = not supported, ✔️* = optional, click for details, ✖️\* ignorable, click for details

| Algorithm         | [Multi-Fidelity](../reference/search_algorithms/multifidelity.md) | [Priors](../reference/search_algorithms/prior.md) | Model-based | [AutoScAn-ready](../reference/autoscan_spaces.md#3-constructing-architecture-spaces) | Multi-Objective |
| :- | :------------: | :----: | :---------: | :-----------------: | :---------------: |
| `Grid Search`|[️️✖️*][autoscan.optimizers.algorithms.grid_search]|❌|❌|✅|❌|
| `Random Search`|[️️✖️*][autoscan.optimizers.algorithms.random_search]|[✔️*][autoscan.optimizers.algorithms.random_search]|❌|✅|❌|
| [`Bayesian Optimization`](../reference/search_algorithms/bayesian_optimization.md)|[️️✖️*][autoscan.optimizers.algorithms.bayesian_optimization]|❌|✅|❌|❌|
| [`Successive Halving`](../reference/search_algorithms/multifidelity.md#1-successive-halfing)|✅|[✔️*][autoscan.optimizers.algorithms.successive_halving]|❌|✅|❌|
| [`ASHA`](../reference/search_algorithms/multifidelity.md#asynchronous-successive-halving)|✅|[✔️*][autoscan.optimizers.algorithms.asha]|❌|✅|❌|
| [`Hyperband`](../reference/search_algorithms/multifidelity.md#2-hyperband)|✅|[✔️*][autoscan.optimizers.algorithms.hyperband]|❌|✅|❌|
| [`Asynch HB`](../reference/search_algorithms/multifidelity.md)|✅|[✔️*][autoscan.optimizers.algorithms.async_hb]|❌|✅|❌|
| [`IfBO`](../reference/search_algorithms/multifidelity.md#3-in-context-freeze-thaw-bayesian-optimization)|✅|[✔️*][autoscan.optimizers.algorithms.ifbo]|✅|❌|❌|
| [`PiBO`](../reference/search_algorithms/prior.md#1-pibo)|[️️✖️*][autoscan.optimizers.algorithms.pibo]|✅|✅|❌|❌|
| [`PriorBand`](../reference/search_algorithms/multifidelity_prior.md#1-priorband)|✅|✅|✅|✅|❌|
| [`PriMO`](../reference/search_algorithms/multi_objective.md/#primo-prior-informed-multi-objective-optimizer)|✅|✅|❌|❌|✅|

If you prefer not to specify a particular optimizer for your AutoML task, you can simply pass `"auto"` or `None`
for the autoscan optimizer. This provides a hassle-free way to get started quickly, as AutoScAn will automatically choose the best optimizer based on the characteristics of your search
space:

- If it has fidelity: [`hyperband`](../reference/search_algorithms/multifidelity.md#2-hyperband)
- If it has both fidelity and a prior: [`priorband`](../reference/search_algorithms/multifidelity_prior.md#1-priorband)
- If it has a prior: [`pibo`](../reference/search_algorithms/prior.md#1-pibo)
- If it has neither: [`bayesian_optimization`](../reference/search_algorithms/bayesian_optimization.md)

For example, running the following format, without specifying a optimizer will choose an optimizer depending on
the `pipeline_space` passed.

```python
autoscan.run(
    evaluate_pipeline=run_function,
    pipeline_space=pipeline_space,
    root_directory="results/",
    total_evaluations_to_spend=25,
    # no optimizer specified
)
```

### 2.2 Choosing one of AutoScAn Optimizers

We have also prepared some optimizers with specific hyperparameters that we believe can generalize well to most AutoML tasks and use cases. The available optimizers are imported via the `autoscan.algorithms` module.
You can use either the optimizer name or the optimizer class itself as the optimizer argument.

```python
autoscan.run(
    evaluate_pipeline=run_function,
    pipeline_space=pipeline_space,
    root_directory="results/",
    total_evaluations_to_spend=25,
    # optimizer specified, along with an argument
    optimizer=autoscan.algorithms.bayesian_optimization, # or as string: "bayesian_optimization"
)
```

For a list of available optimizers, please refer [here](./search_algorithms/landing_page_algo.md).

!!! tip "Interactive tutorial"

    The Optimizer Selection section of the [Efficient Optimization tutorial](https://colab.research.google.com/github/automl/auto-scan/blob/master/tutorials/3_efficiency_techniques.ipynb) lists all built-in optimizers with their default hyperparameters and runs several of them on the same task.

### 2.3 Hyperparameter Overrides

For users who want more control over the optimizer's hyperparameters, you can input a dictionary with your parameter choices together with the optimizer name.

```python
autoscan.run(
    evaluate_pipeline=run_function,
    pipeline_space=pipeline_space,
    root_directory="results/",
    total_evaluations_to_spend=25,
    optimizer=("bayesian_optimization", {"initial_design_size": 5})
)
```

### 2.4 Loading Optimizer Information

AutoScAn automatically saves the optimizer metadata (name and configuration) when you run an optimization. You can retrieve this information later using `autoscan.load_optimizer_info()`:

```python
import autoscan

# Load the optimizer info from a previous run
optimizer_info = autoscan.load_optimizer_info("path/to/autoscan_folder")

# Access the optimizer name and configuration
print(f"Optimizer: {optimizer_info['name']}")
print(f"Configuration: {optimizer_info['info']}")
```

This is useful for:

- **Inspecting** what optimizer and settings were used in a previous run
- **Reproducing** experiments with the same optimizer configuration
- **Comparing** different optimizer settings across runs

!!! tip "Reconstructing a Complete Run"

    Combine `load_optimizer_info()` with `load_pipeline_space()` to fully reconstruct a previous optimization. See [Reconstructing and Reproducing Runs](autoscan_run.md#reconstructing-and-reproducing-runs) for a complete example.

## 3 Scaling Optimizers

Scaling optimizers build scaling laws into the optimization itself. Instead of finding
one good configuration for a fixed setting, they spend a search budget `C_search` on
cheaper configurations and learn how the best configuration changes with cost. From this
they predict the best configuration for a larger target cost.

They draw on three arguments of `autoscan.run()`, see
[estimating the cost of a configuration](autoscan_run.md#scaling-studies-estimating-the-cost-of-a-configuration):

- `cost_estimator`: the cost of a configuration, e.g. its FLOPs, known before it is
  evaluated. It is computed from the parameters marked with `is_compute_scaling=True`.
- `max_cost_per_config`: no configuration estimated to cost more than this is sampled.
- `total_cost_to_spend`: the search budget `C_search`, summed over all evaluations.

| Optimizer | Approach | Supports | Install |
| :- | :- | :- | :- |
| [`carbs`][autoscan.optimizers.algorithms.carbs] | Cost-aware BO along the pareto front of objective and cost | `Integer` and `Float` parameters, no fidelities | `pip install auto-scan[carbs]` |
| [`chinchilla`][autoscan.optimizers.algorithms.chinchilla] | Splits compute between model size and data on ISO-FLOP levels | Needs `cost_estimator` and `total_cost_to_spend`, no fidelities | Included |

### 3.1 CARBS

[CARBS](https://github.com/imbue-ai/carbs) searches around the pareto front of objective
and cost. It starts from a search center and moves to more expensive configurations as
it goes, which yields good configurations at every cost, not just one. It needs the cost
of every evaluation. That is either the `cost` returned by `evaluate_pipeline` or the
estimate of the `cost_estimator`:

```python
autoscan.run(
    evaluate_pipeline=run_function,
    pipeline_space=pipeline_space,
    root_directory="results/",
    optimizer=("carbs", {"initial_search_radius": 0.3}),
    cost_estimator=estimate_flops,
    max_cost_per_config=1e12,
)
```

If the center of the space is too expensive to start from, set
`search_center_cost_fraction` and `search_center_shrink_param`. The given parameter,
e.g. the size of the dataset, is halved until the search center costs at most that
fraction of `max_cost_per_config`.

CARBS wraps an external library, so it is not installed with AutoScAn. Install it with
`pip install auto-scan[carbs]`. It does not keep any files of its own. Its state is
rebuilt from the trials in the `root_directory` whenever a configuration is sampled, so
runs can be continued and parallelized like with any other optimizer.

### 3.2 Chinchilla Guided Scaling

[`chinchilla`][autoscan.optimizers.algorithms.chinchilla] follows
[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556). It
lays out a static list of configurations that grow the model and the amount of data together,
to find how a compute budget should be split between the two. The parameters marked
with `is_compute_scaling=True` make up the total scaling parameters and `data_parameter_name` names the one that sets the amount of data. Every other parameter is drawn at random once and held
fixed.

```python
def estimate_params(width: int, layers: int) -> float:
    return ...

autoscan.run(
    evaluate_pipeline=run_function,
    pipeline_space=pipeline_space,
    root_directory="results/",
    optimizer=(
        "chinchilla",
        {"data_parameter_name": "n_data", "params_estimator": estimate_params},
    ),
    cost_estimator=estimate_flops,  # (1)!
    max_cost_per_config=1e12,  # (2)!
    total_cost_to_spend=1e14,  # (3)!
)
```

1.  Required, see the
    [`cost_estimator`](autoscan_run.md#scaling-studies-estimating-the-cost-of-a-configuration).
2.  Optional. No configuration estimated to cost more than this is laid out.
3.  Required, the configurations are laid out to fit into this budget.
    `worker_cost_to_spend` is used if it is not given.

| `strategy` | Layout |
| :- | :- |
| `"approach2"` (default) | Configurations on `iso_flop_steps` ISO-FLOP levels, up to `candidates_per_iso_flop` on each |
| `"approach1"` | Every model is trained on `data_multipliers` times its parameter count |

The configurations are laid out on the first ask, and all of them fit into the budget.
The run therefore ends with a `Grid search exhausted` error once every one of them was
evaluated, if the budget is not used up before. With `live_plots=True`, the fitted
scaling of parameters and data with the compute is written to the `summary` folder.

!!! note

    The values of the parameters that are not scaled are drawn when the configurations
    are laid out, so every worker of a run draws its own.

## 4 Custom Optimizers

To design entirely new optimizers, you can define them as class with a `__call__` method outside of AutoScAn and pass them to the `autoscan.run()` function:

```python
@dataclass
class MyOptimizer:
    space: P
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

The class is then passed to the `autoscan.run()` function just like the built-in optimizers and can be configured the same way, using a dictionary:

```python
autoscan.run(
    evaluate_pipeline=run_function,
    pipeline_space=pipeline_space,
    root_directory="results/",
    total_evaluations_to_spend=25,
    optimizer=MyOptimizer,
)
```

For more details on how to define a custom optimizer see the [Optimizer Interface][autoscan.optimizers.optimizer.AskFunction].

To receive the `cost_estimator` and `max_cost_per_config` given to `autoscan.run()`, like the [scaling optimizers](#3-scaling-optimizers), inherit from [`ScalingOptimizer`][autoscan.optimizers.optimizer.ScalingOptimizer]:

```python
class MyOptimizer(ScalingOptimizer):
    def __call__(self, trials, budget_info, n=None):
        if self.cost_estimator is not None:
            cost = self.cost_estimator(**config)
        ...
```
