# Optimizer Configuration

## 1 What optimizer works best for my problem?

For a scaling study, the [scaling optimizers](#3-scaling-optimizers) build the scaling law into the search: [`chinchilla`](#32-chinchilla-guided-scaling) splits the compute between model size and data, and [`carbs`](#31-carbs) searches along the pareto front of objective and cost. To sample a space without assumptions on how it scales, use `random_search` or `grid_search`, and `bayesian_optimization` to find good configurations with fewer evaluations.

Fidelity parameters and priors in the search space are optional, see [priors](../reference/search_algorithms/landing_page_algo.md#priors) for the optimizers that use them.

## 2 ScaleFM Optimizer Selection

### 2.1 Automatic Optimizer Selection

The [algorithms](../reference/search_algorithms/landing_page_algo.md) section goes into detail on each optimizer. This chapter focusses on how to select them when using ScaleFM.

✅ = supported/necessary, ❌ = not supported, ✔️* = optional, click for details

| Algorithm         | [Priors](../reference/search_algorithms/landing_page_algo.md#priors) | Model-based |
| :- | :----: | :---------: |
| `Grid Search`|❌|❌|
| `Random Search`|[✔️*][scalefm.optimizers.algorithms.random_search]|❌|
| `Complex Random Search`|❌|❌|
| [`Bayesian Optimization`](../reference/search_algorithms/bayesian_optimization.md)|❌|✅|
| [`CARBS`](#31-carbs)|[✔️*][scalefm.optimizers.algorithms.carbs]|✅|
| [`Chinchilla`](#32-chinchilla-guided-scaling)|❌|❌|

Without an `optimizer`, `scalefm.acquire()` uses [`random_search`][scalefm.optimizers.algorithms.random_search].
Pass `"auto"` instead to let ScaleFM choose the optimizer based on your search space
(or, when continuing a run, to reuse the optimizer it was started with):

- If it has neither a fidelity nor a prior: [`bayesian_optimization`](../reference/search_algorithms/bayesian_optimization.md)
- Otherwise: [`random_search`][scalefm.optimizers.algorithms.random_search], sampling around the priors, if there are any
- If it cannot be converted to a classic search space, e.g. because it has an `Operation`: [`complex_random_search`][scalefm.optimizers.algorithms.complex_random_search]

A fidelity is always set to its maximum.

For example, the following chooses an optimizer depending on the `pipeline_space` passed.

```python
scalefm.acquire(
    evaluate_pipeline=run_function,
    pipeline_space=pipeline_space,
    root_directory="results/",
    total_evaluations_to_spend=25,
    optimizer="auto",
)
```

### 2.2 Choosing one of ScaleFM Optimizers

The available optimizers are imported via the `scalefm.algorithms` module.
You can use either the optimizer name or the optimizer class itself as the optimizer argument.

```python
scalefm.acquire(
    evaluate_pipeline=run_function,
    pipeline_space=pipeline_space,
    root_directory="results/",
    total_evaluations_to_spend=25,
    # optimizer specified, along with an argument
    optimizer=scalefm.algorithms.bayesian_optimization, # or as string: "bayesian_optimization"
)
```

For a list of available optimizers, please refer [here](./search_algorithms/landing_page_algo.md).

!!! tip "Interactive tutorial"

    The [ScaleFM demo](https://colab.research.google.com/github/automl/scalefm/blob/master/tutorials/scalefm_demo.ipynb) shows the basic pipeline end to end, with Chinchilla approach 1 and the default random search.

### 2.3 Hyperparameter Overrides

For users who want more control over the optimizer's hyperparameters, you can input a dictionary with your parameter choices together with the optimizer name.

```python
scalefm.acquire(
    evaluate_pipeline=run_function,
    pipeline_space=pipeline_space,
    root_directory="results/",
    total_evaluations_to_spend=25,
    optimizer=("bayesian_optimization", {"initial_design_size": 5})
)
```

### 2.4 Loading Optimizer Information

ScaleFM automatically saves the optimizer metadata (name and configuration) when you run an optimization. You can retrieve this information later using `scalefm.load_optimizer_info()`:

```python
import scalefm

# Load the optimizer info from a previous run
optimizer_info = scalefm.load_optimizer_info("path/to/scalefm_folder")

# Access the optimizer name and configuration
print(f"Optimizer: {optimizer_info['name']}")
print(f"Configuration: {optimizer_info['info']}")
```

This is useful for:

- **Inspecting** what optimizer and settings were used in a previous run
- **Reproducing** experiments with the same optimizer configuration
- **Comparing** different optimizer settings across runs

!!! tip "Reconstructing a Complete Run"

    Combine `load_optimizer_info()` with `load_pipeline_space()` to fully reconstruct a previous optimization. See [Reconstructing and Reproducing Runs](scalefm_acquire.md#reconstructing-and-reproducing-runs) for a complete example.

## 3 Scaling Optimizers

Scaling optimizers build scaling laws into the optimization itself. Instead of finding
one good configuration for a fixed setting, they spend a search budget `C_search` on
cheaper configurations and learn how the best configuration changes with cost. From this
they predict the best configuration for a larger target cost.

They draw on three arguments of `scalefm.acquire()`, see
[estimating the cost of a configuration](scalefm_acquire.md#scaling-studies-estimating-the-cost-of-a-configuration):

- `cost_estimator`: the cost of a configuration, e.g. its FLOPs, known before it is
  evaluated. It is computed from the parameters marked with `is_compute_scaling=True`.
- `max_cost_per_config`: no configuration estimated to cost more than this is sampled.
- `total_cost_to_spend`: the search budget `C_search`, summed over all evaluations.

| Optimizer | Approach | Supports | Install |
| :- | :- | :- | :- |
| [`carbs`][scalefm.optimizers.algorithms.carbs] | Cost-aware BO along the pareto front of objective and cost | `Integer` and `Float` parameters, no fidelities | `pip install scalefm[carbs]` |
| [`chinchilla`][scalefm.optimizers.algorithms.chinchilla] | Splits compute between model size and data on ISO-FLOP levels | Needs `cost_estimator` and `total_cost_to_spend`, no fidelities | Included |

### 3.1 CARBS

[CARBS](https://github.com/imbue-ai/carbs) searches around the pareto front of objective
and cost. It starts from a search center and moves to more expensive configurations as
it goes, which yields good configurations at every cost, not just one. It needs the cost
of every evaluation. That is either the `cost` returned by `evaluate_pipeline` or the
estimate of the `cost_estimator`:

```python
scalefm.acquire(
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

CARBS wraps an external library, so it is not installed with ScaleFM. Install it with
`pip install scalefm[carbs]`. It does not keep any files of its own. Its state is
rebuilt from the trials in the `root_directory` whenever a configuration is sampled, so
runs can be continued and parallelized like with any other optimizer.

### 3.2 Chinchilla Guided Scaling

[`chinchilla`][scalefm.optimizers.algorithms.chinchilla] follows
[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556). It
lays out a static list of configurations that grow the model and the amount of data together,
to find how a compute budget should be split between the two. The parameters marked
with `is_compute_scaling=True` make up the total scaling parameters and `data_parameter_name` names the one that sets the amount of data. Every other parameter is drawn at random once and held
fixed.

```python
def estimate_params(width: int, layers: int) -> float:
    return ...

scalefm.acquire(
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
    [`cost_estimator`](scalefm_acquire.md#scaling-studies-estimating-the-cost-of-a-configuration).
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

To design entirely new optimizers, you can define them as class with a `__call__` method outside of ScaleFM and pass them to the `scalefm.acquire()` function:

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

The class is then passed to the `scalefm.acquire()` function just like the built-in optimizers and can be configured the same way, using a dictionary:

```python
scalefm.acquire(
    evaluate_pipeline=run_function,
    pipeline_space=pipeline_space,
    root_directory="results/",
    total_evaluations_to_spend=25,
    optimizer=MyOptimizer,
)
```

For more details on how to define a custom optimizer see the [Optimizer Interface][scalefm.optimizers.optimizer.AskFunction].

To receive the `cost_estimator` and `max_cost_per_config` given to `scalefm.acquire()`, like the [scaling optimizers](#3-scaling-optimizers), inherit from [`ScalingOptimizer`][scalefm.optimizers.optimizer.ScalingOptimizer]:

```python
class MyOptimizer(ScalingOptimizer):
    def __call__(self, trials, budget_info, n=None):
        if self.cost_estimator is not None:
            cost = self.cost_estimator(**config)
        ...
```
