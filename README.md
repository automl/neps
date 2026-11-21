# Extrapolaw

[![PyPI version](https://img.shields.io/pypi/v/extrapolaw?color=informational)](https://pypi.org/project/extrapolaw/)
[![Python versions](https://img.shields.io/pypi/pyversions/extrapolaw)](https://pypi.org/project/extrapolaw/)
[![License](https://img.shields.io/pypi/l/extrapolaw?color=informational)](LICENSE)
[![Tests](https://github.com/automl/extrapolaw/actions/workflows/tests.yaml/badge.svg)](https://github.com/automl/extrapolaw/actions)

Extrapolaw is a tool for scaling studies of deep learning pipelines: it searches how architectural and hyperparameter choices behave as you scale models, data, and compute, and fits scaling laws to predict the best choices at larger scales.
Use it from a single GPU to a multi-node cluster or even multiple clusters.

Extrapolaw brings together [years of our algorithmic advances](https://github.com/automl/extrapolaw/blob/master/docs/citations.md) (e.g., in NeurIPS, ICML, or ICLR) with a runtime tailored to large scale models. Extrapolaw is actively maintained and used to run on many different clusters, tuning even billion-parameter scale models with many concurrent trials.

To learn about Extrapolaw, check out [the documentation](https://automl.github.io/extrapolaw/latest/) or [our examples](extrapolaw_examples/).

## Why Extrapolaw

### Scaling laws built in

- **Search on a budget, predict at scale:** scaling laws are integrated directly into the optimization loop, so that results obtained at small scale inform predictions of optimal configurations at larger scales.
- **Cost-constrained sampling:** optimizers restrict sampling to the feasible region of the search space, i.e., configurations that satisfy a user-specified black-box cost constraint.

### Tailored to large scale models

- **Tuning distributed models:** Extrapolaw works with [DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [FSDP](https://docs.pytorch.org/tutorials/intermediate/FSDP1_tutorial.html), on a single node or across multiple nodes, out of the box ([examples](extrapolaw_examples/efficiency/)).
- **Zero-effort to run many concurrent models:** start more workers on the same machine or in a multi-node setup. As long as they share the results directory, they coordinate on their own, with no server to set up.
- **Live monitoring and interventions:** follow a run with `extrapolaw.status`, live plots, or [TensorBoard](https://automl.github.io/extrapolaw/latest/reference/analyse/#visualizing-results), and steer it without starting over: add workers, extend the budget, re-run failed trials, or import tuning results from anywhere (even cross-cluster).

### Search algorithms

- **Expert knowledge and prior studies:** use your intuition as priors, and import results from earlier studies, when you have them.
- **Model-based search:** strategies such as Bayesian optimization choose promising configurations smartly instead of sampling blindly.
- **[Scaling optimizers](https://automl.github.io/extrapolaw/latest/reference/optimizers/#3-scaling-optimizers):** grounded scaling-law prescriptions, adapted to budget-aware search.

### Generally applicable

- **Any design space:** hyperparameters, architectures, resource allocation, or any component of the pipeline.
- **Any scaling dimension:** scale model size, dataset size, or any other quantity that sets the compute (`is_compute_scaling=True`).
- **Downstream tasks:** follow how downstream performances scale next to the pre-training loss.

## Installation

Extrapolaw supports Python 3.11 to 3.14. Install the latest release from PyPI:

```bash
pip install extrapolaw
```

## Basic Usage

Using `extrapolaw` is based on the following pattern:

1. Define an `evaluate_pipeline` function that evaluates a configuration of your pipeline.
1. Define a `pipeline_space` of the parameters to optimize.
1. Call `extrapolaw.acquire(evaluate_pipeline, pipeline_space)`.
1. Optionally, pass a `cost_estimator` and a cost budget to constrain the search by cost.

In code, the usage pattern can look like this:

```python
import extrapolaw
import logging

logging.basicConfig(level=logging.INFO)


# 1. Define a function that accepts hyperparameters and computes the validation error
def evaluate_pipeline(lr: float, alpha: int, optimizer: str, width: int):
    # Create your model
    model = MyModel(lr=lr, alpha=alpha, optimizer=optimizer, width=width)

    # Train and evaluate the model with your training pipeline
    validation_error = train_and_eval(model)
    # The training cost comes from `estimate_flops` below
    return {"objective_to_minimize": validation_error}


# 2. Define a search space of parameters; use the same parameter names as in evaluate_pipeline
class ExampleSpace(extrapolaw.PipelineSpace):
    lr = extrapolaw.Float(
        lower=1e-5,
        upper=1e-1,
        log=True,  # Log spaces
        log_base=10,  # Logarithm base, by default it's natural log
        prior=1e-3,  # Incorporate your knowledge to help optimization
    )
    alpha = extrapolaw.Integer(lower=1, upper=42)
    optimizer = extrapolaw.Categorical(choices=["sgd", "adam"])
    width = extrapolaw.Integer(lower=64, upper=1024, log=True, is_compute_scaling=True)


# Optional: the cost of a configuration, e.g. its FLOPs, known before it is evaluated
def estimate_flops(width: int) -> float:
    return 6 * width**2 * 1e6


# 3. Run the Extrapolaw optimization
extrapolaw.acquire(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=ExampleSpace(),
    root_directory="path/to/save/results",  # Replace with the actual path.
    cost_estimator=estimate_flops,  # Cost of every configuration
    max_cost_per_config=1e12,  # Never sample a configuration costing more than this
    total_cost_to_spend=1e14,  # Budget for the whole search
)
```

To also fit scaling laws on top of the cost constraints, use one of the [scaling optimizers](https://automl.github.io/extrapolaw/latest/reference/optimizers/#3-scaling-optimizers), e.g. `optimizer="carbs"`.

## Resources to Get Started

### Examples

- **[Hyperparameter optimization](extrapolaw_examples/basic_usage/1_hyperparameters.py):** the essentials of HPO with Extrapolaw.
- **[Scaling study of TabPFN](extrapolaw_examples/real_world/tabpfn_scan_bench.py):** a random search under a FLOP budget, with the scaling law fitted and extrapolated.
- **[Custom runtime with AskAndTell](extrapolaw_examples/experimental/ask_and_tell_example.py):** use Extrapolaw optimizers and state with your own evaluation loop.
- **[All examples](extrapolaw_examples/):** more use cases and advanced configurations.

### Documentation

- [Getting started](https://automl.github.io/extrapolaw/latest/getting_started/)
- [Reference](https://automl.github.io/extrapolaw/latest/reference/extrapolaw_acquire/): running Extrapolaw, search spaces, optimizers, and analysing runs
- [Scaling optimizers](https://automl.github.io/extrapolaw/latest/reference/optimizers/#3-scaling-optimizers): cost-constrained search guided by scaling laws
- [Algorithms](https://automl.github.io/extrapolaw/latest/reference/search_algorithms/landing_page_algo/)
- [API](https://automl.github.io/extrapolaw/latest/api/extrapolaw/api/)

## Contributing

Please see the [documentation for contributors](https://automl.github.io/extrapolaw/latest/dev_docs/contributing/).

## Citing Extrapolaw

To cite Extrapolaw or the papers behind its algorithms, see our [citation guide](https://automl.github.io/extrapolaw/latest/citations/).
