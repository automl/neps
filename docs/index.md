# Neural Pipeline Search (NePS)

[![PyPI version](https://img.shields.io/pypi/v/neural-pipeline-search?color=informational)](https://pypi.org/project/neural-pipeline-search/)
[![Python versions](https://img.shields.io/pypi/pyversions/neural-pipeline-search)](https://pypi.org/project/neural-pipeline-search/)
[![License](https://img.shields.io/pypi/l/neural-pipeline-search?color=informational)](https://github.com/automl/neps/blob/master/LICENSE)
[![Tests](https://github.com/automl/neps/actions/workflows/tests.yaml/badge.svg)](https://github.com/automl/neps/actions)

NePS is a tool for tuning the design choices of deep learning pipelines efficiently and across scales.
Use it for hyperparameter optimization (HPO), neural architecture search (NAS), or any other design choice in your pipeline, from a single GPU to a multi-node cluster or even multiple clusters.

NePS brings together [years of our algorithmic advances](citations.md) (e.g., in NeurIPS, ICML, or ICLR) with a runtime tailored to large scale models. NePS is actively maintained and used to run on many different clusters, tuning even billion-parameter scale models with many concurrent trials.

To learn about NePS, check out [the documentation](getting_started.md), [our examples](examples/index.md), or our [Colab tutorials](#tutorials).

## Why NePS

### Tailored to large scales models

- **Tuning distributed models:** NePS works with [DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [FSDP](https://docs.pytorch.org/tutorials/intermediate/FSDP1_tutorial.html), on a single node or across multiple nodes, out of the box ([examples](examples/efficiency/index.md)).
- **Zero-effort to run many concurrent models:** start more workers on the same machine or in a multi-node setup. As long as they share the results directory, they coordinate on their own, with no server to set up.
- **Live monitoring and interventions:** follow a run with `neps.status`, live plots, or [TensorBoard](reference/analyse.md#visualizing-results), and steer it without starting over: add workers, extend the budget, re-run failed trials, or import tuning results from anywhere (even cross-cluster).

### Efficient tuning algorithms

- **Low-fidelity evaluations:** principled use of cheap evaluations, such as fewer epochs or less data, to rule out bad configurations early.
- **Expert knowledge and prior studies:** use your intuition as priors, and results from earlier studies, when you have them.
- **Model-based search:** strategies such as Bayesian optimization choose promising configurations smartly instead of sampling blindly.

### Generally applicable

- **Any design space:** hyperparameters, architectures, resource allocation, or any component of the pipeline.
- **Any scaling dimension:** use epochs, dataset size, model size, or any other quantity as the fidelity.
- **Any and multiple objective:** optimize pre-training loss, downstream tasks, resource usage, or several of them at once.

## Installation

NePS supports Python 3.11 to 3.14. Install the latest release from PyPI:

```bash
pip install neural-pipeline-search
```

## Basic Usage

Using `neps` is based on the following pattern:

1. Define an `evaluate_pipeline` function that evaluates a configuration of your pipeline.
1. Define a `pipeline_space` of the parameters to optimize.
1. Call `neps.run(evaluate_pipeline, pipeline_space)`.

In code, the usage pattern can look like this:

```python
import neps
import logging

logging.basicConfig(level=logging.INFO)


# 1. Define a function that accepts hyperparameters and computes the validation error
def evaluate_pipeline(lr: float, alpha: int, optimizer: str):
    # Create your model
    model = MyModel(lr=lr, alpha=alpha, optimizer=optimizer)

    # Train and evaluate the model with your training pipeline
    validation_error = train_and_eval(model)
    return validation_error


# 2. Define a search space of parameters; use the same parameter names as in evaluate_pipeline
class ExampleSpace(neps.PipelineSpace):
    lr = neps.Float(
        lower=1e-5,
        upper=1e-1,
        log=True,  # Log spaces
        log_base=10,  # Logarithm base, by default it's natural log
        prior=1e-3,  # Incorporate your knowledge to help optimization
    )
    alpha = neps.Integer(lower=1, upper=42)
    optimizer = neps.Categorical(choices=["sgd", "adam"])


# 3. Run the NePS optimization
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=ExampleSpace(),
    root_directory="path/to/save/results",  # Replace with the actual path.
    total_evaluations_to_spend=100,
)
```

## Resources to Get Started

### Tutorials

Interactive notebooks that run in Google Colab:

| Tutorial | What it covers | Run |
|----------|----------------|-----|
| **1. Getting Started with HPO** | Basic HPO workflow, synthetic functions, and deep learning tasks | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/automl/neps/blob/master/tutorials/1_getting_started_hpo.ipynb) |
| **2. Defining Search Spaces** | Parameter types, fidelity parameters, and `PipelineSpace` classes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/automl/neps/blob/master/tutorials/2_search_spaces.ipynb) |
| **3. Efficient Optimization** | Multi-fidelity optimization, expert priors, optimizer selection, and parallelization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/automl/neps/blob/master/tutorials/3_efficiency_techniques.ipynb) |
| **4. Multi-Objective Optimization** | Multi-objective optimization with PriMO, including per-objective expert priors | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/automl/neps/blob/master/tutorials/4_multi_objective.ipynb) |

To run them locally instead, see the [tutorials folder](https://github.com/automl/neps/tree/master/tutorials).

### Examples

- **[Hyperparameter optimization](examples/basic_usage/1_hyperparameters.md):** the essentials of HPO with NePS.
- **[Multi-fidelity optimization](examples/efficiency/multi_fidelity.md):** speed up tuning with cheap, low-fidelity evaluations.
- **[Multi-objective optimization](examples/efficiency/multi_objective.md):** optimize competing objectives with PriMO, using expert priors and multi-fidelity.
- **[Expert priors](examples/efficiency/expert_priors_for_hyperparameters.md):** use what you already know to focus the search.
- **[Custom runtime with AskAndTell](examples/experimental/ask_and_tell_example.md):** use NePS optimizers and state with your own evaluation loop.
- **[All examples](examples/index.md):** more use cases and advanced configurations.

### Documentation

- [Getting started](getting_started.md)
- [Reference](reference/neps_run.md): running NePS, search spaces, optimizers, and analysing runs
- [Algorithms](reference/search_algorithms/landing_page_algo.md)
- [API](api/neps/api.md)

## Contributing

Please see the [documentation for contributors](dev_docs/contributing.md).

## Citing NePS

To cite NePS or the papers behind its algorithms, see our [citation guide](citations.md).
