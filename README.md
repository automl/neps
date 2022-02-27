# Neural Pipeline Search

Neural Pipeline Search helps deep learning experts find the best neural pipeline.

Features:

- Hyperparameter optimization (HPO)
- Neural architecture search (NAS): cell-based and hierarchical
- Joint NAS and HPO
- Expert priors to guide the search
- Asynchronous parallelization and distribution
- Fault tolerance for crashes and job time limits

Soon-to-come Features:

- Multi-fidelity
- Cost-aware
- Across code version transfer
- Python 3.8+ support
- Multi-objective

![Python versions](https://img.shields.io/badge/python-3.7-informational)
[![License](https://img.shields.io/badge/license-Apache%202.0-informational)](LICENSE)
[![Tests](https://github.com/automl/neps/actions/workflows/tests.yaml/badge.svg)](https://github.com/automl/neps/actions)

## Installation

Using pip

```bash
pip install git+https://github.com/automl/neps.git
```

To install specific versions of torch (e.g., cuda enabled versions) you might want to use our utility

```bash
python -m neps.utils.install_torch
```

## Usage

Using `neps` always follows the same pattern:

1. Define a `run_pipeline` function that evaluates architectures/hyperparameters for your problem
1. Define a search space `pipeline_space` of architectures/hyperparameters
1. Call `neps.run` to optimize `run_pipeline` over `pipeline_space`

In code the usage pattern can look like this:

```python
import neps

# 1. Define a function that accepts a hyperparameter and computes the validation error
def run_pipeline(some_hyperparameter: float):
    validation_error = -some_hyperparameter
    return validation_error


# 2. Define a search space containing one hyperparameter
pipeline_space = dict(
    some_hyperparameter=neps.FloatParameter(lower=0, upper=1),
)

# 3. Call neps.run to optimize run_pipeline over pipeline_space
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    working_directory="usage_example",
    max_evaluations_total=5,
)
```

For more advanced usage examples have a look at [neps_examples](neps_examples).

### Status information

To show status information about a neural pipeline search use

```bash
python -m neps.status WORKING_DIRECTORY
```

If you need more status information than is printed per default, please have a look at

```bash
python -m neps.status --help
```

### Parallelization

In order to run a neural pipeline search with multiple processes or multiple machines, simply call `neps.run` multiple times.
All calls to `neps.run` need to use the same `working_directory` on the same filesystem, otherwise there is no synchronization between the `neps.run`'s.

## Contributing

Please see our guidelines and guides for contributors at [CONTRIBUTING.md](CONTRIBUTING.md).
