# Neural Pipeline Search

Neural Pipeline Search helps deep learning experts find the best neural pipeline.

Features:

- Hyperparameter optimization (HPO)
- Neural architecture search (NAS), cell-based and hierarchical
- HPO and NAS
- Asynchronous parallelization and distribution
- Fault tolerance

Soon-to-come Features:

- Multi-fidelity
- Cost-aware
- Across code version transfer
- Multi-objective

![Python versions](https://img.shields.io/badge/python-3.7-informational)
[![License](https://img.shields.io/badge/license-MIT-informational)](LICENSE)
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

1. Define a `run_pipeline` function that maps parameters (hyperparameters and/or architectures) to a loss.
1. Define a `pipeline_space` dictionary of parameter spaces
1. Call `neps.run` on `run_pipeline` and `pipeline_space`

In code the usage patterns looks like this:

```python
import neps

# 1. Define a run_pipeline function that maps parameters to a loss.
def run_pipeline(x):
    return {"loss": x}


# 2. Define a pipeline_space dictionary of parameter spaces
pipeline_space = dict(
    x=neps.FloatParameter(lower=0, upper=1, log=False),
)

# 3. Call neps.run on run_pipeline and pipeline_space
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    working_directory="results/usage",
    max_evaluations_total=5,
)
```

## Advanced Usage

Please see our examples in [neps_examples](neps_examples).

## Contributing

Please see our guidelines and guides for contributors at [CONTRIBUTING.md](CONTRIBUTING.md).
