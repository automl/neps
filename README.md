# Neural Pipeline Search (NePS)

[![PyPI version](https://img.shields.io/pypi/v/neural-pipeline-search?color=informational)](https://pypi.org/project/neural-pipeline-search/)
[![Python versions](https://img.shields.io/pypi/pyversions/neural-pipeline-search)](https://pypi.org/project/neural-pipeline-search/)
[![License](https://img.shields.io/pypi/l/neural-pipeline-search?color=informational)](LICENSE)
[![Tests](https://github.com/automl/neps/actions/workflows/tests.yaml/badge.svg)](https://github.com/automl/neps/actions)

Welcome to NePS, a powerful and flexible Python library for hyperparameter optimization (HPO) and neural architecture search (NAS) with its primary goal: enable HPO and NAS for deep learners!

NePS houses recently published and also well-established algorithms that can all be run massively parallel on distributed setups, with tools to analyze runs, restart runs, etc., all tailored to the needs of deep learning experts.

Take a look at our [documentation](https://automl.github.io/neps/latest/) for all the details on how to use NePS!

## Key Features

In addition to the features offered by traditional HPO and NAS libraries, NePS, e.g., stands out with:

1. [**Hyperparameter Optimization (HPO) With Prior Knowledge:**](neps_examples/template/priorband_template.py)

   - NePS excels in efficiently tuning hyperparameters using algorithms that enable users to make use of their prior knowledge within the search space. This is leveraged by the insights presented in:
     - [PriorBand: Practical Hyperparameter Optimization in the Age of Deep Learning](https://arxiv.org/abs/2306.12370)
     - [πBO: Augmenting Acquisition Functions with User Beliefs for Bayesian Optimization](https://arxiv.org/abs/2204.11051)

1. [**Neural Architecture Search (NAS) With Context-free Grammar Search Spaces:**](neps_examples/basic_usage/architecture.py)

   - NePS is equipped to handle context-free grammar search spaces, providing advanced capabilities for designing and optimizing architectures. this is leveraged by the insights presented in:
     - [Construction of Hierarchical Neural Architecture Search Spaces based on Context-free Grammars](https://arxiv.org/abs/2211.01842)

1. [**Easy Parallelization and Resumption of Runs:**](https://automl.github.io/neps/latest/examples/efficiency/)

   - NePS simplifies the process of parallelizing optimization tasks both on individual computers and in distributed
     computing environments. It also allows users to conveniently resume these optimization tasks after completion to
     ensure a seamless and efficient workflow for long-running experiments.

1. [**Seamless User Code Integration:**](neps_examples/template/)

   - NePS's modular design ensures flexibility and extensibility. Integrate NePS effortlessly into existing machine learning workflows.

## Installation

To install the latest release from PyPI run

```bash
pip install neural-pipeline-search
```

To get the latest version from github run

```bash
pip install git+https://github.com/automl/neps.git
```

> Note: As indicated with the `v0.x.x` version number APIs will change in the future.


## Basic Usage

Using `neps` always follows the same pattern:

1. Define a `run_pipeline` function capable of evaluating different architectural and/or hyperparameter configurations
   for your problem.
1. Define a search space named `pipeline_space` of those Parameters e.g. via a dictionary
1. Call `neps.run` to optimize `run_pipeline` over `pipeline_space`

In code, the usage pattern can look like this:

```python
import neps
import logging


# 1. Define a function that accepts hyperparameters and computes the validation error
def run_pipeline(
    hyperparameter_a: float, hyperparameter_b: int, architecture_parameter: str
) -> dict:
    # Create your model
    model = MyModel(architecture_parameter)

    # Train and evaluate the model with your training pipeline
    validation_error, training_error = train_and_eval(
        model, hyperparameter_a, hyperparameter_b
    )

    return {  # dict or float(validation error)
        "loss": validation_error,
        "info_dict": {
            "training_error": training_error
            # + Other metrics
        },
    }


# 2. Define a search space of parameters; use the same parameter names as in run_pipeline
pipeline_space = dict(
    hyperparameter_a=neps.FloatParameter(
        lower=0.001, upper=0.1, log=True  # The search space is sampled in log space
    ),
    hyperparameter_b=neps.IntegerParameter(lower=1, upper=42),
    architecture_parameter=neps.CategoricalParameter(["option_a", "option_b"]),
)


# 3. Run the NePS optimization
logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="path/to/save/results",  # Replace with the actual path.
    max_evaluations_total=100,
)
```

## Examples

Discover how NePS works through these practical examples:

- **[Hyperparameter Optimization (HPO)](neps_examples/basic_usage/hyperparameters.py)**: Learn the essentials of hyperparameter optimization with NePS.

- **[Architecture Search with Primitives](neps_examples/basic_usage/architecture.py)**: Dive into architecture search using primitives in NePS.

- **[Multi-Fidelity Optimization](neps_examples/efficiency/multi_fidelity.py)**: Understand how to leverage multi-fidelity optimization for efficient model tuning.

- **[Utilizing Expert Priors for Hyperparameters](neps_examples/efficiency/expert_priors_for_hyperparameters.py)**: Learn how to incorporate expert priors for more efficient hyperparameter selection.

- **[Additional NePS Examples](neps_examples/)**: Explore more examples, including various use cases and advanced configurations in NePS.

## Contributing

Please see the [documentation for contributors](https://automl.github.io/neps/latest/dev_docs/contributing/).

## Citations

For pointers on citing the NePS package and papers refer to our [documentation on citations](https://automl.github.io/neps/latest/citations/).
