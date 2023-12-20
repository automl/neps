# Neural Pipeline Search (NePS)

[![PyPI version](https://img.shields.io/pypi/v/neural-pipeline-search?color=informational)](https://pypi.org/project/neural-pipeline-search/)
[![Python versions](https://img.shields.io/pypi/pyversions/neural-pipeline-search)](https://pypi.org/project/neural-pipeline-search/)
[![License](https://img.shields.io/pypi/l/neural-pipeline-search?color=informational)](LICENSE)
[![Tests](https://github.com/automl/neps/actions/workflows/tests.yaml/badge.svg)](https://github.com/automl/neps/actions)

Welcome to NePS, a powerful and flexible Python library for hyperparameter optimization (HPO) and neural architecture search (NAS) with its primary goal: enable HPO adoption in practice for deep learners!

NePS houses recently published and some more well-established algorithms that are all capable of being run massively parallel on any distributed setup, with tools to analyze runs, restart runs, etc.

Read our documentation and the following document for how to use NePS!


## Key Features

In addition to the common features offered by traditional HPO and NAS libraries, NePS stands out with the following key features:

1. [**Hyperparameter Optimization (HPO) With Prior Knowledge:**](neps_examples/template/priorband_template.py) 
    - NePS excels in efficiently tuning hyperparameters using algorithms that enable users to leverage their prior knowledge within the search space.

2. [**Neural Architecture Search (NAS) With Context-free Grammar Search Spaces:**](neps_examples/basic_usage/architecture.py) 
    - NePS is equipped to handle context-free grammar search spaces, providing advanced capabilities for designing and optimizing architectures.

3. [**Easy Parallelization:**](https://automl.github.io/neps/latest/parallelization/) 
    - NePS simplifies the parallelization of optimization tasks. Whether experiments are running on a single machine or a distributed computing environment.

4. [**Resume Runs After Termination:**](https://automl.github.io/neps/latest/parallelization/) 
    - NePS allows users to easily resume optimization runs after termination, providing a convenient and efficient workflow for long-running experiments.

5. [**Seamless User Code Integration:**](neps_examples/template/) 
    - NePS's modular design ensures flexibility and extensibility. Integrate NePS effortlessly into existing machine learning workflows.

## Getting Started

### 1. Installation

Using pip:

```bash
pip install neural-pipeline-search
```

> Note: As indicated with the `v0.x.x` version number, NePS is early stage code and APIs might change in the future.

### 2. Basic Usage

Using `neps` always follows the same pattern:

1. Define a `run_pipeline` function that evaluates architectures/hyperparameters for your problem
1. Define a search space `pipeline_space` of architectures/hyperparameters
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
    validation_error, test_error = train_and_eval(
        model, hyperparameter_a, hyperparameter_b
    )

    return {  # dict or float(validation error)
        "loss": validation_error,
        "info_dict": {
            "test_error": test_error
            # + Other metrics
        },
    }


# 2. Define a search space of hyperparameters; use the same names as in run_pipeline
pipeline_space = dict(
    hyperparameter_b=neps.IntegerParameter(
        lower=1, upper=42, is_fidelity=True
    ),  # Mark 'is_fidelity' as true for a multi-fidelity approach.
    hyperparameter_a=neps.FloatParameter(
        lower=0.001, upper=0.1, log=True
    ),  # If True, the search space is sampled in log space.
    architecture_parameter=neps.CategoricalParameter(
        ["option_a", "option_b", "option_c"]
    ),
)

if __name__ == "__main__":
    # 3. Run the NePS optimization
    logging.basicConfig(level=logging.INFO)
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="path/to/save/results",  # Replace with the actual path.
        max_evaluations_total=100,
        searcher="hyperband"  # Optional specifies the search strategy,
        # otherwise NePs decides based on your data.
    )
```

## Documentation

For more details and features please have a look at our [documentation](https://automl.github.io/neps/latest/) and [examples](neps_examples)

## Analysing runs

See our [documentation on analysing runs](https://automl.github.io/neps/latest/analyse).

## Alternatives

NePS does not cover your use-case? Have a look at [some alternatives](https://automl.github.io/neps/latest/alternatives).

## Contributing

Please see the [documentation for contributors](https://automl.github.io/neps/latest/contributing/).

## Citations

Please consider citing us if you use our tool!

Refer to our [documentation on citations](docs/citations.md).
