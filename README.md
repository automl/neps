# Neural Pipeline Search (NePS)

[![PyPI version](https://img.shields.io/pypi/v/neural-pipeline-search?color=informational)](https://pypi.org/project/neural-pipeline-search/)
[![Python versions](https://img.shields.io/pypi/pyversions/neural-pipeline-search)](https://pypi.org/project/neural-pipeline-search/)
[![License](https://img.shields.io/pypi/l/neural-pipeline-search?color=informational)](LICENSE)
[![Tests](https://github.com/automl/neps/actions/workflows/tests.yaml/badge.svg)](https://github.com/automl/neps/actions)

NePS helps deep learning experts to optimize the hyperparameters and/or architecture of their deep learning pipeline with:

- Hyperparameter Optimization (HPO) ([example](neps_examples/basic_usage/hyperparameters.py))
- Neural Architecture Search (NAS) ([example](neps_examples/basic_usage/architecture.py), [paper](https://openreview.net/forum?id=Ok58hMNXIQ))
- Joint Architecture and Hyperparameter Search (JAHS) ([example](neps_examples/basic_usage/architecture_and_hyperparameters.py), [paper](https://openreview.net/forum?id=_HLcjaVlqJ))

For efficiency and convenience NePS allows you to

- Add your intuition as priors for the search ([example HPO](neps_examples/efficiency/expert_priors_for_hyperparameters.py), [example JAHS](neps_examples/experimental/expert_priors_for_architecture_and_hyperparameters.py), [paper](https://openreview.net/forum?id=MMAeCXIa89))
- Utilize low fidelity (e.g., low epoch) evaluations to focus on promising configurations ([example](neps_examples/efficiency/multi_fidelity.py), [paper](https://openreview.net/forum?id=ds21dwfBBH))
- Trivially parallelize across machines ([example](neps_examples/efficiency/parallelization.md), [documentation](https://automl.github.io/neps/latest/parallelization/))

Or [all of the above](neps_examples/efficiency/multi_fidelity_and_expert_priors.py) for maximum efficiency!

### Note

As indicated with the `v0.x.x` version number, NePS is early stage code and APIs might change in the future.

## Getting Started

### 1. Installation

Using pip:

```bash
pip install neural-pipeline-search
```

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
    validation_error, test_error = train_and_eval(model, hyperparameter_a, hyperparameter_b)

    return {
        "loss": validation_error,
        "info_dict": {
            "test_error": test_error
            # + Other metrics
        }
    }

# 2. Define a search space of hyperparameters; use the same names as in run_pipeline
pipeline_space = dict(
    hyperparameter_b=neps.IntegerParameter(
        lower=1,
        upper=100,
        is_fidelity=True), # Mark 'is_fidelity' as true for a multi-fidelity approach.
    hyperparameter_a=neps.FloatParameter(
        lower=0.0,
        upper=1.0,
        log=True), # If True, the search space is sampled in log space.
    architecture_parameter=neps.CategoricalParameter(["option_a", "option_b", "option_c"]),
)

if __name__=="__main__":
    # 3. Run the NePS optimization
    logging.basicConfig(level=logging.INFO)
    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space,
        root_directory="path/to/save/results", # Replace with the actual path.
        max_evaluations_total=100,
        searcher="hyperband" # Optional specifies the search strategy,
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
