# Neural Pipeline Search (NePS)

[![PyPI version](https://img.shields.io/pypi/v/neural-pipeline-search?color=informational)](https://pypi.org/project/neural-pipeline-search/)
[![Python versions](https://img.shields.io/pypi/pyversions/neural-pipeline-search)](https://pypi.org/project/neural-pipeline-search/)
[![License](https://img.shields.io/pypi/l/neural-pipeline-search?color=informational)](LICENSE)
[![Tests](https://github.com/automl/neps/actions/workflows/tests.yaml/badge.svg)](https://github.com/automl/neps/actions)

Welcome to NePS, a powerful and flexible Python library for hyperparameter optimization (HPO) and neural architecture search (NAS) that **makes HPO and NAS practical for deep learners**.

NePS houses recently published and also well-established algorithms that can all be run massively parallel on distributed setups and, in general, NePS is tailored to the needs of deep learning experts.

To learn about NePS, check-out [the documentation](https://automl.github.io/neps/latest/), [our examples](neps_examples/), or a [colab tutorial](https://colab.research.google.com/drive/11IOhkmMKsIUhWbHyMYzT0v786O9TPWlH?usp=sharing).

## Key Features

In addition to the features offered by traditional HPO and NAS libraries, NePS stands out with:

1. **Hyperparameter Optimization (HPO) Efficient Enough For Deep Learning:** <br />
    NePS excels in efficiently tuning hyperparameters using algorithms that enable users to make use of their prior knowledge, while also using many other efficiency boosters.
     - [PriorBand: Practical Hyperparameter Optimization in the Age of Deep Learning (NeurIPS 2023)](https://arxiv.org/abs/2306.12370)
     - [πBO: Augmenting Acquisition Functions with User Beliefs for Bayesian Optimization (ICLR 2022)](https://arxiv.org/abs/2204.11051) <br /> <br />
1. **Neural Architecture Search (NAS) with Expressive Search Spaces:** <br />
    NePS provides capabilities for designing and optimizing architectures in an expressive and natural fashion.
     - [Construction of Hierarchical Neural Architecture Search Spaces based on Context-free Grammars (NeurIPS 2023)](https://arxiv.org/abs/2211.01842) <br /> <br />
1. **Zero-effort Parallelization and an Experience Tailored to DL:** <br />
     NePS simplifies the process of parallelizing optimization tasks both on individual computers and in distributed
     computing environments. As NePS is made for deep learners, all technical choices are made with DL in mind and common
     DL tools such as Tensorboard are [embraced](https://automl.github.io/neps/latest/reference/analyse/#visualizing-results).

## Installation

To install the latest release from PyPI run

```bash
pip install neural-pipeline-search
```

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
    validation_error = train_and_eval(
        model, hyperparameter_a, hyperparameter_b
    )
    return validation_error


# 2. Define a search space of parameters; use the same parameter names as in run_pipeline
pipeline_space = dict(
    hyperparameter_a=neps.Float(
        lower=0.001, upper=0.1, log=True  # The search space is sampled in log space
    ),
    hyperparameter_b=neps.Integer(lower=1, upper=42),
    architecture_parameter=neps.Categorical(["option_a", "option_b"]),
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

Discover how NePS works through these examples:

- **[Hyperparameter Optimization](neps_examples/basic_usage/hyperparameters.py)**: Learn the essentials of hyperparameter optimization with NePS.

- **[Multi-Fidelity Optimization](neps_examples/efficiency/multi_fidelity.py)**: Understand how to leverage multi-fidelity optimization for efficient model tuning.

- **[Utilizing Expert Priors for Hyperparameters](neps_examples/efficiency/expert_priors_for_hyperparameters.py)**: Learn how to incorporate expert priors for more efficient hyperparameter selection.

- **[Architecture Search](neps_examples/basic_usage/architecture.py)**: Dive into (hierarchical) architecture search in NePS.

- **[Additional NePS Examples](neps_examples/)**: Explore more examples, including various use cases and advanced configurations in NePS.

## Contributing

Please see the [documentation for contributors](https://automl.github.io/neps/latest/dev_docs/contributing/).

## Citations

For pointers on citing the NePS package and papers refer to our [documentation on citations](https://automl.github.io/neps/latest/citations/).
