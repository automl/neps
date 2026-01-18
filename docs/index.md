# Neural Pipeline Search (NePS)

[![PyPI version](https://img.shields.io/pypi/v/neural-pipeline-search?color=informational)](https://pypi.org/project/neural-pipeline-search/)
[![Python versions](https://img.shields.io/pypi/pyversions/neural-pipeline-search)](https://pypi.org/project/neural-pipeline-search/)
[![License](https://img.shields.io/pypi/l/neural-pipeline-search?color=informational)](https://github.com/automl/neps/blob/master/LICENSE)
[![Tests](https://github.com/automl/neps/actions/workflows/tests.yaml/badge.svg)](https://github.com/automl/neps/actions)

Welcome to NePS, a powerful and flexible Python library for hyperparameter optimization (HPO) and neural architecture search (NAS) with its primary goal: **make HPO and NAS usable for deep learners in practice**.

NePS houses recently published and also well-established algorithms that can all be run massively parallel on distributed setups, with tools to analyze runs, restart runs, etc., all **tailored to the needs of deep learning experts**.

## Key Features

In addition to the features offered by traditional HPO and NAS libraries, NePS stands out with:

1. **Hyperparameter Optimization (HPO) Efficient Enough For Deep Learning:** <br />
    NePS excels in efficiently tuning hyperparameters using algorithms that enable users to make use of their prior knowledge, while also using many other efficiency boosters.
     - [PriorBand: Practical Hyperparameter Optimization in the Age of Deep Learning (NeurIPS 2023)](https://arxiv.org/abs/2306.12370)
     - [πBO: Augmenting Acquisition Functions with User Beliefs for Bayesian Optimization (ICLR 2022)](https://arxiv.org/abs/2204.11051) <br /> <br />
2. **Neural Architecture Search (NAS) with Expressive Search Spaces:** <br />
    NePS provides capabilities for designing and optimizing architectures in an expressive and natural fashion.
     - [Construction of Hierarchical Neural Architecture Search Spaces based on Context-free Grammars (NeurIPS 2023)](https://arxiv.org/abs/2211.01842) <br /> <br />
3. **Zero-effort Parallelization and an Experience Tailored to DL:** <br />
     NePS simplifies the process of parallelizing optimization tasks both on individual computers and in distributed
     computing environments. As NePS is made for deep learners, all technical choices are made with DL in mind and common
     DL tools such as Tensorboard are [embraced](https://automl.github.io/neps/latest/reference/analyse/#visualizing-results).

!!! tip

    Check out:

    * [Reference documentation](./reference/neps_run.md) for a quick overview.
    * [API](api/neps/api.md) for a more detailed reference.
    * [Colab Tutorial](https://colab.research.google.com/drive/11IOhkmMKsIUhWbHyMYzT0v786O9TPWlH?usp=sharing) walking through NePS's main features.
    * [Examples](examples/index.md) for basic code snippets to get started.

## Installation

To install the latest release from PyPI run

```bash
pip install neural-pipeline-search
```

## Basic Usage

Using `neps` always follows the same pattern:

1. Define a `evaluate_pipeline` function capable of evaluating different architectural and/or hyperparameter configurations
   for your problem.
2. Define a search space named `pipeline_space` of those Parameters e.g. via a dictionary
3. Call `neps.run` to optimize `evaluate_pipeline` over `pipeline_space`

In code, the usage pattern can look like this:

```python
import neps
import logging


# 1. Define a function that accepts hyperparameters and computes the validation error
def evaluate_pipeline(hyperparameter_a: float, hyperparameter_b: int, architecture_parameter: str):
    # Create your model
    model = MyModel(architecture_parameter)

    # Train and evaluate the model with your training pipeline
    validation_error = train_and_eval(model, hyperparameter_a, hyperparameter_b)
    return validation_error


# 2. Define a search space of parameters; use the same parameter names as in evaluate_pipeline
class ExampleSpace(neps.PipelineSpace):
    hyperparameter_a = neps.Float(lower=0.001, upper=0.1, log=True)  # Log scale parameter
    hyperparameter_b = neps.Integer(lower=1, upper=42)
    architecture_parameter = neps.Categorical(choices=("option_a", "option_b"))

# 3. Run the NePS optimization
logging.basicConfig(level=logging.INFO)
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=ExampleSpace(),
    root_directory="path/to/save/results",  # Replace with the actual path.
    evaluations_to_spend=100,
)

# 4. status information about a neural pipeline search run, using:
# python -m neps.status path/to/save/results
```

## Examples

Discover how NePS works through these examples:

- **[Hyperparameter Optimization](examples/basic_usage/1_hyperparameters.md)**: Learn the essentials of hyperparameter optimization with NePS.

- **[Multi-Fidelity Optimization](examples/efficiency/multi_fidelity.md)**: Understand how to leverage multi-fidelity optimization for efficient model tuning.

- **[Utilizing Expert Priors for Hyperparameters](examples/efficiency/expert_priors_for_hyperparameters.md)**: Learn how to incorporate expert priors for more efficient hyperparameter selection.

- **[Benefiting NePS State and Optimizers with custom runtime](examples/experimental/ask_and_tell_example.md)**: Learn how to use AskAndTell, an advanced tool for leveraging optimizers and states while enabling a custom runtime for trial execution.

- **[Integration with TensorBoard](examples/convenience/neps_tblogger_tutorial.md)**: Discover how to leverage NePS's built-in TensorBoard support and seamlessly incorporate your own custom TensorBoard data for enhanced experiment tracking.

- **[Additional NePS Examples](examples/index.md)**: Explore more examples, including various use cases and advanced configurations in NePS.

## Contributing

Please see the [documentation for contributors](dev_docs/contributing.md).

## Citations

For pointers on citing the NePS package and papers refer to our [documentation on citations](citations.md).
