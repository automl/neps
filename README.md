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

1. **Hyperparameter Optimization (HPO) Efficient Enough for Deep Learning:** <br />
    NePS excels in efficiently tuning hyperparameters using algorithms that enable users to make use of their prior knowledge, while also using many other efficiency boosters.
     - [PriorBand: Practical Hyperparameter Optimization in the Age of Deep Learning (NeurIPS 2023)](https://arxiv.org/abs/2306.12370)
     - [πBO: Augmenting Acquisition Functions with User Beliefs for Bayesian Optimization (ICLR 2022)](https://arxiv.org/abs/2204.11051) <br /> <br />
1. **Neural Architecture Search (NAS) with Expressive Search Spaces:** <br />
    NePS provides capabilities for optimizing DL architectures in an expressive and natural fashion.
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

1. Define a `evaluate_pipeline` function capable of evaluating different architectural and/or hyperparameter configurations
   for your problem.
2. Define a `pipeline_space` of those Parameters
3. Call `neps.run(evaluate_pipeline, pipeline_space)`

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
        log=True,   # Log spaces
        prior=1e-3, # Incorporate your knowledge to help optimization
    )
    alpha = neps.Integer(lower=1, upper=42)
    optimizer = neps.Categorical(choices=["sgd", "adam"])

# 3. Run the NePS optimization
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=ExampleSpace(),
    root_directory="path/to/save/results",  # Replace with the actual path.
    evaluations_to_spend=100,
)
```

## Examples

Discover how NePS works through these examples:

- **[Hyperparameter Optimization](neps_examples/basic_usage/hyperparameters.py)**: Learn the essentials of hyperparameter optimization with NePS.

- **[Multi-Fidelity Optimization](neps_examples/efficiency/multi_fidelity.py)**: Understand how to leverage multi-fidelity optimization for efficient model tuning.

- **[Utilizing Expert Priors for Hyperparameters](neps_examples/efficiency/expert_priors_for_hyperparameters.py)**: Learn how to incorporate expert priors for more efficient hyperparameter selection.

- **[Benefiting NePS State and Optimizers with custom runtime](neps_examples/experimental/ask_and_tell_example.py)**: Learn how to use AskAndTell, an advanced tool for leveraging optimizers and states while enabling a custom runtime for trial execution.

- **[Additional NePS Examples](neps_examples/)**: Explore more examples, including various use cases and advanced configurations in NePS.

## Contributing

Please see the [documentation for contributors](https://automl.github.io/neps/latest/dev_docs/contributing/).

## Citations

For pointers on citing the NePS package and papers refer to our [documentation on citations](https://automl.github.io/neps/latest/citations/).
