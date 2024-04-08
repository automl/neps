# Getting Started

Getting started with NePS involves a straightforward yet powerful process, centering around its three main components.
This approach ensures flexibility and efficiency in evaluating different architecture and hyperparameter configurations
for your problem.

## The 3 Main Components

1. **Define a [`run_pipeline`](https://automl.github.io/neps/latest/run_pipeline) Function**: This function is essential
   for evaluating different configurations. You'll implement the specific logic for your problem within this function.
   For detailed instructions on initializing and effectively using `run_pipeline`, refer to the guide.

1. **Establish a [`pipeline_space`](https://automl.github.io/neps/latest/pipeline_space)**: Your search space for
   defining parameters. You can structure this in various formats, including dictionaries, YAML, or ConfigSpace.
   The guide offers insights into defining and configuring your search space.

1. **Execute with [`neps.run`](https://automl.github.io/neps/latest/neps_run)**: Optimize your `run_pipeline` over
   the `pipeline_space` using this function. For a thorough overview of the arguments and their explanations,
   check out the detailed documentation.

By following these steps and utilizing the extensive resources provided in the guides, you can tailor NePS to meet
your specific requirements, ensuring a streamlined and effective optimization process.

## Basic Usage

In code, the usage pattern can look like this:

```python
import neps
import logging


# 1. Define a function that accepts hyperparameters and computes the validation error
def run_pipeline(
    hyperparameter_a: float, hyperparameter_b: int, architecture_parameter: str
) -> dict:
    # insert here your own model
    model = MyModel(architecture_parameter)

    # insert here your training/evaluation pipeline
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


# 2. Define a search space of the parameters of interest; ensure that the names are consistent with those defined
# in the run_pipeline function
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

## Examples

Discover the features of NePS through these practical examples:

- **[Hyperparameter Optimization (HPO)](https://github.com/automl/neps/blob/master/neps_examples/template/basic_template.py)**: Learn the essentials of
  hyperparameter optimization with NePS.

- **[Architecture Search with Primitives](https://github.com/automl/neps/tree/master/neps_examples/basic_usage/architecture.py)**: Dive into architecture search
  using primitives in NePS.

- **[Multi-Fidelity Optimization](https://github.com/automl/neps/tree/master/neps_examples/efficiency/multi_fidelity.py)**: Understand how to leverage
  multi-fidelity optimization for efficient model tuning.

- **[Utilizing Expert Priors for Hyperparameters](https://github.com/automl/neps/blob/master/neps_examples/template/priorband_template.py)**:
  Learn how to incorporate expert priors for more efficient hyperparameter selection.

- **[Additional NePS Examples](https://github.com/automl/neps/tree/master/neps_examples/)**: Explore more examples, including various use cases and
  advanced configurations in NePS.
