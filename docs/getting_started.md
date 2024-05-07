# Getting Started

Getting started with NePS involves a straightforward yet powerful process, centering around its three main components.
This approach ensures flexibility and efficiency in evaluating different architecture and hyperparameter configurations
for your problem.

NePS requires Python 3.8 or higher. You can install it via pip or from source.

```bash
pip install neural-pipeline-search
```

## The 3 Main Components
1. **Execute with [`neps.run()`](./reference/neps_run.md)**:
Optimize your `run_pipeline=` over the `pipeline_space=` using this function.
For a thorough overview of the arguments and their explanations, check out the detailed documentation.

2. **Define a [`run_pipeline=`](./reference/run_pipeline.md) Function**:
This function is essential for evaluating different configurations.
You'll implement the specific logic for your problem within this function.
For detailed instructions on initializing and effectively using `run_pipeline=`, refer to the guide.

3. **Establish a [`pipeline_space=`](./reference/pipeline_space.md)**:
Your search space for defining parameters.
You can structure this in various formats, including dictionaries, YAML, or ConfigSpace.
The guide offers insights into defining and configuring your search space.

By following these steps and utilizing the extensive resources provided in the guides, you can tailor NePS to meet
your specific requirements, ensuring a streamlined and effective optimization process.

## Basic Usage
In code, the usage pattern can look like this:

```python
import neps
import logging

def run_pipeline( # (1)!
    hyperparameter_a: float,
    hyperparameter_b: int,
    architecture_parameter: str,
) -> dict:
    # insert here your own model
    model = MyModel(architecture_parameter)

    # insert here your training/evaluation pipeline
    validation_error, training_error = train_and_eval(
        model, hyperparameter_a, hyperparameter_b
    )

    return {
        "loss": validation_error, #! (2)
        "info_dict": {
            "training_error": training_error
            # + Other metrics
        },
    }


pipeline_space = {  # (3)!
    "hyperparameter_b":neps.IntegerParameter(1, 42, is_fidelity=True), #! (4)
    "hyperparameter_a":neps.FloatParameter(1e-3, 1e-1, log=True) #! (5)
    "architecture_parameter": neps.CategoricalParameter(["option_a", "option_b", "option_c"]),
}

if __name__ == "__main__":
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

1.  Define a function that accepts hyperparameters and computes the validation error.
2.  Return a dictionary with the objective to minimize and any additional information.
3.  Define a search space of the parameters of interest; ensure that the names are consistent with those defined in the run_pipeline function.
4.  Use `is_fidelity=True` for a multi-fidelity approach.
5.  Use `log=True` for a log-spaced hyperparameter.

!!! tip

    Please visit the [full reference](./reference/neps_run.md) for a more comprehensive walkthrough of defining budgets,
    optimizers, YAML configuration, parallelism, and more.

## Examples
Discover the features of NePS through these practical examples:

* **[Hyperparameter Optimization (HPO)](./examples/template/basic_template.md)**:
Learn the essentials of hyperparameter optimization with NePS.

* **[Architecture Search with Primitives](./examples/basic_usage/architecture.md)**:
Dive into architecture search using primitives in NePS.

* **[Multi-Fidelity Optimization](./examples/efficiency/multi_fidelity.md)**:
Understand how to leverage multi-fidelity optimization for efficient model tuning.

* **[Utilizing Expert Priors for Hyperparameters](./examples/template/priorband_template.md)**:
Learn how to incorporate expert priors for more efficient hyperparameter selection.

* **[Additional NePS Examples](./examples/index.md)**:
Explore more examples, including various use cases and advanced configurations in NePS.
