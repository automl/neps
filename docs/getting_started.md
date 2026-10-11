# Getting Started

Getting started with AutoScAn involves a straightforward yet powerful process, centering around its three main components.
This approach ensures flexibility and efficiency in evaluating different architecture and hyperparameter configurations
for your problem.

AutoScAn requires Python 3.11 or higher.
You can install it via `pip` or from [source](https://github.com/automl/auto-scan/).

```bash
pip install auto-scan
```

## The 3 Main Components

1. **Establish a [`pipeline_space=`](reference/autoscan_spaces.md)**:

    ```python
    class ExampleSpace(autoscan.PipelineSpace):
        # Define the parameters of your search space
        some_parameter = autoscan.Float(lower=0.0, upper=1.0)       # float
        another_parameter = autoscan.Integer(lower=0, upper=10)     # integer
        optimizer = autoscan.Categorical(choices=("sgd", "adam"))           # categorical
        epoch = autoscan.IntegerFidelity(lower=1, upper=100)
        learning_rate = autoscan.Float(lower=1e-5, upper=1, log=True)
        alpha = autoscan.Float(lower=0.1, upper=1.0, prior=0.99, prior_confidence="high")
    ```

1. **Define an `evaluate_pipeline()` function**:

    ```python
    def evaluate_pipeline(some_parameter: float,
                     another_parameter: float,
                     optimizer: str, epoch: int,
                     learning_rate: float, alpha: float) -> float:
        model = make_model(...)
        loss = eval_model(model)
        return loss
    ```

1. **Execute with [`autoscan.run()`](reference/autoscan_run.md)**:

    ```python
    autoscan.run(evaluate_pipeline, ExampleSpace(), live_plots=True)
    ```

---

## What's Next?

The [reference](reference/autoscan_run.md) section provides detailed information on the individual components of AutoScAn.

1. How to use the [**`autoscan.run()`** function](reference/autoscan_run.md) to start the optimization process.
2. The different [search space](reference/autoscan_spaces.md) options available.
3. How to choose and configure the [optimizer](reference/optimizers.md) used.
4. How to define the [`evaluate_pipeline()` function](reference/evaluate_pipeline.md).
5. How to [analyze](reference/analyse.md) the optimization runs.

!!! tip "Interactive tutorials"

    Or try AutoScAn hands-on with our tutorials:

    1. [Getting Started with HPO](https://colab.research.google.com/github/automl/auto-scan/blob/master/tutorials/1_getting_started_hpo.ipynb): the basic AutoScAn workflow, from a synthetic function to a deep learning task
    2. [Defining Search Spaces](https://colab.research.google.com/github/automl/auto-scan/blob/master/tutorials/2_search_spaces.ipynb): parameter types, fidelity parameters, priors, and `PipelineSpace` classes
    3. [Efficient Optimization](https://colab.research.google.com/github/automl/auto-scan/blob/master/tutorials/3_efficiency_techniques.ipynb): multi-fidelity optimization, expert priors, optimizer selection, and parallelization
    4. [Multi-Objective Optimization](https://colab.research.google.com/github/automl/auto-scan/blob/master/tutorials/4_multi_objective.ipynb): Multi-objective optimization with PriMO, including per-objective expert priors

Or discover the features of AutoScAn through these practical examples:

* **[Hyperparameter Optimization (HPO)](examples/basic_usage/1_hyperparameters.md)**:
Learn the essentials of hyperparameter optimization with AutoScAn.

* **[Multi-Fidelity Optimization](examples/efficiency/multi_fidelity.md)**:
Understand how to leverage multi-fidelity optimization for efficient model tuning.

* **[Utilizing Expert Priors for Hyperparameters](examples/efficiency/expert_priors_for_hyperparameters.md)**:
Learn how to incorporate expert priors for more efficient hyperparameter selection.

* **[Benefiting AutoScAn State and Optimizers with custom runtime](examples/experimental/ask_and_tell_example.md)**:
Learn how to use AskAndTell, an advanced tool for leveraging optimizers and states while enabling a custom runtime for trial execution.

- **[Integration with TensorBoard](examples/convenience/autoscan_tblogger_tutorial.md)**: Discover how to leverage AutoScAn's built-in TensorBoard support and seamlessly incorporate your own custom TensorBoard data for enhanced experiment tracking.

* **[Additional AutoScAn Examples](examples/index.md)**:
Explore more examples, including various use cases and advanced configurations in AutoScAn.
