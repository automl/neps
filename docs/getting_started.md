# Getting Started

Getting started with NePS involves a straightforward yet powerful process, centering around its three main components.
This approach ensures flexibility and efficiency in evaluating different architecture and hyperparameter configurations
for your problem.

NePS requires Python 3.10 or higher.
You can install it via `pip` or from [source](https://github.com/automl/neps/).

```bash
pip install neural-pipeline-search
```

## The 3 Main Components

1. **Establish a [`pipeline_space=`](reference/neps_spaces.md)**:

```python
class ExampleSpace(neps.PipelineSpace):
    # Define the parameters of your search space
    some_parameter = neps.Float(min_value=0.0, max_value=1.0)       # float
    another_parameter = neps.Integer(min_value=0, max_value=10)     # integer
    optimizer = neps.Categorical(choices=("sgd", "adam"))           # categorical
    epoch = neps.Fidelity(neps.Integer(min_value=1, max_value=100))
    learning_rate = neps.Float(min_value=1e-5, max_value=1, log=True)
    alpha = neps.Float(min_value=0.1, max_value=1.0, prior=0.99, prior_confidence="high")
```

2. **Define an `evaluate_pipeline()` function**:

```python
def evaluate_pipeline(some_parameter: float,
                 another_parameter: float,
                 optimizer: str, epoch: int,
                 learning_rate: float, alpha: float) -> float:
    model = make_model(...)
    loss = eval_model(model)
    return loss
```

3. **Execute with [`neps.run()`](reference/neps_run.md)**:

```python
neps.run(evaluate_pipeline, ExampleSpace())
```

---

## What's Next?

The [reference](reference/neps_run.md) section provides detailed information on the individual components of NePS.

1. How to use the [**`neps.run()`** function](reference/neps_run.md) to start the optimization process.
2. The different [search space](reference/neps_spaces.md) options available.
3. How to choose and configure the [optimizer](reference/optimizers.md) used.
4. How to define the [`evaluate_pipeline()` function](reference/evaluate_pipeline.md).
5. How to [analyze](reference/analyse.md) the optimization runs.

Or discover the features of NePS through these practical examples:

* **[Hyperparameter Optimization (HPO)](examples/basic_usage/hyperparameters.md)**:
Learn the essentials of hyperparameter optimization with NePS.

* **[Multi-Fidelity Optimization](examples/efficiency/multi_fidelity.md)**:
Understand how to leverage multi-fidelity optimization for efficient model tuning.

* **[Utilizing Expert Priors for Hyperparameters](examples/efficiency/expert_priors_for_hyperparameters.md)**:
Learn how to incorporate expert priors for more efficient hyperparameter selection.

* **[Benefiting NePS State and Optimizers with custom runtime](examples/experimental/ask_and_tell_example.md)**:
Learn how to use AskAndTell, an advanced tool for leveraging optimizers and states while enabling a custom runtime for trial execution.

* **[Additional NePS Examples](examples/index.md)**:
Explore more examples, including various use cases and advanced configurations in NePS.
