# Getting Started

Getting started with Extrapolaw involves a straightforward yet powerful process, centering around its three main components.
This approach ensures flexibility and efficiency in evaluating different architecture and hyperparameter configurations
for your problem.

Extrapolaw requires Python 3.11 or higher.
You can install it via `pip` or from [source](https://github.com/automl/extrapolaw/).

```bash
pip install extrapolaw
```

## The 3 Main Components

1. **Establish a [`pipeline_space=`](reference/extrapolaw_spaces.md)**:

    ```python
    class ExampleSpace(extrapolaw.PipelineSpace):
        # Define the parameters of your search space
        some_parameter = extrapolaw.Float(lower=0.0, upper=1.0)       # float
        another_parameter = extrapolaw.Integer(lower=0, upper=10)     # integer
        optimizer = extrapolaw.Categorical(choices=("sgd", "adam"))           # categorical
        epoch = extrapolaw.Integer(lower=1, upper=100)
        learning_rate = extrapolaw.Float(lower=1e-5, upper=1, log=True)
        alpha = extrapolaw.Float(lower=0.1, upper=1.0)
    ```

1. **Define an `evaluate_pipeline()` function**:

    ```python
    def evaluate_pipeline(some_parameter: float,
                    another_parameter: float,
                    optimizer: str, epoch: int,
                    learning_rate: float, alpha: float) -> dict:
        model = make_model(...)
        loss = eval_model(model)
        return {"objective_to_minimize": loss, "cost": training_flops(model)}
    ```

1. **Execute with [`extrapolaw.acquire()`](reference/extrapolaw_acquire.md)**:

    ```python
    extrapolaw.acquire(evaluate_pipeline, ExampleSpace(), live_plots=True)
    ```

---

## What's Next?

The [reference](reference/extrapolaw_acquire.md) section provides detailed information on the individual components of Extrapolaw.

1. How to use the [**`extrapolaw.acquire()`** function](reference/extrapolaw_acquire.md) to start the optimization process.
2. The different [search space](reference/extrapolaw_spaces.md) options available.
3. How to choose and configure the [optimizer](reference/optimizers.md) used.
4. How to define the [`evaluate_pipeline()` function](reference/evaluate_pipeline.md).
5. How to [analyze](reference/analyse.md) the optimization runs.

Discover the features of Extrapolaw through these practical examples:

* **[Hyperparameter Optimization (HPO)](examples/basic_usage/1_hyperparameters.md)**:
Learn the essentials of hyperparameter optimization with Extrapolaw.

* **[Scaling Study of TabPFN](examples/real_world/tabpfn_scan_bench.md)**:
Learn how to run a scaling study under a FLOP budget, and fit and extrapolate its scaling law.

* **[Benefiting Extrapolaw State and Optimizers with custom runtime](examples/experimental/ask_and_tell_example.md)**:
Learn how to use AskAndTell, an advanced tool for leveraging optimizers and states while enabling a custom runtime for trial execution.

- **[Integration with TensorBoard](examples/convenience/extrapolaw_tblogger_tutorial.md)**: Discover how to leverage Extrapolaw's built-in TensorBoard support and seamlessly incorporate your own custom TensorBoard data for enhanced experiment tracking.

* **[Additional Extrapolaw Examples](examples/index.md)**:
Explore more examples, including various use cases and advanced configurations in Extrapolaw.
