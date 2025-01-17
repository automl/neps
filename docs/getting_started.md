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
1. **Establish a [`pipeline_space=`](reference/pipeline_space.md)**:
```python
pipeline_space={
    "some_parameter": (0.0, 1.0),   # float
    "another_parameter": (0, 10),   # integer
    "optimizer": ["sgd", "adam"],   # categorical
    "epoch": neps.Integer(lower=1, upper=100, is_fidelity=True),
    "learning_rate": neps.Float(lower=1e-5, uperr=1, log=True),
    "alpha": neps.Float(lower=0.1, upper=1.0, prior=0.99, prior_confidence="high")
}

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

1. **Execute with [`neps.run()`](reference/neps_run.md)**:

```python
neps.run(evaluate_pipeline, pipeline_space)
```

---

You can find a longer walk through in the [reference](reference/neps_run.md)!

## Examples
Discover the features of NePS through these practical examples:

* **[Hyperparameter Optimization (HPO)](examples/template/basic.md)**:
Learn the essentials of hyperparameter optimization with NePS.

* **[Multi-Fidelity Optimization](examples/efficiency/multi_fidelity.md)**:
Understand how to leverage multi-fidelity optimization for efficient model tuning.

* **[Utilizing Expert Priors for Hyperparameters](examples/template/priorband.md)**:
Learn how to incorporate expert priors for more efficient hyperparameter selection.

* **[Additional NePS Examples](examples/index.md)**:
Explore more examples, including various use cases and advanced configurations in NePS.
