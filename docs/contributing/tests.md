# Checks and Tests

We have setup checks and tests at several points in the development flow:

- At every commit we automatically run a suite of [pre-commit](https://pre-commit.com/) hooks that perform static code analysis, autoformating, and sanity checks. This is setup during our [installation process](https://automl.github.io/neps/contributing/installation/).
- At every commit / push locally running a minimal suite of integration tests is encouraged. The tests correspond directly to examples in [neps_examples](https://github.com/automl/neps/tree/master/neps_examples) and only check for crash-causing errors.
- At every push all integration tests and regression tests are run automatically using [github actions](https://github.com/automl/neps/actions).

## Examples and Integration Tests

We use examples in [neps_examples](https://github.com/automl/neps/tree/master/neps_examples) as integration tests, which we run from the main directory via

```bash
pytest
```

before every critical push.

### Creating an Integration Test

If you want an implementation to be included in the above testing procedure:

1. Create an example in [neps_examples](https://github.com/automl/neps/tree/master/neps_examples).
1. Add the example to [test_examples.py](https://github.com/automl/neps/blob/377049fe57ba46d061790933baf35214fab6f11e/tests/test_examples.py#L33).

### Running all integration tests locally

To speedup testing for developers, we only run a core set of tests per default. To run all tests use

```bash
pytest -m all_examples
```

On github, we always run all examples.

### What to do if tests fail

If tests fail for you on the master:

1. Try running the tests with a fresh environment install.
1. If issues persist, notify others in the neps developers chat on mattermost.

## Regression Tests

Regression tests are run on each push to the repository to assure the performance of the optimizers don't degrade.

Currently, regression runs are recorded on JAHS-Bench-201 data for 2 tasks: `cifar10` and `fashion_mnist` and only for optimizers: `random_search`, `bayesian_optimization`, `mf_bayesian_optimization`, `regularized_evolution`.
This information is stored in the `tests/regression_runner.py` as two lists: `TASKS`, `OPTIMIZERS`.
The recorded results are stored as a json dictionary in the `tests/losses.json` file.

### Adding new optimizer algorithms

Once a new algorithm is added to NEPS library, we need to first record the performance of the algorithm for 100 optimization runs.

- If the algorithm expects standard loss function (pipeline) and accepts fidelity hyperparameters in pipeline space, then recording results only requires adding the optimizer name into `OPTIMIZERS` list in `tests/regression_runner.py` and running `tests/regression_runner.py`

- In case your algorithm requires custom pipeline and/or pipeline space you can modify the `runner.run_pipeline` and `runner.pipeline_space` attributes of the `RegressionRunner` after initialization (around line `#322` in `tests/regression_runner.py`)

You can verify the optimizer is recorded by rerunning the `regression_runner.py`.
Now regression test will be run on your new optimizer as well on every push.

### Regression test metrics

For each regression test the algorithm is run 10 times to sample its performance, then they are statistically compared to the 100 recorded runs. We use these 3 boolean metrics to define the performance of the algorithm on any task:

1. [Kolmogorov-Smirnov test for goodness of fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html) - `pvalue` >= 10%
1. Absolute median distance - bounded within 92.5% confidence range of the expected median distance
1. Median improvement - Median improvement over the recorded median

Test metrics are run for each `(optimizer, task)` combination separately and then collected.
The collected metrics are then further combined into 2 metrics

1. Task pass - either both `Kolmogorov-Smirnov test` and `Absolute median distance` test passes or just `Median improvement`
1. Test aggregate - Sum_over_tasks(`Kolmogorov-Smirnov test` + `Absolute median distance` + 2 * `Median improvement`)

Finally, a test for an optimizer only passes when at least for one of the tasks `Task pass` is true, and `Test aggregate` is higher than 1 + `number of tasks`

### On regression test failures

Regression tests are stochastic by nature, so they might fail occasionally even the algorithm performance didn't degrade.
In the case of regression test failure, try running it again first, if the problem still persists, then you can contact [Danny Stoll](mailto:stolld@cs.uni-freiburg.de) or [Samir](mailto:garibovs@cs.uni-freiburg.de).
You can also run tests locally by running:

```
poetry run pytest -m regression_all
```

## Disabling and Skipping Checks etc.

### Pre-commit: How to not run hooks?

To commit without running `pre-commit` use `git commit --no-verify -m <COMMIT MESSAGE>`.

### Mypy: How to ignore warnings?

There are two options:

- Disable the warning locally:

  ```python
  code = "foo"  # type: ignore[ERROR_CODE]
  ```

- If you know what you are doing, you can add the whole module to the `[[tool.mypy.overrides]]` section.
  This is useful e.g., when adding new files that are in early stage development.

### Black: How to not format code parts?

```python
x = 2  # fmt: off
```

or for blocks

```python
# fmt: off
x = 2
y = x + 1
# fmt: on
```
