# Introduction

## Getting Help

Please use our github and raise an issue at: [https://github.com/automl/neps](https://github.com/automl/neps)

## Development Workflow

We use one main branch `master` and feature branches for development.
We use pull requests to merge feature branches into `master`.
Versions released to PyPI are tagged with a version number.

Automatic checks are run on every pull request and on every commit to `master`.

## Installation

There are three required steps and one optional:

1. Install uv
1. Install the neps package using uv
1. Activate pre-commit for the repository

For instructions see below.

## 1. Install uv

First, install uv, e.g., via

```bash
curl -sSL https://install.python-poetry.org | python3 -
# or directly into your virtual env using `pip install poetry`
```

## 2. Create a virtual environment

```bash
uv venv --python 3.11
source .venv/bin/activate
```

## 3. Install the neps Package Using uv

Clone the repository, e.g.,

```bash
git clone https://github.com/automl/neps.git
cd neps
```

Then, inside the main directory of neps run

```bash
uv pip install -e ".[dev]"
```

This will installthe neps package but also additional dev dependencies.

## 4. Activate pre-commit for the repository

With the python environment used to install the neps package run in the main directory of neps

```bash
pre-commit install
```

This install a set of hooks that will run basic linting and type checking before every comment.
If you ever need to unsinstall the hooks, you can do so with `pre-commit uninstall`.
These mostly consist of `ruff` for formatting and linting and `mypy` for type checking.

We highly recommend you install at least [`ruff`](https://github.com/astral-sh/ruff) either on command line, or in the editor of
your choice, e.g.
[VSCode](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff),
[PyCharm](https://plugins.jetbrains.com/plugin/20574-ruff).


# Checks and Tests

We have setup checks and tests at several points in the development flow:

- At every commit we automatically run a suite of [pre-commit](https://pre-commit.com/) hooks that perform static code analysis, autoformating, and sanity checks.
This is setup during our [installation process](https://automl.github.io/neps/contributing/installation/).
- At every commit / push locally running a minimal suite of integration tests is encouraged.
The tests correspond directly to examples in [neps_examples](https://github.com/automl/neps/tree/master/neps_examples) and only check for crash-causing errors.
- At every push all integration tests and regression tests are run automatically using [github actions](https://github.com/automl/neps/actions).

## Linting (Ruff)
For linting we use `ruff` for checking code quality. You can install it locally and use it as so:

```bash
uv pip install ruff
ruff check --fix neps  # the --fix flag will try to fix issues it can automatically
```


This will also be run using `pre-commit` hooks.

To ignore a rule for a specific line, you can add a comment with `ruff: disable` at the end of the line, e.g.

```python
for x, y in zip(a, b):  # noqa: <ERRCODE>
    pass
```

The configuration of `ruff` is in the `pyproject.toml` file and we refer you to the
[documentation](https://docs.astral.sh/ruff/) if you require any changes to be made.

There you can find the documentation for all of the rules employed.

## Type Checking (Mypy)
For type checking we use `mypy`. You can install it locally and use it as so:

```bash
uv pip install mypy
mypy neps
```

Types are helpful for making your code more understandable by your editor and tools, allowing them to warn you of
potential issues, as well as allow for safer refactoring. Copilot also works better with types.

To ignore some error you can use `# type: ignore` at the end of the line, e.g.

```python
code = "foo"  # type: ignore
```

A common place to ignore types is when dealing with numpy arrays, tensors and pandas, where the type
checker can not be sure of the return type.

```python
df.mean()  # Is this another dataframe, a series or a single number?
```

In the worse case, please just use `Any` and move on with your life, the type checker is meant to help you catch bugs,
not hinder you. However it will take some experience to know whe it's trying to tell you something useful vs. something
it just can not infer properly. A good rule of thumb is that you're only dealing with simple native types from python
or types defined from NePS, there is probably a good reason for a mypy error.

If you have issues regarding typing, please feel free to reach out for help `@eddiebergman`.


## Examples and Integration Tests

We use examples in [neps_examples](https://github.com/automl/neps/tree/master/neps_examples) as integration tests, which we run from the main directory via

```bash
pytest
```

If tests fail for you on the master, please raise an issue on github, preferabbly with some informationon the error,
traceback and the environment in which you are running, i.e. python version, OS, etc.

## Regression Tests

Regression tests are run on each push to the repository to assure the performance of the optimizers don't degrade.

Currently, regression runs are recorded on JAHS-Bench-201 data for 2 tasks: `cifar10` and `fashion_mnist` and only for optimizers: `random_search`, `bayesian_optimization`, `mf_bayesian_optimization`.
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
uv run pytest -m regression_all
```

## Disabling and Skipping Checks etc.

### Pre-commit: How to not run hooks?

To commit without running `pre-commit` use `git commit --no-verify -m <COMMIT MESSAGE>`.

### Mypy: How to ignore warnings?

There are two options:

- Disable the warning locally:

  ```python
  code = "foo"  # type: ignore
  ```

## Managing Dependencies

To manage dependencies we use [uv](https://docs.astral.sh/uv/getting-started/) (replaces pip).

## Add dependencies

To install a dependency use

```bash
uv add dependency
```

and commit the updated `pyproject.toml` to git.

For more advanced dependency management see examples in `pyproject.toml` or have a look at the [uv documentation](https://docs.astral.sh/uv/getting-started/).

## Install dependencies added by others

When other contributors added dependencies to `pyproject.toml`, you can install them via

```bash
uv pip install -e ".[dev]"
```

# Documentation

We use [MkDocs](https://www.mkdocs.org/getting-started/), more specifically [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) for documentation.
To support documentation for multiple versions, we use the plugin [mike](https://github.com/jimporter/mike).

Source files for the documentation are under `/docs` and configuration at  [mkdocs.yml](https://github.com/automl/neps/tree/master/mkdocs.yml).

To build and view the documentation run

```bash
mike deploy 0.5.1 latest
mike serve
```

and open the URL shown by the `mike serve` command.

To publish the documentation run

```bash
mike deploy 0.5.1 latest -p
```

# Releasing a New Version

There are four steps to releasing a new version of neps:

0. Understand Semantic Versioning
1. Update the Package Version
2. Commit and Push With a Version Tag
3. Update Documentation
4. Publish on PyPI

## 0. Understand Semantic Versioning

We follow the [semantic versioning](https://semver.org) scheme.

## 1. Update the Package Version and CITATION.cff

```bash
bump-my-version bump <major | minor | patch>
```

## 2. Commit with a Version Tag

First commit and test

```bash
git add pyproject.toml
git commit -m "Bump version from v0.8.4 to v0.9.0"
uv run pytest
```

Then tag and push

```bash
git tag v0.9.0
git push --tags
git push
```

## 3. Update Documentation

First check if the documentation has any issues via

```bash
mike deploy 0.9.0 latest -u
mike serve
```

and then looking at it.

Afterwards, publish it via

```bash
mike deploy 0.9.0 latest -up
```

## 4. Publish on PyPI

To publish to PyPI:

1. Get publishing rights, e.g., asking Danny or Maciej or Neeratyoy.
2. Be careful, once on PyPI we can not change things.
3. Run

```bash
poetry publish
```

This will ask for your PyPI credentials.
