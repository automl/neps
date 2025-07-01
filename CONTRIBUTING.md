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

### 1. Install uv

First, install uv, e.g., via

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone the neps repository

```bash
git clone https://github.com/automl/neps.git
cd neps
```

### 3. Create a virtual environment and install the neps package

```bash
uv venv --python 3.11
source .venv/bin/activate
```

Then, inside the main directory of neps run

```bash
uv pip install -e ".[dev]"
```

This will installthe neps package but also additional dev dependencies.

### 4. Activate pre-commit for the repository

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

We have setup checks and tests at several points in the development flow:

- At every commit we automatically run a suite of [pre-commit](https://pre-commit.com/) hooks that perform static code analysis, autoformating, and sanity checks.
This is setup during our [installation process](https://automl.github.io/neps/contributing/installation/).
- At every commit / push locally running a minimal suite of integration tests is encouraged.
The tests correspond directly to examples in [neps_examples](https://github.com/automl/neps/tree/master/neps_examples) and only check for crash-causing errors.

## Checks and tests

### Linting (Ruff)
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

### Type Checking (Mypy)
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


### Examples and Integration Tests

We use some examples in [neps_examples](https://github.com/automl/neps/tree/master/neps_examples) as integration tests, which we run from the main directory via

```bash
pytest
```

If tests fail for you on the master, please raise an issue on github, preferably with some information on the error,
traceback and the environment in which you are running, i.e. python version, OS, etc.

## Disabling and Skipping Checks etc.

### Pre-commit: How to not run hooks?

To commit without running `pre-commit` use `git commit --no-verify -m <COMMIT MESSAGE>`.

### Mypy: How to ignore warnings?

There are two options:

- Disable the warning locally:

  ```python
  code = "foo"  # type: ignore
  ```

### Managing Dependencies

To manage dependencies we use [uv](https://docs.astral.sh/uv/getting-started/) (replaces pip).

#### Add dependencies

To install a dependency use

```bash
uv add dependency
```

and commit the updated `pyproject.toml` to git.

For more advanced dependency management see examples in `pyproject.toml` or have a look at the [uv documentation](https://docs.astral.sh/uv/getting-started/).

#### Install dependencies added by others

When other contributors added dependencies to `pyproject.toml`, you can install them via

```bash
uv pip install -e ".[dev]"
```

## Documentation

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

## Releasing a New Version

There are four steps to releasing a new version of neps:

0. Understand Semantic Versioning
1. Update the Package Version
2. Commit and Push With a Version Tag
3. Update Documentation
4. Publish on PyPI

### 0. Understand Semantic Versioning

We follow the [semantic versioning](https://semver.org) scheme.

## 1. Run tests

```bash
uv run pytest
```

## 2. Update the Package Version and CITATION.cff

```bash
bump-my-version bump <major | minor | patch>
```

This will automatically update the version in `pyproject.toml` and `CITATION.cff`, tag the commit and push it to the remote repository.

### 3. Update Documentation

First check if the documentation has any issues via

```bash
mike deploy <current version> latest -u
mike serve
```

and then looking at it.

Afterwards, publish it via

```bash
mike deploy <current version> latest -up
```

### 4. Publish on PyPI

To publish to PyPI:

1. Get publishing rights, e.g., asking Danny or Neeratyoy.
2. Be careful, once on PyPI we can not change things.
3. Run

```bash
uv build
uv publish
```

This will ask for your PyPI credentials.
