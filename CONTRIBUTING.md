# Contributing

## Development Practices and Tooling

We loosely practice [trunk-based-development](https://trunkbaseddevelopment.com/):
* We work almost exclusively on the master branch
* We commit, push, and pull often
* We use [pre-commit](https://pre-commit.com/) to run code quality checks at every commit

To manage dependencies and for package distribution we use [poetry](https://python-poetry.org/docs/) (replaces pip).

We document major features with an example (see cnas_examples).
These examples also serve as integration tests, which we will run automatically in the future.

## Python Coding Guidelines

* We use relative imports inside our library
* We use the black style with line length 90, enforced by our autoformatter as part of our pre-commit hooks

## Developer Installation

```bash
bash install_dev_utils/miniconda.sh    # For managing python versions
bash install_dev_utils/poetry.sh       # Python package manager
```

Create and activate python environment. For example, using conda:

```bash
conda create -n neps python=3.7.5
conda activate neps
```

```bash
poetry install
pre-commit install
```

## Tooling Tips

### Add dependencies

To install a dependency use

```bash
poetry add dependency
```

and commit the updated `pyproject.toml` to git.

For more advanced dependency management see examples in `pyproject.toml` or have a look at the [poetry documentation](https://python-poetry.org/).

### Do not run pre-commit hooks

To commit without running `pre-commit` use `git commit --no-verify -m <COMMIT MESSAGE>`.

### Ignore pylint warning

```python
code = "foo"  # pylint: disable=bar
```

Or remove warnings in `pyproject.toml` that we do not consider useful (do not catch bugs, do not increase code quality).

### Do not format with black

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

### Editorconfig

You might want to install an [editorconfig](https://editorconfig.org/) plugin for your text editor to automatically set line lengths etc.
