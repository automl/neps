# Contributing

## Getting Help

Ask in the neps developer chat on mattermost or any contributor directly.
If you are not in the mattermost chat yet, ask to get access.

## Development Practices and Tooling

### Development Workflow

We loosely practice [trunk-based-development](https://trunkbaseddevelopment.com/):

- We work almost exclusively on the master branch
- We commit, push, and pull often
- We automatically run code quality checks before every commit (using [pre-commit](https://pre-commit.com/))
- We manually run tests (using `pytest`) before every critical push and automatically afterwards (using [github actions](https://github.com/automl-private/neps/actions))

### Dependency Management

To manage dependencies and for package distribution we use [poetry](https://python-poetry.org/docs/) (replaces pip).

### Examples and Tests

We document major features with an example (see [neps_examples](neps_examples)).
When adding a new example also include it in the [example README](neps_examples/README.md).

These examples also serve as integration tests, which we run from the main directory via

```bash
pytest
```

before every critical push.
Running the tests will create a temporary directory `tests_tmpdir` that includes the output of the last three test executions.

If tests fail for you on the master:

1. Try running the tests with a fresh environment install.
1. If issues persist, notify others in the neps developers chat on mattermost.

## Developer Installation

There are three required steps and one optional:

1. Optional: Install miniconda and create an environment
1. Install poetry
1. Install the neps package using poetry
1. Activate pre-commit for the repository

For instructions see below.

### 1. Optional: Install miniconda and create an environment

To manage python versions install e.g., miniconda with

```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh
bash install_miniconda.sh -b -p $HOME/.conda  # Change to place of preference
rm install_miniconda.sh
```

Consider running `~/.conda/bin/conda init` or `~/.conda/bin/conda init zsh` .

Then finally create the environment and activate it

```bash
conda create -n neps python=3.7.5
conda activate neps
```

### 2. Install poetry

First, install poetry, e.g., via

```bash
wget https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py -O get-poetry.py
python get-poetry.py
rm get-poetry.py
```

Then consider appending

```bash
export PATH="$HOME/.poetry/bin:$PATH"
```

to your `.zshrc` / `.bashrc` or alternatively simply running the export manually.

### 3. Install the neps Package Using poetry

Inside the main directory of neps run

```bash
poetry install
```

To install specific versions of torch (e.g., cuda enabled versions) you might want to use our utility

```bash
python -m neps.utils.install_torch
```

### 4. Activate pre-commit for the repository

With the python environment used to install the neps package run in the main directory of neps

```bash
pre-commit install
```

## Tooling Tips

### Poetry: Add dependencies

To install a dependency use

```bash
poetry add dependency
```

and commit the updated `pyproject.toml` to git.

For more advanced dependency management see examples in `pyproject.toml` or have a look at the [poetry documentation](https://python-poetry.org/).

### Poetry: Install dependencies added by others

When other contributors added dependencies to `pyproject.toml`, you can install them via

```bash
rm poetry.lock
poetry install
```

or also updating dependencies alongside via

```bash
poetry update
```

### Pre-commit: Do not run hooks

To commit without running `pre-commit` use `git commit --no-verify -m <COMMIT MESSAGE>`.

### Pylint: Ignore warnings

There are two options:

- Disable the warning locally:

  ```python
  code = "foo"  # pylint: disable=bar
  ```

  Make sure to use the named version of the error (e.g., `unspecified-encoding`, not `W1514`).

- Remove warning in `pyproject.toml` that we do not consider useful (do not catch bugs, do not increase code quality).

### Black: Do not format code parts

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

[Editorconfig](https://editorconfig.org/) allows to set line lengths and other display parameters automatically based on a `.editorconfig` file.
Many editors have [native support](https://editorconfig.org/#pre-installed) (e.g., PyCharm) so you do not need to do anything.
For other editors (e.g., VSCode), you need to install a [plugin](https://editorconfig.org/#download).
