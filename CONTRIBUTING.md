# Contributing

## Development Practices and Tooling

### Development Workflow

We loosely practice [trunk-based-development](https://trunkbaseddevelopment.com/):

- We work almost exclusively on the master branch
- We commit, push, and pull often
- We automatically run code quality checks at every commit (using [pre-commit](https://pre-commit.com/))

### Dependency Management

To manage dependencies and for package distribution we use [poetry](https://python-poetry.org/docs/) (replaces pip).

### Examples and Tests

We document major features with an example (see [neps_examples](neps_examples)). When adding a new example also include it in the [example README](neps_examples/README.md)

These examples also serve as integration tests, which we will run automatically in the future and currently run via
simply `pytest` in the main directory.

### Python Coding Guidelines

- We use relative imports inside our library
- We use the black style with line length 90, enforced by our autoformatter as part of our pre-commit hooks

## Developer Installation

### 0. Optional: Install Miniconda and Create an Environment

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

### 1. Install poetry

First, install poetry, e.g., via

```bash
wget https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py -O get-poetry.py
python get-poetry.py
rm get-poetry.py
```

Then append to your `.zshrc` / `.bashrc` or run: `export PATH="$HOME/.poetry/bin:$PATH"`

### 2. Install the neps Package Using poetry

Inside the main directory of neps run

```bash
poetry install
```

To install specific versions of torch (e.g., cuda enabled versions) you might want to use our utility

```bash
python -m neps.utils.install_torch
```

### 3. Activate pre-commit for the repository

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

### Pre-commit: Do not run hooks

To commit without running `pre-commit` use `git commit --no-verify -m <COMMIT MESSAGE>`.

### Pylint: Ignore warnings

```python
code = "foo"  # pylint: disable=bar
```

Or remove warnings in `pyproject.toml` that we do not consider useful (do not catch bugs, do not increase code quality).

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

You might want to install an [editorconfig](https://editorconfig.org/) plugin for your text editor to automatically set line lengths etc.
