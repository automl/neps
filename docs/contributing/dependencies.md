# Managing Dependencies

To manage dependencies and for package distribution we use [poetry](https://python-poetry.org/docs/) (replaces pip).

## Add dependencies

To install a dependency use

```bash
poetry add dependency
```

and commit the updated `pyproject.toml` to git.

For more advanced dependency management see examples in `pyproject.toml` or have a look at the [poetry documentation](https://python-poetry.org/).

## Install dependencies added by others

When other contributors added dependencies to `pyproject.toml`, you can install them via

```bash
poetry lock
poetry install
```
