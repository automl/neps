# Tooling

## Pre-commit: How to not run hooks?

To commit without running `pre-commit` use `git commit --no-verify -m <COMMIT MESSAGE>`.

## Pylint: How to ignore warnings?

There are two options:

- Disable the warning locally:

  ```python
  code = "foo"  # pylint: disable=ERROR_CODE
  ```

  Make sure to use the named version of the error (e.g., `unspecified-encoding`, not `W1514`).

- Remove warning in `pyproject.toml` that we do not consider useful (do not catch bugs, do not increase code quality).

## Mypy: How to ignore warnings?

There are two options:

- Disable the warning locally:

  ```python
  code = "foo"  # type: ignore[ERROR_CODE]
  ```

- If you know what you are doing, you can add the whole module to the `[[tool.mypy.overrides]]` section.
  This is useful e.g., when adding new files that are in early stage development.

## Black: How to not format code parts?

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

## What is Editorconfig about?

[Editorconfig](https://editorconfig.org/) allows to set line lengths and other display parameters automatically based on a `.editorconfig` file.
Many editors have [native support](https://editorconfig.org/#pre-installed) (e.g., PyCharm) so you do not need to do anything.
For other editors (e.g., VSCode), you need to install a [plugin](https://editorconfig.org/#download).
