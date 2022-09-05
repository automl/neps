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

## Disabling and Skipping Checks etc.

### Pre-commit: How to not run hooks?

To commit without running `pre-commit` use `git commit --no-verify -m <COMMIT MESSAGE>`.

### Pylint: How to ignore warnings?

There are two options:

- Disable the warning locally:

  ```python
  code = "foo"  # pylint: disable=ERROR_CODE
  ```

  Make sure to use the named version of the error (e.g., `unspecified-encoding`, not `W1514`).

- Remove warning in `pyproject.toml` that we do not consider useful (do not catch bugs, do not increase code quality).

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
