# Introduction

## Getting Help

Ask in the neps contributor chat on mattermost or any contributor directly.
If you are not in the mattermost chat yet, ask to get access.

## Development Workflow

We loosely practice [trunk-based-development](https://trunkbaseddevelopment.com/):

- We work almost exclusively on the master branch
- We commit, push, and pull often
- We automatically run code quality checks before every commit (using [pre-commit](https://pre-commit.com/))
- We manually run tests (using `pytest`) before every critical push and automatically afterwards (using [github actions](https://github.com/automl/neps/actions))

## Examples and Tests

We document major features with an example (see [neps_examples](https://github.com/automl/neps/tree/master/neps_examples).
When adding a new example also include it in the [example README](https://github.com/automl/neps/tree/master/neps_examples/README.md).

These examples also serve as integration tests, which we run from the main directory via

```bash
pytest
```

before every critical push.
Running the tests will create a temporary directory `tests_tmpdir` that includes the output of the last three test executions.

To speedup testing for developers, we only run a core set of tests per default. To run all tests use

```bash
pytest -m all_examples
```

On github, we always run all examples.

If tests fail for you on the master:

1. Try running the tests with a fresh environment install.
1. If issues persist, notify others in the neps developers chat on mattermost.
