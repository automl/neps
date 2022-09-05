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

## Installation

For the contributor installation process see [Installation](https://automl.github.io/neps/contributing/installation/).

## Checks and tests

The documentation also includes [an overview](https://automl.github.io/neps/contributing/tests/) on the protocols and tools we use for code quality checks and tests.
