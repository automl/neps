# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NePS (Neural Pipeline Search) is a Python library for hyperparameter optimization (HPO)
and neural architecture search (NAS), published as `neural-pipeline-search` on PyPI.
Users call `neps.run(evaluate_pipeline, pipeline_space, ...)` to launch an optimization;
NePS coordinates one or more parallel workers (processes/machines) that share state
through the filesystem under `root_directory`.

## Common commands

Environment is managed with `uv` (not plain pip/venv).

```bash
# Setup
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install

# Lint (ruff, line-length 90, config in pyproject.toml)
ruff check --fix neps
ruff format neps

# Type check (mypy, strict — all functions must be typed)
mypy neps

# Tests (pytest). Default addopts exclude the `ci_examples` marker.
pytest                                  # full suite minus ci_examples
pytest tests/test_state/test_trial.py   # single file
pytest tests/test_state/test_trial.py::test_name  # single test
pytest -m ""                            # run everything, including ci_examples (what CI does)
pytest -m ci_examples                   # only the example-based integration tests

# Skip pre-commit hooks for a WIP commit (not for normal use)
git commit --no-verify -m "..."
```

CI (`.github/workflows/tests.yaml`) runs `uv run --all-extras pytest -m ""` across
Python 3.10–3.13 on Linux/macOS/Windows, so don't assume a test only needs to pass
locally with the default marker filter.

Docs use MkDocs Material + `mike` for versioning (`mike deploy <version> latest && mike serve`),
source under `docs/`, config in `mkdocs.yml`. `docs/dev_docs/contributing.md` just
includes the root `CONTRIBUTING.md`.

## Architecture

### Two parallel search-space systems (mid-migration)

The codebase is actively transitioning from a legacy space API to a new one; both
exist simultaneously and `neps.run()`'s `pipeline_space` argument accepts either:

- **Legacy**: `neps/space/search_space.py` (`SearchSpace`) plus `neps/space/parameters.py`
  (`HPOFloat`, `HPOInteger`, `HPOCategorical`, `HPOConstant`). Config-space-like, dict-driven.
- **New**: `neps/space/neps_spaces/` (`PipelineSpace`, `Float`, `Integer`, `Categorical`,
  `Fidelity`/`FloatFidelity`/`IntegerFidelity`, `Resample`, `Operation`). Pipeline spaces are
  defined by subclassing `PipelineSpace` as a class body (see README usage example).
  `neps/space/neps_spaces/neps_space.py` (`NepsCompatConverter`,
  `convert_neps_to_classic_search_space`) bridges the two so legacy optimizers can consume
  the new space type. When touching space/sampling code, check which system a given
  optimizer expects before assuming compatibility.

### Optimizer plugin protocol

Optimizers implement the `AskFunction` protocol (`neps/optimizers/optimizer.py`): a
callable `(trials, budget_info, n=None) -> SampledConfig | list[SampledConfig]`. Users can
pass a string name, a `(name, kwargs)` tuple, a raw callable, or a `CustomOptimizer` to
`neps.run(optimizer=...)`; resolution happens in `neps/optimizers/__init__.py::load_optimizer`.

Built-in optimizers are implemented as individual modules under `neps/optimizers/`
(e.g. `bayesian_optimization.py`, `bracket_optimizer.py` for successive
halving/hyperband/ASHA-family algorithms, `priorband.py`, `primo.py`, `ifbo.py`,
`random_search.py`, `grid_search.py`, plus the `neps_*` variants that operate on the new
`PipelineSpace`). They are registered by name in `neps/optimizers/algorithms.py` in
`PredefinedOptimizers` and the `OptimizerChoice` literal — when adding a new optimizer,
update both, add a documented factory function in `algorithms.py`, and add a section to
`neps.run()`'s docstring (there's a checklist comment at the top of `algorithms.py`).
`neps/sampling/` (`Prior`, `Sampler`, `Uniform`, distributions) provides the sampling
primitives optimizers build on; `neps/optimizers/acquisition/` and `neps/optimizers/models/`
hold BO acquisition functions and surrogate models (e.g. FTPFN for `ifbo`).

### Shared filesystem state & the worker runtime

There is no central server: parallel workers coordinate purely through
`root_directory` on a shared filesystem, using file locks (`portalocker`/`filelock`) and
atomic writes. This is the core design constraint for anything touching `neps/state/` or
`neps/runtime.py`:

- `neps/state/neps_state.py` (`NePSState`) is the source of truth — an object each worker
  independently constructs (`NePSState.create_or_load`) that reads/writes trials, optimizer
  state, and errors atomically without a central coordinator.
- `neps/state/filebased.py` defines the on-disk `ReaderWriterTrial` / `ReaderWriterErrDump`
  format (each trial is a directory with `config.yaml`, `report.yaml`, `metadata.json`) and
  `FileLocker`.
- `neps/state/trial.py` defines the `Trial`/`Report` dataclasses; `neps/state/optimizer.py`
  holds `OptimizationState`/`BudgetInfo`; `neps/state/err_dump.py` tracks worker errors.
- `neps/runtime.py` is the worker loop: it repeatedly asks the optimizer for the next
  trial(s), locks/claims one, evaluates `evaluate_pipeline`, writes the report, and checks
  stopping criteria (`evaluations_to_spend`, `cost_to_spend`, `fidelities_to_spend`,
  `continue_until_max_evaluation_completed`). It also special-cases PyTorch DDP so only
  rank-zero drives the NePS loop (`_is_ddp_and_not_rank_zero`).
- Timeouts/poll intervals/retry counts for all the above are tunable via `NEPS_*` env vars
  defined in `neps/env.py` (e.g. `NEPS_TRIAL_FILELOCK_TIMEOUT`, `NEPS_STATE_FILELOCK_TIMEOUT`) —
  useful when debugging flaky filesystem-lock behavior, especially on slower/networked FS.
- `neps/optimizers/ask_and_tell.py` (`AskAndTell`) exposes a lower-level ask/tell interface
  for driving optimizers/state under a custom runtime, bypassing `neps.run()`'s worker loop.

### Other entry points

- `neps/api.py` — all public top-level functions (`run`, `create_config`, `load_config`,
  `import_trials`, `save_pipeline_results`, `load_pipeline_space`, `load_optimizer_info`);
  this is what `neps/__init__.py` re-exports as the `neps.*` public API.
- `neps/status/status.py` — `neps.status(...)`, inspects a `root_directory` to summarize
  progress/incumbents (also runnable as `python -m neps.status`).
- `neps/plot/` — plotting utilities (`neps.plot`) and TensorBoard integration (`tblogger`);
  runnable as `python -m neps.plot`.
- `neps/clean/` — utilities for cleaning up a `root_directory` (`python -m neps.clean`).

### Tests mirror examples

`tests/test_neps_space/` covers the new `PipelineSpace` system in depth (grammar-like,
recursive, resampled, HNAS-like spaces, backward compatibility with the legacy space).
`tests/test_runtime/` and `tests/test_state/` exercise the worker loop and filebased state
directly. Additionally, many `neps_examples/` scripts double as integration tests — pytest
markers (`ci_examples`, `core_examples`, `runtime`, `neps_api`, `summary_csv`) partition
these; the default local run excludes `ci_examples` (see `addopts` in `pyproject.toml`),
but CI runs with all markers enabled.

## Code style notes specific to this repo

- Every module must start `from __future__ import annotations` (enforced by ruff's isort
  `required-imports`).
- mypy is strict: `disallow_untyped_defs`, `disallow_incomplete_defs`,
  `disallow_untyped_decorators` are all on for `neps/` (not `tests/`).
- Docstrings follow the Google convention (`tool.ruff.lint.pydocstyle`).
- `neps/optimizers/**.py` and `tests/*.py` have relaxed ruff rules (see
  `[tool.ruff.lint.per-file-ignores]` in `pyproject.toml`) — don't assume the strict rule
  set applies uniformly across the repo.
