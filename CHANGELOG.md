# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `extrapolaw.acquire(cost_estimator=..., max_cost_per_config=...)` for scaling studies. The cost estimator is a function or, for finite spaces, a JSON table. It fills in the `cost` of evaluations that do not report one and is handed to optimizers inheriting from `ScalingOptimizer`.
- `carbs` optimizer, wrapping [CARBS](https://github.com/imbue-ai/carbs).
- `chinchilla` optimizer for [Chinchilla](https://arxiv.org/abs/2203.15556) guided scaling studies.
- Added scaling plots also with the inference cost in with CSV outputs. `evaluate_pipeline` can return an `inference_cost` and `downstream_performances` next to the `cost`.
- `evaluate_pipeline` can return the model size `n_params` and the amount of training data `n_data`, which `extrapolaw.predict()` uses.
- Each scaling optimizer names the scaling law fitted on its runs (`ScalingOptimizer.predictor`, `chinchilla3` for `chinchilla`). `extrapolaw.predict()` and the `loss_over_cost` plot both use it.
- The scaling plots are per task, in `summary/pre_train/` and `summary/downstream/<name>/`, with the fit extended to the target costs, the Pareto front and the predictions as text and CSV files, and parameter plots with what the fit predicts for each parameter.
- `extrapolaw.predict()` predicts at 10 and 100 times the largest cost of the runs by default, takes `target_inference_costs`, and saves the target costs to `summary/targets.json` and how the predictions converged to `summary/convergence/`.

### Changed
- **BREAKING:** Renamed `extrapolaw.run()` to `extrapolaw.acquire()`.
- **BREAKING:** `evaluate_pipeline` must return a dictionary with `objective_to_minimize` and the training `cost`, unless a `cost_estimator` provides the cost.
- **BREAKING:** Forked from NePS 0.17.0 and renamed the package `neps` → `extrapolaw` (PyPI package, import name, CLI, `EXTRAPOLAW_*` env vars, optimizer names, default result dirs, examples in `extrapolaw_examples/`).
- `bayesian_optimization` passes the space's constants to `constraints_func`, like `random_search` does.
- **BREAKING:** The summary no longer has `best_config.txt`, `best_config_trajectory.txt`, `short.csv` and the incumbent trajectory, and the scaling data is no longer logged to TensorBoard.

Extrapolaw was forked from [NePS](https://github.com/automl/neps) 0.17.0. The earlier history is in [CHANGELOG_NEPS.md](CHANGELOG_NEPS.md).

[unreleased]: https://github.com/automl/extrapolaw/compare/v0.17.0...HEAD
