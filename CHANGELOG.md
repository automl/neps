# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-11-25

### Added
- `scalefm.acquire(cost_estimator=..., max_cost_per_config=...)` for scaling studies. The cost estimator is a function or, for finite spaces, a JSON table. It fills in the `cost` of evaluations that do not report one and is handed to optimizers inheriting from `ScalingOptimizer`.
- `carbs` optimizer, wrapping [CARBS](https://github.com/imbue-ai/carbs). CARBS is not on PyPI, so it is not an extra of ScaleFM; install it from GitHub, see the [optimizer docs](https://automl.github.io/scalefm/latest/reference/optimizers/#31-carbs).
- `chinchilla` optimizer for [Chinchilla](https://arxiv.org/abs/2203.15556) guided scaling studies.
- Added scaling plots also with the inference cost in with CSV outputs. `evaluate_pipeline` can return an `inference_cost` and `downstream_performances` next to the `cost`.
- `evaluate_pipeline` can return the model size `n_params` and the amount of training data `n_data`, which `scalefm.predict()` uses.
- Each scaling optimizer names the scaling law fitted on its runs (`ScalingOptimizer.predictor`, `chinchilla3` for `chinchilla`). `scalefm.predict()` and the `loss_over_cost` plot both use it.
- The scaling plots are per task, in `summary/pre_train/` and `summary/downstream/<name>/`, with the fit extended to the target costs, the Pareto front and the predictions as CSV files, and parameter plots with what the fit predicts for each parameter.
- `scalefm.predict()` predicts at 10 and 100 times the largest cost of the runs by default, takes `target_inference_costs`, and saves the target costs to `summary/targets.json` and how the predictions converged to `summary/convergence/`.
- An LLM pre-training scaling study as a real-world example with asynchronous evaluation, training each config on up to 4 GPUs.

### Changed
- **BREAKING:** Renamed `scalefm.run()` to `scalefm.acquire()`.
- **BREAKING:** `scalefm.acquire()` uses `random_search` without an `optimizer`, instead of `"auto"`. Pass `optimizer="auto"` to choose the optimizer from the space, or to continue a run with the optimizer it was started with.
- **BREAKING:** `scalefm.acquire(is_async=True)` only samples configurations and takes no `evaluate_pipeline` (it must be `None`); launch a job for each `configs/config_<id>/config.yaml` and report its result with `scalefm.save_pipeline_results()`. Without `is_async`, `evaluate_pipeline` is required, and returning `None` from it raises a `ValueError`.
- **BREAKING:** `evaluate_pipeline` must return a dictionary with `objective_to_minimize` and the training `cost`, unless a `cost_estimator` provides the cost.
- **BREAKING:** Forked from NePS 0.17.0 and renamed the package `neps` → `scalefm` (PyPI package, import name, CLI, `SCALEFM_*` env vars, optimizer names, default result dirs, examples in `scalefm_examples/`).
- `bayesian_optimization` passes the space's constants to `constraints_func`, like `random_search` does.
- **BREAKING:** The summary no longer has `best_config.txt`, `best_config_trajectory.txt`, `short.csv` and the incumbent trajectory, and the scaling data is no longer logged to TensorBoard.
- `optimizer="auto"` picks `bayesian_optimization` for a space without priors and fidelities, `random_search` otherwise (sampling from the priors), and `complex_random_search` for a space that cannot be converted to a classic one. A fidelity is set to its maximum as it's not used in any optimizer for now.
- The TabPFN example runs only `scalefm.predict()`, since `acquire()` and `predict()` already write the summary.

### Fixed
- Workers started together on a new `root_directory` no longer crash with `EOFError` reading a half-written `optimizer_state.pkl`: the state files are replaced whole, and `optimizer_info.yaml` is written last.
- Grid search and `chinchilla` runs stop normally once every configuration was evaluated, instead of failing with `WorkerFailedToGetPendingTrialsError`.
- Parameter plots draw the power-law fit as a straight line instead of steps.


### Removed
- **BREAKING:** The HPO optimizers that scaling studies do not use: `successive_halving`, `hyperband`, `mo_hyperband`, `asha`, `moasha`, `async_hb`, `priorband`, `ifbo`, `pibo`, `primo`, `scalefm_hyperband`, `scalefm_priorband`, `scalefm_regularized_evolution` and `scalefm_local_and_incumbent`. `bayesian_optimization`, `random_search`, `grid_search`, `complex_random_search`, `scalefm_random_search`, `carbs` and `chinchilla` remain.
- **BREAKING:** Multi-objective optimization. `objective_to_minimize` must be a single float, and `bayesian_optimization` no longer takes a `reference_point`.
- **BREAKING:** `worker_fidelities_to_spend` and `total_fidelities_to_spend` of `scalefm.acquire()`, and the checkpoint helpers `load_checkpoint`, `save_checkpoint` and `load_lightning_checkpoint`.
- **BREAKING:** `scalefm.plot()` and `python -m scalefm.plot`, the incumbent trajectory plots carried over from NePS.
- The NePS tutorials, the basic usage and ask-and-tell examples, and the examples on expert priors, multi-fidelity, multi-objective, freeze-thaw and image segmentation HPO.
- The `ifbo` and `seaborn` dependencies.

ScaleFM was forked from [NePS](https://github.com/automl/neps) 0.17.0. The earlier history is in [CHANGELOG_NEPS.md](CHANGELOG_NEPS.md).

[0.2.0]: https://github.com/automl/scalefm/compare/v0.1.0...HEAD
