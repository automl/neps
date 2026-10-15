# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `autoscan.run(cost_estimator=..., max_cost_per_config=...)` for scaling studies. The cost estimator is a function or, for finite spaces, a JSON table. It fills in the `cost` of evaluations that do not report one and is handed to optimizers inheriting from `ScalingOptimizer`.
- `carbs` optimizer, wrapping [CARBS](https://github.com/imbue-ai/carbs).
- `chinchilla` optimizer for [Chinchilla](https://arxiv.org/abs/2203.15556) guided scaling studies.

### Changed
- **BREAKING:** Forked from NePS 0.17.0 and renamed the package `neps` → `autoscan` (import name, CLI, `AUTOSCAN_*` env vars, optimizer names, default result dirs).
- `bayesian_optimization` passes the space's constants to `constraints_func`, like `random_search` does.


## [0.1.0] - <date>
- First AutoScAn release, based on NePS 0.17.0.

AutoScAn was forked from [NePS](https://github.com/automl/neps) 0.17.0. The earlier history is in [CHANGELOG_NEPS.md](CHANGELOG_NEPS.md).

[unreleased]: https://github.com/automl/auto_scan/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/automl/auto_scan/releases/tag/v0.1.0
