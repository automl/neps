# Changelog

All notable changes will be documented in this file.


## [unreleased]

### 🚀 Features

- Import pre-evaluated trials (#244)
- Remove global (cross-worker) stopping criteria (#252)

### 🐛 Bug Fixes

- Update contributors (#240)
- Remove .trace.lock (#243)
- Remove write_summary_to_disk (#249)
- Update ifbo dependency version in pyproject.toml (#254)

### 📚 Documentation

- Enhancing analysis docs discoverability (#245)

### ⚙️ Miscellaneous Tasks

- Cache uv prune unused wheels (#241)

## [0.14.0] - 2025-11-30

### 🚀 Features

- Add MOASHA - multi-objective brackets for promotion using MO strategies
- Add MOASHA - add epsnet MO promotion strategy
- Add MOASHA - add optimizer in algorithms and modify bracketoptimizer
- Add MOASHA - add MOASHA to test neps state
- Add MO_Hyperband
- Add Multiobjective algorithms: syne-tune apache license
- Add Multi-objective Bayesian Optimization using qLNEHVI
- Fidelity and prior compatibility checks in optimization algorithms (#212)
- Upgrade to Python 3.13 (#221)
- Multi-objective algorithms (#222)
- Update api with MO and fix runtime issues
- Add primo
- Add primo
- Add primo
- Add primo to algorithms
- Change tests for MO
- Add PriMO (#228)
- Update primo
- Allow None confidence centers for MO priors
- Modify tests for MOMF opts
- Fix ruff-format in status.py
- Fix more ruff-formatting
- Fix Async and HB brackets
- Txt files fidelity stopping crit (#230)
- Change Worker ID (#235)
- Async Saving Evaluation Result (#229)

### 🐛 Bug Fixes

- Tensor decoded values overflow for parameters with log with torch.float64
- Tensor decoded value overflow for parameters with log (#215)
- Insert pibo in requires prior in tests
- Set default value for ignore_fidelity in grid_search function
- Improve error messages for prior usage in optimization functions
- Simplify error message formatting in random_search function
- Clarify prior parameter requirements and error handling in documentation
- Add noqa comments to suppress linting warnings in algorithms and bayesian optimization modules
- Add requires_prior test case
- Remove extras from prior test
- Trial_report not being updated in Ask-Tell (#208)
- Merge branch 'master' into neps-mo
- Partial priors in space
- Allow partial priors in priors test
- Allow partial priors (#224)
- 123-optimizers-multifidelity-algorithms-should-check-for-min_budget-0 (#225)
- Disable info_dict logging
- Disable info_dict logging
- Skip tests for PriMO for now
- HyperBand bracket priority
- Remove upper bounds clip warning
- Fix Multi-fidelity algorithms (#232)

### 📚 Documentation

- Update LICENSE (#209)
- Update Documentation (#211)
- Add ask and tell Documentations (#236)

## [0.13.0] - 2025-04-11

### 🚀 Features

- Ask and tell (#174)
- Added support for PyTorch Lightning in the DDP backend. (#162)
- Initial multi-objective support (#180)
- *(api)* Add warning if no stopping criterion are set (#185)
- Model based bracket optimizers (#181)

### 🐛 Bug Fixes

- Use absolute path for previous config location (#130)
- Deprecation warning over concat with empty dataframe (#132)
- Timeout set explicitly to 120 seconds, debug on env vars (#141)
- Debug env vars better (#142)
- *(runtime)* Filelock issues (#161)
- *(ifbo)* Transformation of categoricals and handling of unusual fidelity bounds (#184)
- *(space)* Populate prior (#188)
- *(space)* `Categorical` domain is `is_categorical=True` (#190)
- *(RandomSearch)* Use initial passed in value (#192)
- Allowing None as constant HP in yamls + minor fix to tblogger
- *(GridSearch)* Making config ordering deterministic
- *(ruff)* Removing comment and unused import
- Allowing None as constant HP in yamls (#195)
- *(ifbo)* Example and post-run plotting
- Ifbo plotting path and keys
- *(ifbo)* Example and post-run plotting (#196)
- *(ifbo)* Explicit path print on successful 3D plotting'
- *(ifbo)* Explicit path print on successful 3D plotting (#198)
- *(GridSearch)* Making config ordering deterministic (#199)
- Spelling mistakes
- Add mypy error ignore comments for portalocker lock module attributes, making pre-commit usable again
- Prevent prior value assignment for fidelity parameters in Float and Integer classes
- Prevent prior definition for fidelity parameters (#204)
- Add mypy error ignore comments for portalocker lock module attributes (#205)

### 💼 Other

- Improve message on how to deal with an error from a worker (#128)
- Provide a log message indicating summary data has been generated (#131)
- *(Optimizers)* Use `ask()` instead of two-stage `load_optimization_state()` and `get_config_and_ids()` (#146)
- *(SearchSpace)* Remove as many methods from SearchSpace as possible (#148)
- Remove unused deps (#172)
- *(brackets)* Lower the number of first stage samples for BO strat (#186)

### 🚜 Refactor

- Introduce Settings Class and Default() Arguments (#116)
- Modularize file state (#126)
- BO and ifBO (#134)
- Use `vulture` to remove dead code (#147)
- Rename `neps.XParameter` to `neps.X`
- Rename `neps.XParameter` to `neps.X` (#149)
- *(SearchSpace)* Removes a lot of methods from `SearchSpace` (#150)
- Renaming several functions (#166)
- A lot (#167)
- Simplify `Prior`, `Sampler` and `ConfigEncoder` construction (#182)
- *(space)* Remove `centers` and `priors` (#189)
- NePS tblogger feature refactoring (#200)

### 📚 Documentation

- Fix typo (#173)
- Added example for DDP with PyTorch Lightning
- Added native Pytorch DDP example with neps (#163)
- Added FSDP examples with Neps (#194)
- Updated optimizer.md to reflect recent changes to how optimizers are defined, passed and configured. Removed mentions of yaml-based customization.
- Real world examples (#201)
- Enhance optimizer documentation and structure (#202)

### 🎨 Styling

- Fixup ruff pre-commit (#191)

### ⚙️ Miscellaneous Tasks

- Add python 3.12 support (#158)
- Enforcing conventional commit naming in PR titles (#160)
- Switch from poetry to uv (#152)
- Enable uv cache (#171)
- Enable uv caching for github actions workflows (#176)
## [0.12.2] - 2024-07-09

### 💼 Other

- 0.12.2
## [0.12.1] - 2024-07-03

### 🚀 Features

- *(yaml)* Declarative usage examples + simplified yaml usage (#91)
- *(yaml)* Enable mixed usage for run_args and neps func arguments + new design for pre-load-hooks  + new design for searcher config (yaml) (#102)

### 🐛 Bug Fixes

- Use 3.8 compatibly type aliases
- *(rng)* Ensure new state is saved to path
- *(runtime)* Lock by default for
- *(runtime)* Boolean check for cost
- *(runtime)* Explicitly load previous results (#106)
- *(seeding)* Save seed state by default
- *(seeding)* Save seed state by default (#109)

### 💼 Other

- *(rng)* Use binarized formats for de/serialization
- No arbitrary object loading `torch.load`
- *(rng)* Use binarized formats for de/serialization of rng (#92)
- *(graph)* Minor optimizations to examples using graph parameters (#95)

### 🚜 Refactor

- *(runtime)* Partial update of state
- *(report)* Just two seperate classes
- *(runtime)* Partial update of state (#93)
- *(SearchSpace)* Switch to `clone()` for search space. (#94)
- Integrate pipeline space into run args yaml + new design for defining constant parameter (#96)

### 📚 Documentation

- Directly link to API docs in search_space docs
- Use function API instead of duplicating docs
- Move `neps.run()` to be the first recommended page
- Directly link to API in docs, remove repetitions
- Remove sleep from examples

### 🧪 Testing

- Fix test relying on boolean positionals
- *(locking)* Allow for a worker not to have evaluated
## [0.12.0] - 2024-05-02

### 🚀 Features

- *(yaml)* First implementation of using yaml from `run_args=`

### 🐛 Bug Fixes

- Broken import
- *(api)* No need to pass logger anymore

### 💼 Other

- Use Union for <=3.9 compatibility
- Use `Dict` in `TypeAlias` for <=3.8 compat
- Fix up some files, ignore others
- Fix import
- Version 0.12.0

### 🚜 Refactor

- *(runtime)* Integrate metahyper closely
- *(_Locker)* Use `poll` everywhere
- *(runtime)* Integrate metahyper as neps.runtime
- *(SearchSpace)* Improve validation and `==`

### 📚 Documentation

- Type fixes
- Document neps.runtime
- *(ruff)* Remove refernces to black
- Update Contributing
- *(CONTRIBUTING)* Add sections regarding linting
- Big documentation cleanup
- Use relative links instead of https ones
- Big documentation cleanup

### 🎨 Styling

- Ruff pass on types.py
- *(ruff)* Fix up linting errors with status dir
- *(ruff)* Fix up neps.__init__
- *(ruff)* Fixup `neps.plot`
- *(ruff)* Some progress towards neps.utils
- Cleanup pyproject.toml
- *(ruff)* Cleanup common.py
- *(ruff)* Remove references to pylint

### ⚙️ Miscellaneous Tasks

- Add concurrency group to save resources on github action tests
- Run tests on any push/PR to master
- *(ruff)* Add ruff config
- Add ruff as dev dep
- *(ruff)* Ignore all files
- Fix ruff versioning
- Add action to run pre-commit
- Update pre-commit yaml
- *(mypy)* Limit to just src dir
- *(pre-commit)* Re-enable pre-commit
## [0.11.0] - 2023-12-08

### 💼 Other

- Remove overly restrictive dependancies
- Export `py.typed` for `neps` + `metahyper`
## [0.7.0] - 2022-12-18

### 🐛 Bug Fixes

- *(locking)* [**breaking**] Change to cross-plat portalocker
- Try unlock instead of lock for windows segfaults
- `os.setsid` not available on windows

### 🧪 Testing

- *(metahyper)* Add test for filelocking

### ⚙️ Miscellaneous Tasks

- *(workflow)* Add dependabot for github actions
- *(workflow)* Update actions
- *(workflow)* Add macos and windows to tests
- *(CI)* Enable metahyper tests
- *(tests)* Trigger on update to workflows too
- *(tests)* Fix mark selection for pytest
## [0.6.0] - 2022-10-22

### 💼 Other

- Regression tests
## [0.2.0] - 2022-01-24
