# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add changelog and the workflow

### Changed
- Use git-cliff for tracing changes
- Update ifbo dependency version in pyproject.toml
- Remove global (cross-worker) stopping criteria
- Enhancing analysis docs discoverability
- Import pre-evaluated trials
- Cache uv prune unused wheels

### Fixed
- A test
- Update ifbo dependency version in pyproject.toml
- Docs & remove .trace.lock
- Update contributors

### Removed
- Remove write_summary_to_disk

## [0.14.0] - 2025-09-16

### Added
- Add ask and tell Documentations
- Add ask and tell example
- Add primo to algorithms
- Add primo
- Add primo
- Add primo
- Added trajectory and best incumbent. Solved warning in plot
- Added documentation and logging of incumbent
- Add requires_prior test case
- Add noqa comments to suppress linting warnings in algorithms and bayesian optimization modules
- Add tests fail comment

### Changed
- Bump version: 0.13.0 → 0.14.0
- Update Contributors
- Async Saving Evaluation Result
- Change Worker ID
- Make defualt worker-id shorter
- Make worker id chosen by user
- Txt files fidelity stopping crit
- Fix Async and HB brackets
- Numpy>=2
- Modify tests for MOMF opts
- Allow None confidence centers for MO priors
- Update primo
- Add PriMO
- Change tests for MO
- Update api with MO and fix runtime issues
- Numpy version update
- Example fixed
- Fixed tests
- Fixed trace error in case of multiple runs
- Logging messages changed
- Merged branch with new txt files
- Merge remote-tracking branch 'origin/master' into trace_csv_file
- Neps.plot fixed
- Rename overwrite_working_directory => overwrite_root_directory
- Fixed multi-fidelity stopping criteria
- Optimizer - fidelities to spend
- Mf example
- Introduction of fidelities_to_spend
- Renaming max_cost_total to cost_to_spend
- Renaming max_evalutaions_total to evaluations_to_spend
- Multi-objective algorithms
- Upgrade to Python 3.13
- Fidelity and prior compatibility checks in optimization algorithms
- Update Documentation
- Enhances fidelity handling in optimization algorithms
- Assign trial_report to trial in tell and tell_custom
- Add Multi-objective Bayesian Optimization using qLNEHVI
- Add Multiobjective algorithms: syne-tune apache license
- Add MO_Hyperband
- Add MOASHA - add MOASHA to test neps state
- Add MOASHA - add optimizer in algorithms and modify bracketoptimizer
- Add MOASHA - add epsnet MO promotion strategy
- Add MOASHA - multi-objective brackets for promotion using MO strategies
- Clip decoded value oveflow after torch.exp
- Update LICENSE
- Update LICENSE

### Fixed
- Fix Multi-fidelity algorithms
- HyperBand bracket priority
- Skip tests for PriMO for now
- Fix more ruff-formatting
- Fix ruff-format in status.py
- Disable info_dict logging
- Disable info_dict logging
- Fix CONTRIBUTING.md headers
- 123-optimizers-multifidelity-algorithms-should-check-for-min_budget-0
- Allow partial priors
- Allow partial priors in priors test
- Partial priors in space
- Merge branch 'master' into neps-mo
- Trial_report not being updated in Ask-Tell
- Clarify prior parameter requirements and error handling in documentation
- Simplify error message formatting in random_search function
- Improve error messages for prior usage in optimization functions
- Set default value for ignore_fidelity in grid_search function
- Insert pibo in requires prior in tests
- Tensor decoded value overflow for parameters with log
- Tensor decoded values overflow for parameters with log with torch.float64

### Removed
- Remove upper bounds clip warning
- Removed info_dict from logging
- Removed comment
- Remove cli
- Remove roadmap and fix contributing docs
- Remove extras from prior test

## [0.13.0] - 2025-04-11

### Added
- Adding only incumbent tblogger and docs
- Add mypy error ignore comments for portalocker lock module attributes, making pre-commit usable again
- Add image, removed code from README
- Added image-segmentation example
- Added image-segmentation pipeline example
- Added landing page
- Added FSDP examples with Neps
- Added FSDP examples with Neps
- Add warning if no stopping criterion are set
- Added native Pytorch DDP example with neps
- Added example for DDP with PyTorch Lightning
- Added support for PyTorch Lightning in the DDP backend.
- Added BOHB tips
- Add more details and links
- Add equation support
- Add more content and image captions
- Add images and SH explanation
- Added ifBO, changed graph
- Added diagram
- Added optimizer pages outline
- Add python 3.12 support
- Add ifbo
- Add cli to declarative usage docs + add declarative usage to README
- Add run-pipeline to neps run cli argument
- Add template of 'run_args' to neps init + add pipeline_space to neps run arguments

### Changed
- Bump version: 0.12.2 → 0.13.0
- NePS tblogger feature refactoring
- Updating Docs
- Fix exception handling with tblogger
- Handle TensorBoard tracking errors and disable feature
- Running end trial callbacks under lock
- NePS tblogger feature refactoring
- Enhance optimizer documentation and structure
- Real world examples
- Updated optimizer.md to reflect recent changes to how optimizers are defined, passed and configured. Removed mentions of yaml-based customization.
- Finished restructuring algorithm pages
- Catch up optimizers page with changes to neps.run
- Landing page for algorithm section
- Merge remote-tracking branch 'origin/master' into optimizers_documentation
- TOC in Getting Started
- Cut down MF and Prior explanation
- Expandable boxes
- Algorithms as chapter point
- Whats next section in getting started page
- Convert pngs to jpgs
- Merge remote-tracking branch 'origin/master' into optimizers_documentation
- Lower the number of first stage samples for BO strat
- Model based bracket optimizers
- Simplify `Prior`, `Sampler` and `ConfigEncoder` construction
- Initial multi-objective support
- Added support for PyTorch Lightning in the DDP backend.
- Added native Pytorch DDP example with neps
- Merge master into ddp-pytorch-lightning
- Enable uv caching for github actions workflows
- Use env var to share config with higher rank workers
- Hacky fix for multiple DDP launches
- Reformatted algorithm names
- Filled MF and MF+Prior pages
- Small fixes and ideas
- Ask and tell
- Bump ytanikin/PRConventionalCommits from 1.2.0 to 1.3.0
- A lot
- Enable uv cache
- Renaming several functions
- Changed equation optics
- BO page
- Prior explanation
- Rename default to prior
- Restructuring optimizers page, inserting low-level-MF and prior explanations
- Newline in optimizers.md
- More detail and fixes for optimizer pages
- Switch from poetry to uv
- Enforcing conventional commit naming in PR titles
- Improve examples
- Update roadmap
- Update citations
- Improve conciseness of README
- Fix structuring of CONTRIBUTING docs
- Fixed `priorband_template.py` example
- Fixed pytest with new fidelity parameter checks
- Fixed `priorband_template.py` example
- New commands for CLI: sample-config, report-config, init, results ; Refactoring CLI
- Update roadmap
- Rename `neps.XParameter` to `neps.X`
- Rename `neps.XParameter` to `neps.X`
- Remove as many methods from SearchSpace as possible
- Use `vulture` to remove dead code
- Use `ask()` instead of two-stage `load_optimization_state()` and `get_config_and_ids()`
- BO and ifBO
- New cli commands help, info-config, errors, status, results + Documentation
- Refactor Declarative Usage (CLI) + Documentation
- Enable pipeline_space keyword in seperate yaml
- Change run-args to optional argument for neps run
- Provide a log message indicating summary data has been generated
- Improve message on how to deal with an error from a worker
- Modularize file state
- Introduce CLI
- Derive type(int, float, str) from str input of searcher-kwargs cli
- Introduce cli neps (init, run)
- Introduce Settings Class and Default() Arguments

### Fixed
- Add mypy error ignore comments for portalocker lock module attributes
- Prevent prior definition for fidelity parameters
- Prevent prior value assignment for fidelity parameters in Float and Integer classes
- Spelling mistakes
- Making config ordering deterministic
- Explicit path print on successful 3D plotting
- Explicit path print on successful 3D plotting'
- Example and post-run plotting
- Ifbo plotting path and keys
- Example and post-run plotting
- Allowing None as constant HP in yamls
- Removing comment and unused import
- Making config ordering deterministic
- Allowing None as constant HP in yamls + minor fix to tblogger
- Use initial passed in value
- Fixup ruff pre-commit
- `Categorical` domain is `is_categorical=True`
- Populate prior
- Fixed bracket display
- Transformation of categoricals and handling of unusual fidelity bounds
- Fixed math display
- Fix headings
- Fix typo
- Filelock issues
- Fix typo in README
- Fix indendation in roadmap
- Debug env vars better
- Timeout set explicitly to 120 seconds, debug on env vars
- Fix np converting lists to np types
- Deprecation warning over concat with empty dataframe
- Use absolute path for previous config location

### Removed
- Remove unused files
- Remove `centers` and `priors`
- Remove unused deps
- Removes a lot of methods from `SearchSpace`

## [0.12.2] - 2024-07-09

### Added
- Add example for running neps on slurm scripts

### Changed
- 0.12.2
- Update pyproject.toml
- Refactor declarative usage
- Merge of master
- Improve rendering of README and docs landing page
- Improve README.md and update docs landing page

## [0.12.1] - 2024-07-03

### Added
- Add tests for checking optimizer loading via run_args
- Add python -m neps_examples utility to print examples
- Add torchvision to non-dev dependencies
- Add tests + fix docs
- Add searcher_args to searcher_info for custom class optimizer loaded via yaml
- Add providing arguments for loaded class BaseOptimizer via yaml + fix pre-commit related errors
- Add docstring rm solved ToDos
- Add mixed usage functionality for run_args and neps func arguments

### Changed
- Bump version from v0.12.0 to v0.12.1
- Allow use of neps.Float instead of neps.FloatParameter
- Only log searcher as info level, log searcher_alg to debug
- Set post_run_summary to True always
- Fix incumbent tblogger bug
- Minor code fix + comment
- Minor change in test
- Minor change in test
- Update to master
- Readd sleep to hyperparameters example
- Maintain roadmap
- Move examples lists from test to neps_examples
- Improve readme, fix broken link, make slimmer
- Maintain roadmap
- Improve README writing, fix links, remove dead weight
- Minor bug fix
- Bug fix
- Have PiBO as an explicit searcher
- Change path of yaml test directories to tests_tmpdir
- Change path of yaml test directories to tests_tmpdir
- Change boolean values in yamls from [True, False] to [true, false]
- Update declarative example
- Code clean up + add notes to docs
- Clean up code and docsctrings
- Adapt SearcherConfigs to the new dict design of optimizers
- Update docs
- Update rm searcher_kwargs key from yaml for user
- Update rm searcher_kwargs key from yaml for user
- Change algorithm to strategy
- Change searcher key algorithm to strategy + rm searcher_kwargs argument
- Define dict in run_args yaml
- Update docs to the new design
- Merge master + fix loading neps-searchers from yaml
- Rm neps argument searcher_path
- Change design for searcher config
- Simplify code
- Change design for pre-load-hooks
- Align the tests for declarative usage to the ones that are in the docs + try fix test
- Update pyproject.toml for PyTorch versions
- Fix AttributeError: 'int' object has no attribute 'nodes'
- Cast Path to str
- Update pyproject.toml
- Update pyproject.toml
- Update pyproject.toml for PyTorch versions
- Update precision of best loss in logs
- Enable mixed usage for run_args and neps func arguments + new design for pre-load-hooks  + new design for searcher config (yaml)
- Integrate pipeline space into run args yaml + new design for defining constant parameter
- Minor optimizations to examples using graph parameters
- Switch to `clone()` for search space.
- Declarative usage examples + simplified yaml usage
- Partial update of state
- Just two seperate classes
- Partial update of state
- Use binarized formats for de/serialization of rng
- Remove sleep from examples
- No arbitrary object loading `torch.load`
- Use binarized formats for de/serialization
- Directly link to API in docs, remove repetitions
- Move `neps.run()` to be the first recommended page
- Use function API instead of duplicating docs
- Directly link to API docs in search_space docs

### Fixed
- Fix yaml dumping and default args selection for optimizer loaded via run_args
- Fix double logging and move post_eval_hook to runtime
- Save seed state by default
- Save seed state by default
- Fix path reference
- Fix test
- Fix pre-commit
- Fix pre-commit error
- Fix pipeline_space example
- Fix errors in tests
- Explicitly load previous results
- Allow for a worker not to have evaluated
- Boolean check for cost
- Lock by default for
- Ensure new state is saved to path
- Use 3.8 compatibly type aliases
- Fix test relying on boolean positionals

### Removed
- Remove arg from examples
- Remove unnecessary torchvision requirement
- Remove neps argument 'searcher_path' and yaml argument searcher_kwargs + New loading design for optimizer

## [0.12.0] - 2024-05-02

### Added
- Add ruff as dev dep
- Add ruff config
- Add valid type check for choices in categorical parameter
- Add logging info for parameter that uses e notation to inform them that the parameter gets deduced as float
- Add global variable to handle path location
- Add comment to check_arg_defaults
- Add type hints for arguments and returnn for all run_args functions
- Add cli neps(init, run) + modify custom optimizer usage, providing just a class
- Add tests to check functionality within neps.run() when using run_args.yaml to provide arguments + comment code
- Adding sleep time to inaccessible locks for summary_csv and sampler_info

### Changed
- Version 0.12.0
- Big documentation cleanup
- Use relative links instead of https ones
- Big documentation cleanup
- Re-enable pre-commit
- Limit to just src dir
- Add sections regarding linting
- Update Contributing
- Fix up some files, ignore others
- Remove refernces to black
- Remove references to pylint
- Cleanup common.py
- Cleanup pyproject.toml
- Update pre-commit yaml
- Add action to run pre-commit
- Some progress towards neps.utils
- Fix up linting errors with status dir
- Document neps.runtime
- Ruff pass on types.py
- Ignore all files
- Improve validation and `==`
- Improved error feedback for wrong path
- Change search_space: to pipeline_space: for tests
- Update to master
- Log boolean check
- Update __eq__ function for parameters to use it to cover test cases
- First implementation of using yaml from `run_args=`
- Switch to relative import for module utils
- Implement strict argument usage check; raise errors as needed
- Minor fix
- Reorganizing doc structure for running NePS
- Update neps_run.md
- Update neps_run.md
- Doc fix for run_arg usage
- Rm cli command
- Delete cli
- Enhanced usability by supporting module paths with and without '.py' extension + general improved error feedback
- Rework of the yaml_usage examples + adding readme for guidance
- Documentation for run_args
- Rework of some of the code + comment and doc string changes
- Increase code maintenance + enabled strict usage of run_args, overwritting all other arguments that are provided via neps.run(...) + Enable loading the pipeline_space provided as a dictionary from YAML and the loading of the BaseOptimizer as searcher from YAML
- Implementation of run_args, loading and checking settings from yaml + tests
- Integrate metahyper as neps.runtime
- Type fixes
- Use `Dict` in `TypeAlias` for <=3.8 compat
- Use Union for <=3.9 compatibility
- Use `poll` everywhere
- Integrate metahyper closely
- Merge pull request #72 from automl/ci-concurreny-groups
- Run tests on any push/PR to master
- Add concurrency group to save resources on github action tests
- Update author list

### Fixed
- Fix import
- No need to pass logger anymore
- Fix ruff versioning
- Fixup `neps.plot`
- Fix up neps.__init__
- Fix link in docs
- Fix issue #35 + boolean checks for log and is_fidelity
- Fix issue #35
- Fix wrong key name for example search space yaml
- Fix type hints for 3.8
- Fix type hints for 3.8
- Fix test + add run_args: constraint structure for yaml file
- Fix python environment call in windows for test cases
- Fix module import BaseOptimizer
- Fix provided arguments check, special case searcher-pipeline_space
- Fix provided arguments check
- Fix provided arguments check
- Broken import
- Fix example

### Removed
- Removed unfinished declerative docs to another branch + added missing usage information for providing pipeline_space to neps.run()

## [0.11.1] - 2024-02-19

### Added
- Adding first iteration to the  run_pipeline documentation
- Add Examples to Readme
- Added citation + analyse.md iteration
- Add comment in code example
- Add comment in code example

### Changed
- Bump version from v0.11.0 to v0.11.1
- Updating run_pipeline docs
- Updating docs for citations
- Doc link fixes + some minor text changes
- Merge pull request #56 from automl/configspace-patch-1
- Update pyproject.toml
- Documentation iteration + minor tblogger example change
- Merge pull request #50 from automl/next_doc_iteration
- Adjusting summary_csv with task_id and dev_stage_id args
- Big Documentation Update + align naming of configspace for yaml usage
- Closes #52: summary_csv compatible with development_stage_id
- Merge pull request #34 from automl/readme_changes_danrgll
- Updating parallelization documentation and README links
- Update branch to current master
- Fixing merge conflicts with master
- Enhancing README: Adding paper links as insights to key features
- Landing page README iteration + installation guide
- Update README.md
- Iterating analyse.md; format + doc changes
- Update README.md with additional key features and improved structure
- Update README.md
- Update citations.md
- Update README.md
- Tblogger doc iteration + adding images
- Merge remote-tracking branch 'origin/master' into readme_changes_danrgll
- Tblogger doc iteration + light/dark mode doc switch
- Iterating through examples README + tblogger doc
- Fixing conflicts from main and README iteration
- Change format
- Change format
- More compact format
- Change order of documentation section
- Adapt Indentation
- Changes for code usage example
- Changes for code usage example
- Merge pull request #48 from automl/dependabot/github_actions/actions/setup-python-5
- Bump actions/setup-python from 4 to 5
- Merge pull request #46 from automl/yaml_config_space
- Update to current master
- Changing root README.md Citations link to docs

### Fixed
- Fix issue #44

## [0.11.0] - 2023-12-08

### Added
- Add citations to the documentation
- Adding edits from #34 from @danrgll and @TarekAbouChakra
- Add type specification for arguments + add more detailed DocStrings for paramter validation functions
- Add examples + test adaptation to new functionalities + outsorcing utils for yaml search space in own file
- Add usage of e and 10^ for exponent format as input for yaml file + new tests for optional type argument
- Add documentation for config_space
- Adding callable function for post-run csv analysis
- Add tests, adapt pipeline_space_from_yaml(), fix some errors raised by pre-commit in search space
- Add yaml to dict function and make it callable in api run
- Adding .optimizer_info.yaml checks for catching user errors
- Adding post-run summary files + testing
- Added whoami function and a test case for searcher as BaseOptimizer
- Added return values

### Changed
- Bump version from v0.10.0 to v0.11.0
- Fixing roadmap version
- Updating roadmap pre-release
- Merge pull request #47 from automl/Neeratyoy-patch-1
- Update README.md
- Enable usage of Path object for yaml_file config_space
- Merge master into branch
- Fixing flattening merge issues between src/ and neps/
- Merge pull request #45 from automl/repo-flattening
- Merge remote-tracking branch 'origin/master' into repo-flattening
- Minor file import changes for clarity and consistency
- Updating pyproject.toml with new repo directories location
- Repo flattening + adjustments
- Merge pull request #36 from automl/typing-export-types
- Export `py.typed` for `neps` + `metahyper`
- Resolves #26; resolving deprecation warning
- Made code more readable for validate parameter inputs + add tests + comment functions
- Changes in yaml_search_space examples
- Resolve merge conflicts
- Change tests to marker neps-api and resolve merge conflicts
- Merge pull request #33 from automl/api-checks
- Better testing + api checks before starting the search
- Fixing merge conflicts with master
- API refactoring + API tests and checks
- Clean up tests
- Summary_csv test fix
- Iterate over summary_csv, fix warning, and convert strings to lowercase
- Adapt tests and documentation for yaml_search_space
- Adapt tests and documentation for yaml_search_space
- Merge pull request #17 from automl/dependabot/github_actions/actions/checkout-4
- Bump actions/checkout from 3 to 4
- Minor tblogger image logging bug fix
- Fix merge error and move MissingDependencyError into metahyper
- Fixes #24 for the next release
- Merge pull request #38 from automl/deps-remove-pandas-upperbound
- Remove overly restrictive dependancies
- Enable to run tests for yaml
- Merge pull request #22 from automl/csv_summary
- Refactoring unit test + stochastic failure debug
- Minor update to neps/status/status.py
- Minor updates to neps/api.py
- Resolving merge conflicts with master
- Minor updates
- Fixing multi-thread .optimizer_info.yaml using decision_locker
- Tblogger example iteration
- Tblogger example iteration
- Merge pull request #21 from automl/tabular-space-support
- Merge conflict resolution
- Removing neps api test from GitHub actions
- Minor cleanup
- Allowing full table sampling in acquisition
- Merging updates
- Budget id >= 0
- Minor clean up, and suggested changes
- Increase pytest timeout to 15mins
- Minor logic updates
- Removing stray iimport
- Making PFN conditioned on local availability
- Merge resolution + removing timing code
- Merging with master
- Tabular support + DyHPO - joint commit with @karibbov
- Merge remote-tracking branch 'refs/remotes/origin/master'
- Merge remote
- Set normalize_y to False
- Utility functions update + example templates iteration
- Reducing tblogger example runtime
- Update releasing.md

### Fixed
- Fix test cases
- Fix format of Pipeline Space Documentation for mkdocs
- Fix naming of parameters in test
- Fix issue regarding tests and search space from yaml file

### Removed
- Remove 10^ notation + introduce key checking for parameters +  enable e notation usage for all parameters and arguments
- Remove extra pop

## [0.10.0] - 2023-11-06

### Added
- Adding warning filter for tensorboard in pytest
- Adding Py3.11 test on Mac
- Add support for python 3.11
- Add cloning to docs of contributor install
- Adding utility functions for checkpoints and the lightning template
- Adding priorband template
- Adding template for basic neps usage
- Add input-output scaling to DeepGP, changes from Heri
- Add functionality to MFObservedData
- Added ignore_missing_defaults option to sample_default_configuration()

### Changed
- Bump version from v0.9.0 to v0.10.0
- Change license from MIT to Apache-2.0
- Increasing GitHub pytest timeout from 10 to 15 minutes
- Minor typo fix
- Removing argparse from tblogger example
- Shortening tblogger example runtime
- Removing argument parsing in Tensorboard example
- Template update + torchvision in deps
- Including new examples for testing
- Adjusting tblogger tutorial
- Adjustments for Tensorboard and saving configuration data from Metahyper
- Fixing PyTorch2.0 issue with pyproject
- Updating dependency, torch version
- Exclude macos python 3.11 test for now
- Update author order in CITATION
- Update author order
- Reformatting optimizer_info yaml with fewer restrictions on Metahyper
- Merge pull request #18 from automl/yaml-api
- Stricter conditions on the optimizer
- Iterating yaml api PR
- Neps with pytorch lightning example
- Making new markers for the yaml_api tests
- Handle Failure mode of the DeepGP: use untrained model
- Tabular SearchSpace for MFEIBO
- Merge remote-tracking branch 'refs/remotes/origin/master'
- DeepGP checkpoint and refine
- Minibatch DeepGP training
- Yaml test cases and minor fixes
- Api yaml new feature
- Merge remote-tracking branch 'origin/master' into tensorboard-neps
- Minor tblogger example changes
- Minor tblogger example changes
- Record learning curves explicitly
- Expose DeepGP parameters, make early stopping optional, disable default normalize_y, add TODOs
- Avoid duplicates when sampling new configs
- Merge pull request #14 from automl/tensorboard-neps
- Tblogger cleanup
- Merge remote-tracking branch 'origin/master' into tensorboard-neps
- Merge remote-tracking branch 'origin/master' into tensorboard-neps
- Tblogger changes in class and example
- Merge remote-tracking branch 'origin/master' into tensorboard-neps
- Fixing issues in tblogger and example
- Tensorboard example and class minor changes
- Tensorboard functionality
- __future__.annotations
- Deep gp with MFEIBO
- Merge remote-tracking branch 'refs/remotes/origin/master'
- Update init design for MF-EI-BO
- Bug fix to MFI + budget-based initial design
- Initial deep gaussian process
- Merge fix
- Simplify MFEIBO get_config_and_ids, anf mf_ei.py
- Fixing issues with MFEIBO
- Update __init__.py
- Dyhpo Acquisition Function MF-EI implementation
- Minor cleanup
- Cleaning changes to MF-EI
- Synthetic Regression Tests added
- Initial MF-EI changes
- Regularized Evolution default assisted = False
- Resolve python version version based issues
- DyHPOBase draft version, with a random acq
- Bug fix in set_hyperparameters_from_dict() for ConstantParameter
- Bug fix in set_hyperparameters_from_dict()
- Modified set_hyperparameters_from_dict() to allow setting it from another search space as well
- Changed data_loading utility functions to be more flexible and use json instead of yaml
- Tensorboard Evaluation File
- Removing deprecated import
- Removing deprecated import
- Removing deprecated files
- Bug fix: log transform with log as False

### Fixed
- Fixing optimizer yaml file
- Fixing tblogger example

### Removed
- Remove support for python 3.7 (end of life)
- Remove metahyper doc stubs
- Remove the budget_id column from observed data

## [0.8.4] - 2023-05-03

### Changed
- Bump version from v0.8.3 to v0.8.4
- Updating PriorBand defaults
- Docstring update
- Simplifying PB decay
- Un-exclude macos tests
- Adjust numpy version
- Un-exclude some windows / macos tests
- Adjust torch version for macOS, add OS classifiers
- Bound torch version to try fix macOS issue
- Allowing prior-based MF algos to sample default at max budget
- Fix poetry version for github actions
- Bump poetry version for github actions
- Merge pull request #9 from automl/dependabot/github_actions/abatilo/actions-poetry-2.3.0
- Bump abatilo/actions-poetry from 2.1.6 to 2.3.0
- Run pre-commit on all files

### Removed
- Remove continuation of crashed configs from metahyper
- Remove torch version constraint

## [0.8.3] - 2023-03-21

### Added
- Add -u flag to publish documentation

### Changed
- Bump version from v0.8.2 to v0.8.3
- Bug fix in SH
- Fixing failing test case for mutation on IntParam
- Replace prints with logger

## [0.8.2] - 2023-03-18

### Added
- Add eddie to CITATION.cff
- Add Eddie as author
- Adding mutation as incumbent sampler
- Add utils and prototype optimizer for meta_neps
- Adding PriorBand+BO
- Add result hallucination for pending

### Changed
- Bump version from v0.8.1 to v0.8.2
- Adjust sleep in example
- Updating PB decay method
- Exposing prior confidence param in BO
- Typo fix
- Bug fix: IntParam mutable object not updating
- Ignore *.DS_Store in .gitignore
- Bug fix in set_hyperparameters_from_dict() preventing None as default with has_prior=True
- Updating mutation function flow per HP + local mutation func. in PB utils
- Manual merge of changes from hnas branch
- Fixing MF-BO joint 2-step AF
- Bug fix: PB BO joint
- Revamping MF BO
- Minor update to PB
- Fixing log transform trigger for float HPs
- Major update to PriorBand family + Bug fixes
- Async PriorBand param update
- Setting default log prior weights to False
- MFBO data subset bug fix + PB decay_t
- PriorBand design changes for ablation
- Fixing baseline logic
- Updating HB+inc baseline
- Minor update to dynamic inc sampling
- Changes to mobster
- Minor logic update in inc activation (PriorBand)
- Code refining and cleaning (mf-priors)
- Minor update to SH sampling default
- Fixing import from merge conflict
- Merging to master a highly diverged commit

### Fixed
- Fix missing pyyaml dependency
- Fix example links

## [0.8.1] - 2023-01-08

### Added
- Add small fixes for repetitive architecture spaces

### Changed
- Bump version from 0.8.0 to 0.8.1

## [0.8.0] - 2022-12-29

### Changed
- Bump version from 0.7.0 to 0.8.0
- Clean architecture and hps example
- Clean architecture example
- Clean up NAS api further
- Ignore grakel warning
- Rework NAS api
- Update roadmap

### Fixed
- Fix __init__ structure

## [0.7.0] - 2022-12-18

### Added
- Add mobster
- Added delete_previous_defaults anf delete_previous_values options to set_hyperparameters_from_dict()
- Added function set_default_confidence_score to float, integer and categorical
- Add get_summary_dict
- Add budget stop criterion to multi_fidelity
- Add fixing of autograd warning to roadmap
- Add rename proposal to roadmap
- Adding async HB baseline
- Adding async PB exp
- Adding PB to async SH/HB
- Add incumbent plotting
- Adding HB + inc sampling
- Adding sampling from hypersphere
- Add convenience examples and fix example links
- Add parallelization example and remove pipeline_dir from basic_usage
- Add metahyper to this repository directly
- Adding HB,ASHA,ASHA-HB based PriorBand versions
- Adding NePS HB with PriorBand ensembling
- Adding Ensemble Policy and custom HB default

### Changed
- Bump version from v0.6.1 to v0.7.0
- Merge pull request #4 from automl/remove_fnctl_dependancy
- Update roadmap
- Exclude macos 3.7 and 3.8 tests
- Bug fix in set_hyperparameters_from_dict() for categorical parameters
- Improve plotting sciprts
- Update roadmap
- Deactivate macOS 3.8 and 3.7 tests
- Do not create graph for nlml backward pass
- Exclude windows with python 3.10 as a gh-actions test
- Fix mark selection for pytest
- Trigger on update to workflows too
- Enable metahyper tests
- Add macos and windows to tests
- Update actions
- Add dependabot for github actions
- Updating PriorBand with inc sampling type and style + misc changes
- Minor cleanup
- Rename budget to max_cost_total, add to example
- Updating asyncPriorBand
- Fixes for parallel/async MF algos
- Update roadmap
- Improve error message for loss value on error
- Minor param update to async PB
- Fixing metahyper import
- Do not print summary when plotting
- Save previous_config_id to metadata.yaml
- Update old docstring
- Clean up metahyper, add analysis test
- Improve README text
- Minor change to plot structure
- Improve README
- Improve analyse example
- Improve documentation on analysis
- Improve examples readme
- Rework examples with respect to analysis
- Parallelization example as .md
- Improve plotting (now through neps.plot)
- Move AttrDict to common utils
- Revert example
- PriorBand exps and ablations
- Removing deprecated files
- Making new additions restructure compatible
- Make root_directory an arg not a kwarg
- Constrain the torch version due to installation issues
- Refactor examples
- Use pipeline_directory instead of working_directory in examples
- Improve speed of poetry install a bit
- Update to poetry 1.2 syntax
- Updating logs and minor formats
- Update roadmap

### Fixed
- Fix metahyper test
- Fix roadmap formating
- `os.setsid` not available on windows
- Try unlock instead of lock for windows segfaults
- Add test for filelocking
- Change to cross-plat portalocker
- Fix root dir name in example
- Fix examples readme
- Fix python compatibility issues for plotting
- Fix poetry version in github workflow
- Fix verison in citation.cff

### Removed
- Remove dill dependency
- Remove unnecessary metahyper files

## [0.6.1] - 2022-10-25

### Added
- Add python versions to test matrix
- Support python3.8/9/10

### Changed
- Bump version from v0.6.0 to v0.6.1
- Setting multi-fidelity default to model-free HB
- Merge fix
- Minor fixes to HB and asyncHB
- Making sync. SH and HB parallel
- Initial SH update
- Adjusted to 0.6.0
- Merge https://github.com/automl/neps into HEAD

### Fixed
- Fix gh actions semantics for python 3.10
- Fix install for 3.7 and 3.10
- Fix version constraint

## [0.6.0] - 2022-10-22

### Added
- Add ability to ignore errors
- Added ranomized PriorBand Policy
- Added random sparse promotion
- Added default sampling and made peaky prior a non-default setting
- Adding v6 with v5 variations
- Added ablation study functionality - regular sampling and promotion
- Added args for allowing ablation studies in raceband
- Add disable_prior flag to BO optimizer
- Adding variants to v5
- Added raceband - temporary name for possible Hyperband competitor
- Adding v5 to mf-priors
- Adding confidence and random sample HPs in mf-prior opts
- Added EI, toyed with bandwidths to emphasize prior
- Added Expected Improvement-like weighting to the TPE
- Adding random_interleave_prob to multi_fidelity
- Adding new optimizer to decay prior to uniform
- Adding new ASHA-AsyncHB opt
- Adding new opt with small update to Async HB
- Added adaptive correlation between fidelities for MF-TPE - only lacking
- Adding first few trial versions of mf-prior algos
- Added ability to remove samples that have been queried at multiple
- Added async support - registering pending configurations as bad in the
- Add Nils
- Adding func to track incumbent in SH
- Add mike to documentation for versioning support
- Add parallelization example to roadmap
- Add alternatives documenation and improve toc of docs
- Add Note section to readme
- Add carl to CITATION.cff
- Added KDE & Vanilla TPE along with example. KDE comes with additional
- Add constant to pipeline_space_from_confgispace
- Add comments explaining core vs all tests
- Adding Asynch HB with and w/o priors

### Changed
- Bump version from v0.5.1 to v0.6.0
- Jitter=1e-6 for gp.py
- Modified defaults and adjusted sampling according to discussion
- Making BO support sampling of default first
- Appending default as first sample param in MF algos
- Changed so that constants have default values, and so that
- Changed to enable vanilla Raceband, no priors
- Fixed minor budgeting bug
- Changed sampling procedure tolerance
- Changed Raceband to accept max budget and utilize sharper priors
- Fixed constant support and categorical distance bug
- Omit ConstantParameter in GP-opt
- Updating v4 with sync SH/HB changes
- Simplified RaceBand according to suggestions
- Fixed sampling tolerance in neighborhood sampling
- Fixed integer parameter bug in sobol -> config conversion
- Updating synchronous MF algos
- Updating categorical choice confidences
- Introduced learning between brackets
- Merging master
- Option for ultra confidence in mf-prior algos
- Changed some HPs of the KDE, and re-added some plotting utilities (that
- Minor readme edit
- Minor comment
- Updating async HB sampling of brackets/rungs
- Fixing/tuning v3_2
- Merge https://github.com/automl/neps into HEAD
- New opt with prior decay to RS
- Updates to dense graph allowing te be trained. small fixes in the API for graphs
- Now supporting Constant values (and ordinals, but those are considered
- For Prior-based MF algos defaulting to NePS confidence
- Reverted the prior strengths in SH to adhere to the convention
- Making SH priors peaky
- Regression tests
- Regression test's metrics changed
- Ensured to properly differentiate between rung and fidelity
- Fixing configspace mapping and RS parameterization
- Bug fix in compute_prior() for a ConstantParameter
- Categorical, Integer and log variables now all work. V1, should be ready
- Async soft promotions now work as intended. First version of algo done,
- Changed the mf_tpe example to not include categoricals (not quite
- Multi-fidelity promotions incorporated - still need to remove
- Ensured multifidelity splitting is done correctly in TPE, and modelled
- KDE now weights each point according to fidelity - still need to sort
- Option for prior confidence in SH/HB
- Syncing with master
- Started incorporating multifidelity funcitonality
- Update roadmap
- Document mike
- Improve documentation on testing and checking protocols
- Refine roadmap
- Refine roadmap
- Unbold documentation / examples
- Improve readme text organization
- Refactor search spaces sub directories
- Renamed KDE, fixed weighting, added some defaults
- Adjusted some parameters of the KDE, allowed for fitting of fixed
- KDE now runnable
- KDE now runs - fixed imports
- Accidentally changed in old optimizers.__init__.py file, reverted now
- Restructure neps_examples and simplifiy test_examples.py

### Fixed
- Fix docs
- Fix docs
- Fix links in docs
- Fix links in README
- Fixed cost-cooling when estimated costs are zero
- Fix typo in README
- Fix links in documenation
- Fix pyproject and add carl

### Removed
- Removed print statements
- Removed print statements
- Remove deprecated documentation part
- Remove torch installation utils
- Remove old plotting utils

## [0.5.1] - 2022-08-29

### Added
- Add changing of CITATION.cff to docs on version releases

### Changed
- Bump version from v0.5.0 to v0.5.1
- Initial HB commit
- Testing submodule + minor code cleanup
- Fixing failing example
- Fix failing test
- Bump version to 0.5.0 also in CITATION.cff
- Restructuring multi-fidelity - adding SH, ASHA with and w/o priors
- Restructuring multi-fidelity - adding SH, ASHA with and w/o priors
- Improve poetry documentation

## [0.5.0] - 2022-08-26

### Fixed
- Fix version from 0.4.10 to 0.5.0

## [0.4.10] - 2022-08-26

### Added
- Add roadmap to documentation
- Add function set_defaults_to_current_values() to search_space.py
- Add Samir
- Add collection of todos
- Add documentation link to README
- Add dev and task id to run() and pass them to metahyper.run()

### Changed
- Bump version from 0.4.9 to 0.4.10
- Move jahs_bench to dev dependency
- Deprecate working_directory in favor of root_directory for neps.run
- Move roadmap into contributing folder
- Improve readme and documentation
- Improve shields
- Change license
- Update link to documentation in README
- Improve contributing documentation and add releasing howto
- Regression tests
- Bug fix in set_defaults_to_current_values()
- Refactor CategoricalParameter outside numerical
- Working_directory -> root_directory
- Update documentation structure
- Updated poetry installation instructions
- Making error the default behavior for optimizer

### Fixed
- Fix configspace + pipelinespace suppor
- Fix docstrings
- Fix pyproject license
- Fix link to example

## [0.4.9] - 2022-05-31

### Added
- Add install from pypi
- Add authors and info
- Add random interleave prob in MF BO
- Add unparsing
- Add back handling of a new graph represenation (hierarchical)
- Add prior for multiple repetitive grammars

### Changed
- Ignore dist
- Rename project for pypi
- Small updates to take last combo op
- Create CITATION.cff
- Letting MF BO switch to BO after eta evals on max fidelity
- Implement loss_value_on_error and cost_value_on_error base_optimizer.py and all optimizer classes. Values can be passed to api.run() with **searcher_kwargs. status() and _post_evaluation_hook() always use default value inf.
- Introduce parameters loss_value_on_error and cost_value_on_error in result_utils.py and base_optimizer.py

### Fixed
- Fix pip install
- Fix date

## [0.4.8] - 2022-05-12

### Added
- Add handling of grid search exhaustion
- Add grid search to optimizers
- Add grid generation for search space
- Add prior to dense graph

### Changed
- Bump version
- Update RE config id
- True fix
- Bug fix
- Neps search space code improvements
- Lower distance for neighbours in crossover
- Enable crossover for HPs

### Fixed
- Fixed cost_cooling and updated test case

## [0.4.7] - 2022-05-07

### Changed
- Bump version, fix async MF
- Bump version, fix async MF
- Fixing async failure modes for MF BO
- Init design fix to MF + error handling if AF sample fails

## [0.4.6] - 2022-05-05

### Changed
- Bump version, fix asynch MF
- Set to max fidelity in AF optimization
- Small fixes
- Change prior decay and initial sampling, MF-BO bug fix
- Update constrained grammar
- Handle hierarchical graph extraction
- Implemented cost cooling
- More flexible towards constraints
- Extract graphs from new representation
- Fix matplotlib version
- Fix call to parent constructor in graph Parameters
- Raise NotImplementedError when not implemented
- Update regularized evolution

### Fixed
- Fix a bug in extract config

### Removed
- Remove prints
- Remove change

## [0.4.5] - 2022-05-02

### Added
- Adding toggle option for model-search and RS in MF BO

### Changed
- Bump version to 0.4.5
- Check for attribute existing

## [0.4.4] - 2022-05-02

### Added
- Add pibo functionality for constrained grammar
- Add fault tolerance example
- Add demonstration of working_directory and previous_working_directory

### Changed
- Bump version to 0.4.4
- Fix error with values not sampled when not using fidelities
- Put realistic lr search space in multi fidelity example
- Allow down to pytorch 1.7.0 + fix torch.linalg.cholesky
- Allow down to pytorch 1.8.0
- Debugging multi-worker setup
- Small fixes
- Debugging multi-worker setup
- MF BO fixed to handle pending evals
- Improve __repr__ lisibility
- Improve code for normalizing, include log in normalizing, add sample_batch to mutation_sampler
- Debugging MF BO again
- Debugging MF BO
- MF BO default performance as inf
- Fixing MF BO restarts
- Another quick fix
- Quick fix
- Remember where zero op is placed
- Disable fault_tolerance example test as it makes no sense
- Update constrained grammar

### Fixed
- Fix MF ASHA

## [0.4.3] - 2022-04-28

### Added
- Add cost cooling structure
- Adding get_cost utility
- Add handling of fidelity parameters
- Add HP normalization for the GP input
- Adding GP-UCB (with TODOs)
- Add set fidelity to maximum value for the acquisition function optimization
- Add *args to forward call

### Changed
- Bump version
- Trick to access the method from child
- Use searcher kwargs
- Commenting GP-UCB until tested locally
- MF BO allows optimization restarts
- Determine rungs wrt max_budget, add todos
- Do not apply decayingpriorweightedacq everytime
- Move kernel construction out of optimizer class
- Update BO optimizer
- Move NOTICE to source code

### Removed
- Remove surrogate model from acquisition functions init when still present (bug fix)

## [0.4.2] - 2022-04-22

### Added
- Add matplotlib
- Add id property to graph grammar

### Changed
- Use log priors per default, bump version
- Update repetitive graph grammar
- Compose functions directly from string descriptor
- Move to method to utils
- Update example

### Fixed
- Fix user priors for categorical hyperparameters

### Removed
- Remove setup method

## [0.4.1] - 2022-04-21

### Added
- Add Literal types
- Add wrapper class and remove edge_data argument from forward method
- Add surrogate_model_fit_args to GP constructor
- Add example for PiBO with graphs and HPs
- Add shift to avoid negative log values
- Add log-PiBO backend for graphs from grammars
- Add merge operation to functional backend

### Changed
- Bump version
- Use mf_bo per default when space has fidelities
- Move multi fidelity to own fil
- Updating function docstring
- Clip likelihood before saving to list
- Update example
- Merging MF BO with master
- Fixing AF sampling for MF BO
- Edits to make MF BO runnable
- Initial ASHA implementation (not tested)
- Upload example using hierarchical GP
- Also update hierarchical GP accordingly
- Rename work_on to set_state
- Refacot load_results
- Also avoid copy in random_sampler acq_sampler
- Rename use_user_priors to user_priors
- Also avoid copy() in rs
- Move copy() into sample()
- Clean up dependencies
- Only run core tests per default
- Make log parameterizable for PiBO
- Start GP fitting of kernel weights froalways m scratch
- Fix NaN in _unscaled_distance
- Move likelihood clipping into optimization

### Removed
- Remove init
- Remove print from base_optimizer
- Remove last graph since it's equal to the full graph
- Remove file

## [0.4.0] - 2022-04-13

### Added
- Add TF backend
- Added required graph generation for sampling to check if graph is valid
- Add support for intermediate build API for hierarchical NAS
- Add set_recursive_attribute to set channels
- Add dict structure as production
- Support for OrdinalHyperparameter in pipeline_space_from_configspace
- Add connection to search space for debugging
- Add connection to hierarchical GP
- Add explanation for hallucination
- Add some TODOs
- Add surrogate from map

### Changed
- Bump version
- Set gradients to zero to avoid memory leak
- Speed up examples
- Rename arguments to function parameter
- Move stem / head out of grammar
- Change the node attribute to coarse label and debugged on all datasets
- Debug on changing the order of the graphs: [hierarchical graphs] + [final_graph] to [final_graph] + [hierarchical graphs]
- Replace fit_on_model with set_state
- Change arguments in examples
- Refactore some and add graphs over hierarchies backend
- Also remove id_parse_tree in other examples
- Refactor hierarchical NAS api
- Debug the running of hierarchical kernels:
- Refactor optimizer
- Basic CArBO implementation
- Move BO specific parts out of base_optimizer
- Make terminal_to_graph_repr map optional
- Fix small bugs

### Fixed
- Categorical parameters: fix infinite loop and value type
- Fix BO docstring

### Removed
- Remove unpacking
- Remove setup
- Remove id_parse argument
- Remove terminal_to_graph_repr

## [0.3.3] - 2022-03-31

### Changed
- Bump version

### Removed
- Remove large parts of graph_utils/utils.py

## [0.3.2] - 2022-03-30

### Changed
- Bump version
- Typing + small refactors

## [0.3.1] - 2022-03-28

### Changed
- Update metahyper and bump version
- Increase default patience

### Removed
- Remove whitespace

## [0.3.0] - 2022-03-24

### Added
- Adding test for user prior
- Add equal operator for graphs
- Add get_edge_list function for variable topologies
- Add missing files
- Add cost-aware skeleton/api
- Added in robin's additive wl surrogate into the bo framework in neps updates
- Added in robin's additive wl surrogate into the bo framework in neps updates
- Add has_prior flag and compute_prior method
- Add sceleton for utilizing prior in acq
- Add prior weighted acq stub and has_prior attr to search space
- Add link to mkdocs documentation
- Add documentation structure
- Added in robin's additive wl surrogate into the bo framework in neps
- Add multi fidelity BO skeleton
- Add SearchSpace.fidelity and SearchSpace.has_fidelity()
- Add logging of working directory
- Add overwrite option
- Add hint to basic example
- Add nullhandler to neps logger
- Add documentation for showing status repeatedly
- Add documentation for specific torch versions
- Add float userpriors
- Add user_prior flag and categorical user prior
- Add patience from BO class to sampling
- Add documentation to status
- Add a comment for integer parameter sampling
- Support singular result
- Add user prior api
- Add multi fidelity API
- Add primitive operations
- Add get_state and load_state methods for optimizers
- Add cost aware setup
- Add status script
- Add improved logging
- Add python support to soon-to-come
- Add features to readme
- Add documentation to api
- Add arglint hook for the api
- Add stopping control for distributed setting
- Add run_pipeline_args and kwargs
- Add new run_pipeline API
- Add Mapping magic to search space
- Add expected hypervolume improvement AF in develop
- Add from_configspace construction to neps searchspace

### Changed
- Deprecate several args in optimizer API and run_pipipeline
- Change default for the watch
- Fix example
- Ensure loss is a float
- Passing budget argument to metahyper
- Upgrade metahyper version
- Fix floats types
- When model building fails, log error and sample randomly
- Fail tests in case of error logged
- Put gp loggers to debug
- Bump metahyper versio
- Check if cuda seed exists in state before using it
- Bump metahyper version
- Bump metahyper version
- Use dev version of metahyper
- Bump metahyper version
- Also run tests when pyproject changed
- Refactor more code
- Json serialization
- Change default value in example
- Complete pibo implementation
- Apply prior weighted when prior exists
- Shorten docstring
- Compute ei incumbent in update step
- Rename reset_surrogate to update
- Extract propose_location from acq function
- Improve comment
- Install metahyper from pypi
- Refactor docs structure and add explanations to CONTRIB.md
- Get 2 versions of gpwl in
- Small refactors and doctring on BayesianOptimization
- Use original error message again
- Improve variable name in README
- Shorten variable names in readme
- Shorten comment in README
- Shorten comment in README
- Improve basic example
- Improve documentation
- Feature pointer to examples more prominently
- Improve parallelization documentation
- Use validation_error in documentation
- Allow more torchvision versions
- Update license in README
- Update LICENSE
- Tweak categorical prior
- Enable user priors
- Use strings for default confidence
- Enhance support for configspace
- Replace random sampler with searchspace.sample_new()
- Simplify randomsampler
- Improve examples
- Improve example
- Mention examples not supported
- Make status importable
- Ignore usage_example
- Improve readme
- Run pre commit
- Update black
- Update to new metahyper version
- Improve post evaluation logging
- Count config_idsfrom 1
- Make hp_kernels and graph_kernels optional
- Update README.md
- Improve usage part in readme
- Deprecate search space api of run
- Update source adaptation for ehvi
- Bug fixes, update hp example to include integer hps
- Make categorical type more general

### Fixed
- Fix ei docstring
- Fix ucb still being important
- Fix log determinant adding
- Fix invalid patience sampling implementation
- Fix error when raising ValueError
- Fix typo
- Fix warning with python -m neps.status
- Fix wrong mypy override
- Fix readme
- Fix multi fidelity example
- Fix typo
- Fix metahyper version
- Fix example error in README
- Fix typing
- Fix type errors numerical & search space

### Removed
- Remove ids whenever possible. Should also be done for graphs
- Remove ucb and make ehvi private
- Remove dead function
- Remove n_iterations
- Remove unneeded mypy override
- Remove unneeded mypy override
- Remove old api from optimizers
- Remove evaluation and utils.util
- Remove old metahyper
- Remove print statement
- Remove read_and_write
- Remove dead code

## [0.2.0] - 2022-01-24

### Added
- Added crossover to search space and updated BO loop to handle pending evaluations
- Add print in eval mode
- Add graph grammar multiple repetitive
- Add linear n node topology
- Add typing packages to dependencies
- Add deprecate decorator
- Add comments about metahyper API status
- Add empty documentation
- Add note to mypy ignoring
- Add mypy to pre-commit hooks and CONTRIBUTING
- Add tests badge
- Add license and copyright notice for nasbowl
- Add github workflow for ci
- Add mutation on a level of dense graph (representation of child was missing)
- Add information on test_tmpdir
- Add instructions for failing tests
- Add poetry tip to CONTRIBUTING
- Add basic badges
- Add absolufy-imports for automated relative imports
- Add infos on utils.install_torch and add to CONTRIBUTING
- Add option to return dict with subgraphs
- Add log-EI in acquisition function mapping
- Add conv-bn-relu to basic primitives
- Add log-EI
- Add all examples to tests
- Add tests_tmpdir to pytests and teardown
- Add API functions for BO
- Add initial population size
- Add FMNIST augmentations/normalization
- Add new metahyper api
- Add pytest mention to CONTRIBUTING
- Add mdformat
- Add random search for the new api
- Add numerical parameter
- Add GraphGrammarCell
- Added all missing modules and reduced number of iterations in  hierarchical_architecture example to speed up testing
- Add --py37 plus flag for pyupgrade
- Add hierarchical architecture example
- Add graph_dense implementation based on grammars
- Added repetitive graph grammar and added common utility functions for graph grammars
- Add convenient dense n-node DAG topology
- Add more basic primitives and topologies
- Add simple graph grammar search space computation
- Add incumbent printing after a successful run
- Add FashionMNIST
- Add pending evaluations for BO
- Add get_dictionary
- Add HPO example (counting ones) using metahyper
- Add get_dictionary and create_from_id for metahyper communication on search space and hyperparameter level
- Add isomorphism checking
- Add repetitive crossover operation
- Add resolution constrained CFG
- Add unparsing of parse trees
- Add tests for examples
- Add commented pytest to pre-commit for later
- Add example folders
- Add python coding guidelines
- Add local_search mutation for hyperparameters
- Add get_hp_by_name for search_space object by holding all hps in a dictionary
- Add example script for running cnas with new search_space object
- Add objective with api
- Add compatability of BO for metahyper
- Add graph_dense sampling
- Add checkpoint path to exception
- Add graph_dense sampling
- Add hyperparameter class (old parameter) with sampling for child classes
- Add deletion of tensors and mmore workers for data loading
- Add RE
- Add hyperparameter class (old parameter) with sampling for child classes
- Add node subgraph
- Add checkpoint reader
- Add simple API objective base class
- Add seed handling to base class
- Add inv transform for scores
- Add base objective class
- Add checkpointer and id
- Add crossover for constrained grammar
- Add repetitive crossover
- Add depth constrained & constrained grammars. Moved mutation & crossover to grammars
- Add grad clipping
- Add training utilities for DL-based experiments
- Add argument to control if we return opt details, default false.
- Add search space class
- Add metahyper dependency
- Add tabulate dependency
- Added other parameter types
- Added wrapper for graph grammar implementing mutation & crossover
- Add parameter types and pipeline space
- Add direct import of api
- Add initial implementation for evolution acq optimizer
- Add pyupgrade to pre-commit
- Add examples as package to pyproject.toml
- Add NOTICE
- Add simon to license
- Add initial api and dummy example
- Added surrogate model fit args to BO optimizer
- Added random interleaving
- Added seed setting for cuda
- Added incumbent plotting
- Added helper libraries for other helpers
- Added functionality to also check for isomorphisms in history
- Added log file
- Added more logging
- Added error handling for random sampler and removed some deepcopying
- Added some special cases
- Added mutation acq optimizer & moved misc to utils
- Added sum kernel to GP
- Add metahyper to pyproject.toml

### Changed
- Update v0.2.0
- Bug fix
- Update training pipeline
- Fix bug
- Return two booleans in crossover
- Bug fix
- Move theta_vector related stuff to combine kernel, apply SMAC lengthscale
- Check whether pipeline_space object supports dimensionality split, check whether loaded result is failed
- Dynamic programming for search space computation
- Fixed remaining mypy errors in graph grammar
- Bug fix
- Fixed more mypy errors
- Resolved merge conflict
- Try to fix poetry install in actions
- Type api
- Rename base_optimizer
- Allow manual trigger
- Fixed some mypy errors
- Bug fix
- Fix pylint warnings
- Move from automl to automl-private
- Move to automl from automl-private
- Change to automl from automl-private
- Update mypy
- Move build system toml part
- Set copyright to general contributors
- Document ci in CONTRIBUTING
- Activate tests on push
- Rename ci
- Rename ci
- Rename ci
- Improve editorconfig documentation
- Save & read seed states
- Small bug fixes
- Improve readme formulations
- Improve pylint ignore warnings section
- Improve dev practices and tooling section
- Improve developer installation section
- Improve tooling tips in CONTRIBUTING
- Apply absolufiy to force relative imports
- Improve content of CONTRIBUTING.md
- Improve content of CONTRIBUTING.md
- Subgraphs is not unrolled anymore
- Also support list as x input for graphs
- Change default value
- Bug fix for log-EI
- Update CONTRIBUTING format and examples guideline
- Change default mutation to simple
- Move example descriptions to example README
- In readme use new style of searchspace construction
- Update README.md
- Bug fixes for RE
- Adjust example for hierarchical NAS for the new metahyper
- Adjust example for NASHPO for the new metahyper
- Make hyperparameters attribute public
- Let pipeline space be defined as dict
- Apply pre-commit
- Update example to use new metahyper
- Make methods private
- Make function calls private
- Update README
- Slim down readme
- Update readme, add example readme
- Update gitignore
- Delete old optimizer abc
- Use list for pylint config
- Update pylint
- Update pyupgrade
- Update isort
- Apply new black version
- Update black version
- Reorder pre-commit hooks
- Change name to neps
- Exclude hyperparameters_architecture from tests for now
- Adjust graph_dense to the changes in graph_grammar
- Merge
- Update examples script, add hpo+nas
- Last naming changes
- Change of search_space initalization and other changes that follow
- Changes in graphs according to the changes of parent classes
- Adjust numerical parameters to the changes in parent classes
- Naming change hyperparameter to parameter (of a search space)
- Upload augmentation lib
- Relative path
- To f-string
- Sync graph_dense with the new graphgrammer implementation
- Switch to value
- Update training pipeline
- Only allow up to python 3.8
- Fix deprecation warning of cholesky
- Copy ResNet Down Block over to graph dense
- Move constrained CFG
- Bug fix
- Bump version for metahyper
- Update README
- Bump metahyper version
- Allow for plotting of temporary results and added axis argument
- Fix evolution bug
- Fix bugs
- Change metahyper branch to master
- Cleanup
- Live_logging does not work for now
- Also rename RS pool_size -> n_candidates
- Fix evolution sampler fitness standardization
- Adjusted BO including renamings.
- Update contributing guidelines
- Work towards new api
- More options for initial samples of evolution acq optimizer
- Make graph flattening optional and fix to_pytorch bug
- Renaming get_graph -> get_graphs
- Change parsing for search_space
- Base hpo benchmarks on new api
- Check back-compatibility, fix name missmatch
- Cleanup
- Fixes of mutation and crossover
- Fix crossover bug
- Also handle concat comb op and subgraphs
- Small fix
- Moved utility class to hierarchical benchmarks
- More consistent with non-api case
- Rename folder
- Bug fix
- Plot shaded region instead of bars
- Changed name of objective to search space
- More refactoring of benchmarks
- Change BO from max to min
- Update benchmark sampling method name
- Update gitignore
- Refactor of hpo benchamrks
- Refactor of hpo benchamrks
- Small change
- Small fixes
- Small adjustments to new design and minor bug fix
- Fixed bugs in pytorch model generation
- Refactor hartmann6 for seperate objective_fn
- Small check if depth constraints match to nonterminals
- Bug fixes in pytorch model generation
- Refactor evaluation of seed in its own fn
- Update benchmarks for refactoring
- Update benchmarks before refactor
- Small fixes
- Moved graph grammar core here and changed directory structure
- Quick fix
- Changed computational flow to be compatible with new evolutionary acqusition optimizers etc.
- Avoid side effect on save_path
- Renamed bo->bayesia_optimization and rs->random_search
- Sort keys for plotting
- Delete empty __init__
- Move torch install utils to comp_nas/utils
- Make return opt_details truly optional
- Changed python version
- Changed python version
- Moved import to top
- Small bug fix when there are not model proposals
- Small fix if no model-based proposal is used
- Use std error instead of std dev
- Always write out csv
- Online csv writing
- Small changes
- Reuse kernels while kernel fitting and addedparallelized version oof EI
- Shorten argument help, since choices are provided
- Fix typo
- New default values
- Do not use property annotations for getter
- Avoid computing inc in EI over and over again
- Bug fix in mutation acq opt
- Updated README and added error messages
- Bug fix of mutation acq optimizer and minor changes in GP
- Changed default values
- Run formating on torch_install.py
- Avoid error log for HPO kernels and return more information from BO optimizer
- FIX naming bug
- Check if we have list/tuple containing multiple graphs
- Replaced Union with Iterable
- Quick bug fix
- Small changes and fixed minor issue in GP
- Renamed folder & removed initial_design folder
- Move plotting out of BO
- Return opt details dict & added arguments
- ADD nasbenches in NASLib style, query accepts API now.
- Merge
- Move utils to surrogate models and added acqusition optimizer base class
- Renamed kernel folder
- HPO benchmarks follow NASLib now
- HPO benchmarks follow NASLib now
- FIX gp bug
- Complete changes, joint representation for a configuration (before switching to NASLib object). Framework tested on all the benchmarks (nb301 gives an error for the check weights.is_leaf == True, commented out for now). ADD nbconfigfiles for running nasbenches (.pth file needed for nb201, .model file needed for nb301).
- Complete changes, joint representation for a configuration (before switching to NASLib object). Framework tested on all the benchmarks (nb301 gives an error for the check weights.is_leaf == True, commented out for now). ADD nbconfigfiles for running nasbenches (.pth file needed for nb201, .model file needed for nb301).
- Use black profile for isort
- Changed folder name
- New software design for NAS optimizers
- More clean-ups
- ADD nasbenches
- Initial commit
- Initial commit

### Fixed
- Fix access to theta_vector in case of no vectorial features
- Fix test bug
- Fix lengthscale parameter for stationary kernels, add split dimensionality for search space and kernels (for now only default categorical/continuous split
- Fix for-else unnecessary
- Fix mypy for pre-commit run -a
- Fix typo in acquisition
- Fix pylint warning
- Fix bug in the search space mutation
- Fix old name
- Fix typo
- Fix readme indents
- Fix imports in examples
- Fix new_result and get_config for metahyper communication
- Fix seeding for benchmarks
- Fix typo
- Fix isort not_skip depreciation

### Removed
- Remove unnecessary args in README
- Remove unnecessary args in examples
- Remove worker init fn argument
- Remove license section in README
- Remove more legacy torch import guards
- Remove install_dev_utils add commands to CONTRIBUTING
- Remove pytest comments in pre commit
- Remove torch import guard and add it to api.py
- Remove latex from editorconfig
- Remove attr
- Remove query from graph
- Remove _id for hps
- Remove all imports to hierarchical_nas_benchmarks
- Remove hierarchical_nas_benchmarks dependencies in hierarchical architecture example
- Remove other hierarchical_nas_benchmarks dependencies in graph_dense
- Remove duplication of BO class
- Remove rng passing, add simple mutation for hps
- Remove rng passing, add simple mutation for hps
- Remove results from repo
- Removed unnecessary arguments
- Remove previously used argument
- Removed deepcopying config over and over again
- Remove torch packages from pyproject, add install util
- Remove pylint comments
- Removed wildcard import
- Removed logs and results

[unreleased]: https://github.com///compare/v0.14.0...HEAD
[0.14.0]: https://github.com///compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com///compare/v0.12.2...v0.13.0
[0.12.2]: https://github.com///compare/v0.12.1...v0.12.2
[0.12.1]: https://github.com///compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com///compare/v0.11.1...v0.12.0
[0.11.1]: https://github.com///compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com///compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com///compare/v0.8.4...v0.10.0
[0.8.4]: https://github.com///compare/v0.8.3...v0.8.4
[0.8.3]: https://github.com///compare/v0.8.2...v0.8.3
[0.8.2]: https://github.com///compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com///compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com///compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com///compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com///compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com///compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com///compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com///compare/v0.4.10...v0.5.0
[0.4.10]: https://github.com///compare/v0.4.9...v0.4.10
[0.4.9]: https://github.com///compare/v0.4.8...v0.4.9
[0.4.8]: https://github.com///compare/v0.4.7...v0.4.8
[0.4.7]: https://github.com///compare/v0.4.6...v0.4.7
[0.4.6]: https://github.com///compare/v0.4.5...v0.4.6
[0.4.5]: https://github.com///compare/v0.4.4...v0.4.5
[0.4.4]: https://github.com///compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com///compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com///compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com///compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com///compare/v0.3.3...v0.4.0
[0.3.3]: https://github.com///compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com///compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com///compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com///compare/v0.2.0...v0.3.0

<!-- generated by git-cliff -->
