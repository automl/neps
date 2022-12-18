# Roadmap

## Before 0.8.0

### Features

- Utility to get best HPs and architecture to pass to run_pipeline

### Fixes

- Ignore grakel deprecation

### Documentation

- Fill up the core documentation pages
- Fix NAS examples
- remove graph_dense API

## Before 0.9.0

### Features

- Plot in hours / days, but also show grid corresponding to 1x and 5x etc.

### Fixes

- Do not plot log y axis per default

## Before 0.10.0

### Features

- Evolution as acq sampler
- Generate plot after each evaluation

### Fixes

- Open never closes (talk to Nils)
- Deadlock in ASHA-like optimizers (talk to Neeratyoy)

### Documentation

- Document summary function

### Refactoring

- Merge GP and hierarchical GP
- Merge gpytorch branch
- Rethink summary/status API
- Utility to get incumbent losses over time
- Restructure folder structure
- Improve placement of \_post_evaluation_hook_function
- maintained vs unmaintained optimizers
- Read and sample at the same time metahyper
- Metahyper into neps
- Renamings
  - run_pipeline = evaluate_pipeline | evaluate_pipeline_error | compute_pipeline_error | train_and_evaluate
  - loss = validation_error | error | pipeline_error
  - XParameter = XSpace
  - Rename default-x to prior-x
  - Use max_cost_total everywhere instead of budget
  - FunctionParameter = CodeParameter | ModuleParameter

### Tests and tooling

- Add priorband to experimental
- Add simple regression tests to run on each push

## Before 1.0.0 version

### Features

- Seamless ddp via cli launcher
- Finegrained control over HP user prior
- Top vs all vs bottom distribution plots
- Tensorboard visualizations (incumbent plot, ..)
- Loss distribution plot
- Print search space upon run
- Add comprehensive regression tests to run manually on the cluster on each version release
- Utility to generate code for best architecture
- 3.11 support
- Deprecate 3.7
- Core Feature set in terms of research

### Fixes

- Printing architecture search spaces / search spaces in general
- Metahyper Refine jobtimelimit feature
- Optimize dependencies

### Refactoring

- Clean up search spaces classes, unused methods
- Break up search space and config aspect
- Remove hnas branch
- Refactor of constraint grammar

### Documentation

- Keep a changelog

## Later version

### Features

- neps_examples callable for options of examples
- Optional argparse adder like pytorch lightning
- Utility neps.clean to manage existing run results
- Collect data optionally via phone-home to webserver
- Add Info dict to status
- Seed (setting context manager?)
- BO improvements via Hebo tricks + Mll replacement
- Checkout Rich logging

### Miscellaneous

- User Mattermost Channel
- Twitter handle and domain, e.g., neural-pipeline.search
- Doing research with NePS / Documentation on that or full setup
