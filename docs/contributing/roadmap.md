# Roadmap

## Next up

### Features

- Utility to get best HPs and (built) architecture\*
- Utility to get incumbent losses over time\*
- Windows support\*

### Documentation

- Fill up the core documentation pages\*
- Fix NAS examples\*

### Fixes

- Fix autograd warning?

### Refactoring

- remove graph_dense API\*
- run_pipeline = evaluate_pipeline?\*
- loss = validation_error | error | pipeline_error?\*
- IntegerParameter = IntegerSpace
- Clean up unused branches
- Merge GP and hierarchical GP
- Merge gpytorch branch

### Tests and tooling

- Add simple regression tests to run on each push

## Before 1.0.0 version

### Features

- Seamless ddp via cli launcher
- Finegrained control over HP user prior
- Top vs all vs bottom distribution plots
- Tensorboard visualizations (incumbent plot, ..)
- Print search space upon run
- Add comprehensive regression tests to run manually on the cluster on each version release

### Fixes

- Printing architecture search spaces / search spaces in general
- Metahyper Refine jobtimelimit feature
- Optimize dependencies

### Refactoring

- clean up search spaces classes, unused methods
- break up search space and config aspect

### Documentation

- Keep a changelog

## Later version

### Features

- Optional argparse adder like pytorch lightning
- Utility neps.clean to manage existing run results
- Collect data optionally via phone-home to webserver
- Add Info dict to status
- BO improvements via Hebo tricks + Mll replacement

### Miscellaneous

- Twitter handle and domain, e.g., neural-pipeline.search
- Doing research with NePS / Documentation on that or full setup
