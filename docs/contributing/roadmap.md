# Roadmap

## As soon as possible

### Features

- Python 3.8+ support
- Utility to get best HPs and (built) architecture
- Utility to get incumbent losses over time
- Ignore error configurations

### Fixes

- Metahyper Refine jobtimelimit feature
- Optimize dependencies

### Refactoring

- merge GP and hierarchical GP etc.

### Tests

- Add simple regression tests to run on each push
- Add comprehensive regression tests to run manually on the cluster on each version release

### Documentation

- Data reading example
- Working directory example
- Fill up the core documentation pages
- Testing protocol
- version tied documentation

## Before 1.0.0 version

### Features

- Seamless ddp via cli launcher (fix from Fabio / Sam)
- Finegrained control over HP user prior
- Incumbent plot
- Top vs all vs bottom distribution plots
- Tensorboard visualizations (incumbent plot, ..)
- Print search space upon run

### Fixes

- Printing architecture search spaces / search spaces in general

### Refactoring

- run_pipeline = evaluate_pipeline
- loss = validation_error
- remove graph_dense API
- clean up search spaces classes, unused methods
- break up search space and config aspect
- Improve error message printed by Loss value on error

### Documentation

- Changelog

## Later version

### Features

- Rework metahyper
- Optional argparse adder like pytorch lightning
- Utility neps.clean to manage existing run results
- Collect data optionally via phone-home to webserver
- Add Info dict to status

### Documentation

- Doing research with NePS
- Trunk based development techniques
