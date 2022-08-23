# Roadmap

## As soon as possible

### Features

- Python 3.8+ support
- Utility to get best HPs and (built) architecture
- Utility to get incumbent losses over time

### Fixes

- Metahyper Refine jobtimelimit feature

### Refactoring

- merge GP and hierarchical GP etc.


### Tests

- Add simple regression tests to run on each push
- Add comprehensive regression tests to run manually on the cluster on each version release

### Documentation

- Data reading example
- Working directory example
- Fill up the core documentation pages




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
- remove constant parameter
- remove graph_dense API
- clean up search spaces classes, unused methods
- break up search space and config aspect

### Tests

- Clean up examples, create experimental examples and/or move examples for test-purposes to test dir

### Documentation

- Trunk based development techniques
- Point to smac for multi-objective problems and classic ML / have "alternatives" section in readme
- Changelog



## Later version

### Features

- Optional argparse adder like pytorch lightning
- Utility neps.clean to manage existing run results
- Collect data optionally via phone-home to webserver
- Add Info dict to status
