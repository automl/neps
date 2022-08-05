Features

- Seamless ddp via cli launcher (fix from Fabio / Sam)
- Collect data optionally via phone-home to webserver
- Utility neps.clean to manage existing run results
- Utility to get best HPs and (built) architecture
- Utility to get incumbent losses over time
- Finegrained control over HP user prior
- Tensorboard visualizations
- Incumbent plot
- Integrate neps analyze repo
- Add Info dict to status
- Python 3.8+ support
- Print search space upon run?
- Optional argparse adder like pytorch lightning

Fixes

- Metahyper Refine crash scenarios and jobtimelimit feature
- Printing architecture search spaces / search spaces in general
- meassure cost correctly in a crashing scenario

Refactoring

- run_pipeline = evaluate_pipeline
- loss = evaluation_error
- remove constant parameter
- remove graph_dense API
- categorical is not a numerical parameter
- clean up search spaces classes, unused methods
- break up search space and config aspect
- merge GP and hierarchical GP etc.

Tests

- Clean up examples, create experimental examples and/or move examples for test-purposes to test dir
- Add simple regression tests to run on each push
- Add comprehensive regression tests to run manually on the cluster on each version release

Documentation:

- Trunk based development techniques
- Link to individual examples in README (for core features)
- Point to smac for multi-objective problems and classic ML / have "alternatives" section in readme
- Data reading example
- Working directory example
- Fill up the core documentation pages
- Metahyper create docs and ghpage
- Metahyper contrib update
- Roadmap
- Changelog
- Mention somewhere the "beta" status indicated by v0.x.x
- Readme improvements
