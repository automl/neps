# Roadmap

## Next up

### Features

- Utility to get best HPs and (built) architecture\*
- Utility to get incumbent losses over time\*

### Documentation

- Data reading example\*
- Working directory example\*
- Parallelization example\*
- Fill up the core documentation pages\*

### Refactoring

- Move metahyper to the neps repository
- Use undepreciated working dir name for metahyper.run
- Clean up unused branches
- Merge GP and hierarchical GP
- Merge gpytorch branch

### Tests and tooling

- Add simple regression tests to run on each push
- poetry hooks
- poetry update

## Before 1.0.0 version

### Features

- Seamless ddp via cli launcher (fix from Fabio / Sam adapted to new ddp version)
- Finegrained control over HP user prior
- Incumbent plot
- Top vs all vs bottom distribution plots
- Tensorboard visualizations (incumbent plot, ..)
- Print search space upon run
- Add comprehensive regression tests to run manually on the cluster on each version release

### Fixes

- Printing architecture search spaces / search spaces in general
- Metahyper Refine jobtimelimit feature
- Optimize dependencies

### Refactoring

- run_pipeline = evaluate_pipeline
- loss = validation_error
- remove graph_dense API
- clean up search spaces classes, unused methods
- break up search space and config aspect
- Improve error message printed by Loss value on error

### Documentation

- Keep a changelog

## Later version

### Features

- Rework metahyper
- Optional argparse adder like pytorch lightning
- Utility neps.clean to manage existing run results
- Collect data optionally via phone-home to webserver
- Add Info dict to status
- BO improvements via Hebo tricks + Mll replacement

### Documentation

- Documentation on cuda pytorch
- Conda install for meta cluster, put the export in profile

### Miscellaneous

- Twitter handle and domain, e.g., neural-pipeline.search
- Doing research with NePS / Documentation on that or full setup
- Regular contributor meetings
- Maintain issues
