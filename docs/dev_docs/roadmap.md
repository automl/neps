# Roadmap

## Next up

### Features

- Improve handling of multi-fidelity for large scale (slurm script modification)
- Evaluate and maybe improve ease-of-use of NePS and DDP etc.
- Optimize dependencies
- Improved examples

### Fixes

- Acq search mutation for HPs potentially only mutates 1 parameter
- `ignore_errors` should work seamlessly with all optimizers

### Refactoring

- Rename: run_pipeline = evaluate_pipeline | evaluate_pipeline_error | compute_pipeline_error | train_and_evaluate
- Rename: loss = validation_error | error | pipeline_error
- Rename: XParameter = XSpace or just X?
- Rename: default-x to prior-x
- Rename: Use max_cost_total everywhere instead of budget

### Documentation

- Keep citations doc up to date

### Tests

- Regression tests to run on each push


## Before 1.0.0 version

### Features

- Generate pdf plot after each evaluation
- Finegrained control over user prior
- Print search space upon run
- Utility to generate code for best architecture
- Core algorithmic feature set (research)

### Fixes

- Contact https://pypi.org/project/neps/ to free up `pip install neps`

### Refactoring

- Improve neps.optimizers:
    - Maintained vs unmaintained optimizers
    - Remove unnecessary / broken optimizers
    - Merge GP and hierarchical GP
- Break up search space and config aspect

### Documentation

- NAS documentation

## After 1.0.0

### Features

- Utility neps.clean to manage existing run results
- Collect data optionally via phone-home to webserver

### Documentation

- Keep a changelog
