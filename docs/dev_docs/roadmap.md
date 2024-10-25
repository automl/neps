# Roadmap

## Next up

### Features

- Improve large scale experience
    - Result saving function (Samir)
    - Priorband default sampling / pass evaluated configs to neps.run (Samir)
    - Document large scale
    - Evaluate and maybe improve ease-of-use of NePS for DDP (Gopalji)
- Optimize dependencies (Anton)
- Tensorboard st no one has to Touch it anymore (Tarek)

### Fixes

- ignore_errors should work seamlessly with all optimizers, also check different error handling Flags (Gopalji)
- Install all dependencies to run core examples always (Anton)

### Refactoring

(Anton)

- Rename: run_pipeline = evaluate_pipeline
- Rename: loss = objective_to_minimize
- Rename: default = prior, default_confidence = prior_confidence
- Rename: budget = max_cost_total

### Documentation

- Update citations (also docs) (Danny)
- Notebooks add (Danny)
- Remove templates (Danny)
- Rework readme (remove declarative API) (Danny)
- Improved examples
    - New Lightning example (Gopalji)
    - DDP examples (Gopalji)
    - Larger examples (Gopalji)
    - Tensorboard into new lightning example (Tarek)
    - Example spawning cloud instances via run pipeline

### Tests

- Pytest needs to work on a fresh install (Anton)
- Regression tests to run on cluster on each version release

## Before 1.0.0 version

### Features

- Utility neps.clean to manage existing run results
- Generate pdf plot after each evaluation
- Finegrained control over user prior
- Print search space upon run
- Utility to generate code for best architecture
- Core algorithmic feature set (research)

### Documentation

- NAS documentation
- Optimizer pages (Anton, Neeratyoy)
- Keep a changelog, add to it before each release
