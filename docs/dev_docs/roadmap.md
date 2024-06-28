# Roadmap

## Before 0.14.0

### Features

- ? Allow yaml based input of search space and the target function source to `neps.run`
- Generate plot after each evaluation
- ? Support conditionals in ConfigSpace search space
- ? Support logging of optimizer state details
- Nice way to handle Multi-fidelity large scale (slurm script modification)
- Seamless ddp via cli launcher
- ? Log priors include again

### Fixes

- acq search mutation only mutates 1 parameter?
- Optimize dependencies
- Double log on neps output
- Are continuations of errored-out runs removed already?

### Refactoring

- ? Merge GP and hierarchical GP
- Branch cleanup
- Rethink summary/status API
- Improve placement of \_post_evaluation_hook_function
- maintained vs unmaintained optimizers
- ? Renamings
    - run_pipeline = evaluate_pipeline | evaluate_pipeline_error | compute_pipeline_error | train_and_evaluate
    - loss = validation_error | error | pipeline_error
    - XParameter = XSpace
    - Rename default-x to prior-x
    - Use max_cost_total everywhere instead of budget

### Documentation

- Keep a changelog
- Keep citations doc up to date
- Role of analysing runs needs to be higher in docs
- Explain what optimizers are run per default / papers higher in docs
- Hier NAS documentation
- Rework readme
    - for keyfeatures, 3+4? Mention prior + multi fidelity (cheap approx)
    - Code example of readme should work when copied
    - ? Sync readme with docs landingpage more nicely
    - Parallelization link is broken


## Before 1.0.0 version

### Features

- Finegrained control over HP user prior
- ? Top vs all vs bottom distribution plots
- ? Tensorboard visualizations (incumbent plot, ..)
- ? Loss distribution plot
- Print search space upon run
- Utility to generate code for best architecture
- Core Feature set in terms of research
- ? Modular plug-and-play of BoTorch acquisition functions
- Generate analysis pdf?

### Fixes

- Printing architecture search spaces / search spaces in general

### Refactoring

- Clean up search spaces classes, unused methods
- Break up search space and config aspect

### Tests and tooling

- Add comprehensive regression tests to run manually on the cluster on each version release
- Regression tests to run on each push
- ? mdformat for readme
- ? Darglint

## Later version

### Documentation

- ? Doing research with NePS / Documentation on that or full setup

### Features

- neps_examples callable for options of examples
- Utility neps.clean to manage existing run results
- Collect data optionally via phone-home to webserver
- Add Info dict to status



