
This content tree contains the core components and utilities of NePS, designed to simplify and enhance the process of running optimization experiments. Below is an overview of the files and their purposes:

- [`neps.run`](neps/api.md): Provides the built-in NePS runtime to **sample new trials** and **evaluate them automatically**.
- [`neps.runtime`](neps/runtime.md): Implements the `Worker`, offering functions to create a worker, sample new trials, and evaluate them.
- `neps.optimizers`:
    - [`neps.algorithms`](neps/optimizers/algorithms.md): Contains a collection of optimization algorithms, such as random search, ASHA, PriorBand, HyperBand, and more, for sampling new trials.
    - [`neps.AskAndTell`](neps/optimizers/ask_and_tell.md): An alternative to `neps.run` that allows full control of the evaluation loop. This is useful when you don’t want to use NePS’ runtime but still want to benefit from its optimizers and state management.
- [`neps.state`](neps/state/neps_state.md): Manages the state of workers, trials, and optimizers, ensuring reproducibility and continuity.
- [`neps.status`](neps/status/status.md): Provides functions to retrieve the status of a run and export it to CSV files for analysis.
- [`neps.plot`](neps/plot/plot.md): Includes tools to visualize the results of a neural pipeline search run.