
This content tree contains the core components and utilities of ScaleFM, designed to simplify and enhance the process of running optimization experiments. Below is an overview of the files and their purposes:

- [`scalefm.acquire`](scalefm/api.md): Provides the built-in ScaleFM runtime to **sample new trials** and optionally **evaluate them automatically**.
- [`scalefm.runtime`](scalefm/runtime.md): Implements the `Worker`, offering functions to create a worker, sample new trials, and evaluate them.
- `scalefm.optimizers`:
    - [`scalefm.algorithms`](scalefm/optimizers/algorithms.md): Contains a collection of optimization algorithms, such as random search, grid search, bayesian optimization, CARBS and Chinchilla, for sampling new trials.
    - [`scalefm.AskAndTell`](scalefm/optimizers/ask_and_tell.md): An alternative to `scalefm.acquire` that allows full control of the evaluation loop. This is useful when you don’t want to use ScaleFM’ runtime but still want to benefit from its optimizers and state management.
- [`scalefm.state`](scalefm/state/scalefm_state.md): Manages the state of workers, trials, and optimizers, ensuring reproducibility and continuity.
- [`scalefm.status`](scalefm/status/status.md): Provides functions to retrieve the status of a run and export it to CSV files for analysis.
- [`scalefm.plot`](scalefm/plot/scaling_plots.md): Includes the scaling plots of a run and `tblogger` for TensorBoard.