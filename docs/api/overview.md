
This content tree contains the core components and utilities of Extrapolaw, designed to simplify and enhance the process of running optimization experiments. Below is an overview of the files and their purposes:

- [`extrapolaw.acquire`](extrapolaw/api.md): Provides the built-in Extrapolaw runtime to **sample new trials** and **evaluate them automatically**.
- [`extrapolaw.runtime`](extrapolaw/runtime.md): Implements the `Worker`, offering functions to create a worker, sample new trials, and evaluate them.
- `extrapolaw.optimizers`:
    - [`extrapolaw.algorithms`](extrapolaw/optimizers/algorithms.md): Contains a collection of optimization algorithms, such as random search, grid search, bayesian optimization, CARBS and Chinchilla, for sampling new trials.
    - [`extrapolaw.AskAndTell`](extrapolaw/optimizers/ask_and_tell.md): An alternative to `extrapolaw.acquire` that allows full control of the evaluation loop. This is useful when you don’t want to use Extrapolaw’ runtime but still want to benefit from its optimizers and state management.
- [`extrapolaw.state`](extrapolaw/state/extrapolaw_state.md): Manages the state of workers, trials, and optimizers, ensuring reproducibility and continuity.
- [`extrapolaw.status`](extrapolaw/status/status.md): Provides functions to retrieve the status of a run and export it to CSV files for analysis.
- [`extrapolaw.plot`](extrapolaw/plot/scaling_plots.md): Includes the scaling plots of a run and `tblogger` for TensorBoard.