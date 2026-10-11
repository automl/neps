
This content tree contains the core components and utilities of AutoScAn, designed to simplify and enhance the process of running optimization experiments. Below is an overview of the files and their purposes:

- [`autoscan.run`](autoscan/api.md): Provides the built-in AutoScAn runtime to **sample new trials** and **evaluate them automatically**.
- [`autoscan.runtime`](autoscan/runtime.md): Implements the `Worker`, offering functions to create a worker, sample new trials, and evaluate them.
- `autoscan.optimizers`:
    - [`autoscan.algorithms`](autoscan/optimizers/algorithms.md): Contains a collection of optimization algorithms, such as random search, ASHA, PriorBand, HyperBand, and more, for sampling new trials.
    - [`autoscan.AskAndTell`](autoscan/optimizers/ask_and_tell.md): An alternative to `autoscan.run` that allows full control of the evaluation loop. This is useful when you don’t want to use AutoScAn’ runtime but still want to benefit from its optimizers and state management.
- [`autoscan.state`](autoscan/state/autoscan_state.md): Manages the state of workers, trials, and optimizers, ensuring reproducibility and continuity.
- [`autoscan.status`](autoscan/status/status.md): Provides functions to retrieve the status of a run and export it to CSV files for analysis.
- [`autoscan.plot`](autoscan/plot/plot.md): Includes tools to visualize the results of a neural pipeline search run.