""" How to generate a summary (neps.status) and visualizations (neps.plot) of a run.

Before running this example analysis, run the hyperparameters example with:

    python -m neps_examples.basic_usage.hyperparameters
"""
import neps

# During the run,

# Printing a summary and reading in results.
# Alternatively use `python -m neps.status results/hyperparameters_example`
results, pending_configs = neps.status("results/hyperparameters_example")
config_id = "1"
print(results[config_id].config)
print(results[config_id].result)
print(results[config_id].metadata)

# Generating plots to the root directory (results/hyperparameters_example)
# Alternatively use `python -m neps.plot results/hyperparameters_example`
neps.plot("results/hyperparameters_example")
