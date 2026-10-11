"""How to generate a summary (autoscan.status) of a run.
Before running this example analysis, run the hyperparameters example with:
    python -m autoscan_examples.basic_usage.hyperparameters
"""

import autoscan

# 1. At all times, AutoScAn maintains several files in the root directory that are human
# read-able and can be useful

# 2. Printing a summary and reading in results.
full, summary = autoscan.status("results/hyperparameters_example", print_summary=True)
config_id = "1"

print("\n", full.head(), "\n")
print(full.loc[config_id])
