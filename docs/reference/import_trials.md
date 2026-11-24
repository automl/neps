# Importing External Trials

When optimizing with ScaleFM, you might have evaluations from previous studies that you want to incorporate into your current optimization run. The [`scalefm.import_trials()`][scalefm.api.import_trials] function enables seamless integration of externally evaluated configurations into ScaleFM.

## When to Import Trials

There are several scenarios where importing trials is valuable:

### Scenario 1: Switching Optimization Algorithms
You've already run evaluations with one algorithm (e.g., random search) and want to leverage them with a more sophisticated optimizer (e.g., Bayesian optimization):

```python
# Random search was used for initial exploration
# Now switch to Bayesian optimization with those results
import scalefm

evaluated_trials = [
    ({"learning_rate": 0.001, "batch_size": 32}, scalefm.UserResultDict(objective_to_minimize=0.45)),
    ({"learning_rate": 0.01, "batch_size": 64}, scalefm.UserResultDict(objective_to_minimize=0.38)),
    ({"learning_rate": 0.0001, "batch_size": 16}, scalefm.UserResultDict(objective_to_minimize=0.52)),
]

scalefm.import_trials(
    evaluated_trials=evaluated_trials,
    root_directory="bayesian_study",
    pipeline_space=my_pipeline_space, # need to define pipeline space
    optimizer="bayesian_optimization"
)

# Continue optimization with Bayesian optimization
scalefm.acquire(
    evaluate_pipeline=evaluate_pipeline, # need to define evaluate pipeline
    pipeline_space=my_pipeline_space, # need to define pipeline space
    root_directory="bayesian_study",
    total_evaluations_to_spend=5,  # 5 more evaluations
)
```

### Scenario 2: Warm-starting from External Optimization
You've optimized a model using a non-ScaleFM tool and want to continue with ScaleFM:

```python
# Results from your own optimization framework
external_results = [
    ({"model_depth": 5, "learning_rate": 0.01}, scalefm.UserResultDict(objective_to_minimize=0.35)),
    ({"model_depth": 10, "learning_rate": 0.001}, scalefm.UserResultDict(objective_to_minimize=0.32)),
]

scalefm.import_trials(
    evaluated_trials=external_results,
    root_directory="continued_study",
    pipeline_space=my_pipeline_space,
    optimizer="bayesian_optimization"
)
```

### Scenario 3: Combining Multiple Optimization Runs
You've performed separate optimization studies and want to merge results:

```python
# Load results from previous run
previous_trials = load_previous_results()  # Your custom loading logic

# Import into new unified study
scalefm.import_trials(
    evaluated_trials=previous_trials,
    root_directory="unified_study",
    pipeline_space=my_pipeline_space,
    optimizer="bayesian_optimization"
)
```

## Import Formats

### Format 1: Manual Dictionary Format

The simplest format is a list of tuples containing configurations and results:

```python
from scalefm import UserResultDict

evaluated_trials = [
    # (configuration_dict, result_dict)
    (
        {"learning_rate": 0.001, "num_layers": 5},
        UserResultDict(objective_to_minimize=0.45)
    ),
    (
        {"learning_rate": 0.01, "num_layers": 10},
        UserResultDict(objective_to_minimize=0.38)
    ),
]

scalefm.import_trials(
    evaluated_trials=evaluated_trials,
    root_directory="my_study",
    pipeline_space=my_pipeline_space,
    optimizer="bayesian_optimization"
)
```

### Result Dictionary Structure

The result dictionary should be a `UserResultDict`:

```python
scalefm.UserResultDict(objective_to_minimize=0.45)  # Required

# Optional fields:
scalefm.UserResultDict(
    objective_to_minimize=0.45, 
    cost=1000, 
    exception=None, 
    learning_curve=[0.32,0.35],
    info_dict={"something": "whatever information we want to save for this config"},
)

```

### Format 2: From Previous ScaleFM Run (Using `load_trials_from_pickle`)

If your external evaluations were done with ScaleFM previously, use `load_trials_from_pickle` to easily extract trials:

```python
import scalefm
from scalefm.utils import load_trials_from_pickle

# Load from previous ScaleFM run
evaluated_trials = load_trials_from_pickle(
    root_dir="path/to/old_study"
)

# Import into new study
scalefm.import_trials(
    evaluated_trials=evaluated_trials,
    root_directory="new_study",
    pipeline_space=my_pipeline_space,
    optimizer="bayesian_optimization" # or other optimizer desired for the study
)
```
