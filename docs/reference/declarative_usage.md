# Introduction to Declarative Usage with NePS
## Configuring with YAML

Configure your experiments by specifying settings in a YAML file. This file becomes the single source for
your project setup, making it easy to share, reproduce, and modify them.

Should i explain how to create a yaml??

## Simple YAML Example
Here’s a basic example of how a YAML configuration for NePS looks:
```yaml
# Basic NEPS Configuration Example
run_pipeline:
  path: path/to/your/run_pipeline.py  # Path to the function file
  name: example_run                   # Function name within the file

pipeline_space:
  learning_rate:
    lower: 1e-5
    upper: 1e-1
    log: True  # Log scale for learning rate
  optimizer:
    choices: [adam, sgd, adamw]
  epochs: 50

root_directory: path/to/results       # Directory for result storage
max_evaluations_total: 20               # Budget


```
## Executing the Configuration
To run the experiment defined in your YAML, use this simple command in Python:
```python
import neps
neps.run(run_args="path/to/your/config.yaml")
```


## Extended Configuration
```yaml
run_pipeline:
  path: path/to/your/run_pipeline.py  # Path to the function file
  name: example_run                   # Function name within the file

pipeline_space:
  learning_rate:
    lower: 1e-5
    upper: 1e-1
    log: True  # Log scale for learning rate
  optimizer:
    choices: [adam, sgd, adamw]
  epochs: 50

root_directory: path/to/results       # Directory for result storage
max_evaluations_total: 20               # Budget
max_cost_total:

# Debug and Monitoring
overwrite_working_directory: True
post_run_summary: True
development_stage_id:
task_id:

# parallelization_setup
max_evaluations_per_run:
continue_until_max_evaluation_completed: False

# error_handling
loss_value_on_error:
cost_value_on_error:
ignore_errors:

searcher: bayesian_optimization # To select only the searcher; for more options, see [here](#customizing-your-own-searcher).

pre_load_hooks:

```
explain what happens with undefined keys?
## Customizing neps searcher
```yaml
run_pipeline:
  path: path/to/your/run_pipeline.py  # Path to the function file
  name: example_run                   # Function name within the file

pipeline_space:
  learning_rate:
    lower: 1e-5
    upper: 1e-1
    log: True  # Log scale for learning rate
  optimizer:
    choices: [adam, sgd, adamw]
  epochs: 50

root_directory: path/to/results       # Directory for result storage
max_evaluations_total: 20               # Budget
searcher:
  algorithm: bayesian_optimization    # name linked with neps keywords, more information click here..?
  # Specific arguments depending on the searcher
  initial_design_size: 7
  surrogate_model: gp
  acquisition: EI
  log_prior_weighted: false
  acquisition_sampler: random
  random_interleave_prob: 0.1
  disable_priors: false
  prior_confidence: high
  sample_default_first: false


```
TODO
information where to find parameters of included searcher, where to find optimizers names...link


## Load your optimizer
```yaml
run_pipeline:
  path: path/to/your/run_pipeline.py  # Path to the function file
  name: example_run                   # Function name within the file

pipeline_space:
  learning_rate:
    lower: 1e-5
    upper: 1e-1
    log: True  # Log scale for learning rate
  optimizer:
    choices: [adam, sgd, adamw]
  epochs: 50

root_directory: path/to/results       # Directory for result storage
max_evaluations_total: 20               # Budget
searcher:
  path: path/to/your/searcher.py  # Path to the class
  name: CustomOptimizer           # class name within the file
  # Specific arguments depending on your searcher
  initial_design_size: 7
  surrogate_model: gp
  acquisition: EI


```
## How to define hooks?

```yaml
# Basic NEPS Configuration Example
run_pipeline:
  path: path/to/your/run_pipeline.py  # Path to the function file
  name: example_run                   # Function name within the file

pipeline_space:
  learning_rate:
    lower: 1e-5
    upper: 1e-1
    log: True  # Log scale for learning rate
  optimizer:
    choices: [adam, sgd, adamw]
  epochs: 50

root_directory: path/to/results       # Directory for result storage
max_evaluations_total: 20               # Budget

pre_load_hooks:
    hook1: path/to/your/hooks.py # (function_name: Path to the function's file)
    hook2: path/to/your/hooks.py # Different function name from the same file source

```
## What if your search space is big?
```yaml
run_pipeline:
  path: path/to/your/run_pipeline.py  # Path to the function file
  name: example_run                   # Function name within the file

pipeline_space: path/to/your/pipeline_space.yaml

root_directory: path/to/results       # Directory for result storage
max_evaluations_total: 20               # Budget


```

pipeline_space.yaml
```yaml
pipeline_space:
  learning_rate:
    lower: 1e-5
    upper: 1e-1
    log: True  # Log scale for learning rate
  optimizer:
    choices: [adam, sgd, adamw]
  epochs: 50
...
```

## If your use case involves experimenting with different searcher settings
```yaml
# Basic NEPS Configuration Example
run_pipeline:
  path: path/to/your/run_pipeline.py  # Path to the function file
  name: example_run                   # Function name within the file

pipeline_space:
  learning_rate:
    lower: 1e-5
    upper: 1e-1
    log: True  # Log scale for learning rate
  optimizer:
    choices: [adam, sgd, adamw]
  epochs: 50

root_directory: path/to/results       # Directory for result storage
max_evaluations_total: 20               # Budget

searcher: path/to/your/searcher_setup.yaml
```

searcher_setup.yaml:
```yaml
algorithm: bayesian_optimization
# Specific arguments depending on the searcher
initial_design_size: 7
surrogate_model: gp
acquisition: EI
log_prior_weighted: false
acquisition_sampler: random
random_interleave_prob: 0.1
disable_priors: false
prior_confidence: high
sample_default_first: false
```

{{ include('test_yaml_test.yaml') }}

