# Optimizer Configuration

Before running the optimizer for your AutoML tasks, you have several configuration options to tailor the optimization
process to your specific needs. These options allow you to customize the optimizer's behavior according to your
preferences and requirements.

### 1. Automatic Optimizer Selection

If you prefer not to specify a particular optimizer for your AutoML task, you can simply pass `"default"` or `None`
for the neps optimizer. NePS will automatically choose the best optimizer based on the characteristics of your search
space. This provides a hassle-free way to get started quickly.

The optimizer selection is based on the following characteristics of your `pipeline_space`:

- If it has fidelity: `hyperband`
- If it has both fidelity and a prior: `priorband`
- If it has a prior: `pibo`
- If it has neither: `bayesian_optimization`

For example, running the following format, without specifying a optimizer will choose an optimizer depending on
the `pipeline_space` passed.
```python
neps.run(
    evalute_pipeline=run_function,
    pipeline_space=pipeline_space,
    root_directory="results/",
    max_evaluations_total=25,
    # no optimizer specified
)
```

### 2. Choosing one of NePS Optimizers

We have also prepared some optimizers with specific hyperparameters that we believe can generalize well to most AutoML
tasks and use cases. For more details on the available default optimizers and the algorithms that can be called,
please refer to the next section on [PredefinedOptimizerConfigs](#optimizer-configurations).

```python
neps.run(
    evalute_pipeline=run_function,
    pipeline_space=pipeline_space,
    root_directory="results/",
    max_evaluations_total=25,
    # optimizer specified, along with an argument
    optimizer="bayesian_optimization",
    initial_design_size=5,
)
```

For more optimizers, please refer [here](#list-available-optimizers) .

### 3. Custom Optimizer Configuration via YAML

For users who want more control over the optimizer's hyperparameters, you can create your own YAML configuration file.
In this file, you can specify the hyperparameters for your preferred optimizer. To use this custom configuration,
provide the path to your YAML file using the `optimizer` parameter when running the optimizer.
The library will then load your custom settings and use them for optimization.

Here's the format of a custom YAML (`custom_bo.yaml`) configuration using `Bayesian Optimization` as an example:

```yaml
name: bayesian_optimization
initial_design_size: 7
surrogate_model: gp
acquisition: EI
log_prior_weighted: false
random_interleave_prob: 0.1
disable_priors: false
prior_confidence: high
sample_prior_first: false
```

```python
neps.run(
    evalute_pipeline=run_function,
    pipeline_space=pipeline_space,
    root_directory="results/",
    max_evaluations_total=25,
    optimizer="path/to/custom_bo.yaml",
)
```

### 4. Hyperparameter Overrides

If you want to make on-the-fly adjustments to the optimizer's hyperparameters without modifying the YAML configuration
file, you can do so by passing keyword arguments (kwargs) to the neps.run function itself. This enables you to fine-tune
specific hyperparameters without the need for YAML file updates. Any hyperparameter values provided as kwargs will take
precedence over those specified in the YAML configuration.

```python
neps.run(
    evalute_pipeline=run_function,
    pipeline_space=pipeline_space,
    root_directory="results/",
    max_evaluations_total=25,
    optimizer="path/to/custom_bo.yaml",
    initial_design_size=5,        # overrides value in custom_bo.yaml
    random_interleave_prob=0.25  # overrides value in custom_bo.yaml
)
```

## Note for Contributors

When designing a new optimizer, it's essential to create a YAML configuration file in the `optimizer_yaml_files` folder under `neps.src.optimizers`. This YAML file should contain the default configuration settings that you believe should be used when the user chooses the optimizer.

Even when many hyperparameters might be set to their default values as specified in the code, it is still considered good practice to include them in the YAML file. This is because the `PredefinedOptimizerConfigs` method relies on the arguments from the YAML file to display the optimizer's configuration to the user.

## Optimizer Configurations

The `PredefinedOptimizerConfigs` class provides a set of useful functions to manage and retrieve default configuration details for NePS optimizers. These functions can help you understand and interact with the available optimizers and their associated algorithms and configurations.

### Importing `PredefinedOptimizerConfigs`

Before you can use the `PredefinedOptimizerConfigs` class to manage and retrieve default configuration details for NePS optimizers, make sure to import it into your Python script. You can do this with the following import statement:

```python
from neps.optimizers.info import PredefinedOptimizerConfigs
```

Once you have imported the class, you can proceed to use its functions to explore the available optimizers, algorithms, and configuration details.

### List Available Optimizers

To list all the available optimizers that can be used in NePS runs, you can use the `get_optimizers` function. It provides you with a list of optimizer names:

```python
optimizers = PredefinedOptimizerConfigs.get_optimizers()
print("Available optimizers:", optimizers)
```

### List Available Searching Algorithms

The `get_available_algorithms` function helps you discover the searching algorithms available within the NePS optimizers:

```python
algorithms = PredefinedOptimizerConfigs.get_available_algorithms()
print("Available searching algorithms:", algorithms)
```

### Find Optimizers Using a Specific Algorithm

If you want to identify which NePS optimizers are using a specific searching algorithm (e.g., Bayesian Optimization, Hyperband, PriorBand...), you can use the `get_optimizer_from_algorithm` function. It returns a list of optimizers utilizing the specified algorithm:

```python
algorithm = "bayesian_optimization"  # Replace with the desired algorithm
optimizers = PredefinedOptimizerConfigs.get_optimizer_from_algorithm(algorithm)
print(f"optimizers using {algorithm}:", optimizers)
```

### Retrieve Optimizer Configuration Details

To access the configuration details of a specific optimizer, you can use the `get_optimizer_kwargs` function. Provide the name of the optimizer you are interested in, and it will return the optimizer's configuration:

```python
optimizer_name = "pibo"  # Replace with the desired NePS optimizer name
optimizer_kwargs = PredefinedOptimizerConfigs.get_optimizer_kwargs(optimizer_name)
print(f"Configuration of {optimizer_name}:", optimizer_kwargs)
```

These functions empower you to explore and manage the available NePS optimizers and their configurations effectively.
