# Optimizer Configuration

## 1 What optimizer works best for my problem?

The best optimizers utilizes all information available in the search space to guide the optimization process. Besides a fully black-box search, there are two sources of information an optimizer can draw from: using small scale proxies ([Multi-Fidelity](#11-multi-fidelity-mf)) and intuition ([Priors](#12-priors)).

### 1.1 Multi-Fidelity (MF)

Multi-Fidelity uses small scale version of the problem, which run cheaper and faster. This could mean training models for a shorter time, using only a subset of the training data, or a smaller model entirely. From these *low fidelity* runs, MF-algorithms can infer which configurations are likely to perform well on the full problem.

It is defined using the `is_fidelity` parameter in the `pipeline_space` definition.

```python
pipeline_space = {
    "epoch": neps.Integer(lower=1, upper=100, is_fidelity=True),
    # epoch will be available as fidelity to the optimizer
}
```

For a more detailed explanation of Multi-Fidelity and a list of NePS-optimizers using MF please refer [here](../reference/search_algorithms/multifidelity.md).

### 1.2 Priors

Optimization with Priors is used, when there already exists an intuition for what region or specific value of a hyperparameter _could_ work well. By providing this intuition as Prior (knowledge) to the optimizer, it can prioritize these most promising regions of the search space, potentially saving a lot of compute.

It is defined using the `prior` parameter in the `pipeline_space` definition.

```python
pipeline_space = {
    "alpha": neps.Float(lower=0.1, upper=1.0, prior=0.4, prior_confidence="high"),
    # alpha will have a prior pointing towards 0.4 with high confidence
}
```

For a more detailed explanation of Priors and a list of NePS-optimizers using Priors please refer [here](../reference/search_algorithms/prior.md).

## 2 NePS Optimizer Selection

For any given search space, NePS provides a set of predefined optimizers from the literature, that work with Multi-Fidelity, Priors or both.

| Algorithm         | [Multi-Fidelity](../reference/search_algorithms/multifidelity.md) | [Priors](../reference/search_algorithms/prior.md) | Model-based | Asynchronous |
| :- | :------------: | :----: | :---------: | :-: |
| `Grid Search`||||✅|
| `Random Search`||||✅|
| [`Successive Halving`](../reference/search_algorithms/multifidelity.md#1-successive-halfing)|✅||||
| [`ASHA`](../reference/search_algorithms/multifidelity.md#asynchronous-successive-halving)|✅|||✅|
| [`Hyperband`](../reference/search_algorithms/multifidelity.md#2-hyperband)|✅||||
| [`Asynch HB`](../reference/search_algorithms/multifidelity.md)|✅|||✅|
| [`IfBO`](../reference/search_algorithms/multifidelity.md#5-in-context-freeze-thaw-bayesian-optimization)|✅||✅||
| [`PiBO`](../reference/search_algorithms/prior.md#1-pibo)||✅|✅||
| [`PriorBand`](../reference/search_algorithms/multifidelity_prior.md#1-priorband)|✅|✅|✅||

The [algorithms](../reference/search_algorithms/landing_page_algo.md) section goes into detail on the different optimizers, while the rest of this chapter will focus on how to select them when using NePS.

### 2.1 Automatic Optimizer Selection

If you prefer not to specify a particular optimizer for your AutoML task, you can simply pass `"auto"` or `None`
for the neps optimizer. NePS will automatically choose the best optimizer based on the characteristics of your search
space. This provides a hassle-free way to get started quickly.

The optimizer selection is based on the following characteristics of your `pipeline_space`:

- If it has fidelity: [`hyperband`](../reference/search_algorithms/multifidelity.md#2-hyperband)
- If it has both fidelity and a prior: [`priorband`](../reference/search_algorithms/multifidelity_prior.md#1-priorband)
- If it has a prior: [`pibo`](../reference/search_algorithms/prior.md#1-pibo)
- If it has neither: [`bayesian_optimization`](../reference/search_algorithms/bayesian_optimization.md)

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

### 2.2 Choosing one of NePS Optimizers

We have also prepared some optimizers with specific hyperparameters that we believe can generalize well to most AutoML tasks and use cases. The available optimizers are imported via the `neps.algorithms` module.
You can use either the optimizer name or the optimizer class itself as the optimizer argument.

```python
neps.run(
    evalute_pipeline=run_function,
    pipeline_space=pipeline_space,
    root_directory="results/",
    max_evaluations_total=25,
    # optimizer specified, along with an argument
    optimizer=neps.algorithms.bayesian_optimization, # or as string: "bayesian_optimization"
    initial_design_size=5,
)
```

For more optimizers, please refer [here](./optimizers.md#41-list-available-optimizers) .

### 2.3 Custom Optimizer Configuration via YAML

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

For small changes, the user can also directly input a YAML-style dictionary as optimizer argument:

```python
neps.run(
    evalute_pipeline=run_function,
    pipeline_space=pipeline_space,
    root_directory="results/",
    max_evaluations_total=25,
    optimizer={"bayesian_optimization", {"initial_design_size": 5}}
)
```


### 2.4 Hyperparameter Overrides

If you want to make on-the-fly adjustments to the optimizer's hyperparameters without modifying the YAML configuration
file, you can do so by passing keyword arguments (kwargs) to the `neps.run` function itself. This enables you to fine-tune
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

## 3 Note for Contributors

When designing a new optimizer, it's essential to create a YAML configuration file in the `optimizer_yaml_files` folder under `neps.src.optimizers`. This YAML file should contain the default configuration settings that you believe should be used when the user chooses the optimizer.

Even when many hyperparameters might be set to their default values as specified in the code, it is still considered good practice to include them in the YAML file. This is because the `PredefinedOptimizerConfigs` method relies on the arguments from the YAML file to display the optimizer's configuration to the user.

## 4 Optimizer Configurations

The `PredefinedOptimizerConfigs` class provides a set of useful functions to manage and retrieve default configuration details for NePS optimizers. These functions can help you understand and interact with the available optimizers and their associated algorithms and configurations.

### 4.1 Importing `PredefinedOptimizerConfigs`

Before you can use the `PredefinedOptimizerConfigs` class to manage and retrieve default configuration details for NePS optimizers, make sure to import it into your Python script. You can do this with the following import statement:

```python
from neps.optimizers.info import PredefinedOptimizerConfigs
```

Once you have imported the class, you can proceed to use its functions to explore the available optimizers, algorithms, and configuration details.

### 4.1 List Available Optimizers

To list all the available optimizers that can be used in NePS runs, you can use the `get_optimizers` function. It provides you with a list of optimizer names:

```python
optimizers = PredefinedOptimizerConfigs.get_optimizers()
print("Available optimizers:", optimizers)
```

### 4.3 List Available Searching Algorithms

The `get_available_algorithms` function helps you discover the searching algorithms available within the NePS optimizers:

```python
algorithms = PredefinedOptimizerConfigs.get_available_algorithms()
print("Available searching algorithms:", algorithms)
```

### 4.4 Find Optimizers Using a Specific Algorithm

If you want to identify which NePS optimizers are using a specific searching algorithm (e.g., Bayesian Optimization, Hyperband, PriorBand...), you can use the `get_optimizer_from_algorithm` function. It returns a list of optimizers utilizing the specified algorithm:

```python
algorithm = "bayesian_optimization"  # Replace with the desired algorithm
optimizers = PredefinedOptimizerConfigs.get_optimizer_from_algorithm(algorithm)
print(f"optimizers using {algorithm}:", optimizers)
```

### 4.5 Retrieve Optimizer Configuration Details

To access the configuration details of a specific optimizer, you can use the `get_optimizer_kwargs` function. Provide the name of the optimizer you are interested in, and it will return the optimizer's configuration:

```python
optimizer_name = "pibo"  # Replace with the desired NePS optimizer name
optimizer_kwargs = PredefinedOptimizerConfigs.get_optimizer_kwargs(optimizer_name)
print(f"Configuration of {optimizer_name}:", optimizer_kwargs)
```

These functions empower you to explore and manage the available NePS optimizers and their configurations effectively.
