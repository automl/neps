# Declarative Usage in NePS for Neural Network Optimization

This folder contains examples and templates for optimizing neural networks using NePS, configured via YAML files.
These configurations allow easy adjustments to experiment parameters and search spaces, enabling fine-tuning of your
models without modifying Python code.
### `hpo_example.py`

This Python script demonstrates how to integrate NePS with a neural network training pipeline for hyperparameter
optimization. It utilizes a YAML configuration file to set up and run the experiments.

```python
--8<-- "neps_examples/convenience/declarative_usage/hpo_example.py"
```

### `config.yaml`

This YAML file defines the NePS arguments for the experiment. By editing this file, users can customize their
experiments without modifying the Python script.

```yaml
--8<-- "neps_examples/convenience/declarative_usage/config.yaml"
```

## Quick Start Guide

1. **Review the YAML File:** Examine `config.yaml` to understand the available configurations and how they are structured.
2. **Run the Example Script:** Execute hpo_example.py, by providing `config.yaml` via the run_args agrument to NePS.
   This will initiate a hyperparameter optimization task based on your YAML configurations.
3. **Modify YAML File:** Experiment with adjusting the parameters in the YAML file to see how changes affect your
   search experiments. This is a great way to learn about the impact of different configurations on your results.

By following these steps and utilizing the provided YAML files, you'll be able to efficiently set up, run, and modify your NePS experiments. Enjoy the flexibility and simplicity that comes with managing your experiment configurations in YAML!
