# YAML Usage in NePS for MNIST Classification

Within this folder, you'll find examples and templates tailored for optimizing neural networks on the MNIST dataset
through NePS, all configured via YAML files. These YAML files allow for the easy adjustment of experiment parameters
and search spaces, enabling you to fine-tune your classification model without directly modifying any Python code.
Below is a succinct overview of each provided file and its role in the optimization process:
### `hpo_example.py`

This Python script demonstrates how to utilize a YAML file to configure a hyperparameter optimization (HPO) task in
NePS. It's designed to show you how to integrate YAML configurations with your Python code to set up HPO experiments.
The script reads YAML files to get its parameters and search spaces, illustrating how to run HPO tasks with minimal
coding effort. Additionally, it includes an example of a run_pipeline function, guiding users on how to define their
model training and evaluation process within the optimization framework.
### `pipeline_space.yaml`

This YAML file defines the search space for your pipeline, including the parameters and their ranges to be explored
during the search. It's structured to be easily readable and modifiable, allowing you to quickly adjust the search
space for your experiments. This file is referenced by the configuration YAML file `run_args.yaml` to dynamically load
the search configuration.

### `run_args.yaml`

This file showcases an example set of NePS arguments, simplifying its usage. By editing this file,
users can customize their experiments without needing to modify the Python script.

### `run_args_alternative.yaml`

An alternative set of arguments. This file has listed the full set of arguments and includes explanations for each of
them. Additionally, it demonstrates how you can structure your YAML file for improved clarity.

## Quick Start Guide

1. **Review the YAML Files:** Start by looking at `pipeline_space.yaml`, `run_args.yaml`, and `run_args_alternative.yaml` to understand the available configurations and how they're structured.
2. **Run the Example Script:** Execute `hpo_example.py`, specifying which YAML file to use for the run arguments. This will initiate an HPO task based on your YAML configurations.
3. **Modify YAML Files:** Experiment with adjusting the parameters in the YAML files to see how changes affect your search experiments. This is a great way to learn about the impact of different configurations on your results.

By following these steps and utilizing the provided YAML files, you'll be able to efficiently set up, run, and modify your NePS experiments. Enjoy the flexibility and simplicity that comes with managing your experiment configurations in YAML!
