!!! note "Work in Progress"
    This document is currently a work in progress and may contain incomplete or preliminary information.

## Introduction
### Configuring with YAML

Configure your experiments by specifying settings in a YAML file. This file becomes the single source for
your project setup, making it easy to share, reproduce, and modify them.

Add yaml tutorial link

#### Simple YAML Example
Here’s a basic example of how a YAML configuration for NePS looks:
```yaml
--8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/simple_example.yaml"
```

#### Executing the Configuration
To run the experiment defined in your YAML, use this simple command in Python:
```python
import neps
def run_pipeline(learning_rate, optimizer, epochs):
    pass
neps.run(run_pipeline, run_args="path/to/your/config.yaml")
```


### Including run_pipeline
```yaml
--8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/simple_example_including_run_pipeline.yaml"
```
```python
import neps
def run_pipeline():
    pass
neps.run(run_args="path/to/your/config.yaml")
```

### Extended Configuration
```yaml
--8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/full_configuration_template.yaml"
```
explain what happens with undefined keys?

## Different Use Cases
### Customizing neps optimizer
```yaml
--8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/customizing_neps_optimizer.yaml"
```
TODO
information where to find parameters of included searcher, where to find optimizers names...link


### Load your own optimizer
```yaml
--8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/loading_own_optimizer.yaml"
```
### How to define hooks?

```yaml
--8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/defining_hooks.yaml"
```
### What if your search space is big?
```yaml
--8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/outsourcing_pipeline_space.yaml"
```

pipeline_space.yaml
```yaml
--8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/pipeline_space.yaml"
```

### If you experimenting a lot with different optimizer settings
```yaml
--8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/outsourcing_optimizer.yaml"
```

searcher_setup.yaml:
```yaml
--8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/set_up_optimizer.yaml"
```

### Architecture search space (Loading Dict)
```yaml
--8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/loading_pipeline_space_dict.yaml"
```
search_space.py
```python
search_space = {}
```


