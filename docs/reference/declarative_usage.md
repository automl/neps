## Introduction
If you prefer to use yaml for experiment configuration,
[`neps.run()`][neps.api.run] supports yaml serialized input.

We make no assumption on how you'd like to structure you experimentation
and you are free to run it as you wish!

Please check [`neps.run()`][neps.api.run] for complete information on the arguments.

#### Simple YAML Example
Below is a YAML configuration example for NePS covering the required arguments.
The arguments match those that you can pass to [`neps.run()`][neps.api.run].

=== "config.yaml"

    ```yaml
    # Basic NePS Configuration Example
    pipeline_space:

      batch_size: 64  # Constant

      learning_rate:  # Float
        lower: 1e-5
        upper: 1e-1
        log: true  # Log scale for learning rate

      optimizer:      # Categorical
        choices: [adam, sgd, adamw]

      epochs:         # Integer
        lower: 5
        upper: 20
        is_fidelity: true

    root_directory: path/to/results       # Directory for result storage
    max_evaluations_total: 20             # Budget

    optimizer:
        name: hyperband  # Which optimizer to use
    ```

=== "run_neps.py"

    ```python
    import neps
    import yaml

    def run_pipeline(learning_rate, optimizer, epochs, batch_size):
        model = initialize_model()
        training_loss = train_model(model, optimizer, learning_rate, epochs)
        evaluation_loss = evaluate_model(model)
        return {"loss": evaluation_loss, "training_loss": training_loss}

    if __name__ == "__main__":
        with open("path/config.yaml") as f:
            settings = yaml.safe_load(f)

        neps.run(run_pipeline, **settings)
    ```

!!! tip "Merging multiple yaml files"

    If you would like to seperate parts of your configuration into multiple yamls,
    for example, to seperate out your search spaces and optimizers,
    you can use the `neps.load_yamls` function to merge them, checking for conflicts.

    ```python
    import neps

    def run_pipeline(...):
        ...

    if __name__ == "__main__":
        settings = neps.load_yamls("path/to/your/config.yaml", "path/to/your/optimizer.yaml")
        neps.run(run_pipeline, **settings)
    ```


#### Comprehensive YAML Configuration Template
This example showcases a more comprehensive YAML configuration, which includes not only the essential parameters
but also advanced settings for more complex setups.

=== "config.yaml"

    ```yaml
    # Full Configuration Template for NePS
    evaluate_pipeline: path/to/your/evaluate_pipeline.py::example_pipeline

    pipeline_space:
      learning_rate:
        lower: 1e-5
        upper: 1e-1
        log: true
      epochs:
        lower: 5
        upper: 20
        is_fidelity: true
      optimizer:
        choices: [adam, sgd, adamw]
      batch_size: 64

    root_directory: path/to/results       # Directory for result storage
    max_evaluations_total: 20             # Budget
    max_cost_total:

    # Debug and Monitoring
    overwrite_working_directory: true
    post_run_summary: false

    # Parallelization Setup
    max_evaluations_per_run:
    continue_until_max_evaluation_completed: false

    # Error Handling
    objective_value_on_error:
    cost_value_on_error:
    ignore_errors:

    optimizer:
        name: hyperband
    ```

=== "run_neps.py"

    ```python

    if __name__ == "__main__":
        import neps

        with open("path/config.yaml") as f:
            settings = yaml.safe_load(f)

        # Note, we specified our run function in the yaml itself!
        neps.run(**settings)
    ```

## CLI Usage

!!! warning "CLI Usage"

    The CLI is still in development and may not be fully functional.
