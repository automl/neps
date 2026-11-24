# Getting Started

Getting started with ScaleFM involves a straightforward yet powerful process, centering around its three main components.
This approach ensures flexibility and efficiency in evaluating different architecture and hyperparameter configurations
for your problem.

ScaleFM requires Python 3.11 or higher.
You can install it via `pip` or from [source](https://github.com/automl/scalefm/).

```bash
pip install scalefm
```

## The 3 Main Components

1. **Establish a [`pipeline_space=`](reference/scalefm_spaces.md)**:

    ```python
    class ExampleSpace(scalefm.PipelineSpace):
        # Define the parameters of your search space
        some_parameter = scalefm.Float(lower=0.0, upper=1.0)       # float
        another_parameter = scalefm.Integer(lower=0, upper=10)     # integer
        optimizer = scalefm.Categorical(choices=("sgd", "adam"))           # categorical
        epoch = scalefm.Integer(lower=1, upper=100)
        learning_rate = scalefm.Float(lower=1e-5, upper=1, log=True)
        alpha = scalefm.Float(lower=0.1, upper=1.0)
    ```

1. **Define an `evaluate_pipeline()` function**:

    ```python
    def evaluate_pipeline(some_parameter: float,
                    another_parameter: float,
                    optimizer: str, epoch: int,
                    learning_rate: float, alpha: float) -> dict:
        model = make_model(...)
        loss = eval_model(model)
        return {"objective_to_minimize": loss, "cost": training_flops(model)}
    ```

1. **Execute with [`scalefm.acquire()`](reference/scalefm_acquire.md)**:

    ```python
    scalefm.acquire(evaluate_pipeline, ExampleSpace(), live_plots=True)
    ```

---

## What's Next?

The [reference](reference/scalefm_acquire.md) section provides detailed information on the individual components of ScaleFM.

1. How to use the [**`scalefm.acquire()`** function](reference/scalefm_acquire.md) to start the optimization process.
2. The different [search space](reference/scalefm_spaces.md) options available.
3. How to choose and configure the [optimizer](reference/optimizers.md) used.
4. How to define the [`evaluate_pipeline()` function](reference/evaluate_pipeline.md).
5. How to [analyze](reference/analyse.md) the optimization runs.

!!! tip "Interactive tutorial"

    Or try ScaleFM hands-on with the [ScaleFM Demo](https://colab.research.google.com/github/automl/scalefm/blob/master/tutorials/scalefm_demo.ipynb), a basic demo of the pipeline, the visualization and the parallelisation of the runs.

Or discover the features of ScaleFM through these practical examples:

- **[Scaling study of TabPFN](examples/real_world/tabpfn_scan_bench.md):** a random search under a FLOP budget, with the scaling law fitted and extrapolated.
- **[Asynchronous evaluation](examples/async_evaluation/submit.md):** submit each configuration as a Slurm job and report its result back with `scalefm.save_pipeline_results()`.
- **[Scaling study of an OpenCLIP VLM](https://github.com/automl/scalefm/tree/master/scalefm_examples/real_world/vlm_openclip):** vision and text towers of varying width and depth, each trained in a Slurm job sized to its GPU memory needs.
- **[Distributed training](examples/efficiency/index.md):** evaluate configurations with DDP or FSDP, in native PyTorch or PyTorch Lightning.
