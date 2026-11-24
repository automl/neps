# ScaleFM

[![PyPI version](https://img.shields.io/pypi/v/scalefm?color=informational)](https://pypi.org/project/scalefm/)
[![Python versions](https://img.shields.io/pypi/pyversions/scalefm)](https://pypi.org/project/scalefm/)
[![License](https://img.shields.io/pypi/l/scalefm?color=informational)](LICENSE)
[![Tests](https://github.com/automl/scalefm/actions/workflows/tests.yaml/badge.svg)](https://github.com/automl/scalefm/actions)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/automl/scalefm/blob/master/tutorials/scalefm_demo.ipynb)

ScaleFM is a tool for scaling studies of deep learning pipelines: it searches how architectural and hyperparameter choices behave as you scale models, data, and compute, and fits scaling laws to predict the best choices at larger scales.
Use it from a single GPU to a multi-node cluster or even multiple clusters.

ScaleFM unifies established and novel scaling prescriptions within the ScAn framework and pairs them with a runtime tailored to large-scale models. ScaleFM is actively maintained and used to run on many different clusters, tuning even billion-parameter scale models with many concurrent trials.

To learn about ScaleFM, check out [the documentation](https://automl.github.io/scalefm/latest/) or [our examples](scalefm_examples/).

## Why ScaleFM

### Scaling laws built in

- **Search on a budget, predict at scale:** scaling laws are integrated directly into the optimization loop, so that results obtained at small scale inform predictions of optimal configurations at larger scales.
- **Cost-constrained sampling:** optimizers restrict sampling to the feasible region of the search space, i.e., configurations that satisfy a user-specified black-box cost constraint.
- **Predicting the optimal configuration at scale:** ScaleFM answers the central question of any scaling study, namely which configuration to train at a target scale where a direct search is prohibitively expensive.

### Tailored to large scale models

- **Tuning distributed models:** ScaleFM works with [DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [FSDP](https://docs.pytorch.org/tutorials/intermediate/FSDP1_tutorial.html), on a single node or across multiple nodes, out of the box ([examples](scalefm_examples/efficiency/)).
- **Zero-effort to run many concurrent models:** start more workers on the same machine or in a multi-node setup. As long as they share the results directory, they coordinate on their own, with no server to set up.
- **Live monitoring and interventions:** follow a run with live plots or [TensorBoard](https://automl.github.io/scalefm/latest/reference/analyse/#visualizing-results), and steer it without starting over: add workers, extend the budget, re-run failed trials, or import tuning results from anywhere (even cross-cluster).

### Data acquisition and prediction algorithms

- **[Grounded prescriptions](https://automl.github.io/scalefm/latest/reference/optimizers/#3-scaling-optimizers):** established prescriptions from the literature, namely Chinchilla (Hoffmann et al., 2022), Kaplan (Kaplan et al., 2020), and CARBS (Fetterman et al., 2023), adapted to budget-aware search and complemented by our novel algorithm, ScAn-BO.
- **Model-based search:** strategies such as Bayesian optimization choose promising configurations smartly instead of sampling statically (e.g., Chinchilla) or at random.

### Generally applicable

- **Any design space:** hyperparameters, architectures, resource allocation, or any component of the pipeline.
- **Any scaling dimension:** scale model size, dataset size, or any other quantity that sets the compute (`is_compute_scaling=True`).
- **Downstream tasks:** follow how downstream performances scale next to the pre-training loss.
- **Training or inference cost:** predictions can be made with respect to either training or inference compute.

## Installation

ScaleFM supports Python 3.11 to 3.14. Install the latest release from PyPI:

```bash
pip install scalefm
```

## Basic Usage

Using `scalefm` is based on the following pattern:

1. Define a `pipeline_space` of the parameters to search, marking those that scale the compute with `is_compute_scaling=True`.
1. Call `scalefm.acquire()` with a `cost_estimator` and a cost budget to acquire data under a cost constraint. With `is_async=True` it only samples the configurations; train each one as a job, which reports its result back with `scalefm.save_pipeline_results()`. Without it, pass an `evaluate_pipeline` function that trains a configuration and returns its result.
1. Call `scalefm.predict()` to extrapolate the results to the target scale(s).

In code, the usage pattern can look like this:

```python
import subprocess
from pathlib import Path

import yaml

import scalefm

ROOT_DIRECTORY = Path("path/to/save/results")  # Replace with the actual path


# 1. Define the search space; the names match the arguments of the training job
class LMSpace(scalefm.PipelineSpace):
    lr = scalefm.Float(lower=1e-4, upper=1e-2, log=True, prior=3e-4)
    batch_size = scalefm.Integer(lower=64, upper=1024, log=True, log_base=2)
    width = scalefm.Integer(lower=256, upper=4096, log=True, is_compute_scaling=True)
    n_layers = scalefm.Integer(lower=4, upper=48, is_compute_scaling=True)
    n_tokens = scalefm.Integer(lower=10**8, upper=10**11, log=True, is_compute_scaling=True)


# The training FLOPs of a configuration, known before it is evaluated
def estimate_flops(width: int, n_layers: int, n_tokens: int) -> float:
    return 6 * (12 * n_layers * width**2) * n_tokens


# 2. Acquire data under the cost constraint: sample a round of configurations
scalefm.acquire(
    pipeline_space=LMSpace(),
    root_directory=ROOT_DIRECTORY,
    cost_estimator=estimate_flops,  # Cost of every configuration
    max_cost_per_config=1e19,  # Never sample a configuration costing more than this
    total_cost_to_spend=1e20,  # Budget for the whole search
    is_async=True,  # Only sample, the jobs below train the configurations
    worker_evaluations_to_spend=8,  # Configurations in this round
)

# 3. Train every new configuration as a multi-GPU job, which reports its result itself
for config_dir in sorted((ROOT_DIRECTORY / "configs").glob("config_*")):
    submitted = config_dir / "submitted"
    if submitted.exists():
        continue
    config = yaml.safe_load((config_dir / "config.yaml").read_text())
    args = " ".join(f"--{name} {value}" for name, value in config.items())
    pipeline_id = config_dir.name.removeprefix("config_")
    subprocess.run(
        f"sbatch --gpus=8 --wrap='torchrun --nproc_per_node=8 train.py "
        f"--pipeline_id {pipeline_id} --root_directory {ROOT_DIRECTORY} {args}'",
        shell=True,
        check=True,
    )
    submitted.touch()

# 4. Once the jobs have reported, predict the optimal performance and configuration
#    at the target scale(s)
predictions = scalefm.predict(
    root_directory=ROOT_DIRECTORY,
    target_costs=[1e22, 1e23],
)  # The plots and their data are also saved to `root_directory/summary/`
```

The training job, `train.py`, reports its result back to ScaleFM when it finishes:

```python
from pathlib import Path

import scalefm

args = parse_args()  # lr, batch_size, width, n_layers, n_tokens, pipeline_id, root_directory
model = build_transformer(width=args.width, n_layers=args.n_layers)  # Wrapped in DDP or FSDP
val_loss = train_and_eval(model, lr=args.lr, batch_size=args.batch_size, n_tokens=args.n_tokens)

if is_main_process():
    scalefm.save_pipeline_results(
        user_result={
            "objective_to_minimize": val_loss,
            "cost": 6 * model.n_params * args.n_tokens,  # Training FLOPs
            "n_params": model.n_params,
            "n_data": args.n_tokens,
        },
        pipeline_id=args.pipeline_id,
        root_directory=Path(args.root_directory),
        live_plots=True,  # Redraw the scaling plots with this result
    )
```

See the [`evaluate_pipeline`](https://automl.github.io/scalefm/latest/reference/evaluate_pipeline/) reference for the full asynchronous workflow.

## Resources to Get Started

### Tutorial

The [ScaleFM Demo](https://colab.research.google.com/github/automl/scalefm/blob/master/tutorials/scalefm_demo.ipynb) is a basic demo of what ScaleFM can do, on a TabPFN surrogate benchmark that runs on a CPU in a few minutes. It covers the basic 
pipeline, visualization, and parallelisation of the runs.

To run it locally instead, see the [tutorials folder](tutorials/).

### Examples

- **[Scaling study of TabPFN](scalefm_examples/real_world/tabpfn_scan_bench.py):** a random search under a FLOP budget, with the scaling law fitted and extrapolated.
- **[Asynchronous evaluation](scalefm_examples/async_evaluation/):** submit each configuration as a Slurm job and report its result back with `scalefm.save_pipeline_results()`.
- **[Scaling study of an OpenCLIP VLM](scalefm_examples/real_world/vlm_openclip/):** vision and text towers of varying width and depth, each trained in a Slurm job sized to its GPU memory needs.
- **[Distributed training](scalefm_examples/efficiency/):** evaluate configurations with DDP or FSDP, in native PyTorch or PyTorch Lightning.

### Documentation

- **[Getting started](https://automl.github.io/scalefm/latest/getting_started/)**
- **[Reference](https://automl.github.io/scalefm/latest/reference/scalefm_acquire/):** running ScaleFM, search spaces, optimizers, and analysing runs
- **[Scaling optimizers](https://automl.github.io/scalefm/latest/reference/optimizers/#3-scaling-optimizers):** cost-constrained search guided by scaling laws
- **[Algorithms](https://automl.github.io/scalefm/latest/reference/search_algorithms/landing_page_algo/)**
- **[API](https://automl.github.io/scalefm/latest/api/scalefm/api/)**

## Contributing

Please see the [documentation for contributors](https://automl.github.io/scalefm/latest/dev_docs/contributing/).

## Citing ScaleFM

To cite ScaleFM or the papers behind its algorithms, see our [citation guide](https://automl.github.io/scalefm/latest/citations/).
