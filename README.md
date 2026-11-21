# Extrapolaw

[![PyPI version](https://img.shields.io/pypi/v/extrapolaw?color=informational)](https://pypi.org/project/extrapolaw/)
[![Python versions](https://img.shields.io/pypi/pyversions/extrapolaw)](https://pypi.org/project/extrapolaw/)
[![License](https://img.shields.io/pypi/l/extrapolaw?color=informational)](LICENSE)
[![Tests](https://github.com/automl/extrapolaw/actions/workflows/tests.yaml/badge.svg)](https://github.com/automl/extrapolaw/actions)

Extrapolaw is a tool for scaling studies of deep learning pipelines: it searches how architectural and hyperparameter choices behave as you scale models, data, and compute, and fits scaling laws to predict the best choices at larger scales.
Use it from a single GPU to a multi-node cluster or even multiple clusters.

Extrapolaw unifies established and novel scaling prescriptions within the ScAn framework and pairs them with a runtime tailored to large-scale models. Extrapolaw is actively maintained and used to run on many different clusters, tuning even billion-parameter scale models with many concurrent trials.

To learn about Extrapolaw, check out [the documentation](https://automl.github.io/extrapolaw/latest/) or [our examples](extrapolaw_examples/).

## Why Extrapolaw

### Scaling laws built in

- **Search on a budget, predict at scale:** scaling laws are integrated directly into the optimization loop, so that results obtained at small scale inform predictions of optimal configurations at larger scales.
- **Cost-constrained sampling:** optimizers restrict sampling to the feasible region of the search space, i.e., configurations that satisfy a user-specified black-box cost constraint.
- **Predicting the optimal configuration at scale:** Extrapolaw answers the central question of any scaling study, namely which configuration to train at a target scale where a direct search is prohibitively expensive.

### Tailored to large scale models

- **Tuning distributed models:** Extrapolaw works with [DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [FSDP](https://docs.pytorch.org/tutorials/intermediate/FSDP1_tutorial.html), on a single node or across multiple nodes, out of the box ([examples](extrapolaw_examples/efficiency/)).
- **Zero-effort to run many concurrent models:** start more workers on the same machine or in a multi-node setup. As long as they share the results directory, they coordinate on their own, with no server to set up.
- **Live monitoring and interventions:** follow a run with live plots or [TensorBoard](https://automl.github.io/extrapolaw/latest/reference/analyse/#visualizing-results), and steer it without starting over: add workers, extend the budget, re-run failed trials, or import tuning results from anywhere (even cross-cluster).

### Data acquisition and prediction algorithms

- **[Grounded prescriptions](https://automl.github.io/extrapolaw/latest/reference/optimizers/#3-scaling-optimizers):** established prescriptions from the literature, namely Chinchilla (Hoffmann et al., 2022), Kaplan (Kaplan et al., 2020), and CARBS (Fetterman et al., 2023), adapted to budget-aware search and complemented by our novel algorithm, ScAn-BO.
- **Model-based search:** strategies such as Bayesian optimization choose promising configurations smartly instead of sampling statically (e.g., Chinchilla) or at random.

### Generally applicable

- **Any design space:** hyperparameters, architectures, resource allocation, or any component of the pipeline.
- **Any scaling dimension:** scale model size, dataset size, or any other quantity that sets the compute (`is_compute_scaling=True`).
- **Downstream tasks:** follow how downstream performances scale next to the pre-training loss.
- **Training or inference cost:** predictions can be made with respect to either training or inference compute.

## Installation

Extrapolaw supports Python 3.11 to 3.14. Install the latest release from PyPI:

```bash
pip install extrapolaw
```

## Basic Usage

Using `extrapolaw` is based on the following pattern:

1. Define an `evaluate_pipeline` function that submits a training job for one configuration; the job reports its result back with `extrapolaw.save_pipeline_results()`.
1. Define a `pipeline_space` of the parameters to search, marking those that scale the compute with `is_compute_scaling=True`.
1. Call `extrapolaw.acquire()` with a `cost_estimator` and a cost budget to acquire data under a cost constraint.
1. Call `extrapolaw.predict()` to extrapolate the results to the target scale(s).

In code, the usage pattern can look like this:

```python
import subprocess

import extrapolaw

ROOT_DIRECTORY = "path/to/save/results"  # Replace with the actual path


# 1. Submit one configuration as a multi-GPU job and return immediately (async)
def evaluate_pipeline(pipeline_id: str, **config):  # Extrapolaw injects `pipeline_id`
    args = " ".join(f"--{name} {value}" for name, value in config.items())
    subprocess.run(
        f"sbatch --gpus=8 --wrap='torchrun --nproc_per_node=8 train.py "
        f"--pipeline_id {pipeline_id} --root_directory {ROOT_DIRECTORY} {args}'",
        shell=True,
        check=True,
    )
    return None  # The job reports its result itself, see train.py below


# 2. Define the search space; the names match the arguments of the training job
class LMSpace(extrapolaw.PipelineSpace):
    lr = extrapolaw.Float(lower=1e-4, upper=1e-2, log=True, prior=3e-4)
    batch_size = extrapolaw.Integer(lower=64, upper=1024, log=True, log_base=2)
    width = extrapolaw.Integer(lower=256, upper=4096, log=True, is_compute_scaling=True)
    n_layers = extrapolaw.Integer(lower=4, upper=48, is_compute_scaling=True)
    n_tokens = extrapolaw.Integer(lower=10**8, upper=10**11, log=True, is_compute_scaling=True)


# The training FLOPs of a configuration, known before it is evaluated
def estimate_flops(width: int, n_layers: int, n_tokens: int) -> float:
    return 6 * (12 * n_layers * width**2) * n_tokens


# 3. Acquire data under the cost constraint
extrapolaw.acquire(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=LMSpace(),
    root_directory=ROOT_DIRECTORY,
    cost_estimator=estimate_flops,  # Cost of every configuration
    max_cost_per_config=1e19,  # Never sample a configuration costing more than this
    total_cost_to_spend=1e20,  # Budget for the whole search
    is_async=True,  # Keep sampling while earlier jobs are still running
    live_plots=True,  # Update the data-acquisition plots during the run
)

# 4. Predict the optimal performance and configuration at the target scale(s)
predictions = extrapolaw.predict(
    root_directory=ROOT_DIRECTORY,
    target_costs=[1e22, 1e23],
)  # The plots and their data are also saved to `root_directory/summary/`
```

The training job, `train.py`, reports its result back to Extrapolaw when it finishes:

```python
from pathlib import Path

import extrapolaw

args = parse_args()  # lr, batch_size, width, n_layers, n_tokens, pipeline_id, root_directory
model = build_transformer(width=args.width, n_layers=args.n_layers)  # Wrapped in DDP or FSDP
val_loss = train_and_eval(model, lr=args.lr, batch_size=args.batch_size, n_tokens=args.n_tokens)

if is_main_process():
    extrapolaw.save_pipeline_results(
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

See [the `evaluate_pipeline` reference](https://automl.github.io/extrapolaw/latest/reference/evaluate_pipeline/) for the full asynchronous workflow.

## Resources to Get Started

### Examples

- **[Scaling study of TabPFN](extrapolaw_examples/real_world/tabpfn_scan_bench.py):** a random search under a FLOP budget, with the scaling law fitted and extrapolated.
- **[Asynchronous evaluation](extrapolaw_examples/async_evaluation/):** submit each configuration as a Slurm job and report its result back with `extrapolaw.save_pipeline_results()`.
- **[Scaling study of an OpenCLIP VLM](extrapolaw_examples/real_world/vlm_openclip/):** vision and text towers of varying width and depth, each trained in a Slurm job sized to its GPU memory needs.
- **[Distributed training](extrapolaw_examples/efficiency/):** evaluate configurations with DDP or FSDP, in native PyTorch or PyTorch Lightning.

### Documentation

- [Getting started](https://automl.github.io/extrapolaw/latest/getting_started/)
- [Reference](https://automl.github.io/extrapolaw/latest/reference/extrapolaw_acquire/): running Extrapolaw, search spaces, optimizers, and analysing runs
- [Scaling optimizers](https://automl.github.io/extrapolaw/latest/reference/optimizers/#3-scaling-optimizers): cost-constrained search guided by scaling laws
- [Algorithms](https://automl.github.io/extrapolaw/latest/reference/search_algorithms/landing_page_algo/)
- [API](https://automl.github.io/extrapolaw/latest/api/extrapolaw/api/)

## Contributing

Please see the [documentation for contributors](https://automl.github.io/extrapolaw/latest/dev_docs/contributing/).

## Citing Extrapolaw

To cite Extrapolaw or the papers behind its algorithms, see our [citation guide](https://automl.github.io/extrapolaw/latest/citations/).
