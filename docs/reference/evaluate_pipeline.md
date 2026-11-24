# The `evaluate_pipeline` function

> **TL;DR**
> *Sync*: return a dict with at least the loss and the training cost ⟶ ScaleFM records it automatically.
> *Async*: call `scalefm.acquire(evaluate_pipeline=None, is_async=True)` to only sample configurations, launch a job for each, and call `scalefm.save_pipeline_results()` when the job finishes.
> *Plots*: pass `live_plots=True` to `scalefm.acquire()` (sync) or to `scalefm.save_pipeline_results()` (async) to redraw them after every result.

---

## 1 Return types

| Allowed return  | When to use                                 | Minimal example                                  |
| --------------- | ------------------------------------------- | ------------------------------------------------ |
| **Dict**        | every evaluation                            | `{"objective_to_minimize": loss, "cost": flops}` |
| **Exception**   | the evaluation failed (raise or return it)  | `raise RuntimeError("diverged")`                 |

A plain number is not enough for a scaling study, so any other value, `None` or a
single float included, stops the run with a `ValueError`. To train the configurations
elsewhere (Slurm, k8s, ...), use the asynchronous mode below instead. Only with a `cost_estimator`
passed to `scalefm.acquire()` may you return just the loss, ScaleFM then adds
the estimated cost.

## 2 Result dictionary keys

| key                       | purpose                                                                                          | required?                                      |
| ------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------- |
| `objective_to_minimize`   | the loss ScaleFM minimises                                                                    | **yes**                                        |
| `cost`                    | the training cost, e.g. FLOPs; also counts towards `total_cost_to_spend`                         | **yes**, unless a `cost_estimator` provides it |
| `inference_cost`          | the cost of using the model once, e.g. the FLOPs of one forward pass; the scaling plots are also drawn over it | optional                         |
| `n_params`                | the model size; `scalefm.predict()` fits `chinchilla3` and the model size and data on it       | optional, **yes** for those fits               |
| `n_data`                  | the amount of training data, e.g. tokens, samples or table cells; as `n_params`                   | optional, **yes** for those fits               |
| `downstream_performances` | the model's performance on downstream tasks, by name, e.g. `{"hellaswag_acc": 0.41}`; each gets its own scaling plot | optional                   |
| `learning_curve`          | list of intermediate objectives                                                                  | optional                                       |
| `info_dict`               | any YAML‑serialisable dict, saved as `extra` in the trial’s `report.yaml`                        | optional                                       |
| `exception`               | an exception describing a failed evaluation                                                      | optional                                       |

```python
def evaluate_pipeline(width: int, n_steps: int, learning_rate: float) -> dict:
    ...  # train the model
    return {
        "objective_to_minimize": val_loss,
        "cost": 6 * n_params * n_tokens,  # training FLOPs
        "inference_cost": 2 * n_params,  # FLOPs of one forward pass
        "n_params": n_params,
        "n_data": n_tokens,
        "downstream_performances": {"val_acc": val_acc},
    }
```

Metrics whose name contains `_acc` or `recall` count as higher-is-better and are
plotted as `1 - value`, so that lower is better in every scaling plot. See
[Scaling plots](analyse.md#scaling-plots) for what is drawn from these values.

> **Live plots**  When `evaluate_pipeline` returns its result (sync), the worker records it
> and refreshes `root_directory/summary/`. Pass `live_plots=True` to `scalefm.acquire()` to also
> redraw the plots every time: the scaling plots, plus anything the optimizer
> provides. See
> [Analysing Runs](analyse.md#live-plots-during-a-run).

---

## 3 Asynchronous evaluation (advanced)

### 3.1 Design

1. **`scalefm.acquire(evaluate_pipeline=None, is_async=True)`** only samples
   configurations. Each one is written to `root_directory/configs/config_<id>/config.yaml`
   and stays pending. Passing an `evaluate_pipeline` together with `is_async=True`
   raises a `ValueError`.
2. **You launch a job** for every pending configuration, reading its `config.yaml`.
3. **The job** calls

   ```python
   scalefm.save_pipeline_results(
       user_result=result_dict,
       pipeline_id=pipeline_id,
       root_directory=root_directory,
   )
   ```

   when it finishes.
   This writes `report.yaml` and marks the trial *SUCCESS* / *CRASHED*.

### 3.2 Code walk‑through

`submit.py` – samples a round of configurations and submits a Slurm job for each

```python
from pathlib import Path
import os

import yaml

import scalefm

ROOT_DIRECTORY = Path("path/to/save/results")


class Space(scalefm.PipelineSpace):
    learning_rate = scalefm.Float(1e-5, 1e-1, log=True)
    optimizer = scalefm.Categorical(["sgd", "adam"])


scalefm.acquire(
    pipeline_space=Space(),
    root_directory=ROOT_DIRECTORY,
    is_async=True,  # Only sample, `evaluate_pipeline` stays None
    worker_evaluations_to_spend=4,  # Configurations in this round
)

for config_dir in sorted((ROOT_DIRECTORY / "configs").glob("config_*")):
    script_path = config_dir / "submit.sh"
    if script_path.exists():  # Submitted in an earlier round
        continue
    config = yaml.safe_load((config_dir / "config.yaml").read_text())
    pipeline_id = config_dir.name.removeprefix("config_")
    # Replace CHANGE_ME__PARTITION_NAME with your cluster's partition
    script_path.write_text(f"""#!/bin/bash
#SBATCH --time=0-00:10
#SBATCH --job-name=trial_{pipeline_id}
#SBATCH --partition=CHANGE_ME__PARTITION_NAME
#SBATCH --output={config_dir}/%j.out
#SBATCH --error={config_dir}/%j.err

python run_pipeline.py \\
       --learning_rate {config["learning_rate"]} \\
       --optimizer {config["optimizer"]} \\
       --pipeline_id {pipeline_id} \\
       --root_dir {ROOT_DIRECTORY}
""")
    os.system(f"sbatch {script_path}")
```

Run it again for the next round, it only submits the new configurations.

`run_pipeline.py` – executed on the compute node

```python
import argparse, json, time, scalefm
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float)
parser.add_argument("--optimizer")
parser.add_argument("--pipeline_id")
parser.add_argument("--root_dir")
args = parser.parse_args()
try:
    # … do heavy training …
    val_loss = 0.1234
    wall_clock_cost = 180  # seconds
    result = {
        "objective_to_minimize": val_loss,
        "cost": wall_clock_cost,
    }
except Exception as e:
    result = {
        "objective_to_minimize": None,
        "cost": None,
        "exception": e,
    }

scalefm.save_pipeline_results(
    user_result=result,
    pipeline_id=args.pipeline_id,
    root_directory=Path(args.root_dir),
)
```

* No worker idles while your job is in the queue ➜ better throughput.
* The job must report `cost` itself: the `cost_estimator` passed to `scalefm.acquire()`
  only fills in a missing cost for results returned by `evaluate_pipeline`, not for those
  saved with `scalefm.save_pipeline_results()`.
* Pending configurations have no cost yet, so they do not count towards
  `total_cost_to_spend`: bound every round with `worker_evaluations_to_spend`.
* Crashes inside the job still mark the trial *CRASHED* instead of hanging.

### 3.3 Live plots

In async mode the result is recorded by `scalefm.save_pipeline_results()` in your job, not by
the worker, so that is where you ask for the plots to be redrawn:

```python
scalefm.save_pipeline_results(
    user_result=result,
    pipeline_id=args.pipeline_id,
    root_directory=Path(args.root_dir),
    live_plots=True,
)
```

This refreshes the general plots (the scaling plots) in
`root_directory/summary/`. Optimizer-specific artifacts need the optimizer object, so
they are only drawn by `scalefm.acquire(..., live_plots=True)`. To redraw everything once all
jobs are done, call `scalefm.analyze(root_directory)`; see
[Analysing Runs](analyse.md#plots-and-reports-scalefmanalyze).

### 3.4 Common pitfalls

* When using async approach, one worker, may create as many trials as possible, of course that in `Slurm` or other workload managers it's impossible to overload the system because of limitations set for each user, but if you want to control resources used for optimization, it's crucial to set desired stopping criteria (e.g `total_evaluations_to_spend` and/or `total_cost_to_spend`) when calling `scalefm.acquire`.

## 4 Extra injected arguments

| name                          | provided when           | description                                                |
| ----------------------------- | ----------------------- | ---------------------------------------------------------- |
| `pipeline_directory`          | always                  | per‑trial working dir (`…/trials/<id>/`)                   |
| `previous_pipeline_directory` | always                  | reserved, always `None` with the built-in optimizers       |
| `pipeline_id`                 | always                  | trial id string you pass to `save_pipeline_results`        |

Use them to handle logging and result persistence.

---

## 5 Checklist

* [x] Return a dict, **or** pass no `evaluate_pipeline` with `is_async=True` and report from the job.
* [x] Include `cost` when using cost budgets.
* [x] With `is_async=True`, make sure **exactly one** call to `scalefm.save_pipeline_results` happens per configuration.
* [x] Save checkpoints and artefacts in `pipeline_directory`.
