# The `evaluate_pipeline` function

> **TL;DR**
> *Sync*: return a scalar or a dict ⟶ AutoScAn records it automatically.
> *Async*: return `None`, launch a job, and call `autoscan.save_pipeline_results()` when the job finishes.
> *Plots*: pass `live_plots=True` to `autoscan.run()` (sync) or to `autoscan.save_pipeline_results()` (async) to redraw them after every result.

---

## 1 Return types

| Allowed return | When to use                                 | Minimal example                              |
| -------------- | ------------------------------------------- | -------------------------------------------- |
| **Scalar**     | simple objective, single fidelity           | `return loss`                                |
| **Dict**       | need cost/extra metrics                     | `{"objective_to_minimize": loss, "cost": 3}` |
| **`None`**     | you launch the job elsewhere (SLURM, k8s …) | *see § 3 Async*                              |

All other values raise a `TypeError` inside AutoScAn.

## 2 Result dictionary keys

| key                     | purpose                                                                      | required?                     |
| ----------------------- | ---------------------------------------------------------------------------- | ----------------------------- |
| `objective_to_minimize` | scalar AutoScAn will minimise                                                    | **yes**                       |
| `cost`                  | wall‑clock, GPU‑hours, … —  Stopping criterion when passed `total_cost_to_spend` to `autoscan.run` | yes *iff* cost budget enabled |
| `learning_curve`        | list/np.array of intermediate objectives                                     | optional                      |
| `extra`                 | any JSON‑serialisable blob                                                   | optional                      |
| `exception`                 | any Exception illustrating the error in evaluation                                                   | optional                      |

> **Tip**  Return exactly what you need; extra keys are preserved in the trial’s `report.yaml`.

> **Live plots**  When `evaluate_pipeline` returns its result (sync), the worker records it
> and refreshes `root_directory/summary/`. Pass `live_plots=True` to `autoscan.run()` to also
> redraw the plots every time: the incumbent trajectory, or the Pareto front for two
> objectives, plus anything the optimizer provides. See
> [Analysing Runs](analyse.md#live-plots-during-a-run).

---

## 3 Asynchronous evaluation (advanced)

### 3.1 Design

1. **The Python side** (your `evaluate_pipeline` function)

   * **creates & submits** a job script.
   * returns `None` so the worker thread isn’t blocked.
2. **The submit script** or **the job** must call

   ```python
   autoscan.save_pipeline_results(
       user_result=result_dict,
       pipeline_id=pipeline_id,
       root_directory=root_directory,
   )
   ```

   when it finishes.
   This writes `report.yaml` and marks the trial *SUCCESS* / *CRASHED*.

### 3.2 Code walk‑through

`submit.py` – called by AutoScAn synchronously

```python
from pathlib import Path
import autoscan
import os

def evaluate_pipeline(
    pipeline_directory: Path,
    pipeline_id: str,          # AutoScAn injects this automatically
    root_directory: Path,      # idem
    learning_rate: float,
    optimizer: str,
):
    # 1) write a Slurm script (replace CHANGE_ME__PARTITION_NAME with your cluster's partition)
    script = f"""#!/bin/bash
#SBATCH --time=0-00:10
#SBATCH --job-name=trial_{pipeline_id}
#SBATCH --partition=CHANGE_ME__PARTITION_NAME
#SBATCH --output={pipeline_directory}/%j.out
#SBATCH --error={pipeline_directory}/%j.err

python run_pipeline.py \
       --learning_rate {learning_rate} \
       --optimizer {optimizer} \
       --pipeline_id {pipeline_id} \
       --root_dir {root_directory}
""")

    # 2) submit and RETURN None (async)
    script_path = pipeline_directory / "submit.sh"
    script_path.write_text(script)
    os.system(f"sbatch {script_path}")

    return None  # ⟵ signals async mode
```

`run_pipeline.py` – executed on the compute node

```python
import argparse, json, time, autoscan
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
        "objective_to_minimize": val_loss,
        "cost": wall_clock_cost,
        "exception": e
    }

autoscan.save_pipeline_results(
    user_result=result,
    pipeline_id=args.pipeline_id,
    root_directory=Path(args.root_dir),
)
```

* No worker idles while your job is in the queue ➜ better throughput.
* Crashes inside the job still mark the trial *CRASHED* instead of hanging.
* Compatible with Successive‑Halving/ASHA — AutoScAn just waits for `report.yaml`.

### 3.3 Live plots

In async mode the result is recorded by `autoscan.save_pipeline_results()` in your job, not by
the worker, so that is where you ask for the plots to be redrawn:

```python
autoscan.save_pipeline_results(
    user_result=result,
    pipeline_id=args.pipeline_id,
    root_directory=Path(args.root_dir),
    live_plots=True,
)
```

This refreshes the general plots (incumbent trajectory or Pareto front) in
`root_directory/summary/`. Optimizer-specific artifacts need the optimizer object, so
they are only drawn by `autoscan.run(..., live_plots=True)`. To redraw everything once all
jobs are done, call `autoscan.analyze(root_directory)`; see
[Analysing Runs](analyse.md#plots-and-reports-autoscananalyze).

### 3.4 Common pitfalls

* When using async approach, one worker, may create as many trials as possible, of course that in `Slurm` or other workload managers it's impossible to overload the system because of limitations set for each user, but if you want to control resources used for optimization, it's crucial to set desired stopping criteria (e.g `total_evaluations_to_spend` and/or `total_cost_to_spend`) when calling `autoscan.run`.

## 4 Extra injected arguments

| name                          | provided when           | description                                                |
| ----------------------------- | ----------------------- | ---------------------------------------------------------- |
| `pipeline_directory`          | always                  | per‑trial working dir (`…/trials/<id>/`)                   |
| `previous_pipeline_directory` | always                  | directory of the lower‑fidelity checkpoint. Can be `None`. |
| `pipeline_id`                 | always                  | trial id string you pass to `save_evaluation_results`      |

Use them to handle warm‑starts, logging and result persistence.

---

## 5 Checklist

* [x] Return scalar **or** dict **or** `None`.
* [x] Include `cost` when using cost budgets.
* [x] When returning `None`, make sure **exactly one** call to `autoscan.save_pipeline_results` happens.
* [x] Save checkpoints and artefacts in `pipeline_directory`.
* [x] Handle resume via `previous_pipeline_directory`.
