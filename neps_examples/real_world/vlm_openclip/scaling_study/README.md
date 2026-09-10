# Scaling study

How much faster does a fixed HPO sweep finish when it gets more parallel workers?

## Setup

- **Task:** evaluate the same 8 configs every time, a grid over `lr` × `wd` × `batch_size` (two values each).
- **Data:** 100k LAION CC12M image/caption pairs (plus 2k for validation). Each config trains a 22.3M-parameter OpenCLIP model for 3 epochs.
- **Workers:** the sweep runs four times, with 1, 2, 4 and 8 workers. Each worker is its own single-GPU Slurm job (NVIDIA H200) calling `neps.run()` on a shared root directory, with `evaluations_to_spend = 8 / n_workers`.

![Throughput against number of workers](scaling_study.png)

| Workers | Wall clock | Throughput | Speedup |
|---|---|---|---|
| 1 | 15.3 min | 2,589 samples/s | 1.00× |
| 2 |  8.0 min | 4,929 samples/s | 1.90× |
| 4 |  4.2 min | 9,360 samples/s | 3.62× |
| 8 |  2.2 min | 18,383 samples/s | 7.10× |

Throughput is all training samples in the sweep divided by its wall clock (first trial start to last trial end).

## Running it

First set `PARTITION`, `MEM_PER_GPU` and `TIME_LIMIT` in `run_scaling_study.py` (the `#CHANGE_ME` markers) for your cluster. Then:

```bash
python ../download_data.py --n_samples 102000   # 1. data cache (no-op if it's already big enough)
python run_scaling_study.py                     # 2. submits all 15 worker jobs (1 + 2 + 4 + 8)
python visualization.py                         # 3. after the jobs finish: table and plot
```

Everything is written to `../results/scaling_study/`: one `workers_<n>/` per setting, `jobs/` for the Slurm scripts and logs, and `summary/` for your table and plot. The `scaling_study.png` above is the reference run and is not overwritten.
