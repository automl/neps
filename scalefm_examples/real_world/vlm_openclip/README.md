# VLM OpenCLIP Scaling Study

A real-world scaling study with ScaleFM: how the contrastive loss of an
[OpenCLIP](https://github.com/mlfoundations/open_clip) vision-language model scales
with model size (the width and depth of its vision and text towers) and data (the
number of LAION image/caption pairs it is trained on), within a FLOP budget, and
what loss and model size to expect at 10-100x more compute.

Configs need very different amounts of GPU memory, so sampling and training are
decoupled: ScaleFM only samples configs, and each one goes to a right-sized
Slurm job.

* `generate_configs.py` defines the search space, the parameter and FLOP
  estimators, and calls `scalefm.acquire()` in asynchronous mode. Pick the
  acquisition with `--optimizer`: `chinchilla2` (ISO-FLOP, default), `chinchilla1`,
  or `scan` (ScAn, Bayesian optimization aimed at `--target_flops`).
* `array_job.py` groups the sampled configs by model size into the tiers of
  `resource_map.json` and writes one Slurm array job per tier. Tiers with more than
  one GPU train each config with DDP through `torchrun`.
* `pipeline/train.py` trains one config on `n_samples` samples and reports the
  validation loss, the training FLOPs as its cost, the parameter count and data, and
  a learning curve, with `scalefm.save_pipeline_results`. Under `torchrun` only
  rank 0 reports.
* `pipeline/post_hoc_downstream_eval.py` scores the trained checkpoints on
  CIFAR-100 zero-shot classification.

All commands are run from this directory:

```bash
cd scalefm_examples/real_world/vlm_openclip
python -m pip install -r requirements.txt
python pipeline/download_data.py            # one-time LAION cache, see below
python generate_configs.py --optimizer chinchilla2 --budget 2e17
python array_job.py
sbatch results/scaling_vlm_openclip/array_jobs/array_job_small.sh   # and/or medium, large
python pipeline/post_hoc_downstream_eval.py --root_dir results/scaling_vlm_openclip
```

Before running `array_job.py`, replace the `CHANGE_ME__*_PARTITION` values in
`resource_map.json` with Slurm partitions on your cluster (it refuses to run
until you do). Nodes to avoid can go in `EXCLUDE_NODES` in `array_job.py`.

## Rounds with ScAn

The Chinchilla approaches lay out all their configs up front. ScAn chooses each
config from the results so far, so it samples in rounds:

```bash
python generate_configs.py --optimizer scan --target_flops 1e19   # first round: the initial design
python array_job.py && sbatch ...                                 # train it
python generate_configs.py --optimizer scan --target_flops 1e19   # next round, once results are in
```

Every call against the same `--root_directory` spends the same global
`total_cost_to_spend`, so rounds, or several generators at once, share one budget.

## Predicting

Once runs have reported, predict the loss and compute-optimal model size at larger
budgets:

```python
import scalefm
from generate_configs import cost_estimator, params_estimator

for p in scalefm.predict(
    "results/scaling_vlm_openclip",
    target_costs=[1e18, 1e19],
    predictor="power_law",  # or "chinchilla3", "gp_posterior"
    params_estimator=params_estimator,  # the GP needs both estimators
    cost_estimator=cost_estimator,
):
    print(p.target_cost, p.loss, p.n_params, p.config)
```

`chinchilla3` derives the compute-optimal model size and data from `C = 6ND` with
`D` in tokens; this example counts data in samples of about 273 tokens.

## Pre-training data

This example pre-trains on LAION image/caption pairs in webdataset-shard
format. Shards hold full-size JPEGs, and decoding those in the training loop
would make the CPU input pipeline, not the GPUs, the bottleneck. So
`download_data.py` does the decode/resize once, up front, and writes a compact
local parquet cache of `common.IMAGE_SIZE` JPEGs (~530 MB for 100k samples).
Shards are never copied to disk; training only ever decodes the small cached
images.

Nothing is fetched that is already on disk. `download_data.py` tries, in order:

1. **The parquet cache.** If it already holds enough samples, the script exits
   immediately -- no shards read, nothing downloaded. Re-running is free.
2. **Local `.tar` shards**, e.g. a LAION-400M copy already staged on the
   cluster. No network access needed.
3. **The Hugging Face Hub**
   ([`laion/conceptual-captions-12m-webdataset`](https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset)),
   streamed over HTTP -- only when there are no local shards.

```bash
python pipeline/download_data.py                       # 100k samples (the default)
python pipeline/download_data.py --n_samples 100000    # what train.py reads by default (N_TRAIN_CACHE + N_VAL)
python pipeline/download_data.py --shards_dir /path/to/train_data
python pipeline/download_data.py --cache_dir /work/$USER/laion_cache
```

`--n_samples` is the only thing that decides how much data is read: shards are
consumed in order until the target is met, then the run stops mid-shard. A
staged LAION-400M copy is far larger than any of this needs.
Configs that ask for more samples than `N_TRAIN_CACHE` in `pipeline/train.py`
repeat passes over the cache. Raising `N_TRAIN_CACHE` and re-running
`download_data.py` with a matching `--n_samples` avoids the repetition; 100k works
out to ~10 Hub shards (10k samples each) and ~530 MB of cache.

Both locations are `#CHANGE_ME` constants in `pipeline/common.py`
(`LAION_CACHE_DIR`, `LAION_SHARDS_DIR`) and can be overridden
via `SCALEFM_LAION_CACHE_DIR` and `SCALEFM_LAION_SHARDS`. Set those
in the shell you run `sbatch` from and Slurm passes them to the jobs, so
training reads exactly the cache this script wrote:

```bash
export SCALEFM_LAION_SHARDS=/path/to/laion400m/train_data  # optional; unset = download from the Hub
export SCALEFM_LAION_CACHE_DIR=/work/$USER/laion_cache
python pipeline/download_data.py
```
