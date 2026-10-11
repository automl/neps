# VLM OpenCLIP

A real-world example of hyperparameter optimization (HPO) with AutoScAn: searching
an [OpenCLIP](https://github.com/mlfoundations/open_clip) model's optimization
hyperparameters (learning rate, weight decay) together with its vision/text
tower width and depth.

Each sampled config needs a different amount of GPU memory, so sampling and
training are decoupled: AutoScAn just samples configs and hands each one off to a
right-sized Slurm job, instead of one worker blocking on training.

`generate_configs.py` drives the search: it calls `autoscan.run()` in asynchronous mode, so AutoScAn only samples configs without immediate training.
`array_job.py` then groups the sampled configs by GPU memory need and writes
one Slurm array job per resource tier.

Each array job runs the training pipeline under `pipeline/`: preparing the
pre-training data cache, training and evaluating one config, and reporting
the result back to AutoScAn. Once training has finished, a post-hoc step scores
the trained checkpoints on a held-out downstream benchmark.

All commands are run from this directory:

```bash
cd autoscan_examples/real_world/vlm_openclip
python -m pip install -r requirements.txt
python generate_configs.py
python array_job.py
python pipeline/download_data.py            # one-time LAION cache, see below
sbatch results/hpo_vlm_openclip/array_jobs/array_job_small.sh   # and/or medium, large
python pipeline/post_hoc_downstream_eval.py --root_dir results/hpo_vlm_openclip
```

Before running `array_job.py`, replace the `CHANGE_ME__*_PARTITION` values in
`resource_map.json` with Slurm partitions on your cluster (it refuses to run
until you do). Nodes to avoid can go in `EXCLUDE_NODES` in `array_job.py`.

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
python pipeline/download_data.py --n_samples 22000     # smallest cache train.py accepts (N_TRAIN + N_VAL)
python pipeline/download_data.py --shards_dir /path/to/train_data
python pipeline/download_data.py --cache_dir /work/$USER/laion_cache
```

`--n_samples` is the only thing that decides how much data is read: shards are
consumed in order until the target is met, then the run stops mid-shard. A
staged LAION-400M copy is far larger than any of this needs.
Raising `N_TRAIN` in `pipeline/train.py` and re-running `download_data.py`
with a matching `--n_samples` is all it takes to scale up; 100k works out to
~10 Hub shards (10k samples each) and ~530 MB of cache.

Both locations are `#CHANGE_ME` constants in `pipeline/common.py`
(`LAION_CACHE_DIR`, `LAION_SHARDS_DIR`) and can be overridden
via `AUTOSCAN_LAION_CACHE_DIR` and `AUTOSCAN_LAION_SHARDS`. Set those
in the shell you run `sbatch` from and Slurm passes them to the jobs, so
training reads exactly the cache this script wrote:

```bash
export AUTOSCAN_LAION_SHARDS=/path/to/laion400m/train_data  # optional; unset = download from the Hub
export AUTOSCAN_LAION_CACHE_DIR=/work/$USER/laion_cache
python pipeline/download_data.py
```
