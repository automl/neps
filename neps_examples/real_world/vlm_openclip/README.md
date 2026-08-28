# VLM OpenCLIP

Two workflows in this directory, sharing the same model+data.

## 1. Architecture/HPO search

The standard NePS use case: search a space where each sampled config (vision/text
width & depth, `batch_size`, ...) needs a *different* amount of GPU memory.
Sampling and training are decoupled so each config can be routed to a
right-sized Slurm job instead of blocking one worker on training:

- `generate_configs.py` — samples configs, writes each to `configs/config_<id>/config.yaml`.
- `array_job.py` — groups pending configs by `batch_size` into the tiers in
  `resource_map.json`, writes one Slurm array job per tier.
- `download_data.py` — one-time LAION image/caption cache (see *Pre-training data* below).
- `train.py` — trains + evaluates one config on LAION (upstream objective: the
  contrastive loss on held-out pre-training data), saves `checkpoint.pt` next to
  its `config.yaml`, reports back via `neps.save_pipeline_results`.
- `post_hoc_downstream_eval.py` — after training, zero-shot-scores each checkpoint
  on CIFAR-100 (downstream, held out of the HPO objective), writes `report_down.yaml`.

```bash
python -m pip install -r requirements.txt
python download_data.py            # one-time LAION cache, see below
python generate_configs.py
python array_job.py
sbatch results/hpo_vlm_openclip/array_jobs/array_job_small.sh   # and/or medium, large
python post_hoc_downstream_eval.py --root_dir results/hpo_vlm_openclip
```

## 2. Scaling study — resource planning

A different use of NePS: not searching for the best model, but measuring how a
*fixed* representative workload actually scales with GPU count, to answer "how
many GPUs/how long will the real run need" for a resource-grant proposal.

The question is deliberately narrow: **the same sweep of training configs** is
run on 1, 2 and 4 GPUs, and we report how many samples per second all GPUs
together process. `n_gpus` is the only thing that varies between the three
runs. Training is real `torchrun` DDP on 100,000 LAION image/caption pairs, and
`ClipLoss` all-gathers features across ranks, so the contrastive batch is the
same at every GPU count -- the all-gather cost is real CLIP-training cost and
is inside the timed region.

- `scaling_study/run_scaling_study.py` — submits one Slurm job per GPU count and exits.
- `scaling_study/hpo_ddp.py` — one GPU count's NePS sweep, run under `torchrun` inside that job.
- `scaling_study/train_ddp.py` — the DDP training every rank runs in lockstep.
- `scaling_study/visualization.py` — aggregates the three sweeps into the scaling table + figure.

Each job runs its own independent `neps.run()` into its own directory
(`results/scaling_study/gpus_1/`, `gpus_2/`, `gpus_4/`), so the three sweeps
queue and run concurrently without sharing an allocation. Those directories
belong to NePS alone — `NePSState.create_or_load` treats an existing path as an
existing state and looks for an `optimizer_info.yaml` in it — so the job
scripts and Slurm logs live beside them under `results/scaling_study/jobs/`. Inside a job, **rank
0 is the NePS worker**: its `evaluate_pipeline` broadcasts the sampled config
to the other ranks, then every rank trains it together under DDP. The other
ranks never call `neps.run()` — they follow rank 0 until it signals the sweep
is over.

`hpo_ddp.HPOSpace` searches `lr`, `wd` and `batch_size` and holds the
architecture and epoch count fixed. It is a **grid** on purpose: the three GPU
counts must evaluate the same configs in the same order, or their throughputs
would not be comparable — a random search would hand each job a different set.

Because each GPU count therefore produces several throughput measurements
rather than one, `visualization.py` aggregates them by **median** (a single
straggler trial — a slow node, a cold filesystem — should not move the curve)
and draws every individual trial behind the line, so the spread stays visible.

```bash
python download_data.py --n_samples 102000   # if the cache is smaller than the study needs
cd scaling_study
python run_scaling_study.py                  # submits the 1-, 2- and 4-GPU jobs
python visualization.py                      # -> ../results/scaling_study/summary/
```

Edit the `#CHANGE_ME` values in `resource_map.json`,
`scaling_study/run_scaling_study.py` (partition, memory, time limit, GPU
counts) and `scaling_study/hpo_ddp.py` (the search space, evaluations per
sweep) for your cluster before running either workflow.

## Pre-training data

Both workflows pre-train on LAION image/caption pairs in webdataset-shard
format -- the same format a production LAION scaling study trains on.

Shards hold full-size JPEGs, and decoding those in the training loop would make
the CPU input pipeline, not the GPUs, the thing the scaling study measures. So
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
python download_data.py                       # 100k samples (the default)
python download_data.py --n_samples 20000     # smaller cache to try things out
python download_data.py --shards_dir /path/to/train_data
python download_data.py --cache_dir /work/$USER/laion_cache
```

`--n_samples` is the only thing that decides how much data is read: shards are
consumed in order until the target is met, then the run stops mid-shard. A
staged LAION-400M copy is far larger than any of this needs -- the one this
example was developed against holds 36,192 shards at roughly 5-6.8k usable
samples each, about 180M pairs -- so the 100k default is a deliberate choice
about how long a training run should take, not a limit imposed by the data.
Raising `N_TRAIN` in `train.py` / `scaling_study/train_ddp.py` and re-running
`download_data.py` with a matching `--n_samples` is all it takes to scale up;
100k works out to ~20 shards and ~430 MB of cache.

Both locations are `#CHANGE_ME` constants in `common.py`
(`LAION_CACHE_DIR`, `LAION_SHARDS_DIR`) and can be overridden per run without
editing anything, via `NEPS_LAION_CACHE_DIR` and `NEPS_LAION_SHARDS`. Set those
in the shell you run `sbatch` from and Slurm passes them to the jobs, so
training reads exactly the cache this script wrote:

```bash
export NEPS_LAION_SHARDS=/work/dlclarge1/.../pre_training_dataset/laion400m/train_data
export NEPS_LAION_CACHE_DIR=/work/$USER/laion_cache
python download_data.py
```

Shard filenames are looked up by index (`00000000.tar` or `00000.tar`) rather
than by listing the directory, since a staged LAION-400M copy holds >100k files
and globbing it on a network filesystem is slow.

CIFAR-100 is still downloaded, but only by `post_hoc_downstream_eval.py`, as
the zero-shot benchmark. Nothing trains on it.
