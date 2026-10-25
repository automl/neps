# LLM Pre-training

A real-world example of hyperparameter optimization (HPO) with NePS: pre-training
small Llama-style language models while searching their optimization
hyperparameters (learning rate, weight decay) together with their width, depth,
attention heads and number of training tokens, as in a scaling study.

Because model size and token budget are searched together, configs need different
amounts of GPU memory and compute. So sampling and training are decoupled: NePS just
samples configs and hands each one off to a Slurm job with as many GPUs as it needs,
instead of one worker blocking on training.

`generate_configs.py` drives the search. It calls `neps.run()` in asynchronous mode,
so NePS only samples configs without immediate training.
`prepare_config_resources.py` then probes each config on one GPU and decides how many
GPUs it trains on, as in scan_llm_bench (see below). `array_job.py` writes one Slurm
array job per GPU count.

Each array task runs `pipeline/train.py`, with `torchrun` and one process per GPU
(DistributedDataParallel) when a config gets more than one GPU. It trains one config,
measures its validation loss and reports the result back to NePS, once, from the first
process. The report's `cost` is the training time in seconds, and its `extra` holds the
training FLOPs (counted as in scan_llm_bench), parameter count, batch sizes and peak GPU
memory, e.g. to plot loss against compute. Training uses bf16 mixed precision on GPUs
that support it natively (Ampere or newer) and float32 otherwise.

All commands are run from this directory:

```bash
cd neps_examples/real_world/llm_pretraining
python -m pip install -r requirements.txt
python pipeline/download_data.py            # one-time token cache, see below
python generate_configs.py
srun --partition=<gpu partition> --gres=gpu:1 python prepare_config_resources.py
python array_job.py
sbatch results/hpo_llm_pretraining/array_jobs/array_job_1gpu.sh   # and/or 2gpu, 4gpu
```

Before running `array_job.py`, replace the `CHANGE_ME__PARTITION` values in
`resource_map.json` with Slurm partitions on your cluster (it refuses to run until
you do), and adjust the time limits to your GPUs once the first reports show how long
training takes (validation adds a little). Nodes to avoid can go in `EXCLUDE_NODES` in
`array_job.py`.

`array_job.py` queues every config without a report, so running it again also
re-queues trials whose job was killed (e.g. by the time limit) before reporting.
Wait until the previous jobs have finished, or it queues the still-waiting ones
twice. `--rerun_crashed` also re-queues trials reported as crashed.

## GPUs per config

As in scan_llm_bench, every config trains on sequences of 2048 tokens, with a target
global batch of 16 sequences (scan_llm_bench raises it to 32, 64 and 128 at 100M, 1B
and 10B training tokens). `prepare_config_resources.py` builds each config's model on
one GPU and tries micro-batches of 1, 2, 4, ... sequences up to the target batch,
keeping the largest whose training step stays under 88% of GPU memory. Then

- `recommended_num_gpus` = target batch / micro-batch, rounded up, at most 4
  (`--max-gpus-cap`, 1, 2 or 4 to match `resource_map.json`);
- `grad_accumulation_steps` covers the rest of the target batch, if the GPUs can't.

Two differences from scan_llm_bench. The micro-batch never exceeds the target batch, so
every config trains with the same global batch of 16 sequences whatever GPU it was
probed on (scan_llm_bench lets a small model train with a larger batch if it fits),
and probing starts at 1 instead of 4 sequences, so a large model that doesn't fit 4
sequences trains on more GPUs with gradient accumulation instead of being skipped.

It writes `micro_batch_size`, `grad_accumulation_steps` and `recommended_num_gpus` into
the config's `config.yaml` (so they also show up in `summary/full.csv`). Run it on the
GPU type the training jobs will use, as the micro-batch that fits depends on its memory.
Configs it has already prepared are skipped, so after sampling more configs, run it
again before `array_job.py`. `array_job.py` leaves configs without
`recommended_num_gpus` unqueued and lists them.

Training then runs `n_tokens // (2048 × global batch)` steps, where the global batch
is micro-batch × gradient accumulation steps × number of GPUs. `pipeline/train.py`
stops with an error if it runs on a different number of GPUs than
`recommended_num_gpus`. `CPUS_PER_GPU` in `array_job.py` sets the CPUs per training
process.

If one of a config's GPU processes fails on its own (e.g. on a faulty GPU), `torchrun`
stops the others. The trial is then reported as crashed or, if no process got to
report, has no report; `array_job.py` re-queues it either way, as above.

## Pre-training data

This example pre-trains on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
tokenized with GPT-2's tokenizer. `download_data.py` streams documents from the
Hugging Face Hub once and writes two flat token files. The first 1M tokens are the
validation set and the next 50M the training set (~100 MB). Training only reads this
cache, and a trial with a budget of `n_tokens` trains on its first `n_tokens` tokens.

```bash
python pipeline/download_data.py                              # 50M training tokens, the default
python pipeline/download_data.py --n_train_tokens 100000000   # for larger n_tokens choices
python pipeline/download_data.py --cache_dir /work/$USER/llm_tokens
```

Nothing is downloaded if the cache already holds enough tokens from the same dataset
and tokenizer (recorded in its `meta.json`); files are written under a temporary name
first, so an interrupted download never leaves a partial cache. The location is the
`#CHANGE_ME` constant `TOKEN_CACHE_DIR` in `pipeline/common.py` and can be overridden
via `NEPS_LLM_CACHE_DIR`. Set it in the shell you run `sbatch` from and Slurm passes it
to the jobs:

```bash
export NEPS_LLM_CACHE_DIR=/work/$USER/llm_tokens
python pipeline/download_data.py
```

The model and training loop are adapted from scan_llm_bench, which builds on
[PlainLM](https://github.com/Niccolo-Ajroldi/plainLM) by Niccolò Ajroldi (MIT License).
