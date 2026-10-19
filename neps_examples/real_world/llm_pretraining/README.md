# LLM Pre-training

A real-world example of hyperparameter optimization (HPO) with NePS: pre-training
small Llama-style language models while searching their optimization
hyperparameters (learning rate, weight decay) together with their width, depth,
attention heads and number of training tokens, as in a scaling study.

Because model size and token budget are searched together, some configs train in a
minute and others take far longer. So sampling and training are decoupled: NePS just
samples configs and hands each one off to a Slurm job sized for its compute, instead
of one worker blocking on training.

`generate_configs.py` drives the search: it calls `neps.run()` in asynchronous mode,
so NePS only samples configs without immediate training.
`array_job.py` then estimates each pending config's training FLOPs and writes one
Slurm array job per resource tier.

Each array task runs `pipeline/train.py` on one GPU: it trains one config, measures
its validation loss and reports the result back to NePS. The report's `cost` is the
training time in seconds, and its `extra` holds the training FLOPs (counted as in
scan_llm_bench), parameter count and peak GPU memory, e.g. to plot loss against
compute. Training uses bf16 mixed precision on GPUs that support it natively (Ampere
or newer) and float32 otherwise.

All commands are run from this directory:

```bash
cd neps_examples/real_world/llm_pretraining
python -m pip install -r requirements.txt
python pipeline/download_data.py            # one-time token cache, see below
python generate_configs.py
python array_job.py
sbatch results/hpo_llm_pretraining/array_jobs/array_job_small.sh   # and/or medium, large
```

Before running `array_job.py`, replace the `CHANGE_ME__*_PARTITION` values in
`resource_map.json` with Slurm partitions on your cluster (it refuses to run until
you do), and adjust the time limits to your GPUs once the first reports show how long
training takes (validation adds a little). Nodes to avoid can go in `EXCLUDE_NODES` in
`array_job.py`.

`array_job.py` queues every config without a report, so running it again also
re-queues trials whose job was killed (e.g. by the time limit) before reporting.
Wait until the previous jobs have finished, or it queues the still-waiting ones
twice. `--rerun_crashed` also re-queues trials reported as crashed.

## Pre-training data

This example pre-trains on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
tokenized with GPT-2's tokenizer. `download_data.py` streams documents from the
Hugging Face Hub once and writes two flat token files: the first 1M tokens are the
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
