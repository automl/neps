# LLM Pre-training

A real-world example of hyperparameter optimization (HPO) with NePS: pre-training
small Llama-style language models while searching their optimization
hyperparameters (learning rate, weight decay) together with their width, depth,
attention heads and number of training tokens, as in a scaling study.

Because model size and token budget are searched together, configs need different
amounts of GPU memory and compute. So sampling and training are decoupled. NePS just
samples configs and hands each one off to a Slurm job with as many GPUs as it needs,
instead of one worker blocking on training.

`generate_configs.py` drives the search. It calls `neps.run()` in asynchronous mode,
so NePS only samples configs without immediate training.
`prepare_config_resources.py` then probes each config on one GPU and decides how many
GPUs it trains on, as in scan_llm_bench (see below). `array_job.py` writes one Slurm
array job per GPU count.

Each array task runs `pipeline/train.py` (with `torchrun`, one process per GPU, when a
config gets more than one GPU). It trains one config and reports its validation loss
to NePS. The report's `cost` is the training time in seconds, and its `extra` holds the
training FLOPs (counted as in scan_llm_bench), parameter count and peak GPU memory.

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
`resource_map.json` with Slurm partitions on your cluster, and adjust the time limits
once the first reports show how long training takes. To keep jobs off faulty nodes,
pass `--exclude=<node1>,<node2>` to `sbatch`.

Running `array_job.py` again re-queues every config without a report (e.g. jobs killed
by the time limit), so wait until the previous jobs have finished.
`--rerun_crashed` also re-queues trials reported as crashed.

## GPUs per config

As in scan_llm_bench, every config trains on sequences of 2048 tokens with a global
batch of 16 sequences. `prepare_config_resources.py` builds each config's model on one
GPU, finds the largest micro-batch that fits, and writes `micro_batch_size`,
`grad_accumulation_steps` and `recommended_num_gpus` (1, 2 or 4) into its
`config.yaml`. Run it on the GPU type the training jobs will use, and again after
sampling more configs. `array_job.py` skips configs it hasn't prepared.

Two differences from scan_llm_bench: the micro-batch never exceeds the target batch, so
every config trains with the same global batch, and probing starts at 1 instead of 4
sequences, so a large model trains on more GPUs instead of being skipped.

## Pre-training data

This example pre-trains on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
tokenized with GPT-2's tokenizer. `download_data.py` streams it from the Hugging Face
Hub once and writes a token cache: 1M validation and 50M training tokens (~100 MB).
A trial with a budget of `n_tokens` trains on the first `n_tokens` tokens. Nothing is
downloaded if the cache already holds enough tokens.

```bash
python pipeline/download_data.py                              # 50M training tokens, the default
python pipeline/download_data.py --n-train-tokens 100000000   # for larger n_tokens choices
```

The location is the `#CHANGE_ME` constant `TOKEN_CACHE_DIR` in `pipeline/common.py` and
can be overridden via `NEPS_LLM_CACHE_DIR`:

```bash
NEPS_LLM_CACHE_DIR=path/to/cache python pipeline/download_data.py
```

Training reads the same variable, so also set it when you run
`prepare_config_resources.py` and `sbatch`.

The model and training loop are adapted from scan_llm_bench, which builds on
[PlainLM](https://github.com/Niccolo-Ajroldi/plainLM) by Niccolò Ajroldi (MIT License).
