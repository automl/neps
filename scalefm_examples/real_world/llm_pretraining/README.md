# LLM Pre-training

A real-world scaling study with ScaleFM: pre-training small Llama-style language
models of different width, depth and number of training tokens, then fitting the
scaling law of the loss over the training FLOPs and predicting it at larger budgets.

Because model size and token budget are searched together, configs need different
amounts of GPU memory and compute. So sampling and training are decoupled. ScaleFM
just samples configs and hands each one off to a Slurm job with as many GPUs as it
needs, instead of one worker blocking on training.

`generate_configs.py` drives the search. It calls `scalefm.acquire()` in asynchronous
mode, so ScaleFM only samples configs without immediate training. The number of
configs is set by `worker_evaluations_to_spend`: in asynchronous mode a pending config
has no cost yet, so a `total_cost_to_spend` alone would never stop the sampling.
`prepare_config_resources.py` then probes each config on one GPU and decides how many
GPUs it trains on (see below). `array_job.py` writes one Slurm
array job per GPU count.

Each array task runs `pipeline/train.py` (with `torchrun`, one process per GPU, when a
config gets more than one GPU). It trains one config and reports to ScaleFM its
validation loss as `objective_to_minimize`, the training FLOPs as `cost`, and
`n_params` and `n_data` (the tokens trained on). Its
`extra` holds the training time and peak GPU memory.

All commands are run from this directory:

```bash
cd scalefm_examples/real_world/llm_pretraining
python -m pip install -r requirements.txt
python pipeline/download_data.py            # one-time token cache, see below
python generate_configs.py
srun --partition=<gpu partition> --gres=gpu:1 python prepare_config_resources.py
python array_job.py
sbatch results/scaling_llm_pretraining/array_jobs/array_job_1gpu.sh   # and/or 2gpu, 4gpu
python predict.py             # once the jobs have finished
```

Before running `array_job.py`, replace the `CHANGE_ME__PARTITION` values in
`resource_map.json` with Slurm partitions on your cluster, and adjust the time limits
once the first reports show how long training takes. To keep jobs off faulty nodes,
pass `--exclude=<node1>,<node2>` to `sbatch`.

Running `array_job.py` again re-queues every config without a report (e.g. jobs killed
by the time limit), so wait until the previous jobs have finished.

## GPUs per config

Every config trains on sequences of 2048 tokens with a global
batch of 16 sequences. `prepare_config_resources.py` builds each config's model on one
GPU, finds the largest micro-batch that fits, and writes `micro_batch_size`,
`grad_accumulation_steps` and `recommended_num_gpus` (1, 2 or 4) into its
`config.yaml`. Run it on the GPU type the training jobs will use, and again after
sampling more configs. `array_job.py` skips configs it hasn't prepared.

The micro-batch never exceeds the target batch, so every config trains with the same
global batch, and probing starts at 1 sequence, so a large model trains on more GPUs
instead of being skipped.

## Prediction

`predict.py` fits the scaling law on the successful trials and writes the plots,
Pareto fronts and predictions to `results/scaling_llm_pretraining/summary/`
(`pre_train/`, `convergence/`). Change `TARGET_FLOPS` to predict at other budgets.

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
can be overridden via `SCALEFM_LLM_CACHE_DIR`:

```bash
SCALEFM_LLM_CACHE_DIR=path/to/cache python pipeline/download_data.py
```

Training reads the same variable, so also set it when you run
`prepare_config_resources.py` and `sbatch`.

The model and training loop are adapted from scan_llm_bench, which builds on
[PlainLM](https://github.com/Niccolo-Ajroldi/plainLM) by Niccolò Ajroldi (MIT License).