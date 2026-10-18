"""Trains and evaluates one LLM config sampled by `generate_configs.py`"""

import argparse
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from common import (
    BATCH_SIZE,
    DEVICE,
    N_VAL_TOKENS,
    SEQ_LEN,
    VOCAB_SIZE,
    autocast,
    get_batch,
    load_tokens,
)
from model import Transformer, training_flops

import neps

# Fixed training settings, the defaults of scan_llm_bench. The searched ones are in
# `generate_configs.py`.
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
WARMUP_FRACTION = 0.05
COOLDOWN_FRACTION = 0.2
SEED = 0


def wsd_schedule(step, total_steps):
    """Warmup-stable-decay LR multiplier: linear warmup, constant, linear cooldown to 0."""
    warmup_steps = max(1, int(WARMUP_FRACTION * total_steps))
    cooldown_steps = int(COOLDOWN_FRACTION * total_steps)
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step < total_steps - cooldown_steps:
        return 1.0
    return (total_steps - step) / max(1, cooldown_steps)


@torch.no_grad()
def validation_loss(model):
    """Mean next-token cross-entropy over the N_VAL_TOKENS validation tokens."""
    tokens = load_tokens("val", N_VAL_TOKENS)
    starts = np.arange((N_VAL_TOKENS - 1) // SEQ_LEN) * SEQ_LEN
    model.eval()
    total_loss = 0.0
    for i in range(0, len(starts), BATCH_SIZE):
        inputs, targets = get_batch(tokens, starts[i : i + BATCH_SIZE])
        with autocast():
            logits = model(inputs)
        total_loss += F.cross_entropy(
            logits.flatten(0, 1).float(), targets.flatten(), reduction="sum"
        ).item()
    return total_loss / (len(starts) * SEQ_LEN)


def evaluate(lr, weight_decay, d_model, n_heads, n_layers, n_tokens):
    torch.manual_seed(SEED)
    # A trial with budget n_tokens trains on the first n_tokens tokens, in shuffled windows.
    steps = max(1, n_tokens // (BATCH_SIZE * SEQ_LEN))
    train_tokens = load_tokens("train", steps * BATCH_SIZE * SEQ_LEN + 1)
    starts = np.random.default_rng(SEED).permutation(steps * BATCH_SIZE) * SEQ_LEN

    model = Transformer(VOCAB_SIZE, SEQ_LEN, d_model, n_layers, n_heads).to(DEVICE)
    # No weight decay on the norm weights (the only 1-D parameters), as in scan_llm_bench.
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ],
        lr=lr,
        betas=BETAS,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: wsd_schedule(s, steps))

    start_time = time.time()
    model.train()
    for step in range(steps):
        inputs, targets = get_batch(
            train_tokens, starts[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        )
        with autocast():
            loss = F.cross_entropy(model(inputs).flatten(0, 1), targets.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    train_loss = loss.item()  # also waits for the GPU, so the time below is complete
    train_seconds = time.time() - start_time

    val_loss = validation_loss(model)
    if not math.isfinite(val_loss):
        raise FloatingPointError(f"Training diverged: the validation loss is {val_loss}.")

    n_tokens_trained = steps * BATCH_SIZE * SEQ_LEN
    return {
        "objective_to_minimize": val_loss,
        "cost": train_seconds,
        "info_dict": {
            "train_loss": train_loss,
            # Counted as in scan_llm_bench
            "flops": training_flops(d_model, n_layers, n_tokens_trained, VOCAB_SIZE, SEQ_LEN),
            "n_params": sum(p.numel() for p in model.parameters()),
            "n_tokens_trained": n_tokens_trained,
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1e6
            if DEVICE == "cuda"
            else None,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", required=True, type=Path)
    parser.add_argument("--root_dir", required=True, type=Path)
    args = parser.parse_args()

    try:
        config = yaml.safe_load((args.config_dir / "config.yaml").read_text())
        result = evaluate(**config)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        # Marks the trial as crashed instead of leaving it pending forever. The explicit
        # `None` keeps NePS from recording its default of 0, which looks like the best loss.
        result = {"objective_to_minimize": None, "cost": None, "exception": e}

    neps.save_pipeline_results(
        user_result=result,
        pipeline_id=args.config_dir.name.removeprefix("config_"),
        root_directory=args.root_dir,
    )
    if "exception" in result:
        sys.exit(1)  # so Slurm also records the job as failed


if __name__ == "__main__":
    main()
