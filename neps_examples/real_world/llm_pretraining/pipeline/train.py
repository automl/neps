"""Trains and evaluates one LLM config sampled by `generate_configs.py`, on the GPUs
chosen by `prepare_config_resources.py` (one process per GPU, started by torchrun)
"""

import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from common import (
    DEVICE,
    N_VAL_TOKENS,
    SEQ_LEN,
    VOCAB_SIZE,
    autocast,
    get_batch,
    load_tokens,
)
from model import Transformer, training_flops
from torch.nn.parallel import DistributedDataParallel as DDP

import neps

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
        return step / warmup_steps
    if step < total_steps - cooldown_steps:
        return 1.0
    return (total_steps - step) / max(1, cooldown_steps)


def pytorch_setup():
    """Initialize DDP if started by torchrun. Returns (rank, world_size)."""
    if "RANK" not in os.environ:
        return 0, 1
    if DEVICE == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="nccl" if DEVICE == "cuda" else "gloo")
    return dist.get_rank(), dist.get_world_size()


@torch.no_grad()
def validation_loss(model, micro_batch_size, rank, world_size):
    """Mean next-token cross-entropy over the N_VAL_TOKENS validation tokens.

    Each process evaluates every `world_size`-th sequence; the sums are then reduced.
    """
    tokens = load_tokens("val", N_VAL_TOKENS)
    n_sequences = (N_VAL_TOKENS - 1) // SEQ_LEN
    starts = (np.arange(n_sequences) * SEQ_LEN)[rank::world_size]
    model.eval()
    total_loss = 0.0
    for i in range(0, len(starts), micro_batch_size):
        inputs, targets = get_batch(tokens, starts[i : i + micro_batch_size])
        with autocast():
            logits = model(inputs)
        total_loss += F.cross_entropy(
            logits.flatten(0, 1).float(), targets.flatten(), reduction="sum"
        ).item()
    if world_size > 1:
        total_loss_tensor = torch.tensor([total_loss], dtype=torch.float64, device=DEVICE)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        total_loss = total_loss_tensor.item()
    return total_loss / (n_sequences * SEQ_LEN)




def evaluate( lr, weight_decay, d_model, n_heads, n_layers, n_tokens, micro_batch_size, grad_accumulation_steps, recommended_num_gpus, rank=0,
    world_size=1,
):
    if world_size != recommended_num_gpus:
        raise ValueError(
            f"This config was prepared for {recommended_num_gpus} GPU(s) but runs on "
            f"{world_size}; submit it with the Slurm script that array_job.py writes."
        )
    torch.manual_seed(SEED)
    global_batch = micro_batch_size * grad_accumulation_steps * world_size
    steps = max(1, n_tokens // (SEQ_LEN * global_batch))
    train_tokens = load_tokens("train", steps * global_batch * SEQ_LEN + 1)
    starts = np.random.default_rng(SEED).permutation(steps * global_batch) * SEQ_LEN

    model = Transformer(VOCAB_SIZE, SEQ_LEN, d_model, n_layers, n_heads).to(DEVICE)
    ddp_model = DDP(model) if world_size > 1 else model
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
        for micro_step in range(grad_accumulation_steps):
            i = step * global_batch + (micro_step * world_size + rank) * micro_batch_size
            inputs, targets = get_batch(train_tokens, starts[i : i + micro_batch_size])
            if world_size > 1:
                ddp_model.require_backward_grad_sync = micro_step == grad_accumulation_steps - 1
            with autocast():
                loss = F.cross_entropy(ddp_model(inputs).flatten(0, 1), targets.flatten())
            (loss / grad_accumulation_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    train_loss = loss.item()
    train_seconds = time.time() - start_time

    val_loss = validation_loss(model, micro_batch_size, rank, world_size)
    if not math.isfinite(val_loss):
        raise FloatingPointError(f"Training diverged: the validation loss is {val_loss}.")

    n_tokens_trained = steps * global_batch * SEQ_LEN
    return {
        "objective_to_minimize": val_loss,
        "cost": train_seconds,
        "info_dict": {
            "train_loss": train_loss,
            "flops": training_flops(d_model, n_layers, n_tokens_trained, VOCAB_SIZE, SEQ_LEN),
            "n_params": sum(p.numel() for p in model.parameters()),
            "n_tokens_trained": n_tokens_trained,
            "micro_batch_size": micro_batch_size,
            "global_batch_size": global_batch,
            "recommended_num_gpus": recommended_num_gpus,
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
        rank, world_size = pytorch_setup()
        result = evaluate(**config, rank=rank, world_size=world_size)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        result = {"objective_to_minimize": None, "cost": None, "exception": e}

    if int(os.environ.get("RANK", "0")) == 0:
        neps.save_pipeline_results(
            user_result=result,
            pipeline_id=args.config_dir.name.removeprefix("config_"),
            root_directory=args.root_dir,
        )
    if dist.is_initialized():
        dist.destroy_process_group()

    if "exception" in result:
        sys.exit(1)


if __name__ == "__main__":
    main()
