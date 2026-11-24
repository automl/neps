"""Trains and evaluates one OpenCLIP config sampled by `generate_configs.py`.

A run trains on `n_samples` image/caption pairs and reports what a scaling analysis
needs: the validation loss, the training FLOPs as its cost, the parameter count and
data, and a learning curve of `(samples seen, flops spent, val loss)` rows that give
extra scaling points for free.

Single GPU:

    python pipeline/train.py --config_dir results/.../configs/config_3 --root_dir results/...

Several GPUs with DDP, every rank runs this script and rank 0 reports:

    torchrun --nproc_per_node=4 pipeline/train.py --config_dir ... --root_dir ...
"""

import argparse
import itertools
import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from open_clip import ClipLoss, get_tokenizer
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import scalefm
from common import (
    build_model,
    get_gpu_name,
    get_peak_gpu_memory_mb,
    load_data,
    model_stats,
)

# #CHANGE_ME: how many cached samples to train on, passes over them repeat when a
# config asks for more. Must be covered by the cache built with
# `download_data.py --n_samples ...` (plus N_VAL).
N_TRAIN_CACHE = 98_000
N_VAL = 2_000

# Validation runs at a fixed batch size regardless of the trial's training
# batch size: the contrastive loss depends on how many negatives are in the
# batch, so a config-dependent val batch size would make trials incomparable.
VAL_BATCH_SIZE = 256

# Points of the learning curve, as fractions of the run
CURVE_FRACTIONS = (0.25, 0.5, 0.75, 1.0)

NUM_WORKERS = 4


def setup_distributed():
    """`(rank, world_size, device)`, with a process group when run by `torchrun`."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return dist.get_rank(), world_size, device


def contrastive_eval(model, val_loader, loss_fn, device):
    """Mean contrastive loss and in-batch image->text retrieval accuracy.

    Only full `VAL_BATCH_SIZE` batches are scored, so every batch contributes
    the same number of negatives and the numbers stay comparable across trials.
    """
    model.eval()
    total_loss, correct, total, n_batches = 0.0, 0, 0, 0
    with torch.no_grad():
        for images, texts in val_loader:
            if images.shape[0] < VAL_BATCH_SIZE:
                continue
            images, texts = images.to(device), texts.to(device)
            out = model(images, texts)
            total_loss += sum(loss_fn(**out, output_dict=True).values()).item()
            n_batches += 1

            logits = out["logit_scale"] * out["image_features"] @ out["text_features"].t()
            target = torch.arange(images.shape[0], device=device)
            correct += (logits.argmax(dim=-1) == target).sum().item()
            total += images.shape[0]
    model.train()

    if n_batches == 0:
        raise RuntimeError(
            f"No full validation batch of {VAL_BATCH_SIZE}; increase N_VAL or lower"
            " VAL_BATCH_SIZE."
        )
    return total_loss / n_batches, correct / total


def evaluate(  # noqa: PLR0913, PLR0915
    lr,
    wd,
    width,
    layers,
    n_samples,
    batch_size,
    checkpoint_path=None,
    rank=0,
    world_size=1,
    device=None,
):
    """Trains on `n_samples` samples, the batch is split evenly over the ranks.

    Returns the result to report on rank 0 and `None` on the other ranks.
    """
    device = device or torch.device("cpu")
    tokenizer = get_tokenizer("ViT-B-32")
    n_train = min(n_samples, N_TRAIN_CACHE)
    train_set, val_set = load_data(tokenizer, n_train=n_train, n_val=N_VAL)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    model = build_model(width, layers, width, layers).to(device)
    # The same count the cost estimator sampled the config with
    n_params, flops_per_sample = model_stats(width, layers)
    # Training gathers the features of every rank, validation runs on rank 0 alone
    loss_fn = ClipLoss(rank=rank, world_size=world_size)
    val_loss_fn = ClipLoss()
    if world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    per_rank_batch = max(1, batch_size // world_size)
    sampler = DistributedSampler(train_set) if world_size > 1 else None
    loader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")
    train_loader = DataLoader(
        train_set,
        batch_size=per_rank_batch,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_set, batch_size=VAL_BATCH_SIZE, **loader_kwargs)
    eval_model = model.module if world_size > 1 else model

    total_steps = max(1, n_samples // (per_rank_batch * world_size))
    curve_steps = {max(1, round(f * total_steps)) for f in CURVE_FRACTIONS}
    learning_curve = []
    val_retrieval_acc = None

    def batches():
        """Batches over repeated passes of the training set."""
        for epoch in itertools.count():
            if sampler is not None:
                sampler.set_epoch(epoch)
            yield from train_loader

    model.train()
    for step, (images, texts) in enumerate(batches(), start=1):
        images, texts = images.to(device), texts.to(device)
        optimizer.zero_grad()
        out = model(images, texts)
        loss = sum(loss_fn(**out, output_dict=True).values())
        loss.backward()
        optimizer.step()

        if step in curve_steps:
            seen = step * per_rank_batch * world_size
            if rank == 0:
                val_loss, val_retrieval_acc = contrastive_eval(
                    eval_model, val_loader, val_loss_fn, device
                )
                learning_curve.append([seen, seen * flops_per_sample, val_loss])
            if world_size > 1:
                dist.barrier()
        if step >= total_steps:
            break

    if rank != 0:
        return None

    n_seen = total_steps * per_rank_batch * world_size
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(eval_model.state_dict(), checkpoint_path)

    # The last point of the learning curve is the end of the run
    return {
        "objective_to_minimize": learning_curve[-1][-1],
        "cost": n_seen * flops_per_sample,
        "learning_curve": learning_curve,
        "n_params": n_params,
        "n_data": n_seen,
        "info_dict": {
            "val_retrieval_acc": val_retrieval_acc,
            "world_size": world_size,
            "gpu_name": get_gpu_name(),
            "peak_gpu_memory_mb": get_peak_gpu_memory_mb(),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True, type=Path)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--config_dir", type=Path)
    selector.add_argument("--group_file", type=Path)
    parser.add_argument("--task_id", type=int, help="Index into --group_file")
    args = parser.parse_args()

    if args.group_file is not None:
        config_id = yaml.safe_load(args.group_file.read_text())[args.task_id]
        config_dir = args.root_dir / "configs" / f"config_{config_id}"
    else:
        config_dir = args.config_dir

    config = yaml.safe_load((config_dir / "config.yaml").read_text())
    pipeline_id = config_dir.name.removeprefix("config_")

    rank, world_size, device = setup_distributed()
    try:
        result = evaluate(
            **config,
            checkpoint_path=config_dir / "checkpoint.pt",
            rank=rank,
            world_size=world_size,
            device=device,
        )
    except Exception as e:  # noqa: BLE001
        result = {"exception": e}

    # Only one process reports, the others only helped train
    if rank == 0:
        scalefm.save_pipeline_results(
            user_result=result,
            pipeline_id=pipeline_id,
            root_directory=args.root_dir,
        )
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
