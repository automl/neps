"""The DDP training run shared by every rank of one scaling-study trial.

`evaluate_ddp` is called in lockstep by all `n_gpus` ranks of a
`torchrun`-launched job (see `hpo_ddp.py`), and returns the measurement on rank
0 and `None` elsewhere. It does not touch NePS: `hpo_ddp.py` owns the search.

The workload is real LAION CC12M image/caption pre-training, read from the
compact local cache built once by `../download_data.py`. That matters for a
scaling study: CIFAR-sized images and a toy model leave each GPU with so little
to do per step that DDP setup and gradient all-reduce dominate, and the
measurement ends up being about overhead rather than about the GPUs.

What is held fixed, so that the only variable is the GPU count:
  - the dataset (`N_TRAIN` samples, `epoch` passes over it),
  - the global batch size, split evenly as `batch_size // world_size` per rank,
    so every run takes the same number of optimizer steps over the same number
    of samples,
  - the contrastive batch: `ClipLoss` all-gathers features across ranks, so the
    loss at 4 GPUs sees the same negatives as the single-GPU run. The cost of
    that all-gather is real CLIP-training cost and is deliberately inside the
    timed region.

The headline number is therefore total training throughput: how many samples
per second all GPUs together push through one identical job.

Two details keep the measurement honest:
  - `WARMUP_STEPS` optimizer steps run before the timer starts, so CUDA context
    creation, cuDNN autotuning and DDP's first-iteration bucket setup are not
    charged to the measured region,
  - `drop_last=True`, so every rank runs an identical number of full-size steps
    and no rank idles at a ragged final batch.
"""

import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from open_clip import ClipLoss, get_tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# `common.py` lives one directory up (shared with ../train.py); make sure it's
# importable regardless of the cwd torchrun/Slurm was launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import build_model, count_params, get_gpu_name, get_model_flops, get_peak_gpu_memory_mb, load_data

# #CHANGE_ME: the fixed workload size. `N_TRAIN` is a *total* over all GPUs and
# never changes with `n_gpus` -- that is the whole point of the study. It is
# bounded by how long a benchmark run should take, not by the corpus (a staged
# LAION-400M copy holds hundreds of millions of pairs); the only requirement is
# that the cache already covers `N_TRAIN + N_VAL`:
#     python ../download_data.py --n_samples 102000
N_TRAIN = 100_000
N_VAL = 2_000

# Validation runs at a fixed batch size regardless of `n_gpus`, so the
# contrastive loss always sees the same number of negatives.
VAL_BATCH_SIZE = 256

# Optimizer steps to run before starting the timer (excluded from the reported
# throughput), and dataloader workers per rank so that CPU-side image decoding
# does not become the thing being benchmarked.
WARMUP_STEPS = 5
NUM_WORKERS = 4


def evaluate_ddp(
    rank, local_rank, world_size, n_gpus,  # noqa: PLR0913
    lr, wd, vision_width, vision_layers, text_width, text_layers, epoch, batch_size,
    checkpoint_path=None,
):
    is_cuda = torch.cuda.is_available()
    device = f"cuda:{local_rank}" if is_cuda else "cpu"
    if is_cuda:
        torch.cuda.set_device(local_rank)
        torch.cuda.reset_peak_memory_stats(device)

    tokenizer = get_tokenizer("ViT-B-32")
    train_set, val_set = load_data(tokenizer, n_train=N_TRAIN, n_val=N_VAL)

    model = build_model(vision_width, vision_layers, text_width, text_layers).to(device)
    ddp_model = DDP(model, device_ids=[local_rank] if is_cuda else None)
    # Gathering features across ranks keeps the contrastive problem identical at
    # every GPU count: the loss always sees `batch_size` negatives, not
    # `batch_size // world_size`.
    loss_fn = ClipLoss(local_loss=True, gather_with_grad=True, rank=rank, world_size=world_size)
    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=lr, weight_decay=wd)

    # Fixed global batch size, split evenly over the ranks.
    per_rank_batch_size = batch_size // world_size
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_loader = DataLoader(
        train_set,
        batch_size=per_rank_batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=is_cuda,
        persistent_workers=NUM_WORKERS > 0,
    )

    def train_step(images, texts):
        images, texts = images.to(device, non_blocking=True), texts.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = ddp_model(images, texts)
        loss = sum(loss_fn(**out, output_dict=True).values())
        loss.backward()
        optimizer.step()

    def sync():
        if is_cuda:
            torch.cuda.synchronize()
        dist.barrier()

    ddp_model.train()
    step, measured_steps, measured_samples, t0 = 0, 0, 0, None
    for epoch_idx in range(epoch):
        sampler.set_epoch(epoch_idx)
        for images, texts in train_loader:
            if step == WARMUP_STEPS:
                # Everything before this point (context creation, autotuning,
                # DDP bucket setup) is deliberately not measured.
                sync()
                t0 = time.perf_counter()
            train_step(images, texts)
            step += 1
            if t0 is not None:
                measured_steps += 1
                measured_samples += images.shape[0]
    sync()
    if t0 is None:
        raise RuntimeError(
            f"The whole run was {step} steps, which is not more than WARMUP_STEPS="
            f"{WARMUP_STEPS}; nothing was timed. Increase N_TRAIN/epoch or lower WARMUP_STEPS."
        )
    wall_clock_time_sec = time.perf_counter() - t0

    # Total samples pushed through by *all* ranks together during the timed
    # region -- the numerator of the throughput we care about.
    samples_tensor = torch.tensor([float(measured_samples)], device=device)
    dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)
    total_train_samples = samples_tensor.item()

    flops = get_model_flops(model) * total_train_samples
    n_params = count_params(model)

    peak_gpu_memory_mb = get_peak_gpu_memory_mb()
    if is_cuda:
        peak_tensor = torch.tensor([peak_gpu_memory_mb or 0.0], device=device)
        dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
        peak_gpu_memory_mb = peak_tensor.item()

    if rank != 0:
        return None

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

    # Validation is rank-0-only and runs outside the timed region: it is a
    # sanity check that the fixed workload really did train, not part of the
    # scaling measurement. Single-rank `ClipLoss`, since only rank 0 is here.
    ddp_model.eval()
    val_loss_fn = ClipLoss()
    val_loss, correct, total, n_batches = 0.0, 0, 0, 0
    with torch.no_grad():
        for images, texts in DataLoader(val_set, batch_size=VAL_BATCH_SIZE):
            if images.shape[0] < VAL_BATCH_SIZE:
                continue
            images, texts = images.to(device), texts.to(device)
            out = model(images, texts)
            val_loss += sum(val_loss_fn(**out, output_dict=True).values()).item()
            n_batches += 1

            logits = out["logit_scale"] * out["image_features"] @ out["text_features"].t()
            target = torch.arange(images.shape[0], device=device)
            correct += (logits.argmax(dim=-1) == target).sum().item()
            total += images.shape[0]

    return {
        "objective_to_minimize": val_loss / n_batches,
        "cost": epoch,
        "info_dict": {
            "n_gpus": n_gpus,
            "wall_clock_time_sec": wall_clock_time_sec,
            "samples_per_sec": total_train_samples / wall_clock_time_sec,
            "total_train_samples": total_train_samples,
            "optimizer_steps": measured_steps,
            "global_batch_size": batch_size,
            "per_rank_batch_size": per_rank_batch_size,
            "val_retrieval_acc": correct / total,
            "flops": flops,
            "n_params": n_params,
            "gpu_name": get_gpu_name(),
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
        },
    }
