"""One GPU-count's HPO sweep, run under `torchrun` inside a Slurm job.

    torchrun --standalone --nproc_per_node=<n_gpus> hpo_ddp.py \\
        --root_dir <results/scaling_study/gpus_<n_gpus>> --n_gpus <n_gpus>

`run_scaling_study.py` submits one such job per GPU count, each with its own
`root_dir`, so the three sweeps are independent NePS states that can queue and
run concurrently.

Rank 0 is the NePS worker: it calls `neps.run()`, and its `evaluate_pipeline`
broadcasts the sampled config to the other ranks before every rank trains it
together under DDP. The other ranks never call `neps.run()`; they sit in
`_follower_loop`, training whatever rank 0 sends until it signals the sweep is
over.

Why not NePS's own DDP handling in `neps/runtime.py`
(`_is_ddp_and_not_rank_zero` -> `_launch_ddp_runtime`)? That path is reached
under `torchrun` -- but it then reads the trial id from `NEPS_DDP_TRIAL_ID`,
which only exists when the non-zero ranks are *children* of the rank-0 NePS
process and inherited its environment (the PyTorch Lightning `strategy="ddp"`
case in `neps_examples/efficiency/pytorch_lightning_ddp.py`). `torchrun` starts
all ranks as siblings before any trial exists, so the variable is unset and
NePS raises. It is also an infinite loop with no exit condition, so even with
the id supplied the non-zero ranks would never return and the Slurm job would
hang until its time limit. Hence the explicit broadcast below.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist

import neps
from train_ddp import evaluate_ddp

# #CHANGE_ME: how many configs each GPU count evaluates.
EVALUATIONS_TO_SPEND = 5

# Sentinel broadcast by rank 0 once `neps.run` has finished, to release the
# followers from their loop so the job can exit.
_STOP = "__stop__"


class HPOSpace(neps.PipelineSpace):
    """What the sweep searches, and what it holds fixed.

    Categorical + grid search on purpose: the three GPU counts must evaluate
    the *same* configs in the *same* order, or their throughputs are not
    comparable. A random search would hand each job a different set.
    """

    # #CHANGE_ME: the searched hyperparameters.
    lr = neps.Categorical(choices=(3e-4, 1e-3, 3e-3))
    wd = neps.Categorical(choices=(1e-5, 1e-4))
    # Global (all-GPU) batch size. Every choice must divide by every GPU count
    # in `run_scaling_study.N_GPUS_CHOICES`, or the per-GPU split is inexact.
    batch_size = neps.Categorical(choices=(256, 512))

    # #CHANGE_ME: fixed for the whole study -- these are what make a 1-GPU and
    # a 4-GPU trial the same job rather than two different ones.
    vision_width = 256
    vision_layers = 6
    text_width = 256
    text_layers = 6
    epoch = 3


def _broadcast(payload, device):
    """Send `payload` from rank 0 to every rank, and return it everywhere.

    `device` must be this rank's own GPU. NCCL pins the collective to the
    current CUDA device, so without it every rank would land on device 0 and
    the broadcast fails with "Duplicate GPU detected".
    """
    box = [payload]
    dist.broadcast_object_list(box, src=0, device=device)
    return box[0]


def _follower_loop(rank, local_rank, world_size, n_gpus, device):
    """Ranks != 0: train whatever rank 0 sends, until it says stop."""
    while True:
        config = _broadcast(None, device)
        if config == _STOP:
            return
        evaluate_ddp(
            rank=rank, local_rank=local_rank, world_size=world_size, n_gpus=n_gpus, **config,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root_dir", required=True, type=Path)
    parser.add_argument("--n_gpus", required=True, type=int)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != args.n_gpus:
        raise ValueError(
            f"torchrun launched WORLD_SIZE={world_size} but --n_gpus={args.n_gpus}; "
            "fix --nproc_per_node in the generated job script."
        )

    # Bind this rank to its own GPU before any collective runs: NCCL uses the
    # current CUDA device, and every rank defaulting to device 0 makes even the
    # config broadcast below fail.
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}") if is_cuda else torch.device("cpu")

    dist.init_process_group(backend="nccl" if is_cuda else "gloo")
    try:
        if rank != 0:
            _follower_loop(rank, local_rank, world_size, args.n_gpus, device)
            return

        def evaluate_pipeline(pipeline_directory, batch_size, **config):
            if batch_size % world_size:
                raise ValueError(
                    f"batch_size={batch_size} is not divisible by world_size="
                    f"{world_size}; every choice must split evenly over the GPUs."
                )
            # Hand the followers this trial's config, then train it together.
            payload = {
                "batch_size": batch_size,
                "checkpoint_path": Path(pipeline_directory) / "checkpoint.pt",
                **config,
            }
            _broadcast(payload, device)
            return evaluate_ddp(
                rank=rank, local_rank=local_rank, world_size=world_size,
                n_gpus=args.n_gpus, **payload,
            )

        logging.basicConfig(level=logging.INFO)
        neps.run(
            evaluate_pipeline=evaluate_pipeline,
            pipeline_space=HPOSpace(),
            root_directory=args.root_dir,
            optimizer=("grid_search", {}),
            evaluations_to_spend=EVALUATIONS_TO_SPEND,
        )
        _broadcast(_STOP, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
