"""Trains and evaluates one config on a single GPU, timing the training loop.
Returns the NePS result plus the throughput numbers the scaling study plots.
"""

import sys
import time
from pathlib import Path

import torch
from open_clip import ClipLoss, get_tokenizer
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import build_model, count_params, get_gpu_name, get_model_flops, get_peak_gpu_memory_mb, load_data

# #CHANGE_ME: the fixed workload every trial trains on, whatever the worker
# count. The cache must already cover it: `python ../download_data.py --n_samples 102000`.
N_TRAIN = 100_000
N_VAL = 2_000

VAL_BATCH_SIZE = 256
WARMUP_STEPS = 5
NUM_WORKERS = 4


def evaluate(
    lr, wd, vision_width, vision_layers, text_width, text_layers, epoch, batch_size,
    n_workers, checkpoint_path=None,
):
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if is_cuda else "cpu")
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    tokenizer = get_tokenizer("ViT-B-32")
    train_set, val_set = load_data(tokenizer, n_train=N_TRAIN, n_val=N_VAL)

    model = build_model(vision_width, vision_layers, text_width, text_layers).to(device)
    loss_fn = ClipLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=is_cuda,
        persistent_workers=NUM_WORKERS > 0,
    )

    def sync():
        if is_cuda:
            torch.cuda.synchronize()

    model.train()
    step, measured_steps, measured_samples, t0 = 0, 0, 0, None
    for _ in range(epoch):
        for images, texts in train_loader:
            if step == WARMUP_STEPS:
                # Context creation, autotuning and cache warm-up stay unmeasured.
                sync()
                t0 = time.perf_counter()
            images, texts = images.to(device, non_blocking=True), texts.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(images, texts)
            loss = sum(loss_fn(**out, output_dict=True).values())
            loss.backward()
            optimizer.step()
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

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)

    # Validation is outside the timed region
    model.eval()
    val_loss, correct, total, n_batches = 0.0, 0, 0, 0
    with torch.no_grad():
        for images, texts in DataLoader(val_set, batch_size=VAL_BATCH_SIZE):
            if images.shape[0] < VAL_BATCH_SIZE:
                continue
            images, texts = images.to(device), texts.to(device)
            out = model(images, texts)
            val_loss += sum(loss_fn(**out, output_dict=True).values()).item()
            n_batches += 1

            logits = out["logit_scale"] * out["image_features"] @ out["text_features"].t()
            target = torch.arange(images.shape[0], device=device)
            correct += (logits.argmax(dim=-1) == target).sum().item()
            total += images.shape[0]

    return {
        "objective_to_minimize": val_loss / n_batches,
        "cost": epoch,
        "info_dict": {
            "n_workers": n_workers,
            "wall_clock_time_sec": wall_clock_time_sec,
            "samples_per_sec": measured_samples / wall_clock_time_sec,
            "total_train_samples": measured_samples,
            "optimizer_steps": measured_steps,
            "batch_size": batch_size,
            "val_retrieval_acc": correct / total,
            "flops": get_model_flops(model) * measured_samples,
            "n_params": count_params(model),
            "gpu_name": get_gpu_name(),
            "peak_gpu_memory_mb": get_peak_gpu_memory_mb(),
        },
    }
