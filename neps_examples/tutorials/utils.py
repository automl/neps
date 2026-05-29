import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

from typing import Tuple, Union


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def set_seeds(seed: int) -> None:
    """Set seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_mnist_dataloader(
    batch_size: int=64,
    val_fraction: float=0.25,
    subsample_fraction: float=1.0,
) -> Tuple[DataLoader, DataLoader, Tuple[int, int, int], int]:
    """Prepare MNIST dataloader.

    Args:
        batch_size (int): Batch size for training and validation dataloader.
        val_fraction (float): Fraction of the dataset to use for validation.
        subsample_fraction (float): Fraction of the training dataset to use.

    Returns:
        train_loader (DataLoader): Dataloader for training dataset.
        val_loader (DataLoader): Dataloader for validation dataset.
        image_dimensions (Tuple[int, int, int]): Dimensions of the image.
        num_classes (int): Number of classes in the dataset.
    """
    # Transformations applied on each image
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (MNIST_MEAN,), (MNIST_STD,)
            ),  # Mean and Std Deviation for MNIST
        ]
    )
    # Loading MNIST dataset
    dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    val_size = int(val_fraction * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Subsample the training dataset
    if subsample_fraction < 1.0:
        train_size = int(subsample_fraction * len(train_set))
        train_set, _ = torch.utils.data.random_split(
            train_set, [train_size, len(train_set) - train_size]
        )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1000, shuffle=False)

    # Get image dimensions from first sample
    num_channels, image_height, image_width = train_set[0][0].shape

    # Number of classes in the dataset
    num_classes = len(dataset.classes)

    return train_loader, val_loader, (num_channels, image_height, image_width), num_classes


def load_neps_checkpoint(
        previous_pipeline_directory: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> Tuple[int, nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler | None]:
    """Load checkpoint state to be used by NePS.
    
    Args:
        previous_pipeline_directory (Path): Directory where checkpoint is saved.
        model (nn.Module): Model to be loaded.
        optimizer (torch.optim.Optimizer): Optimizer to be loaded.
        scheduler (torch.optim.lr_scheduler.LRScheduler | None): Scheduler to be loaded.

    Returns:
        Steps (int): Number of steps the model was trained for.
        model (nn.Module): Model with loaded state.
        optimizer (torch.optim.Optimizer): Optimizer with loaded state.
        scheduler (torch.optim.lr_scheduler.LRScheduler | None): Scheduler with loaded state.
    """
    steps = None
    if previous_pipeline_directory is not None:
        checkpoint = torch.load(previous_pipeline_directory / "checkpoint.pth", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            scheduler = None
        if "steps" in checkpoint:
            steps = checkpoint["steps"]
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
    return steps, model, optimizer, scheduler


def save_neps_checkpoint(
    pipeline_directory: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    """Save checkpoint state to be used by NePS.
    
    Args:
        pipeline_directory (Path): Directory where checkpoint is saved.
        epoch (int): Number of steps the model was trained for.
        model (nn.Module): Model to be saved.
        optimizer (torch.optim.Optimizer): Optimizer to be saved.
        scheduler (torch.optim.lr_scheduler.LRScheduler | None): Scheduler to be saved.
    """
    _save_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "steps": epoch,
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        _save_dict["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(
        _save_dict,
        pipeline_directory / "checkpoint.pth",
    )


def display_images(dataloader: DataLoader, num_images: int=5):
    """Display random images from a PyTorch DataLoader.

    Args:
        dataloader (DataLoader): The DataLoader to sample from.
        num_images (int): The number of images to display.
    """
    # Get a batch of images from the dataloader
    images, labels = next(iter(dataloader))

    # Choose random images from the batch
    indices = np.random.choice(len(images), size=num_images, replace=False)

    # Display the images
    fig, axs = plt.subplots(1, num_images, figsize=(5*num_images, 5))
    for i, idx in enumerate(indices):
        # Un-normalize the image
        image = images[idx].numpy() * MNIST_STD + MNIST_MEAN

        # PyTorch tensors for images are (C, H, W) but matplotlib expects (H, W, C)
        image = np.transpose(image, (1, 2, 0))

        # Display the image
        if num_images > 1:
            axs[i].imshow(image.squeeze(), cmap='gray')
            axs[i].set_title(f'Label: {labels[idx].item()}')
            axs[i].axis('off')  # Hide the axes
        else:
            axs.imshow(image.squeeze(), cmap='gray')
            axs.set_title(f'Label: {labels[idx].item()}')
            axs.axis('off')  # Hide the axes

    # Show the plot
    plt.show()


def get_optimizer(optimizer: str, model: nn.Module, learning_rate: float, weight_decay: float):
    optimizer_name = optimizer
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise KeyError(f"optimizer {optimizer} is not available")
    return optimizer


def get_scheduler(
    optimizer: torch.optim, scheduler: str = None, scheduler_args: dict = None
) -> torch.optim.lr_scheduler.LRScheduler | None:
    match scheduler:
        case None:
            return None
        case "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_args)
        case "step":
            return optim.lr_scheduler.StepLR(optimizer, **scheduler_args)
        case "expo":
            return optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_args)
        case _:
            raise KeyError(f"Scheduler {scheduler} is not available")


def train_one_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    scheduler=None,
    device: str = "cpu"
) -> Tuple[nn.Module, optim.Optimizer, Union[torch.optim.lr_scheduler.LRScheduler, None], float]:
    loss_per_batch = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_per_batch.append(loss.item())
    return model, optimizer, scheduler, np.mean(loss_per_batch)


def validate_model(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: str = "cpu"
) -> Tuple[float, float]:
    model.eval()
    val_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            val_loss += loss_fn(output, target).item()
    val_loss /= total
    val_accuracy = correct / total
    return val_loss, val_accuracy


def total_gradient_l2_norm(model: nn.Module) -> float:
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm**0.5