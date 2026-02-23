"""Example training script that works with NEPS async evaluation.

This script receives hyperparameters as command-line arguments from
the NepsAsyncEvaluator and runs training on the SLURM cluster.
"""

import argparse
import json
import logging
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

        if batch_idx % 100 == 0:
            logger.info(
                f"Batch {batch_idx}/{len(train_loader)}: "
                f"Loss={loss.item():.4f}, Accuracy={100*correct/total:.1f}%"
            )

    return total_loss / len(train_loader), 100 * correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    return test_loss / len(test_loader), 100 * correct / total


def main(args):
    """Main training loop."""
    logger.info(f"Starting training with args: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup directories
    results_dir = Path("results") / args.root_directory / args.pipeline_id
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results directory: {results_dir}")

    # Save hyperparameters
    hparams = {
        "learning_rate": args.learning_rate,
        "optimizer": args.optimizer,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
    }
    with open(results_dir / "hparams.json", "w") as f:
        json.dump(hparams, f, indent=2)

    # Data setup
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        root="~/pytorch_datasets",
        train=True,
        transform=transform,
        download=True,
    )
    test_dataset = datasets.MNIST(
        root="~/pytorch_datasets",
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Model setup
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    ).to(device)

    # Optimizer setup
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    criterion = nn.CrossEntropyLoss()

    # Training loop
    start_time = time.time()
    best_accuracy = 0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        logger.info(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
            f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%"
        )

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), results_dir / "best_model.pt")

    elapsed_time = time.time() - start_time

    # Save results
    with open(results_dir / "history.json", "w") as f:
        json.dump(history, f)

    results = {
        "test_accuracy": best_accuracy,
        "final_test_loss": history["test_loss"][-1],
        "training_time": elapsed_time,
    }
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Training complete! Best test accuracy: {best_accuracy:.2f}%")
    logger.info(f"Results saved to {results_dir}")

    return best_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MNIST classifier")

    # NEPS pipeline arguments
    parser.add_argument(
        "--pipeline-id",
        type=str,
        required=True,
        help="Unique trial identifier from NEPS",
    )

    # Hyperparameters
    parser.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd", "rmsprop"],
        help="Optimizer choice",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train",
    )

    # Output
    parser.add_argument(
        "--root-directory",
        type=str,
        default="scaling_study",
        help="Root directory for results",
    )

    args = parser.parse_args()
    main(args)
