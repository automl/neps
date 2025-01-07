"""
Exploring NePS Compatibility with PyTorch Lightning
=======================================================

1. Introduction:
----------------
Welcome to this tutorial on utilizing NePS-generated files and directories
in conjunction with PyTorch Lightning.

2. Setup:
---------
Ensure you have the necessary dependencies installed. You can install the 'NePS'
package by executing the following command:

```bash
pip install neural-pipeline-search
```

Additionally, note that 'NePS' does not include 'torchvision' as a dependency.
You can install it with this command:

```bash
pip install torchvision==0.14
```

Make sure to download the torchvision version that is compatible with your
pytorch version. More info on this link:

https://pypi.org/project/torchvision/

Additionally, you will need to install the PyTorch Lightning package. This can
be achieved with the following command:

```bash
pip install lightning
```

These dependencies ensure you have everything you need for this tutorial.
"""
import argparse
import glob
import logging
import random
import time
from pathlib import Path
from typing import Tuple
from warnings import warn

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import neps
from neps.utils.common import get_initial_directory, load_lightning_checkpoint

#############################################################
# Definig the seeds for reproducibility


def set_seed(seed=123):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


#############################################################
# Define the lightning model


class LitMNIST(L.LightningModule):
    def __init__(
        self,
        configuration: dict,
        n_train: int = 8192,
        n_valid: int = 1024,
    ):
        super().__init__()

        # Initialize the model's hyperparameters with the configuration
        self.save_hyperparameters(configuration)

        self.n_train = n_train
        self.n_valid = n_valid

        # Define data transformation and objective_to_minimize function
        self.transform = transforms.ToTensor()
        self.criterion = nn.NLLLoss()

        # Define the model's architecture
        self.linear1 = nn.Linear(in_features=784, out_features=392)
        self.linear2 = nn.Linear(in_features=392, out_features=196)
        self.linear3 = nn.Linear(in_features=196, out_features=10)

        # Define PyTorch Lightning metrics for training, validation, and testing
        metric = Accuracy(task="multiclass", num_classes=10)
        self.train_accuracy = metric.clone()
        self.val_accuracy = metric.clone()
        self.test_accuracy = metric.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass function
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return F.log_softmax(x, dim=1)

    def common_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass and compute objective_to_minimize, predictions, and get the ground
        truth labels for a batch of data.
        """
        x, y = batch
        logits = self.forward(x)
        objective_to_minimize = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        return objective_to_minimize, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> float:
        objective_to_minimize, preds, y = self.common_step(batch, batch_idx)
        self.train_accuracy.update(preds, y)

        self.log_dict(
            {"train_objective_to_minimize": objective_to_minimize, "train_acc": self.val_accuracy.compute()},
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        return objective_to_minimize

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        objective_to_minimize, preds, y = self.common_step(batch, batch_idx)
        self.val_accuracy.update(preds, y)

        self.log_dict(
            {"val_objective_to_minimize": objective_to_minimize, "val_acc": self.val_accuracy.compute()},
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        _, preds, y = self.common_step(batch, batch_idx)
        self.test_accuracy.update(preds, y)

        self.log(name="test_acc", value=self.test_accuracy.compute())

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Configure and return the optimizer based on the configuration
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(
                "The optimizer choices is not one of the available optimizers"
            )
        return optimizer

    def on_train_end(self):
        # Get the metric at the end of the training and log it with respect to
        # it's hyperparameters
        val_acc_metric = {
            "val_accuracy": self.val_accuracy.compute(),
        }

        # Log hyperparameters
        self.logger.log_hyperparams(self.hparams, metrics=val_acc_metric)

    def prepare_data(self) -> None:
        # Downloading the dataste if not already downloaded
        MNIST(self.hparams.data_dir, train=True, download=True)
        MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: str = None) -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.mnist_full = MNIST(
                self.hparams.data_dir, train=True, transform=self.transform
            )

            # Create random subsets of the training dataset for validation.
            self.train_sampler = SubsetRandomSampler(range(self.n_train))
            self.val_sampler = SubsetRandomSampler(
                range(self.n_train, self.n_train + self.n_valid)
            )

        # Assign test dataset for use in dataloader
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.hparams.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_full,
            batch_size=self.hparams.batch_size,
            sampler=self.train_sampler,
            num_workers=16,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_full,
            batch_size=self.hparams.batch_size,
            sampler=self.val_sampler,
            num_workers=16,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.mnist_test,
            batch_size=self.hparams.batch_size,
            num_workers=16,
        )


#############################################################
# Define search space


def search_space() -> dict:
    # Define a dictionary to represent the hyperparameter search space
    space = dict(
        data_dir=neps.Constant("./data"),
        batch_size=neps.Constant(64),
        lr=neps.Float(lower=1e-5, upper=1e-2, log=True, prior=1e-3),
        weight_decay=neps.Float(
            lower=1e-5, upper=1e-3, log=True, prior=5e-4
        ),
        optimizer=neps.Categorical(choices=["Adam", "SGD"], prior="Adam"),
        epochs=neps.Integer(lower=1, upper=9, log=False, is_fidelity=True),
    )
    return space


#############################################################
# Define the run pipeline function

def run_pipeline(pipeline_directory, previous_pipeline_directory, **config):
    # Deprecated function, use evaluate_pipeline instead
    warn("run_pipeline is deprecated, use evaluate_pipeline instead", DeprecationWarning)
    return evaluate_pipeline(
        pipeline_directory,
        previous_pipeline_directory,
        **config,
    )


def evaluate_pipeline(pipeline_directory, previous_pipeline_directory, **config) -> dict:
    # Initialize the first directory to store the event and checkpoints files
    init_dir = get_initial_directory(pipeline_directory)
    checkpoint_dir = init_dir / "checkpoints"

    # Initialize the model and checkpoint dir
    model = LitMNIST(config)

    # Create the TensorBoard logger for logging
    logger = TensorBoardLogger(
        save_dir=init_dir, name="data", version="logs", default_hp_metric=False
    )

    # Add checkpoints at the end of training
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}-{val_objective_to_minimize:.2f}",
    )

    # Use this function to load the previous checkpoint if it exists
    checkpoint_path, checkpoint = load_lightning_checkpoint(
        previous_pipeline_directory=previous_pipeline_directory,
        checkpoint_dir=checkpoint_dir,
    )

    if checkpoint is None:
        previously_spent_epochs = 0
    else:
        previously_spent_epochs = checkpoint["epoch"]

    # Create a PyTorch Lightning Trainer
    epochs = config["epochs"]

    trainer = L.Trainer(
        logger=logger,
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
    )

    # Train the model and retrieve training/validation metrics
    if checkpoint_path:
        trainer.fit(model, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model)

    train_accuracy = trainer.logged_metrics.get("train_acc", None)
    val_objective_to_minimize = trainer.logged_metrics.get("val_objective_to_minimize", None)
    val_accuracy = trainer.logged_metrics.get("val_acc", None)

    # Test the model and retrieve test metrics
    trainer.test(model)

    test_accuracy = trainer.logged_metrics.get("test_acc", None)

    return {
        "objective_to_minimize": val_objective_to_minimize,
        "cost": epochs - previously_spent_epochs,
        "info_dict": {
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
        },
    }


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_evaluations_total",
        type=int,
        default=15,
        help="Number of different configurations to train",
    )
    args = parser.parse_args()

    # Initialize the logger and record start time
    start_time = time.time()
    set_seed(112)
    logging.basicConfig(level=logging.INFO)

    # Run NePS with specified parameters
    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=search_space(),
        root_directory="results/hyperband",
        max_evaluations_total=args.max_evaluations_total,
        searcher="hyperband",
    )

    # Record the end time and calculate execution time
    end_time = time.time()
    execution_time = end_time - start_time

    # Log the execution time
    logging.info(f"Execution time: {execution_time} seconds")
