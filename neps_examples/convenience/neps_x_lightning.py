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

#############################################################
# Definig the seeds for reproducibility


def set_seed(seed=123):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


#############################################################
# Function to get the initial directory used for storing tfevent files and
# checkpoints


def initial_directory(pipeline_directory: Path) -> Path:
    """
    Find the initial directory based on its existence and the presence of
    the "previous_config.id" file.

    Args:
        pipeline_directory (Path): The starting directory to search from.

    Returns:
        Path: The initial directory.
    """
    while True:
        # Get the id of the previous directory
        previous_pipeline_directory_id = pipeline_directory / "previous_config.id"

        # Get the directory where all configs are saved
        optim_result_dir = pipeline_directory.parent

        if previous_pipeline_directory_id.exists():
            # Get and join to the previous path according to the id
            with open(previous_pipeline_directory_id) as config_id_file:
                id = config_id_file.read()
                pipeline_directory = optim_result_dir / f"config_{id}"
        else:
            # Initial directory found
            return pipeline_directory


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

        # Define data transformation and loss function
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
        Perform a forward pass and compute loss, predictions, and get the ground
        truth labels for a batch of data.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> float:
        loss, preds, y = self.common_step(batch, batch_idx)
        self.train_accuracy.update(preds, y)

        self.log_dict(
            {"train_loss": loss, "train_acc": self.val_accuracy.compute()},
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss, preds, y = self.common_step(batch, batch_idx)
        self.val_accuracy.update(preds, y)

        self.log_dict(
            {"val_loss": loss, "val_acc": self.val_accuracy.compute()},
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
        data_dir=neps.ConstantParameter("./data"),
        batch_size=neps.ConstantParameter(64),
        lr=neps.FloatParameter(lower=1e-5, upper=1e-2, log=True, default=1e-3),
        weight_decay=neps.FloatParameter(
            lower=1e-5, upper=1e-3, log=True, default=5e-4
        ),
        optimizer=neps.CategoricalParameter(choices=["Adam", "SGD"], default="Adam"),
        epochs=neps.IntegerParameter(lower=1, upper=9, log=False, is_fidelity=True),
    )
    return space


#############################################################
# Define the run pipeline function


def run_pipeline(pipeline_directory, previous_pipeline_directory, **config) -> dict:
    # Initialize the first directory to store the event and checkpoints files
    init_dir = initial_directory(pipeline_directory)
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
        filename="{epoch}-{val_loss:.2f}",
    )

    # Initialize variables for checkpoint tracking progress
    previously_spent_epochs = 0
    checkpoint_path = None

    if previous_pipeline_directory:
        # Search for possible checkpoints to continue training
        ckpt_files = glob.glob(str(checkpoint_dir / "*.ckpt"))

        if ckpt_files:
            # Load the checkpoint and retrieve necessary data
            checkpoint_path = ckpt_files[-1]
            checkpoint = torch.load(checkpoint_path)
            previously_spent_epochs = checkpoint["epoch"]
        else:
            raise FileNotFoundError(
                "No checkpoint files were located in the checkpoint directory"
            )

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
    val_loss = trainer.logged_metrics.get("val_loss", None)
    val_accuracy = trainer.logged_metrics.get("val_acc", None)

    # Test the model and retrieve test metrics
    trainer.test(model)

    test_accuracy = trainer.logged_metrics.get("test_acc", None)

    return {
        "loss": val_loss,
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
        run_pipeline=run_pipeline,
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
