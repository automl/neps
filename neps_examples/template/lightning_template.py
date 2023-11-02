"""
This code is not runnable but should serve as a guide to a successful neps run
using using pytorch lightning and priorband as a searcher.

Steps:
1. Create search space with a fidelity parameter.
2. Create run_pipeline which includes:
    A. Start by getting the initial directory, which will be used to store TensorBoard
       event files and checkpoint files.
    B. Initialize the lightning model.
    C. Create the TensorBoard logger and the checkpoint callback.
    D. Check for any existing checkpoint files and load checkpoint data.
    E. Create a PyTorch Lightning Trainer.
    F. Train the model, calculate metrics, and test the model.
3. Use neps.run and specify "priorband" as the searcher.

For a more detailed guide, please refer to:
https://github.com/automl/neps/blob/master/neps_examples/convenience/neps_x_lightning.py
"""
import logging

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

import neps
from neps.utils.common import get_initial_directory, load_lightning_checkpoint

# 1. Create the pipeline_space


def pipeline_space() -> dict:
    # Define a dictionary to represent the hyperparameter search space
    space = dict(
        lr=neps.FloatParameter(lower=1e-5, upper=1e-2, log=True, default=1e-3),
        optimizer=neps.CategoricalParameter(choices=["Adam", "SGD"], default="Adam"),
        epochs=neps.IntegerParameter(lower=1, upper=9, log=False, is_fidelity=True),
    )
    return space


# 2. Create the lightning module


class LitModel(L.LightningModule):
    def __init__(self, configuration: dict):
        super().__init__()

        self.save_hyperparameters(configuration)

        # You can now define your criterion, transforms, model layers, and
        # metrics obtained during trainig that configuration

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass function
        pass

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # Training step function
        # Training metric of choice
        pass

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # Validation step function
        # Validation metric of choice
        pass

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # Test step function
        # Test metric of choice
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # Define the optimizer base on the configuration
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr)
        else:
            raise ValueError(f"{self.hparams.optimizer} is not a valid optimizer")
        return optimizer

    # Here one can now configure the dataloaders for the model
    # Further details can be found here:
    # https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    # https://github.com/automl/neps/blob/master/neps_examples/convenience/neps_x_lightning.py


# 3. Define the run pipeline function


def run_pipeline(pipeline_directory, previous_pipeline_directory, **config) -> dict:
    # A. Start by getting the initial directory which will be used to store tensorboard
    # event files and checkpoint files
    init_dir = get_initial_directory(pipeline_directory)
    checkpoint_dir = init_dir / "checkpoints"
    tensorboard_dir = init_dir / "tensorboard"

    # B. Create the model
    model = LitModel(config)

    # C. Create the TensorBoard logger and the checkpoint callback
    logger = TensorBoardLogger(
        save_dir=tensorboard_dir, name="data", version="logs", default_hp_metric=False
    )
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir)

    # D. Checking for any checkpoint files and checkpoint data returns None if
    # no checkpoint files exist.
    checkpoint_path, checkpoint_data = load_lightning_checkpoint(
        previous_pipeline_directory=previous_pipeline_directory,
        checkpoint_dir=checkpoint_dir,
    )

    # E. Create a PyTorch Lightning Trainer
    epochs = config["epochs"]

    trainer = L.Trainer(
        logger=logger,
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
    )

    # F. Train, test, and their corresponding metrics
    if checkpoint_path:
        trainer.fit(model, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model)
    val_loss = trainer.logged_metrics.get("val_loss", None)

    trainer.test(model)
    test_loss = trainer.logged_metrics.get("test_loss", None)

    return {
        "loss": val_loss,
        "info_dict": {
            "test_loss": test_loss,
        },
    }


# 4. Define the neps.run function with the searcher as the argument

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    neps.run(
        run_pipeline=run_pipeline,
        pipeline_space=pipeline_space(),
        root_directory="results",
        max_evaluations_total=15,
        searcher="priorband",
    )
