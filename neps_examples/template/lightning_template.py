""" Boilerplate code to optimize a simple PyTorch Lightning model.

NOTE!!! This code is not meant to be executed.
It is only to serve as a template to help interface NePS with an existing ML/DL pipeline.


The following script describes the crucial components that a user needs to provide
in order to interface with Lightning.

The 3 crucial components are:
* The search space, called the `pipeline_space` in NePS
  * This defines the set of hyperparameters that the optimizer will search over
  * This declaration also allows injecting priors per hyperparameter
* The `lightning module`
  * This defines the training, validation, and testing of the model
  * This distributes the hyperparameters
  * This can be used to create the Dataloaders for training, validation, and testing
* The `evaluate_pipeline` function
  * This function is called by the optimizer and is responsible for running the pipeline
  * The function should at the minimum expect the hyperparameters as keyword arguments
  * The function should return the objective_to_minimize of the pipeline as a float
    * If the return value is a dictionary, it should have a key called "objective_to_minimize" with the objective_to_minimize as a float

Overall, running an optimizer from NePS with Lightning involves 5 clear steps:
1. Importing neccessary packages including NePS and Lightning.
2. Designing the search space as a dictionary.
3. Creating the LightningModule with the required parameters
4. Creating the evaluate_pipeline and returning the objective_to_minimize and other wanted metrics.
5. Using neps run with the optimizer of choice.

For a more detailed guide, please refer to:
https://github.com/automl/neps/blob/master/neps_examples/convenience/neps_x_lightning.py
"""
import logging
from warnings import warn

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

import neps
from neps.utils.common import get_initial_directory, load_lightning_checkpoint

logger = logging.getLogger("neps_template.run")


def pipeline_space() -> dict:
    # Create the search space based on NEPS parameters and return the dictionary.
    # IMPORTANT:
    space = dict(
        lr=neps.Float(
            lower=1e-5,
            upper=1e-2,
            log=True,  # If True, the search space is sampled in log space
            prior=1e-3,  # a non-None value here acts as the mode of the prior distribution
        ),
        optimizer=neps.Categorical(choices=["Adam", "SGD"], prior="Adam"),
        epochs=neps.Integer(
            lower=1,
            upper=9,
            is_fidelity=True,  # IMPORTANT to set this to True for the fidelity parameter
        ),
    )
    return space


class LitModel(L.LightningModule):
    def __init__(self, configuration: dict):
        super().__init__()

        self.save_hyperparameters(configuration)

        # You can now define your criterion, data transforms, model layers, and
        # metrics obtained during training

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


def run_pipeline(pipeline_directory, previous_pipeline_directory, **config) -> dict | float:
    warn("run_pipeline is deprecated, use evaluate_pipeline instead", DeprecationWarning)
    return evaluate_pipeline(pipeline_directory, previous_pipeline_directory, **config)

def evaluate_pipeline(
    pipeline_directory,  # The directory where the config is saved
    previous_pipeline_directory,  # The directory of the config's immediate lower fidelity
    **config,  # The hyperparameters to be used in the pipeline
) -> dict | float:
    # Start by getting the initial directory which will be used to store tensorboard
    # event files and checkpoint files
    init_dir = get_initial_directory(pipeline_directory)
    checkpoint_dir = init_dir / "checkpoints"
    tensorboard_dir = init_dir / "tensorboard"

    # Create the model
    model = LitModel(config)

    # Create the TensorBoard logger and the checkpoint callback
    logger = TensorBoardLogger(
        save_dir=tensorboard_dir, name="data", version="logs", default_hp_metric=False
    )
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir)

    # Checking for any checkpoint files and checkpoint data, returns None if
    # no checkpoint files exist.
    checkpoint_path, checkpoint_data = load_lightning_checkpoint(
        previous_pipeline_directory=previous_pipeline_directory,
        checkpoint_dir=checkpoint_dir,
    )

    # Create a PyTorch Lightning Trainer
    epochs = config["epochs"]

    trainer = L.Trainer(
        logger=logger,
        max_epochs=epochs,
        callbacks=[checkpoint_callback],
    )

    # Train, test, and get their corresponding metrics
    if checkpoint_path:
        trainer.fit(model, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model)
    val_objective_to_minimize = trainer.logged_metrics.get("val_objective_to_minimize", None)

    trainer.test(model)
    test_objective_to_minimize = trainer.logged_metrics.get("test_objective_to_minimize", None)

    # Return a dictionary with the results, or a single float value (objective_to_minimize)
    return {
        "objective_to_minimize": val_objective_to_minimize,
        "info_dict": {
            "test_objective_to_minimize": test_objective_to_minimize,
        },
    }


# end of evaluate_pipeline

if __name__ == "__main__":
    neps.run(
        evaluate_pipeline=evaluate_pipeline,  # User TODO (defined above)
        pipeline_space=pipeline_space(),  # User TODO (defined above)
        root_directory="results",
        max_evaluations_total=25,  # total number of times `evaluate_pipeline` is called
        searcher="priorband",  # "priorband_bo" for longer budgets, and set `initial_design_size``
    )
