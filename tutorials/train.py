from functools import partial
from pathlib import Path
import time
import torch.nn as nn
from typing import Union

from tutorials.model import SimpleCNN
from tutorials.utils import (
    get_optimizer,
    get_scheduler,
    load_neps_checkpoint,
    prepare_mnist_dataloader,
    save_neps_checkpoint,
    train_one_epoch,
    validate_model
)

from neps.plot.tensorboard_eval import tblogger


def training_pipeline(
    # neps parameters for load-save of checkpoints
    out_dir: Union[Path, None] = None,
    load_dir: Union[Path, None] = None,
    # hyperparameters
    batch_size: int = 128,
    num_layers: int = 2,
    num_neurons: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    optimizer: str = "adamw",
    dropout: bool = True,
    # fidelity control
    epochs: int = 10,
    subsample: float = 1.0,
    # other parameters
    val_fraction: float = 0.3,
    log_neps_tensorboard: bool = False,
    verbose: bool = True,
    allow_checkpointing: bool = False,
    use_for_demo: bool = False,
    optimize_over_loss: bool = False,
):
    """Training pipeline for a simple CNN on MNIST dataset.


    This is a standard pipeline to train and validate models. 
    The only exclusive requirement to interface NePS are:
    * Arguments that pass hyperparameters
    * (Optional) Using tblogger to log tensorboard metrics supported by NePS
    * Returning a dictionary with keys "loss", "cost", and "info_dict"
        * "loss" must be a minimizing metric

    Args:
        out_dir: (Union[Path, None])
            Directory to save the checkpoint.
        load_dir: (Union[Path, None])
            Directory to load the checkpoint.
        batch_size: (int)
            Batch size for training and validation dataloader.
        num_layers: (int)
            Number of convolutional layers in the model.
        num_neurons: (int)
            Number of neurons in the hidden layer.
        learning_rate: (float)
            Learning rate for the optimizer.
        weight_decay: (float)
            L2 regularization parameter.
        optimizer: (str)
            Name of the optimizer to use.
        dropout: (bool)
            Whether to use dropout in the model.
        epochs: (int)
            Number of epochs to train the model.
        subsample: (float)
             Fraction of the training data to use.
        log_neps_tensorboard: (bool)
            Whether to log tensorboard metrics.
        verbose: (bool) 
            Whether to print training progress.
        allow_checkpointing: (bool) 
            Whether to save checkpoints.
        optimize_over_loss: (bool) 
            Whether to optimize over loss or accuracy.

        use_for_demo: (bool)
            Whether to use this pipeline for demo purposes.
            This sets the subsampling factor to 10% and epochs to 3, or to the values 
            passed if it is less than 10% and 3 respectively.
    """

    if use_for_demo:
        # For demo purposes, we will use a:
        # smaller dataset
        subsample = min(0.1, subsample)
        # fewer epochs
        epochs = min(3, epochs)
        # smaller training set
        val_fraction = max(0.4, val_fraction)
        # a single layer
        num_layers = min(1, num_layers)

    # Load data
    _start = time.time()
    (
        train_loader,
        val_loader,
        (num_channels, image_height, image_width),
        num_classes
    ) = prepare_mnist_dataloader(
        batch_size=batch_size,
        subsample_fraction=subsample,
        val_fraction=val_fraction
    )
    data_load_time = time.time() - _start

    # Instantiate model
    model = SimpleCNN(
        input_channels=num_channels,
        num_layers=num_layers,
        num_classes=num_classes,
        hidden_dim=num_neurons,
        image_height=image_height,
        image_width=image_width,
        dropout=dropout
    )

    # Instantiate loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize an optimizer
    optimizer_name = optimizer
    optimizer = get_optimizer(
        optimizer=optimizer,
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    # Initialize LR scheduler
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler="cosine",
        scheduler_args={
            "T_max": epochs,
            "eta_min": 1e-6
        }
    )

    # Load possible checkpoint
    start = time.time()
    steps = None
    if allow_checkpointing:
        steps, model, optimizer, scheduler = load_neps_checkpoint(
            load_dir, model, optimizer, scheduler
        )
    checkpoint_load_time = time.time() - start

    train_start = time.time()
    validation_time = 0
    if log_neps_tensorboard:
        writer = tblogger.ConfigWriter(write_summary_incumbent=True)
    else:
        writer = None
    # Training loop
    steps = steps or 0  # accounting for continuation if checkpoint loaded
    for epoch in range(steps, epochs):

        # perform one epoch of training
        model.train()
        model, optimizer, scheduler, mean_loss = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler
        )

        # perform validation per epoch
        start = time.time()
        val_loss, val_accuracy = validate_model(model, val_loader, criterion)
        validation_time += (time.time() - start)

        if verbose:
            print(
                f"Epoch {epoch+1}/{epochs}, "
                f"loss: {mean_loss:.5f}, "
                f"val loss: {val_loss:.5f}"
            )

        minimizing_metric = val_loss if optimize_over_loss else 1-val_accuracy

        # special logging for NePS
        start = time.time()
        if writer is not None:
            writer.add_scalar(tag="loss", scalar_value=minimizing_metric, global_step=epoch)
            writer.add_scalar(tag="mean_loss", scalar_value=mean_loss, global_step=epoch)
            writer.add_scalar(tag="lr_decay", scalar_value=scheduler.get_last_lr()[0], global_step=epoch)
        logging_time = time.time() - start
        
    if log_neps_tensorboard:  
        writer.add_hparams(
            hparam_dict={"lr": learning_rate, "optim": optimizer_name, "wd": weight_decay},
            metric_dict={"loss_val": val_loss}
        )
        writer.close()

    training_time = time.time() - train_start - validation_time - logging_time

    # Save checkpoint
    if allow_checkpointing:
        save_neps_checkpoint(out_dir, epoch, model, optimizer, scheduler)

    return {
        "objective_to_minimize": minimizing_metric,  # validation loss in the last epoch
        "cost": time.time() - _start,
        "info_dict": {
            "training_loss": float(mean_loss),  # training loss in the last epoch
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "training_time": training_time,
            "validation_time": validation_time,
        }
    }


run_pipeline_demo = partial(
    training_pipeline,
    # for a faster run
    batch_size=1024,
    subsample=0.2,
    epochs=3,
    use_for_demo=True,
)

run_pipeline_sf = partial(
    training_pipeline,
    allow_checkpointing = False,
)

run_pipeline_mf = partial(
    training_pipeline,
    allow_checkpointing = True,
)

run_pipeline_half_data = partial(training_pipeline, use_for_demo=False, subsample=0.5)