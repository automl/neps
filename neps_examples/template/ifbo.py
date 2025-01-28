import numpy as np
from pathlib import Path

ASSUMED_MAX_LOSS = 10


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
        wd=neps.Float(
            lower=0,
            upper=1e-1,
            log=True,
            prior=1e-3,
        ),
        epoch=neps.Integer(
            lower=1,
            upper=10,
            is_fidelity=True,  # IMPORTANT to set this to True for the fidelity parameter
        ),
    )
    return space


def evaluate_pipeline(
    pipeline_directory: Path,  # The directory where the config is saved
    previous_pipeline_directory: Path
    | None,  # The directory of the config's immediate lower fidelity
    **config,  # The hyperparameters to be used in the pipeline
) -> dict | float:
    # Defining the model
    #  Can define outside the function or import from a file, package, etc.
    class my_model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear1 = nn.Linear(in_features=224, out_features=512)
            self.linear2 = nn.Linear(in_features=512, out_features=10)

        def forward(self, x):
            x = F.relu(self.linear1(x))
            x = self.linear2(x)
            return x

    # Instantiates the model
    model = my_model()

    # IMPORTANT: Extracting hyperparameters from passed config
    learning_rate = config["lr"]
    weight_decay = config["wd"]

    # Initializing the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    ## Checkpointing
    # loading the checkpoint if it exists
    previous_state = load_checkpoint(  # predefined function from neps
        directory=previous_pipeline_directory,
        model=model,  # relies on pass-by-reference
        optimizer=optimizer,  # relies on pass-by-reference
    )
    # adjusting run budget based on checkpoint
    if previous_state is not None:
        epoch_already_trained = previous_state["epochs"]
        # + Anything else saved in the checkpoint.
    else:
        epoch_already_trained = 0
        # + Anything else with default value.

    # Extracting target epochs from config
    max_epochs = config["epoch"]

    # User TODO:
    #  Load relevant data for training and validation

    # Actual model training
    for epoch in range(epoch_already_trained, max_epochs):
        # Training loop
        ...
        # Validation loop
        ...
        logger.info(f"Epoch: {epoch}, Loss: {...}, Val. acc.: {...}")
        loss = ...

    # Save the checkpoint data in the current directory
    save_checkpoint(
        directory=pipeline_directory,
        values_to_save={"epochs": max_epochs},
        model=model,
        optimizer=optimizer,
    )

    # NOTE: Normalize the loss to be between 0 and 1
    ## crucial for ifBO's FT-PFN surrogate to work as expected
    loss = np.clip(loss, 0, ASSUMED_MAX_LOSS) / ASSUMED_MAX_LOSS

    # Return a dictionary with the results, or a single float value (loss)
    return {
        "objective_to_minimize": loss,
        "info_dict": {
            "train_accuracy": ...,
            "test_accuracy": ...,
        },
    }


if __name__ == "__main__":
    import neps

    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=pipeline_space(),
        root_directory="results",
        max_evaluations_total=50,
        optimizer="ifbo",
    )
# end of evaluate_pipeline
