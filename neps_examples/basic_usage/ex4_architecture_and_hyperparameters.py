"""
This example demonstrates how to combine neural network architecture
search with hyperparameter optimization using NePS.
"""

import neps
import torch
import numpy as np
from ex3_architecture_search import NN_Space

# Extend the architecture search space with a hyperparameter
extended_space = NN_Space().add(neps.Integer(16, 128), name="batch_size")

def evaluate_pipeline(model: torch.nn.Module, batch_size: int) -> float:
    # For demonstration, we return a dummy objective value
    # In practice, you would train and evaluate the model here
    x = torch.ones(size=[1, 3, 220, 220])
    result = np.sum(model(x).detach().numpy().flatten())

    objective_value = batch_size * result  # Dummy computation
    return objective_value


if __name__ == "__main__":
    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=extended_space,
        root_directory="results/architecture_with_hp_example",
        evaluations_to_spend=5,
        overwrite_root_directory=True,
    )
    neps.status(
        root_directory="results/architecture_with_hp_example",
        print_summary=True,
    )
