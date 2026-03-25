import logging

import torch
import torch.nn.functional as F
from torch import nn, optim

import neps


class SimpleNet(nn.Module):
    """Simple neural network for demonstration."""

    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(10, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_model_and_optimizer(learning_rate, hidden_size):
    """Create a simple model and optimizer."""
    model = SimpleNet(hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer


def simulate_training(
    learning_rate: float,
    hidden_size: int,
    epoch: int,
    batch_size: int,
) -> dict:
    """Train a neural network and return multiple objectives.
    
    two Objectives are returned:
    1. Validation error
    2. Training time
    """
    import time
    
    model, optimizer = get_model_and_optimizer(learning_rate, hidden_size)
    criterion = nn.MSELoss()
    
    # Create dummy training data
    num_train_samples = 100
    X_train = torch.randn(num_train_samples, 10)
    y_train = torch.randn(num_train_samples, 1)
    
    X_val = torch.randn(50, 10)
    y_val = torch.randn(50, 1)
    
    # Train the model
    start_time = time.time()
    model.train()
    for _ in range(epoch):
        # Mini-batch training
        for i in range(0, num_train_samples, batch_size):
            batch_X = X_train[i : i + batch_size]
            batch_y = y_train[i : i + batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    training_cost = time.time() - start_time
    
    # Evaluate on validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        validation_error = criterion(val_outputs, y_val).item()
    
    return dict(
        objective_to_minimize=[float(validation_error), training_cost],
        cost=training_cost,
    )


class PriMOSpace(neps.PipelineSpace):
    """
    Contains hyperparameters with priors and a fidelity parameter.
    """
    learning_rate = neps.Float(
        lower=1e-4,
        upper=1e-1,
        log=True,
        prior=1e-3,  # Expert belief: learning rate of ~0.001 works well
        prior_confidence="medium",
    )
    hidden_size = neps.Integer(
        lower=8,
        upper=256,
        log=True,
        prior=64,  # Expert belief: hidden size of 64 is a good balance
        prior_confidence="medium",
    )
    batch_size = neps.Integer(
        lower=8,
        upper=128,
        log=True,
        prior=32,  # Expert belief: batch size of 32 often works well
        prior_confidence="low",
    )
    epoch = neps.Fidelity(neps.Integer(
        lower=1,
        upper=10,  # At max fidelity (10 epochs), we get the best approximation
    ))


logging.basicConfig(level=logging.INFO)

# Run PriMO optimization
neps.run(
    evaluate_pipeline=simulate_training,
    pipeline_space=PriMOSpace(),
    root_directory="results/primo_multi_objective",
    optimizer="primo",  # Specify PriMO as the optimizer
    fidelities_to_spend=20,  # Budget in terms of fidelity units
)
