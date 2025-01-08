from __future__ import annotations

from warnings import warn

import numpy as np


def run_pipeline(learning_rate, optimizer, epochs):
    """Func for test loading of run_pipeline."""
    warn("run_pipeline is deprecated, use evaluate_pipeline instead", DeprecationWarning, stacklevel=2)
    return evaluate_pipeline(learning_rate, optimizer, epochs)

def evaluate_pipeline(learning_rate, optimizer, epochs):
    """Func for test loading of evaluate_pipeline."""
    eval_score = np.random.choice([learning_rate, epochs], 1) if optimizer == "a" else 5.0
    return {"objective_to_minimize": eval_score}


def run_pipeline_constant(learning_rate, optimizer, epochs, batch_size):
    """Func for test loading of run_pipeline."""
    warn("run_pipeline is deprecated, use evaluate_pipeline instead", DeprecationWarning, stacklevel=2)
    return evaluate_pipeline_constant(learning_rate, optimizer, epochs, batch_size)

def evaluate_pipeline_constant(learning_rate, optimizer, epochs, batch_size):
    """Func for test loading of evaluate_pipeline."""
    eval_score = np.random.choice([learning_rate, epochs], 1) if optimizer == "a" else 5.0
    eval_score += batch_size
    return {"objective_to_minimize": eval_score}

