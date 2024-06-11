import numpy as np


def run_pipeline(learning_rate, optimizer, epochs):
    """func for test loading of run_pipeline"""
    if optimizer == "a":
        eval_score = np.random.choice([learning_rate, epochs], 1)
    else:
        eval_score = 5.0
    return {"loss": eval_score}


def run_pipeline_constant(learning_rate, optimizer, epochs, batch_size):
    """func for test loading of run_pipeline"""
    if optimizer == "a":
        eval_score = np.random.choice([learning_rate, epochs], 1)
    else:
        eval_score = 5.0
    eval_score += batch_size
    return {"loss": eval_score}

