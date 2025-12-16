import logging
import numpy as np
import neps
import socket
import os
from neps.optimizers.algorithms import scaling_law_guided_primo

seq_length = 2048

def evaluate_pipeline(hidden_size, num_layers, learning_rate, global_batch_size, num_training_steps):
    
    params = get_number_of_parameters(hidden_size, num_layers, learning_rate, global_batch_size, num_training_steps)
    total_tokens_processed = global_batch_size * num_training_steps * seq_length
    E = 1.7 # Simulate data noise
    # A / (N^a)
    A = 500
    alpha = 0.09
    model_loss_term = A / (params**alpha)
    
    # B / (D^b)
    B = 800
    beta = 0.15
    data_loss_term = B / (total_tokens_processed**beta)
    
    optimal_lr = 1e-4
    lr_penalty = 1.0 + (np.log10(learning_rate) - np.log10(optimal_lr))**2 * 0.5

    final_loss = (E + model_loss_term + data_loss_term) * lr_penalty
    return neps.UserResultDict(
        objective_to_minimize=final_loss, 
        cost=get_total_flops(hidden_size, num_layers, learning_rate, global_batch_size, num_training_steps)/1e12,
    )

def get_number_of_parameters(hidden_size, num_layers, learning_rate, global_batch_size, num_training_steps):
    vocab_size = 50257
    return (12 * num_layers * hidden_size**2) + (vocab_size * hidden_size)

def get_total_flops(hidden_size, num_layers, learning_rate, global_batch_size, num_training_steps):
    return 2 * num_training_steps * global_batch_size * seq_length * (12 * num_layers * hidden_size**2)

class PipeSpace(neps.PipelineSpace):
    hidden_size=neps.Integer(lower=16, upper=128, is_arch_param=True)
    num_layers=neps.Integer(lower=4, upper=8, is_arch_param=True)
    learning_rate=neps.Float(lower=0.000001, upper=0.001, log=True)
    global_batch_size=neps.Integer(lower=32, upper=1024, log=True)
    num_training_steps=neps.Integer(lower=100, upper=200, log=True)

pipeline_space = PipeSpace()

logging.basicConfig(level=logging.DEBUG)
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/hyperparameters_example",
    cost_to_spend=1000,
    worker_id=f"worker_1-{socket.gethostname()}-{os.getpid()}",
    overwrite_root_directory=True,
    optimizer=lambda space: scaling_law_guided_primo(
        space,
        get_number_of_parameters=get_number_of_parameters,
        get_total_flops=get_total_flops,
        prior_centers=None,
    ),
)
