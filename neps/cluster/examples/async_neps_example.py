"""Example of using NepsAsyncEvaluator with NEPS for async SLURM evaluation.

This example shows the recommended pattern for running NEPS optimizations
with async evaluation via SLURM.
"""

from pathlib import Path
import neps
from cluster import NepsAsyncEvaluator, SlurmConfig


# Configure SLURM job settings
# Job names will be auto-generated from root_dir and config_id
slurm_config = SlurmConfig(
    root_dir="results/mnist_scaling_study",  # Used to generate job name
    partition="testdlc2_gpu-l40s",
    gpus_per_node=1,
    cpus_per_task=1,
    mem_gb=32,
    output_dir="logs/slurm",
    workspace_dir="neps_cluster/examples",
    # Environment: using conda (default)
    environment_manager="conda",
    conda_env_name="scaling",
    conda_path="/home/alipourn/miniconda3",
    # Optional: account for billing
    account=None,
)

# Create async evaluator that submits jobs to SLURM
# The evaluator will run 'python run_pipeline.py' with the pipeline arguments
evaluator = NepsAsyncEvaluator(
    training_script="train_example.py",
    config=slurm_config
)

# Define the evaluate_pipeline function for NEPS
# This is called by NEPS and submits the job to SLURM
def evaluate_pipeline_async(pipeline_id, learning_rate, optimizer, **kwargs):
    """
    Called by neps.run() for each configuration to evaluate.
    
    Instead of running training synchronously, this submits a SLURM job
    and returns None immediately (async evaluation).
    
    Args:
        pipeline_id: Unique trial identifier from NEPS
        learning_rate: Learning rate hyperparameter
        optimizer: Optimizer choice ("sgd", "adam", etc.)
        **kwargs: Additional hyperparameters
    
    Returns:
        None (async evaluation - job runs on cluster)
    """
    # Submit SLURM job with these parameters
    # run_pipeline.py will be called as:
    # python run_pipeline.py --pipeline-id <id> --learning-rate <lr> --optimizer <opt> ...
    evaluator.lazy_submit(
        pipeline_id=pipeline_id,
        learning_rate=learning_rate,
        optimizer=optimizer,
        **kwargs
    )
    

import logging
logging.basicConfig(level=logging.DEBUG)
# Define the search space
class ExampleSpace(neps.PipelineSpace):
    """Hyperparameter space for the optimization."""
    optimizer = neps.Categorical(choices=["sgd", "adam", "rmsprop"])
    learning_rate = neps.Float(lower=1e-5, upper=1e-2, log=True)
    batch_size = neps.Categorical(choices=[16, 32, 64])


if __name__ == "__main__":
    # Run NEPS optimization
    neps.run(
        evaluate_pipeline=evaluate_pipeline_async,
        pipeline_space=ExampleSpace(),
        root_directory="results/mnist_scaling_study",
        evaluations_to_spend=1,  # Total configs to try
        overwrite_root_directory=True,
        optimizer="random_search"
    )
