import logging
import numpy as np
import neps
import socket
import os
from neps.optimizers.algorithms import kaplan_guided_scaling
# from lcbench import estimate_lcbench_flops, Objective

dataset_size= 1000 #TODO
# bench_name, bench_desc = list(mfpbench_benches.items())[0]
bench_name, bench_desc= None, None
# bench = bench_desc.load(bench_desc)

conf_const = dict(learning_rate = 0.001,
    batch_size = 87,
    max_dropout = 0.2442070308082,
    momentum = 0.3778629866698,
    num_layers = 4,
    weight_decay = 0.0473271620733)

def _get_layer_widths(max_units, n_layers):
    """Reconstructs the funnel shape: [input, L1, L2, ..., output]"""
    widths = []
    
    current_width = max_units
    
    # LCBench Funnel Strategy: Halve width each layer, but keep >= 16 or num_classes
    for _ in range(n_layers):
        widths.append(current_width)
        current_width = max(16, int(current_width / 2))
    # ignore params in last layer 
    # widths.append(self.num_classes)
    return widths

def get_number_of_parameters(epoch, n_layers, max_units):
    """Calculates total trainable parameters (Weights + Biases)."""
    widths = _get_layer_widths(max_units, n_layers)
    total_params = 0
    
    for i in range(len(widths) - 1):
        n_in = widths[i]
        n_out = widths[i+1]
        
        # Linear Layer: Weights + Biases
        weights = n_in * n_out
        biases = n_out
        total_params += weights + biases
    return total_params

def evaluate_pipeline(epoch, n_layers, max_units):
    conf = dict(epoch=epoch, n_layers=n_layers, max_units=max_units)
    conf.update(conf_const)
    return bench.query(None, epoch=epoch, objectives=["val_accuracy"])[0]


def get_total_flops(epoch, n_layers, max_units):
    print(type(epoch))
    conf = dict(epoch=epoch, n_layers=n_layers, max_units=max_units)
    conf.update(conf_const)
    return None


def seen_datapoints_estimator(epoch, n_layers, max_units):
    return dataset_size * epoch

class PipeSpace(neps.PipelineSpace):
    epoch=neps.Integer(lower=100, upper=1000, is_scaling=True)
    n_layers = neps.Integer(lower=4, upper=8, is_scaling=True)
    max_units = neps.Integer(lower=64, upper=1024, is_scaling=True)

pipeline_space = dict(
    epoch=neps.Integer(lower=100, upper=1000, is_scaling=True),
    n_layers = neps.Integer(lower=4, upper=8, is_scaling=True),
    max_units = neps.Integer(lower=64, upper=1024, is_scaling=True)
)


logging.basicConfig(level=logging.DEBUG)
neps.run(
    evaluate_pipeline=evaluate_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/hyperparameters_example",
    cost_to_spend=1000,
    worker_id=f"worker_1-{socket.gethostname()}-{os.getpid()}",
    overwrite_root_directory=True,
    optimizer=lambda space: kaplan_guided_scaling(
        space,
        params_estimator=get_number_of_parameters,
        flops_estimator=get_total_flops,
        seen_datapoints_estimator=seen_datapoints_estimator,
        max_evaluation_flops=1000,
    ),
)
