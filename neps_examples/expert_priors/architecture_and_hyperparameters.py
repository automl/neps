import logging
import time

from torch import nn

import neps
from neps.optimizers.bayesian_optimization.acquisition_functions import AcquisitionMapping
from neps.search_spaces.architecture import primitives as ops
from neps.search_spaces.architecture import topologies as topos

primitives = {
    "id": ops.Identity(),
    "conv3x3": {"op": ops.ReLUConvBN, "kernel_size": 3, "stride": 1, "padding": 1},
    "conv1x1": {"op": ops.ReLUConvBN, "kernel_size": 1},
    "avg_pool": {"op": ops.AvgPool1x1, "kernel_size": 3, "stride": 1},
    "downsample": {"op": ops.ResNetBasicblock, "stride": 2},
    "residual": topos.Residual,
    "diamond": topos.Diamond,
    "linear": topos.Linear,
    "diamond_mid": topos.DiamondMid,
    "down1": topos.DownsampleBlock,
}

structure = {
    "S": [
        "diamond D2 D2 D1 D1",
        "diamond D1 D2 D2 D1",
        "diamond D1 D1 D2 D2",
        "linear D2 D1",
        "linear D1 D2",
        "diamond_mid D1 D2 D1 D2 D1",
        "diamond_mid D2 D2 Cell D1 D1",
    ],
    "D2": [
        "diamond D1 D1 D1 D1",
        "linear D1 D1",
        "diamond_mid D1 D1 Cell D1 D1",
    ],
    "D1": [
        "diamond D1Helper D1Helper Cell Cell",
        "diamond Cell Cell D1Helper D1Helper",
        "diamond D1Helper Cell Cell D1Helper",
        "linear D1Helper Cell",
        "linear Cell D1Helper",
        "diamond_mid D1Helper D1Helper Cell Cell Cell",
        "diamond_mid Cell D1Helper D1Helper D1Helper Cell",
    ],
    "D1Helper": ["down1 Cell downsample"],
    "Cell": [
        "residual OPS OPS OPS",
        "diamond OPS OPS OPS OPS",
        "linear OPS OPS",
        "diamond_mid OPS OPS OPS OPS OPS",
    ],
    "OPS": ["conv3x3", "conv1x1", "avg_pool", "id"],
}

prior_distr = {
    "S": [1 / 7 for _ in range(7)],
    "D2": [1 / 3 for _ in range(3)],
    "D1": [1 / 7 for _ in range(7)],
    "D1Helper": [1],
    "Cell": [1 / 4 for _ in range(4)],
    "OPS": [1 / 4 for _ in range(4)],
}


def set_recursive_attribute(op_name, predecessor_values):
    in_channels = 64 if predecessor_values is None else predecessor_values["C_out"]
    out_channels = in_channels * 2 if op_name == "ResNetBasicblock" else in_channels
    return dict(C_in=in_channels, C_out=out_channels)


def run_pipeline(some_architecture, some_float, some_integer, some_cat):
    start = time.time()

    in_channels = 3
    n_classes = 20
    base_channels = 64
    out_channels = 512

    model = some_architecture.to_pytorch()
    model = nn.Sequential(
        ops.Stem(base_channels, C_in=in_channels),
        model,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(out_channels, n_classes),
    )

    number_of_params = sum(p.numel() for p in model.parameters())
    y = abs(1.5e7 - number_of_params)

    if some_cat != "a":
        y *= some_float + some_integer
    else:
        y *= -some_float - some_integer

    end = time.time()

    return {
        "loss": y,
        "info_dict": {
            "test_score": y,
            "train_time": end - start,
        },
    }


pipeline_space = dict(
    some_architecture=neps.FunctionParameter(
        set_recursive_attribute=set_recursive_attribute,
        structure=structure,
        primitives=primitives,
        name="pibo",
        prior=prior_distr,
    ),
    some_float=neps.FloatParameter(
        lower=1, upper=1000, log=True, default=900, default_confidence="medium"
    ),
    some_integer=neps.IntegerParameter(
        lower=0, upper=50, default=35, default_confidence="low"
    ),
    some_cat=neps.CategoricalParameter(
        choices=["a", "b", "c"], default="a", default_confidence="high"
    ),
)

acquisition_function = AcquisitionMapping["EI"]

logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/user_priors_with_graphs",
    max_evaluations_total=15,
    acquisition=acquisition_function,
    log_prior_weighted=True,
)

previous_results, pending_configs = neps.status("results/user_priors_with_graphs")
