from __future__ import annotations

import logging

from torch import nn

import neps
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
    "linear": topos.get_sequential_n_edge(2),
    "diamond_mid": topos.DiamondMid,
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
    "D1Helper": ["linear Cell downsample"],
    "Cell": [
        "residual OPS OPS OPS",
        "diamond OPS OPS OPS OPS",
        "linear OPS OPS",
        "diamond_mid OPS OPS OPS OPS OPS",
    ],
    "OPS": ["conv3x3", "conv1x1", "avg_pool", "id"],
}


def set_recursive_attribute(op_name, predecessor_values):
    in_channels = 64 if predecessor_values is None else predecessor_values["C_out"]
    out_channels = in_channels * 2 if op_name == "ResNetBasicblock" else in_channels
    return dict(C_in=in_channels, C_out=out_channels)


def run_pipeline(architecture):
    in_channels = 3
    n_classes = 20
    base_channels = 64
    out_channels = 512

    model = architecture.to_pytorch()
    model = nn.Sequential(
        ops.Stem(base_channels, C_in=in_channels),
        model,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(out_channels, n_classes),
    )

    number_of_params = sum(p.numel() for p in model.parameters())
    validation_error = abs(1.5e7 - number_of_params)

    return validation_error


pipeline_space = dict(
    architecture=neps.FunctionParameter(
        set_recursive_attribute=set_recursive_attribute,
        structure=structure,
        primitives=primitives,
        name="makrograph",
    )
)

logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    root_directory="results/hierarchical_architecture_example",
    max_evaluations_total=15,
)
