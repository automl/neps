from __future__ import annotations

import logging
import time

import networkx as nx
from torch import nn

import neps
from neps.search_spaces.graph_grammar import primitives as ops
from neps.search_spaces.graph_grammar import topologies as topos

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


def build(graph):
    in_channels = 3
    n_classes = 20
    base_channels = 64
    out_channels = 512

    # Assign channels
    in_node = [n for n in graph.nodes if graph.in_degree(n) == 0][0]
    for n in nx.topological_sort(graph):
        for pred in graph.predecessors(n):
            e = (pred, n)
            if pred == in_node:
                channels = base_channels
            else:
                pred_pred = list(graph.predecessors(pred))[0]
                channels = graph.edges[(pred_pred, pred)]["C_out"]
            if graph.edges[e]["op_name"] == "ResNetBasicblock":
                graph.edges[e].update({"C_in": channels, "C_out": channels * 2})
            else:
                graph.edges[e].update({"C_in": channels, "C_out": channels})

    in_node = [n for n in graph.nodes if graph.in_degree(n) == 0][0]
    out_node = [n for n in graph.nodes if graph.out_degree(n) == 0][0]
    max_node_label = max(graph.nodes())
    graph.add_nodes_from([max_node_label + 1, max_node_label + 2])
    graph.add_edge(max_node_label + 1, in_node)
    graph.edges[max_node_label + 1, in_node].update(
        {
            "op": ops.Stem(base_channels, C_in=in_channels),
            "op_name": "Stem",
        }
    )
    graph.add_nodes_from([out_node, max_node_label + 2])
    graph.add_edge(out_node, max_node_label + 2)

    graph.edges[out_node, max_node_label + 2].update(
        {
            "op": ops.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels, n_classes),
            ),
            "op_name": "Out",
        }
    )


def run_pipeline(working_directory, architecture):
    start = time.time()
    model = architecture.to_pytorch()
    number_of_params = sum(p.numel() for p in model.parameters())
    y = abs(1.5e7 - number_of_params)
    end = time.time()

    return {
        "loss": y,
        "info_dict": {
            "test_score": y,
            "train_time": end - start,
        },
    }


pipeline_space = dict(
    architecture=neps.FunctionParameter(
        build_fn=build,
        name="makrograph",
        grammar=structure,
        terminal_to_op_names=primitives,
    )
)

logging.basicConfig(level=logging.INFO)
neps.run(
    run_pipeline=run_pipeline,
    pipeline_space=pipeline_space,
    working_directory="results/hierarchical_architecture_example_new",
    max_evaluations_total=20,
)

previous_results, pending_configs = neps.status(
    "results/hierarchical_architecture_example_new"
)
