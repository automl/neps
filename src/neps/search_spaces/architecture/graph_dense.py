from copy import deepcopy
from functools import partial
from itertools import combinations
from typing import Any, Dict, List

import torch.nn as nn

from . import primitives as ops
from .api import FunctionParameter

# from .cfg import Grammar
from .graph import Graph
from .primitives import ResNetBasicblock
from .topologies import DenseNNodeDAG

PRIMITIVES = {
    "Identity": ops.Identity(),
    "Zero": ops.Zero(stride=1),
    "ReLUConvBN3x3": {
        "op": ops.ReLUConvBN,
        "kernel_size": 3,
        "op_name": "ReLUConvBN3x3",
    },
    "ReLUConvBN1x1": {
        "op": ops.ReLUConvBN,
        "kernel_size": 1,
        "op_name": "ReLUConvBN1x1",
    },
    "AvgPool1x1": {"op": ops.AvgPool1x1, "kernel_size": 3, "stride": 1},
    "Cell": None,
}

TERMINAL_2_GRAPH_REPR: Dict[str, List[Any]] = {
    "Cell": [],
}


def build(
    _cell: Graph, optimizer_scope: List[str], in_channels: int = 3, num_classes: int = 10
):

    # Cell is on the edges
    # 1-2:               Preprocessing
    # 2-3, ..., 6-7:     cells stage 1
    # 7-8:               residual block stride 2
    # 8-9, ..., 12-13:   cells stage 2
    # 13-14:             residual block stride 2
    # 14-15, ..., 18-19: cells stage 3
    # 19-20:             post-processing

    total_num_nodes = 20
    channels = [16, 32, 64]

    graph = Graph()
    graph.add_nodes_from(range(1, total_num_nodes + 1))
    graph.add_edges_from([(i, i + 1) for i in range(1, total_num_nodes)])

    cell = Graph()
    cell.name = "cell"
    for u, v, d in _cell.edges(data=True):
        cell.add_edge(u, v)
        cell.edges[u, v].update(d)

    # preprocessing
    graph.edges[1, 2].set("op", ops.Stem(C_in=in_channels, C_out=channels[0]))

    # stage 1
    for i in range(2, 7):
        cell_copied = deepcopy(cell).set_scope("stage_1")
        graph.edges[i, i + 1].set("op", cell_copied)

    # stage 2
    graph.edges[7, 8].set(
        "op", ResNetBasicblock(C_in=channels[0], C_out=channels[1], stride=2)
    )

    for i in range(8, 13):
        cell_copied = deepcopy(cell).set_scope("stage_2")
        graph.edges[i, i + 1].set("op", cell_copied)

    # stage 3
    graph.edges[13, 14].set(
        "op", ResNetBasicblock(C_in=channels[1], C_out=channels[2], stride=2)
    )

    for i in range(14, 19):
        cell_copied = deepcopy(cell).set_scope("stage_3")
        graph.edges[i, i + 1].set("op", cell_copied)

    # post-processing
    graph.edges[19, 20].set(
        "op",
        ops.Sequential(
            nn.BatchNorm2d(channels[-1]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], num_classes),
        ),
    )
    # # # set the ops at the cells (channel dependent)
    for c, scope in zip(channels, optimizer_scope):
        graph.update_edges(
            update_func=lambda edge: edge.data.update(
                {"C_in": c, "C_out": c}  # pylint: disable=cell-var-from-loop
            ),
            scope=scope,
            private_edge_data=True,
        )

    return graph


class GraphDenseParameter:
    OPTIMIZER_SCOPE = [
        "stage_1",
        "stage_2",
        "stage_3",
    ]

    is_fidelity = False

    def __new__(
        cls,
        num_nodes: int,
        edge_choices: List[str],
        in_channels: int,
        num_classes: int,
        prior: dict = None,
    ):
        assert num_nodes > 1, "DAG has to have more than one node"

        dense_cell = partial(DenseNNodeDAG, number_of_nodes=num_nodes)
        PRIMITIVES.update({"Cell": dense_cell})
        TERMINAL_2_GRAPH_REPR.update({"Cell": dense_cell().edge_list})

        edge_list = list(combinations(list(range(num_nodes)), 2))

        productions = 'S -> "Cell" {}\nOPS -> {}'.format(
            "OPS " * len(edge_list),
            "".join([f'"{op}" | ' for op in edge_choices])[:-2],
        )
        # grammar = Grammar.fromstring(productions)

        build_fn = partial(
            build,
            optimizer_scope=cls.OPTIMIZER_SCOPE,
            in_channels=in_channels,
            num_classes=num_classes,
        )

        return FunctionParameter(
            set_recursive_attribute=build_fn,
            old_build_api=True,
            name=f"nb201_dense_{num_nodes}",
            structure=productions,
            primitives=PRIMITIVES,
            return_graph_per_hierarchy=False,
            constraint_kwargs=None,
            prior=prior,
        )
