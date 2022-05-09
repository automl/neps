from functools import partial
from itertools import combinations
from typing import List

import networkx as nx
import torch.nn as nn

from ..graph_grammar import primitives as ops
from ..graph_grammar.cfg import Grammar
from ..graph_grammar.graph_grammar import GraphGrammar
from ..graph_grammar.topologies import DenseNNodeDAG
from .primitives import ResNetBasicblock

TERMINAL_2_OP_NAMES = {
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

TERMINAL_2_GRAPH_REPR = {
    "Cell": None,
}


class GraphDenseParameter(GraphGrammar):

    OPTIMIZER_SCOPE = [
        "stage_1",
        "stage_2",
        "stage_3",
    ]

    def __init__(self, num_nodes: int, edge_choices: List[str], prior: dict = None):

        assert num_nodes > 1, "DAG has to have more than one node"
        self.num_nodes = num_nodes

        dense_cell = partial(DenseNNodeDAG, number_of_nodes=self.num_nodes)
        TERMINAL_2_OP_NAMES.update({"Cell": dense_cell})
        TERMINAL_2_GRAPH_REPR.update({"Cell": dense_cell().edge_list})

        edge_list = list(combinations(list(range(num_nodes)), 2))
        self.edge_choices = edge_choices

        productions = 'S -> "Cell" {}\nOPS -> {}'.format(
            "OPS " * len(edge_list),
            "".join([f'"{op}" | ' for op in self.edge_choices])[:-2],
        )
        grammar = Grammar.fromstring(productions)
        self.edge_attr = False

        super().__init__(
            grammar=grammar,
            terminal_to_op_names=TERMINAL_2_OP_NAMES,
            terminal_to_graph_edges=TERMINAL_2_GRAPH_REPR,
            edge_attr=self.edge_attr,
            prior=prior,
        )

        self.num_classes = self.NUM_CLASSES if hasattr(self, "NUM_CLASSES") else 10

        self.cell = None
        self.trainable = False
        self.graph_repr = None

    def reset(self):
        self.cell = None
        super().reset()

    def sample(self, user_priors: bool = False):  # pylint: disable=unused-argument
        super().sample()
        # pylint: disable=attribute-defined-outside-init
        self.nxTree = self.create_nx_tree(self.string_tree)
        self.setup(self.nxTree)

    def create_from_id(self, identifier: str):
        super().create_from_id(identifier)
        # pylint: disable=attribute-defined-outside-init
        self.nxTree = self.create_nx_tree(self.string_tree)
        self.setup(self.nxTree)

    def setup(self, tree: nx.DiGraph):
        # remove all nodes
        self.clear_graph()

        self.cell = self.build_graph_from_tree(
            tree=tree,
            terminal_to_torch_map=TERMINAL_2_OP_NAMES,
            return_cell=True,
        )
        self.id = self.string_tree  # pylint: disable=attribute-defined-outside-init
        self.graph_repr = self.to_graph_repr(self.cell, edge_attr=self.edge_attr)
        self.cell.name = "cell"
        self.cell = self.prune_graph(self.cell)
        self.graph_repr = self.prune_graph(self.graph_repr, False)

        if self.trainable:
            # Cell is on the edges
            # 1-2:               Preprocessing
            # 2-3, ..., 6-7:     cells stage 1
            # 7-8:               residual block stride 2
            # 8-9, ..., 12-13:   cells stage 2
            # 13-14:             residual block stride 2
            # 14-15, ..., 18-19: cells stage 3
            # 19-20:             post-processing

            total_num_nodes = 20
            self.add_nodes_from(range(1, total_num_nodes + 1))
            self.add_edges_from([(i, i + 1) for i in range(1, total_num_nodes)])

            channels = [16, 32, 64]

            #
            # operations at the edges
            #

            # preprocessing
            # TODO: make C_in parameterizable
            self.edges[1, 2].set("op", ops.Stem(C_out=channels[0], C_in=3))

            # stage 1
            for i in range(2, 7):
                self.edges[i, i + 1].set("op", self.cell.copy().set_scope("stage_1"))

            # stage 2
            self.edges[7, 8].set(
                "op", ResNetBasicblock(C_in=channels[0], C_out=channels[1], stride=2)
            )
            for i in range(8, 13):
                self.edges[i, i + 1].set("op", self.cell.copy().set_scope("stage_2"))

            # stage 3
            self.edges[13, 14].set(
                "op", ResNetBasicblock(C_in=channels[1], C_out=channels[2], stride=2)
            )
            for i in range(14, 19):
                self.edges[i, i + 1].set("op", self.cell.copy().set_scope("stage_3"))

            # post-processing
            self.edges[19, 20].set(
                "op",
                ops.Sequential(
                    nn.BatchNorm2d(channels[-1]),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(channels[-1], self.num_classes),
                ),
            )

            # set the ops at the cells (channel dependent)
            for c, scope in zip(channels, self.OPTIMIZER_SCOPE):
                # pylint: disable=cell-var-from-loop
                channels_ = {"C_in": c, "C_out": c}
                self.update_edges(
                    update_func=lambda edge: edge.data.update(channels_),
                    scope=scope,
                    private_edge_data=True,
                )

            self.compile()

        self._check_graph(self.graph_repr)

    def get_model_for_evaluation(self, trainable=False):
        self.trainable = trainable
        self.clear_graph()
        if self.nxTree is None:
            # pylint: disable=attribute-defined-outside-init
            self.nxTree = self.create_nx_tree(self.string_tree)
        if len(self.nodes()) == 0:
            self.setup(self.nxTree)
        pytorch_model = self.to_pytorch()
        return pytorch_model

    def create_graph_from_string(self, child: str):
        g = GraphDenseParameter(num_nodes=self.num_nodes, edge_choices=self.edge_choices)
        g.string_tree = child  # pylint: disable=attribute-defined-outside-init
        # pylint: disable=attribute-defined-outside-init
        g.nxTree = g.create_nx_tree(g.string_tree)
        g.setup(g.nxTree)
        return g

    def mutate(
        self,
        parent: GraphGrammar = None,
        mutation_rate: float = 1.0,
        mutation_strategy: str = "bananas",
    ):
        child = super().mutate(parent, mutation_rate, mutation_strategy)
        child.create_representation(child.string_tree)
        return child

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.num_nodes == other.num_nodes
            and self.edge_choices == other.edge_choices
            and self.graph_repr == other.graph_repr
        )

    def __repr__(self):
        return "<Graph, num_nodes: {}, edge_choices: {}>".format(
            self.num_nodes, self.edge_choices
        )
