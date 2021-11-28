from functools import partial
from itertools import combinations
from typing import List

import networkx as nx
import numpy as np

from comprehensive_nas.search_spaces.graph_dense.primitives import ResNetBasicblock
from comprehensive_nas.search_spaces.graph_grammar import primitives as ops
from comprehensive_nas.search_spaces.graph_grammar.cfg import Grammar
from comprehensive_nas.search_spaces.graph_grammar.graph_grammar import GraphGrammar
from comprehensive_nas.search_spaces.graph_grammar.topologies import DenseNNodeDAG

# pylint: disable=C0412
try:
    import torch.nn as nn
except ModuleNotFoundError:
    from comprehensive_nas.utils.torch_error_message import error_message

    raise ModuleNotFoundError(error_message)
# pylint: enable=C0412

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


class GraphDenseHyperparameter(GraphGrammar):
    OPTIMIZER_SCOPE = [
        "stage_1",
        "stage_2",
        "stage_3",
    ]

    def __init__(self, name: str, num_nodes: int, edge_choices: List[str]):

        assert num_nodes > 1, "DAG has to have more than one node"
        self.num_nodes = num_nodes
        TERMINAL_2_OP_NAMES.update(
            {"Cell": partial(DenseNNodeDAG, number_of_nodes=self.num_nodes)}
        )
        self.edge_list = list(combinations(list(range(num_nodes)), 2))
        self.edge_choices = edge_choices

        productions = 'S -> "Cell" {}\nOPS -> {}'.format(
            "OPS " * len(self.edge_list),
            "".join(['"{}" | '.format(op) for op in self.edge_choices])[:-2],
        )
        grammar = Grammar.fromstring(productions)
        self.edge_attr = False

        super().__init__(grammars=[grammar], edge_attr=self.edge_attr, name=name)

        self.num_classes = self.NUM_CLASSES if hasattr(self, "NUM_CLASSES") else 10

        self.string_tree: str = ""
        self.nxTree: nx.DiGraph = None
        self.cell = None
        self.graph_repr = None
        self.trainable = True
        self.id = None

        self.value = None
        self._id = -1

    def sample(self):
        self.string_tree = self.grammars[0].sampler(1)[0]
        self.nxTree = self.from_stringTree_to_nxTree(self.string_tree, self.grammars[0])
        self.setup(self.nxTree, save_trees_to_var=False)
        self._id = np.random.random()

    def setup(self, tree: nx.DiGraph, save_trees_to_var: bool = True):
        if save_trees_to_var:
            self.nxTree = tree
            self.string_tree = self.from_nxTree_to_stringTree(self.nxTree)

        # remove all nodes
        self.clear_graph()

        self.cell = self.build_graph_from_tree(
            tree=tree,
            terminal_to_torch_map=TERMINAL_2_OP_NAMES,
            return_cell=True,
        )
        self.graph_repr = self.to_graph_repr(self.cell, edge_attr=self.edge_attr)
        self.cell.name = "cell"
        self.id = self.string_tree
        self.cell = self.prune_graph(self.cell)
        self.graph_repr = self.prune_graph(self.graph_repr, edge_attr=self.edge_attr)

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
            self.edges[1, 2].set("op", ops.Stem(C_out=channels[0], C_in=1))

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
                channels = {"C_in": c, "C_out": c}
                self.update_edges(
                    update_func=lambda edge: edge.data.update(channels),
                    scope=scope,
                    private_edge_data=True,
                )

            self.compile()

        self._check_graph()

    def create_graph_from_string(self, child: str):
        g = GraphDenseHyperparameter(
            name=self.name, num_nodes=self.num_nodes, edge_choices=self.edge_choices
        )
        g.string_tree = child
        g.nxTree = g.from_stringTree_to_nxTree(g.string_tree, g.grammars[0])
        g.setup(g.nxTree, save_trees_to_var=False)
        return g

    def _check_graph(self):
        if len(self.graph_repr) == 0 or self.graph_repr.number_of_edges() == 0:
            raise ValueError("Invalid DAG")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return (
            self.name == other.name
            and self.num_nodes == other.num_nodes
            and self.edge_list == other.edge_list
            and self.edge_choices == other.edge_choices
            and self.graph_repr == other.graph_repr
        )

    def __hash__(self):
        return hash((self.name, self._id, self.num_nodes, tuple(self.edge_choices)))

    def __repr__(self):
        return "Graph {}-{:.07f}, num_nodes: {}, edge_choices: {}".format(
            self.name, self._id, self.num_nodes, self.edge_choices
        )

    def __copy__(self):
        return self.__class__(
            name=self.name, num_nodes=self.num_nodes, edge_choices=self.edge_choices
        )

    def get_graphs(self):
        return self.graph_repr

    def get_model_for_evaluation(self):
        self.clear_graph()
        if self.nxTree is None:
            self.nxTree = self.create_nx_tree(self.string_tree)
        if len(self.nodes()) == 0:
            self.setup(self.nxTree, save_trees_to_var=False)
        pytorch_model = self.to_pytorch()
        return pytorch_model

    def create_from_id(self, identifier: str):
        self.id = identifier
        self.string_tree = self.id
        if self.id_parse_tree:
            self.nxTree = self.create_nx_tree(self.string_tree)
        else:
            self.graph_repr = self.from_stringTree_to_graph_repr(
                self.string_tree,
                self.grammars[0],
                terminal_to_graph_edges=self.terminal_to_graph_repr,
                valid_terminals=self.terminal_to_op_names.keys(),
                edge_attr=self.edge_attr,
            )
