from __future__ import annotations

import os

import networkx as nx
from path import Path
from torch import nn

from neps.search_spaces.graph_grammar import primitives as ops
from neps.search_spaces.graph_grammar import topologies as topos
from neps.search_spaces.graph_grammar.cfg import Grammar
from neps.search_spaces.graph_grammar.graph_grammar import GraphGrammar
from neps.search_spaces.graph_grammar.primitives import ResNetBasicblock

TERMINAL_2_OP_NAMES = {
    "id": ops.Identity(),
    "conv3x3": {"op": ops.ReLUConvBN, "kernel_size": 3, "stride": 1, "padding": 1},
    "conv1x1": {"op": ops.ReLUConvBN, "kernel_size": 1},
    "avg_pool": {"op": ops.AvgPool1x1, "kernel_size": 3, "stride": 1},
    "downsample": {"op": ResNetBasicblock, "stride": 2},
    "residual": topos.Residual,
    "diamond": topos.Diamond,
    "linear": topos.Linear,
    "diamond_mid": topos.DiamondMid,
    "down1": topos.DownsampleBlock,
}


class HierarchicalArchitectureExample(GraphGrammar):
    TRAINABLE = True
    in_channels = 3
    n_classes = 20

    def __init__(
        self,
        edge_attr: bool = False,
        id_parse_tree: bool = False,
        grammar: Grammar = None,
        base_channels: int = 64,
        out_channels: int = 512,
    ):
        if grammar is None:
            dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
            search_space_path = dir_path / "grammar.cfg"
            with open(search_space_path, encoding="utf-8") as f:
                productions = f.read()
            grammar = Grammar.fromstring(productions)

        super().__init__(
            grammar=grammar,
            terminal_to_op_names=TERMINAL_2_OP_NAMES,
            edge_attr=edge_attr,
            id_parse_tree=id_parse_tree,
        )

        self.base_channels = base_channels
        self.out_channels = out_channels
        self.name: str = ""

        self.nxTree = None
        self.string_tree = ""
        self.id = ""

    def setup(self, tree: nx.DiGraph, save_tree: bool = True):
        if save_tree:
            self.nxTree = tree  # pylint: disable=W0201

        self.build_graph_from_tree(
            tree=tree,
            terminal_to_torch_map=self.terminal_to_op_names,
        )
        self.prune_graph()

        self.name = "makrograph"

        if self.TRAINABLE:
            self._assign_channels()
            in_node = [n for n in self.nodes if self.in_degree(n) == 0][0]
            out_node = [n for n in self.nodes if self.out_degree(n) == 0][0]
            max_node_label = max(self.nodes())
            self.add_nodes_from([max_node_label + 1, max_node_label + 2])
            self.add_edge(max_node_label + 1, in_node)
            self.edges[max_node_label + 1, in_node].update(
                {
                    "op": ops.Stem(self.base_channels, C_in=self.in_channels),
                    "op_name": "Stem",
                }
            )
            self.add_nodes_from([out_node, max_node_label + 2])
            self.add_edge(out_node, max_node_label + 2)

            self.edges[out_node, max_node_label + 2].update(
                {
                    "op": ops.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Flatten(),
                        nn.Linear(self.out_channels, self.n_classes),
                    ),
                    "op_name": "Out",
                }
            )

            self.compile()
            self.update_op_names()

    def _assign_channels(self):
        in_node = [n for n in self.nodes if self.in_degree(n) == 0][0]
        for n in nx.topological_sort(self):
            for pred in self.predecessors(n):
                e = (pred, n)
                if pred == in_node:
                    channels = self.base_channels
                else:
                    pred_pred = list(self.predecessors(pred))[0]
                    channels = self.edges[(pred_pred, pred)]["C_out"]
                if self.edges[e]["op_name"] == "ResNetBasicblock":
                    self.edges[e].update({"C_in": channels, "C_out": channels * 2})
                else:
                    self.edges[e].update({"C_in": channels, "C_out": channels})

    def get_model_for_evaluation(self):
        self.clear_graph()
        if self.nxTree is None:
            self.nxTree = self.create_nx_tree(self.string_tree)
        if len(self.nodes()) == 0:
            self.setup(self.nxTree, save_tree=False)
        pytorch_model = self.to_pytorch()
        return pytorch_model

    def create_graph_from_string(self, child: str):
        g = HierarchicalArchitectureExample(
            edge_attr=self.edge_attr,
            id_parse_tree=self.id_parse_tree,
            grammar=self.grammars[0],
            base_channels=self.base_channels,
            out_channels=self.out_channels,
        )
        g.string_tree = child
        g.id = child
        g.create_representation(g.string_tree)
        return g
