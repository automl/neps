import os

import matplotlib.pyplot as plt
import networkx as nx
from path import Path

from .graph import Graph


def graph_to_digraph(graph: Graph, edge_label: str = "op_name") -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(graph.nodes())
    for u, v in graph.edges():
        if isinstance(graph.edges[u, v]["op"], Graph):
            g.add_edge(u, v, op_name=graph.edges[u, v]["op"].name)
        else:
            # g.add_edge(u, v, op_name=graph.edges[u, v]['op'].get_op_name)
            g.add_edge(u, v, op_name=graph.edges[u, v][edge_label])
    return g


def draw_graph(
    graph,
    write_out: Path,
    edge_attr: bool = True,
    edge_label: str = "op_name",
    node_label: str = "op_name",
    fig_size: tuple = (10, 10),
):
    if isinstance(graph, Graph) and edge_attr:
        g = graph_to_digraph(graph, edge_label)
    else:
        g = graph

    plt.figure(figsize=fig_size)
    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    nx.drawing.nx_agraph.write_dot(g, dir_path / "test.dot")
    pos = nx.drawing.nx_agraph.graphviz_layout(g, prog="dot")
    nx.draw(g, pos, with_labels=edge_attr)
    if edge_attr:
        edge_labels = {e: g.edges[e][edge_label] for e in g.edges()}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
    else:
        node_labels = {n: g.nodes[n][node_label] for n in g.nodes()}
        nx.draw_networkx_labels(g, pos, labels=node_labels)
    plt.savefig(write_out)
    plt.close()
    os.remove(dir_path / "test.dot")


def drawNxTree(
    tree: nx.DiGraph,
    write_out: Path,
    node_label: str = "op_name",
    fig_size: tuple = (10, 10),
) -> None:
    plt.figure(figsize=fig_size)
    dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
    nx.drawing.nx_agraph.write_dot(tree, dir_path / "test.dot")
    pos = nx.drawing.nx_agraph.graphviz_layout(tree, prog="dot")
    nx.draw(
        tree,
        pos,
        with_labels=True,
        labels={k: v[node_label] for k, v in tree.nodes(data=True)},
        arrows=True,
    )
    plt.savefig(write_out)
    plt.close()
    os.remove(dir_path / "test.dot")
