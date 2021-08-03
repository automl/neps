import logging

import matplotlib.pyplot as plt
import networkx as nx


def draw_graph(G: nx.Graph, node_label="op_name", edge_label="op_name"):
    pos = nx.spring_layout(G)
    nx.draw(G, pos)
    try:
        graph_type = G.graph_type
        if graph_type == "edge_attr":
            edge_label = nx.get_edge_attributes(G, edge_label)
            nx.draw_networkx_edge_labels(G, pos, edge_label)
        else:
            label = {n: i[node_label] for n, i in G.nodes(data=True)}
            nx.draw_networkx_labels(G, pos, labels=label)
    except AttributeError:
        logging.warning(
            "misc/draw_graph: G.graph_type is not found as a valid attribute. Falling back to the default"
            "node attribute graphs."
        )
        label = {n: i[node_label] for n, i in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=label)
    plt.show()
