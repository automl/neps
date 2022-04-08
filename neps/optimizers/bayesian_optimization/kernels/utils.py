from typing import Tuple

import networkx as nx
import numpy as np


def transform_to_undirected(gr: list):
    """Transform a list of directed graphs by undirected graphs."""
    undirected_gr = []
    for g in gr:
        if not isinstance(g, nx.Graph):
            continue
        if isinstance(g, nx.DiGraph):
            undirected_gr.append(g.to_undirected())
        else:
            undirected_gr.append(g)
    return undirected_gr


def extract_configs(configs: list) -> Tuple[list, list]:
    """Extracts graph & HPs from configs objects

    Args:
        configs (list): Object holding graph and/or HPs

    Returns:
        Tuple[list, list]: list of graphs, list of HPs
    """
    N = len(configs)
    if N > 0 and "get_graphs" in dir(configs[0]):
        graphs = [c.get_graphs() for c in configs]
    elif N > 0 and "graph" in dir(configs[0]):
        graphs = [c.graph for c in configs]
    elif N > 0 and isinstance(configs, list):  # assumes that the list is meaningful!
        graphs = configs
    else:
        graphs = [None] * N

    if N > 0 and "get_hps" in dir(configs[0]):
        hps = [c.get_hps() for c in configs]
    elif N > 0 and "hps" in dir(configs[0]):
        hps = [c.hps for c in configs]
    else:
        hps = [None] * N

    return graphs, hps


def graph_metrics(graph, metric=None, directed=True):
    if directed:
        G = graph
    else:
        G = graph.to_undirected()

    # global metrics
    if metric == "avg_path_length":
        avg_path_length = nx.average_shortest_path_length(G)
        metric_score = avg_path_length

    elif metric == "density":
        density = nx.density(G)
        metric_score = density

    else:
        raise NotImplementedError

    return metric_score


def extract_configs_hierarchy(
    configs: list, d_graph_features: int, hierarchy_consider=None
) -> Tuple[list, list]:
    """Extracts graph & graph features from configs objects
    Args:
        configs (list): Object holding graph and/or graph features
        d_graph_features (int): Number of global graph features used; if d_graph_features=0, indicate not using global graph features
        hierarchy_consider (list or None): Specify graphs at which earlier hierarchical levels to be considered
    Returns:
        Tuple[list, list]: list of graphs, list of HPs
    """
    N = len(configs)

    if N > 0 and "get_graphs" in dir(configs[0]):
        combined_graphs = [c.get_graphs() for c in configs]
    # final architecture graph
    elif N > 0 and "graph" in configs[0].keys():
        combined_graphs = [c["graph"] for c in configs]  # get the list of final graphs

    if N > 0 and hierarchy_consider is not None and isinstance(combined_graphs[0], list):
        graphs = list(
            map(
                list,
                zip(
                    *[
                        [g[0][0]] + [hg for _, hg in g[0][1].items()]
                        for g in combined_graphs
                    ]
                ),
            )
        )
        ### full graph, 0th hierarchy (high-level, smallest), 1st hierarchy, 2nd hierarchy, 3rd hierarchy, ...
        ### graph gets bigger of hierarchies
        ### list shape: (1+4) x N
    else:
        graphs = [g[0][0] for g in combined_graphs]

    if N > 0 and d_graph_features > 0:
        # graph_features = [c['metafeature'] for c in configs]
        # these feature values are normalised between 0 and 1
        # the two graph features used are 'avg_path_length', 'density'
        graph_features = [
            [
                graph_metrics(g[0][0], metric="avg_path_length"),
                graph_metrics(g[0][0], metric="density"),
            ]
            for g in combined_graphs
        ]
        graph_features_array = np.vstack(graph_features)  # shape n_archs x 2 (nx(2+d_hp))
    else:
        # if not using global graph features of the final architectures, set them to None
        graph_features_array = [None] * N

    # get graph for earlier hierarchical levels
    # if N > 0 and hierarchy_consider is not None:
    #     # note the node feature for graph in earlier hierarchical level should be more coarse
    #     # i.e. for each graph
    #     # original_node_labels = nx.get_node_attributes(G, 'op_name')
    #     # new_node_labels = {k: v.split('(')[1] for k, v in original_node_labels.items() if
    #     #                    '(' in v and ')' in v}
    #     # nx.set_node_attributes(G, new_node_labels, name='op_name')
    #     all_hierarchy_graphs = [
    #         [c["hierarchy_graphs"][k] for c in configs]
    #         for k in configs[0]["hierarchy_graphs"].keys()
    #     ]
    #     # if we get hierarchial levels [0,1,2] , all_hierarchy_graphs is a list of 3 lists
    #     all_hierarchy_graphs = all_hierarchy_graphs + [
    #         graphs
    #     ]  # all_hierarchy_graphs is a list of 4 lists
    #     return all_hierarchy_graphs, graph_features_array
    # else:
    #     return graphs, graph_features_array
    return graphs, graph_features_array
